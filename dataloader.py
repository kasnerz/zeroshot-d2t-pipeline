#!/usr/bin/env python3

import numpy as np
import os
import logging
import re
import json
import argparse
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

from torch.utils.data import DataLoader, Dataset
from data import get_dataset_class
from collections import defaultdict
from datasets import load_dataset, dataset_dict, Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer
from model import add_special_tokens

logger = logging.getLogger(__name__)

class D2TDataModule(pl.LightningDataModule):
    def __init__(self, args, model_name=None, special_tokens=False):
        super().__init__()
        self.args = args
        self.model_name = model_name or self.args.model_name

        # disable the "huggingface/tokenizers: The current process just got forked" warning
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        self.special_tokens = special_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name,
                                                       use_fast=True)
        if special_tokens:
            add_special_tokens(self.tokenizer, None)


    def setup(self, stage):
        data_dir = os.path.join(self.args.dataset)

        if stage == "fit":
            splits = ["train", "dev"]
        elif stage == "predict":
            splits = [self.args.split]
        
        raw_dataset = {
            split : load_dataset("json", 
                data_files=os.path.join(data_dir, f"{split}.json"),
                field="data",
                split="train") for split in splits
        }
        self.dataset = self._process_raw_dataset(raw_dataset)

    
    def _process_raw_dataset(self, raw_dataset):
        dataset = {}

        for split in raw_dataset.keys():
            dataset[split] = raw_dataset[split].map(
                self._convert_to_features,
                batched=True
            )
            dataset[split].set_format(
                type="torch",
                columns=[
                    "attention_mask", "input_ids", "labels"
                ])
        return dataset


    def _convert_to_features(self, example_batch, indices=None):
        return NotImplementedError


    def train_dataloader(self):
        return DataLoader(
            self.dataset['train'],
            batch_size=self.args.batch_size,
            num_workers=self.args.max_threads,
            collate_fn=self._pad_sequence,
        )

    def val_dataloader(self):
        return DataLoader(self.dataset['dev'],
             batch_size=self.args.batch_size,
             num_workers=self.args.max_threads,
             collate_fn=self._pad_sequence
        )

    def test_dataloader(self):
        return DataLoader(self.dataset['test'],
          batch_size=self.args.batch_size,
          num_workers=self.args.max_threads,
          collate_fn=self._pad_sequence
        )

    def _pad_sequence(self, batch):
        batch_collated = {}

        paddings = {
            "input_ids" : self.tokenizer.pad_token_id,
            "attention_mask" : 0,
            "labels" : -100
        }
        for key in ["input_ids", "attention_mask", "labels"]:
            elems = [x[key] for x in batch]
            elems_pad = pad_sequence(elems, batch_first=True, padding_value=paddings[key])
            batch_collated[key] = elems_pad

        return batch_collated


# TODO
class OrdDataModule(D2TDataModule):
    def __init__(self, args, model_name=None):
        super().__init__(args, model_name)




class AggDataModule(D2TDataModule):
    def __init__(self, args, model_name=None):
        super().__init__(args, model_name)


    def _convert_to_features(self, example_batch, indices=None):
        sents = example_batch["sents"]
        labels = example_batch["sep"]

        text = [f" {self.tokenizer.sep_token} ".join(group) for group in sents]

        features = self.tokenizer(text,
            max_length=self.args.max_length,
            truncation=True
        )
        do_not_care_label = -100
        labels_batch = []

        for b in range(len(features["input_ids"])):
            input_ids = torch.tensor(features["input_ids"][b])
            labels_example = labels[b]

            # expand labels in the form [-100, -100, ..., label_i, ..., -100]
            # where -100 means not computing loss for the token and label_i
            # is at the position of i-th sentence separator
            labels_expanded = torch.clone(input_ids)
            labels_expanded[labels_expanded != self.tokenizer.sep_token_id] = do_not_care_label

            # in RoBERTa, eos_token == sep_token, but we do not care about EOS
            labels_expanded[-1] = do_not_care_label

            for i, index in enumerate(torch.nonzero(labels_expanded == self.tokenizer.sep_token_id)):
                labels_expanded[index] = labels_example[i]

            assert len(labels_expanded) == len(input_ids)
            labels_batch.append(labels_expanded.tolist())

        features['labels'] = labels_batch

        return features



class PCDataModule(D2TDataModule):
    def __init__(self, args, model_name=None):
        super().__init__(args, model_name, special_tokens=True)

    def _convert_to_features(self, example_batch, indices=None):
        seps_all = example_batch["sep"]
        sents_all = example_batch["sents"]
        out = []

        for sents, seps in zip(sents_all, seps_all):
            example = [sents[0]]
            for sep, sent in zip(seps, sents[1:]):
                if sep == 1:
                    example.append(self.tokenizer.sep_token)
                example.append(sent)

            text = " ".join(example)
            out.append(text)

        features = self.tokenizer(
            out,
            max_length=self.args.max_length,
            truncation=True
        )
        features["labels"] = self.tokenizer(
            example_batch["text"]
        )["input_ids"]

        return features


class PCAggDataModule(D2TDataModule):
    def __init__(self, args, model_name=None):
        super().__init__(args, model_name, special_tokens=True)


    def _convert_to_features(self, example_batch, indices=None):
        text = example_batch["sents"]
        text = [" ".join(group) for group in text]

        features = self.tokenizer(
            text,
            max_length=self.args.max_length,
            truncation=True
        )
        features["labels"] = self.tokenizer(
            example_batch["text"]
        )["input_ids"]

        return features


class PCOrdAggDataModule(D2TDataModule):
    def __init__(self, args, model_name=None):
        super().__init__(args, model_name, special_tokens=True)
        random.seed(args.seed)

    def _convert_to_features(self, example_batch, indices=None):
        sents_all = []

        for sents in example_batch["sents"]:
            random.shuffle(sents)
            sents_all.append(sents)

        text = [" ".join(group) for group in sents_all]

        features = self.tokenizer(
            text,
            max_length=self.args.max_length,
            truncation=True
        )
        features["labels"] = self.tokenizer(
            example_batch["text"]
        )["input_ids"]

        return features