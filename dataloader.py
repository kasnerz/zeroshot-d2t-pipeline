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

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name,
                                                       use_fast=True)
        self.special_tokens = special_tokens

        if special_tokens:
            add_special_tokens(self.tokenizer, None)


    def setup(self, stage):
        data_dir = os.path.join("data", self.args.dataset)

        if stage == "fit":
            splits = ["train", "dev"]
        elif stage == "predict":
            splits = [self.args.split]

        self.dataset = {
            split : load_dataset("json", 
                data_files=os.path.join(data_dir, f"{split}.json"),
                field="data",
                split="train") for split in splits
        }

        for split in self.dataset.keys():
            for column in self.dataset[split].column_names:
                if column not in ["text", "labels"]:
                    self.dataset[split] = self.dataset[split].remove_columns(column)

            self.dataset[split] = self.dataset[split].map(
                self._convert_to_features,
                batched=True,
                remove_columns=['labels'],
            )
            self.dataset[split].set_format(
                type="torch",
                columns=[
                    "attention_mask", "input_ids", "labels"
                ])

    def _convert_to_features(self, example_batch, indices=None):
        text = example_batch["text"]

        features = self.tokenizer(
            text,
            max_length=self.args.max_length,
            truncation=True
        )
        features["labels"] = self.tokenizer(
            example_batch["labels"]
        )["input_ids"]

        return features

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


class AggDataModule(D2TDataModule):
    def __init__(self, args, model_name=None):
        super().__init__(args, model_name)


    def setup(self, stage):
        data_dir = os.path.join("data", self.args.dataset)

        if stage == "fit":
            splits = ["train", "dev"]
        elif stage == "predict":
            splits = ["dev", "test"]

        self.dataset = {
            split : load_dataset("json", data_files=os.path.join(data_dir, f"{split}.json"), field="data", split="train") 
                    for split in splits
        }

        for split in self.dataset.keys():
            self.dataset[split] = self.dataset[split].map(
                self._convert_to_features,
                batched=True,
                remove_columns=['labels'],
            )
            self.dataset[split].set_format(
                type="torch",
                columns=[
                    "attention_mask", "input_ids", "labels"
                ])


    def _convert_to_features(self, example_batch, indices=None):
        text = example_batch["text"]
        labels = example_batch["labels"]

        text = [f" {self.tokenizer.sep_token} ".join(group) for group in text]

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




# class AggPairsDataModule(D2TDataModule):
#     def __init__(self, args, model_name=None):
#         super().__init__(args, model_name)

#     def setup(self, stage):
#         data_dir = os.path.join("data", self.args.dataset)

#         if stage == "fit":
#             splits = ["train", "dev"]
#         elif stage == "predict":
#             splits = ["dev", "test"]

#         self.dataset = {
#             split : load_dataset("json", 
#                 data_files=os.path.join(data_dir, f"{split}.json"),
#                 field="data",
#                 split="train") for split in splits
#         }

#         for split in self.dataset.keys():
#             for column in self.dataset[split].column_names:
#                 if column not in ["s1", "s2", "label"]:
#                     self.dataset[split] = self.dataset[split].remove_columns(column)

#             self.dataset[split] = self.dataset[split].map(
#                 self._convert_to_features,
#                 batched=True,
#                 remove_columns=["s1", "s2", "label"],
#             )
#             self.dataset[split].set_format(
#                 type="torch",
#                 columns=[
#                     "attention_mask", "input_ids", "labels"
#                 ])

#     def _convert_to_features(self, example_batch, indices=None):
#         s1 = example_batch["s1"]
#         s2 = example_batch["s2"]

#         features = self.tokenizer(
#             s1,
#             s2,
#             max_length=self.args.max_length,
#             truncation=True
#         )
#         features["labels"] = example_batch["label"]

#         return features

#     def _pad_sequence(self, batch):
#         batch_collated = {
#             "labels" : torch.tensor([x["labels"] for x in batch])
#         }

#         paddings = {
#             "input_ids" : self.tokenizer.pad_token_id,
#             "attention_mask" : 0,
#         }
#         for key in ["input_ids", "attention_mask"]:
#             elems = [x[key] for x in batch]
#             elems_pad = pad_sequence(elems, batch_first=True, padding_value=paddings[key])
#             batch_collated[key] = elems_pad

#         return batch_collated


# TODO
class OrdDataModule(D2TDataModule):
    def __init__(self, args, model_name=None):
        super().__init__(args, model_name)



class PCDataModule(D2TDataModule):
    def __init__(self, args, model_name=None):
        super().__init__(args, model_name, special_tokens=True)


    def _convert_to_features(self, example_batch, indices=None):
        text = example_batch["text"]

        if type(text[0]) is list:
            # input is a list of separate sentences -> join 
            text = [" ".join(group) for group in text]

        features = self.tokenizer(
            text,
            max_length=self.args.max_length,
            truncation=True
        )
        features["labels"] = self.tokenizer(
            example_batch["labels"]
        )["input_ids"]

        return features