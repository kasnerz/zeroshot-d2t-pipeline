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

from transformers import (
    AdamW,
    AutoModelForSeq2SeqLM,
    AutoModelForTokenClassification,
    AutoModelForSequenceClassification,
    AutoConfig,
    AutoTokenizer,
    get_scheduler
)

logger = logging.getLogger(__name__)


def add_special_tokens(tokenizer, model):
    special_tokens_dict = {'additional_special_tokens': ['<sep>']}
    tokenizer.add_special_tokens(special_tokens_dict)

    if model is not None:
        model.resize_token_embeddings(len(tokenizer))


class D2TTrainingModule(pl.LightningModule):
    def __init__(self, args, **kwargs):
        super().__init__()
        self.args = args
        self.save_hyperparameters()
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name,
                                                       use_fast=True)

    def forward(self, **inputs):
        out = self.model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=inputs["labels"]
        )
        return {"loss": out["loss"], "logits": out["logits"]}

    def training_step(self, batch, batch_idx):
        labels = batch["labels"]
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        outputs = self(input_ids=input_ids,
                       attention_mask=attention_mask,
                       labels=labels)
        loss = outputs["loss"]

        self.log('loss/train', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        labels = batch["labels"]
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        outputs = self(input_ids=input_ids,
                       attention_mask=attention_mask,
                       labels=labels)
        loss = outputs["loss"]

        self.log('loss/val', loss, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        out = self.model.generate(batch["input_ids"], 
            max_length=self.args.max_length,
            num_beams=1,
            num_return_sequences=1)
        
        out = self.tokenizer.batch_decode(out, 
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        for idx, o in enumerate(out):
            logger.info(f"[{batch_idx * len(out) + idx}] {o}")
            self.out_file_handle.write(o + "\n")

    def configure_optimizers(self):
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.args.learning_rate,
            eps=self.args.adam_epsilon,
            betas=(self.args.adam_beta1, self.args.adam_beta2)
        )

        total_steps = self.args.max_steps if self.args.max_steps else len(
            self.train_dataloader()) * self.args.max_epochs
        warmup_steps = total_steps * self.args.warmup_proportion

        scheduler = get_scheduler(
            "polynomial",
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        logger.info(f"Using Adam optimizer")
        logger.info(f"Learning rate: {self.args.learning_rate}")
        logger.info(f"Total steps: {total_steps}")
        logger.info(f"Warmup steps: {warmup_steps}")

        scheduler = {
            'scheduler': scheduler,
            'interval': 'step',
            'frequency': 1
        }
        return [optimizer], [scheduler]
        

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser],
                                         add_help=False)
        parser.add_argument("--learning_rate", default=3e-5, type=float)
        parser.add_argument("--adam_epsilon", default=1e-9, type=float)
        parser.add_argument("--adam_beta1", default=0.9, type=float)
        parser.add_argument("--adam_beta2", default=0.997, type=float)
        parser.add_argument("--warmup_proportion", default=0.1, type=float)
        parser.add_argument("--label_smoothing", default=0.1, type=float)

        return parser


class AggPairsTrainingModule(D2TTrainingModule):
    def __init__(self, args, **kwargs):
        super().__init__(args, **kwargs)

        self.model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name,
            return_dict=True,
            num_labels=2
        )

    def test_step(self, batch, batch_idx):
        raise NotImplementedError


class AggTrainingModule(D2TTrainingModule):
    def __init__(self, args, **kwargs):
        super().__init__(args, **kwargs)

        self.model = AutoModelForTokenClassification.from_pretrained(
            args.model_name,
            return_dict=True,
            num_labels=2
        )

    def test_step(self, batch, batch_idx):
        raise NotImplementedError


class OrdTrainingModule(D2TTrainingModule):
    def __init__(self, args, **kwargs):
        super().__init__(args, **kwargs)


class PCTrainingModule(D2TTrainingModule):
    def __init__(self, args, **kwargs):
        super().__init__(args, **kwargs)

        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            args.model_name,
            return_dict=True
        )
        add_special_tokens(self.tokenizer, self.model)