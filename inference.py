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

from model import (
    D2TTrainingModule, 
    AggTrainingModule, 
    PCTrainingModule,
)
from model import add_special_tokens
from transformers import (
    AutoConfig,
    AutoTokenizer,
)


logger = logging.getLogger(__name__)

class D2TInferenceModule:
    def __init__(self, args, model_path, training_module_cls):
        self.args = args
        self.model = training_module_cls.load_from_checkpoint(model_path)
        self.model.freeze()
        logger.info(f"Loaded model from {model_path}")

        self.model_name = self.model.model.name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name,
                                                       use_fast=True)

    def predict(self, s, beam_size=1):
        inputs = self.tokenizer(s, return_tensors='pt')

        if hasattr(self.args, "gpus") and self.args.gpus > 0:
            self.model.cuda()
            for key in inputs.keys():
                inputs[key] = inputs[key].cuda()
        else:
            logger.warning("Not using GPU")

        return self.generate(inputs["input_ids"], beam_size)


    def generate(self, input_ids, beam_size):
        out = self.model.model.generate(input_ids, 
            max_length=self.args.max_length,
            num_beams=beam_size,
            num_return_sequences=beam_size
        )

        sentences = self.tokenizer.batch_decode(out, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=True
        )
            
        return sentences


class AggInferenceModule(D2TInferenceModule):
    def __init__(self, args, model_path):
        super().__init__(args, model_path=model_path, training_module_cls=AggTrainingModule)


    def predict(self, sents, beam_size=1):
        text = f" {self.tokenizer.sep_token} ".join(sents)

        inputs = self.tokenizer(text,
            max_length=self.args.max_length,
            truncation=True,
            return_tensors='pt'
        )
        if hasattr(self.args, "gpus") and self.args.gpus > 0:
            self.model.cuda()
            for key in inputs.keys():
                inputs[key] = inputs[key].cuda()
        else:
            logger.warning("Not using GPU")

        logits = self.model.model.forward(inputs["input_ids"])["logits"]
        preds = torch.argmax(logits, axis=2)
        seps = preds[inputs["input_ids"] == self.tokenizer.sep_token_id][:-1]

        return seps.tolist()


class PCInferenceModule(D2TInferenceModule):
    def __init__(self, args, model_path):
        super().__init__(args, model_path=model_path, training_module_cls=PCTrainingModule)

        add_special_tokens(self.tokenizer, None)
