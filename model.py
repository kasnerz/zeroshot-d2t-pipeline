#!/usr/bin/env python3

"""
Code for individual models.

Code for the ordering model is based on https://github.com/airKlizz/passage-ordering

If you use the ordering model, please cite also:
@inproceedings{calizzano2021ordering,
  title={Ordering sentences and paragraphs with pre-trained encoder-decoder transformers and pointer ensembles},
  author={Calizzano, R{\'e}mi and Ostendorff, Malte and Rehm, Georg},
  booktitle={Proceedings of the 21st ACM Symposium on Document Engineering},
  pages={1--9},
  year={2021}
}
"""

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

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

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
    BartModel,
    get_scheduler
)
from transformers.modeling_outputs import ModelOutput
from utils.ordering_utils import OrderingMixin

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
        outputs = self(**batch)
        loss = outputs["loss"]

        self.log('loss/train', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(**batch)
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
        parser.add_argument("--learning_rate", default=2e-5, type=float)
        parser.add_argument("--adam_epsilon", default=1e-9, type=float)
        parser.add_argument("--adam_beta1", default=0.9, type=float)
        parser.add_argument("--adam_beta2", default=0.997, type=float)
        parser.add_argument("--warmup_proportion", default=0.1, type=float)
        parser.add_argument("--label_smoothing", default=0.1, type=float)

        return parser

# =============================
# Start of code based on https://github.com/airKlizz/passage-ordering/blob/main/training/scripts/models/bart_simple.py
# =============================
class PointerHead(nn.Module):
    """Head for pointer ordering task."""

    def __init__(
        self,
        embed_dim,
        bias=True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.scaling = self.embed_dim ** -0.5

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _shape(self, tensor, seq_len, bsz):
        return tensor.contiguous().view(seq_len, bsz, self.embed_dim).transpose(0, 1)

    def forward(
        self,
        query,
        key,
    ):
        """Input shape: Time(SeqLen) x Batch x Channel"""
        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]

        q = self.q_proj(query) * self.scaling
        k = self.k_proj(key)

        q = self._shape(q, tgt_len, bsz)
        k = self._shape(k, -1, bsz)

        assert k is not None
        assert q is not None
        src_len = k.size(1)
        attn_weights = torch.bmm(q, k.transpose(1, 2))
        assert attn_weights.size() == (bsz, tgt_len, src_len)

        return attn_weights


@dataclass
class Seq2SeqOrderingOutput(ModelOutput):
    loss: Optional[torch.FloatTensor]
    logits: torch.FloatTensor = None
    last_hidden_state: Optional[List[torch.FloatTensor]] = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None

class OrdTrainingModule(D2TTrainingModule, OrderingMixin):
    def __init__(self, args, **kwargs):
        super().__init__(args, **kwargs)
        self.model = BartModel.from_pretrained(
            args.model_name,
            use_cache=True,
            return_dict=True
        )
        self.pointer = PointerHead(self.model.config.d_model)

        self.eos_token_id = self.tokenizer.eos_token_id
        self.pad_token_id = self.tokenizer.pad_token_id

    def is_sequence_ordering_model(self):
        return True

    def prepare_inputs_for_generation(
        self, decoder_input_ids, past, input_ids, attention_mask, use_cache, encoder_outputs, **kwargs
    ):
        return {
            "input_ids": input_ids,  # input_ids is needed for sequence mask
            "encoder_outputs": encoder_outputs,
            "past_key_values": past,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "use_cache": use_cache,   # change this to avoid caching (presumably for debugging)
        }

    def forward(self, 
            input_ids,
            attention_mask=None,
            encoder_outputs=None,
            decoder_input_ids=None,
            decoder_attention_mask=None,
            past_key_values=None,
            labels=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None
        ):

        if labels is not None:
            use_cache = False

        
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        use_cache = use_cache if use_cache is not None else self.args.use_cache

        encoder_sequence_last_hidden_state = outputs.encoder_last_hidden_state
        decoder_sequence_last_hidden_state = outputs.last_hidden_state

        encoder_sequence_attention_mask = (input_ids == self.tokenizer.eos_token_id).float()
        if use_cache:
            decoder_sequence_last_hidden_state = decoder_sequence_last_hidden_state[:, -1:]
            decoder_sequence_attention_mask = (decoder_input_ids[:, -1:] == self.eos_token_id).float()
        else:
            decoder_sequence_attention_mask = (decoder_input_ids == self.eos_token_id).float()

        sequence_attention_mask = torch.bmm(
            decoder_sequence_attention_mask.unsqueeze(2), encoder_sequence_attention_mask.unsqueeze(1)
        )

        logits = self.pointer(
            query=decoder_sequence_last_hidden_state.transpose(1, 0),
            key=encoder_sequence_last_hidden_state.transpose(1, 0),
        )
        # logits: shape = (bsz, decoder_len, encoder_len), X_ij = probability of j to be the sentence after i


        assert sequence_attention_mask.size() == logits.size(), f"{sequence_attention_mask.size()}, {logits.size()}"

        logits[sequence_attention_mask == 0] = float("-inf")

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        if return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqOrderingOutput(
            loss=loss,
            logits=logits,
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )

    @staticmethod
    def _reorder_cache(past, beam_idx):
        ((enc_out, enc_mask), past_key_values) = past
        reordered_past = []
        for layer_past in past_key_values:
            # get the correct batch idx from decoder layer's batch dim for cross and self-attn
            layer_past_new = {
                attn_key: _reorder_buffer(attn_cache, beam_idx) for attn_key, attn_cache in layer_past.items()
            }
            reordered_past.append(layer_past_new)

        new_enc_out = enc_out if enc_out is None else enc_out.index_select(0, beam_idx)
        new_enc_mask = enc_mask if enc_mask is None else enc_mask.index_select(0, beam_idx)

        past = ((new_enc_out, new_enc_mask), reordered_past)
        return past

    def get_encoder(self):
        return self.model.encoder


    def test_step(self, batch, batch_idx):
        raise NotImplementedError

# =============================
# End of code based on https://github.com/airKlizz/passage-ordering/blob/main/training/scripts/models/bart_simple.py
# =============================



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


class PCTrainingModule(D2TTrainingModule):
    def __init__(self, args, **kwargs):
        super().__init__(args, **kwargs)

        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            args.model_name,
            return_dict=True
        )
        add_special_tokens(self.tokenizer, self.model)