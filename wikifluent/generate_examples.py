#!/usr/bin/env python3

import argparse
import logging
import numpy as np
import os
import re
import torch
import pytorch_lightning as pl

from pprint import pprint as pp
import sys
sys.path.insert(0, "..")
from model import D2TInferenceModule

from utils.tokenizer import Tokenizer
from allennlp.predictors.predictor import Predictor
import allennlp_models.tagging
import nltk
import json
import random

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO, datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

random.seed(42)

def normalize(line):
    # remove dash-related artefacts
    line = line.replace(" - ", "-") \
               .replace("–", " – ")

    return line

def split_sentence(line, dm):
    all_sents = []
    to_split = [(line, 0)]

    max_nr_of_splits = random.randint(0,2)

    while to_split:
        sent, nr_of_splits = to_split.pop(0)

        # non-deterministic number of splits
        if nr_of_splits > max_nr_of_splits:
            all_sents.append(sent)
            continue

        out = dm.predict(sent, beam_size=1)[0]
        sents = nltk.sent_tokenize(out)

        # a sentence could not be split successfully
        if out == sent \
            or len(sents) != 2 \
            or sents[0] == sents[1] \
            or len(sents[0]) >= len(sent) \
            or len(sents[1]) >= len(sent) \
            or len(sents[0]) + len(sents[1]) < len(sent):
            # use the original sentence
            all_sents.append(sent)
        # a split was successful
        else:
            to_split.insert(0, (sents[1], nr_of_splits+1))
            to_split.insert(0, (sents[0], nr_of_splits+1))
    return all_sents


def resolve_coref(text, coref_predictor, tokenizer):
    res = coref_predictor.predict(document=text)

    replacements = [[tok, None] for tok in res["document"]]
    antencendents = res["predicted_antecedents"]
    spans = res["top_spans"]

    assert len(antencendents) == len(spans)

    for i, ant in enumerate(antencendents):
        if ant != -1:
            pronoun_span = spans[i]
            pronoun_span_beg = pronoun_span[0]
            pronoun_span_end = pronoun_span[1]
            coref_span = replacements[spans[ant][1]][1] if replacements[spans[ant][1]][1] is not None else spans[ant]
            replacements[pronoun_span_beg][1] = coref_span

            for j in range(pronoun_span_beg, pronoun_span_end):
                replacements[j][1] = "DEL"

    resolved_tokens = []

    for token, repl in replacements:
        if repl is None:
            resolved_tokens.append(token)
        elif repl == "DEL":
            continue
        else:
            resolved_tokens += res["document"][repl[0]:repl[1]+1]

    resolved_text = " ".join(resolved_tokens)
    resolved_text = tokenizer.detokenize(resolved_text)

    return resolved_text


def create_example(line, coref_predictor, tokenizer):
    line = line.rstrip()
    line = normalize(line)
    sentences = nltk.sent_tokenize(line)
    sents_all = []
    aggregation = []

    for i, sent in enumerate(sentences):
        sents = split_sentence(sent, dm)
        sents_all += sents

        # TODO change to separators
        aggregation += [i] * len(sents)
    
    text = " ".join(sents_all)
    text = resolve_coref(text, coref_predictor, tokenizer)
    text = normalize(text)
    
    out = nltk.sent_tokenize(text)
    out = [sent[0].upper() + sent[1:] for sent in out]

    # wrong sentence count
    if len(out) != len(aggregation):
        return None

    example = {
        "in" : line,
        "out" : out,
        "aggregation" : aggregation
    }
    logger.info("---------------------")
    logger.info(idx)
    logger.info(line)
    logger.info(out)
    logger.info(aggregation)

    return example


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_filename", type=str, required=True,
        help="Input file with sentences to split.")
    parser.add_argument("--out_dir", type=str, default="chunks",
        help="Output directory.")
    parser.add_argument("--split_model", type=str, default="../experiments/wikisplit/model.ckpt",
        help="Path to the splitting model.")
    parser.add_argument("--gpus", type=int, default=1,
        help="GPUs used for the splitting model.")
    parser.add_argument("--start_line", type=int, default=None,
        help="Line of the input file where to start (used for parallelization).")
    parser.add_argument("--end_line", type=int, default=None,
        help="Line of the input file where to end (used for parallelization).")
    parser.add_argument("--seed", default=42, type=int,
        help="Random seed.")
    parser.add_argument("--max_threads", default=8, type=int,
        help="Maximum number of threads.")
    parser.add_argument("--max_length", type=int, default=1024,
        help="Maximum number of tokens per example")
    args = parser.parse_args()

    logger.info(args)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(args.max_threads)

    dm = D2TInferenceModule(args, model_path=args.split_model)
    out_filename = os.path.join(args.out_dir, "out.jsonl")

    tokenizer = Tokenizer()
    coref_predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/coref-spanbert-large-2021.03.10.tar.gz")
    out = []

    if args.start_line is not None and args.end_line is not None:
        out_filename += f"_{args.start_line:07d}-{args.end_line:07d}"

    with open(args.in_filename) as in_file, open(out_filename, "w") as out_file:
        for idx, line in enumerate(in_file.readlines()):
            # process only lines in the given range
            if args.start_line and idx < args.start_line:
                continue
            if args.end_line and idx >= args.end_line:
                break
            try:
                example = create_example(line, coref_predictor, tokenizer)
                if example is not None:
                    out_file.write(json.dumps(example) + "\n")
            except Exception as e:
                logger.error(f"Couldn't process text: {line}")
                logger.error(e)