#!/usr/bin/env python

import os
import argparse
import contextlib
import logging
import sacrebleu
import sys

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO, datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)


def evaluate_bleu(args):
    with open(args.hyp_file) as f:
        hyps = f.read().rstrip("\n").split("\n")

    with open(args.ref_file) as f:
        refs = f.read().rstrip("\n").split("\n")
        # refs = f.read().rstrip("\n").split("\n\n")
        ref = [ref_group.split("\n") for ref_group in refs]

        max_refs = max([len(refs) for refs in ref])
        refs_transposed = [[refs[i] if len(refs) > i else None for refs in ref] 
                                        for i in range(max_refs)]

    bleu_score = sacrebleu.corpus_bleu(hyps, refs_transposed)
    print(bleu_score)


def evaluate_accuracy(args):
    with open(args.hyp_file) as f:
        hyps = f.read().rstrip("\n").split("\n")

    with open(args.ref_file) as f:
        # refs = f.read().rstrip("\n").split("\n\n")
        refs = f.read().rstrip("\n").split("\n")
        refs = [ref_group.split("\n") for ref_group in refs]

    assert len(hyps) == len(refs)

    correct = 0
    total = 0

    for hyp, ref_list in zip(hyps, refs):
        if hyp in ref_list:
            correct += 1
        total += 1

    acc = correct/total
    print(f"Accuracy: {acc:.2f} ({correct}/{total})")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref_file", type=str, required=True,
        help="Dataset to compare the results with.")
    parser.add_argument("--hyp_file", type=str, default=None,
        help="File with output from the model.")
    parser.add_argument("--mode", type=str, default="bleu",
        help="Evaluation mode (\"bleu\"=BLEU-2 / \"acc\"=accuracy)")
    args = parser.parse_args()

    if args.mode == "bleu":
        evaluate_bleu(args)
    elif args.mode == "acc":
        evaluate_accuracy(args)
    else:
        raise ValueError(f"Unknown evaluation mode: {args.mode}")