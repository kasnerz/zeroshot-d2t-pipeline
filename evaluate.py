#!/usr/bin/env python

import os
import argparse
import contextlib
import logging
import sacrebleu
import sys
sys.path.insert(0, 'e2e_metrics')
import measure_scores

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO, datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

def evaluate_e2e_metrics(args):
    data_src, data_ref, data_sys = measure_scores.load_data(args.ref_file, args.hyp_file, None)

    if args.lowercase:
        data_src = [sent.lower() for sent in data_src]
        data_ref = [[sent.lower() for sent in sent_list] for sent_list in data_ref]
        data_sys = [sent.lower() for sent in data_sys]

    measure_scores.evaluate(data_src, data_ref, data_sys, print_as_table=True, print_table_header=True)


def evaluate_sacrebleu(args):
    with open(args.hyp_file) as f:
        hyps = f.read().rstrip("\n").split("\n")

    with open(args.ref_file) as f:
        refs = f.read().rstrip("\n").split("\n\n")
        ref = [ref_group.split("\n") for ref_group in refs]

        max_refs = max([len(refs) for refs in ref])
        refs_transposed = [[refs[i] if len(refs) > i else None for refs in ref] 
                                        for i in range(max_refs)]

    bleu_score = sacrebleu.corpus_bleu(hyps, refs_transposed, lowercase=args.lowercase)

    print(bleu_score)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref_file", type=str, required=True,
        help="Dataset to compare the results with.")
    parser.add_argument("--hyp_file", type=str, default=None,
        help="File with output from the model.")
    parser.add_argument("--lowercase", action="store_true", default=False,
        help="Evaluate on lower-cased files.")
    parser.add_argument("--use_e2e_metrics", action="store_true", default=False,
        help="Evaluate using E2E metrics instead of SacreBLEU.")
    args = parser.parse_args()

    if args.use_e2e_metrics:
        evaluate_e2e_metrics(args)
    else:
        evaluate_sacrebleu(args)
