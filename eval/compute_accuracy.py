#!/usr/bin/env python3
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import json
import torch
import sys
import os
import nltk
import argparse


class AccuracyMetric:
    def __init__(self, args):
        self.args = args
        self.nli_map = {
            0 : "CONTRADICTION",
            1 : "NEUTRAL",
            2 : "ENTAILMENT"
        }
        model_name = "roberta-large-mnli"
        self.tokenizer = RobertaTokenizer.from_pretrained(
            model_name,
            use_fast=True
        )
        self.roberta = RobertaForSequenceClassification.from_pretrained(
            model_name,
            return_dict=True
        )
        if self.args.gpu:
            self.roberta = self.roberta.cuda()

        self.hyp_file = open(args.hyp_file)
        self.ref_file = open(args.ref_file)
        self.acc_file = open(args.hyp_file + ".acc", "w")


    def predict(self, ref, hyp):
        inputs = self.tokenizer(ref, hyp, return_tensors="pt")
        if self.args.gpu:
            for key in inputs.keys():
                inputs[key] = inputs[key].cuda()

        out = self.roberta(**inputs)
        prediction = torch.argmax(out["logits"], axis=1).item()

        return prediction

    def run(self):
        lines = self.hyp_file.readlines()
        ref_data = json.load(self.ref_file)["data"]

        assert(len(lines) == len(ref_data))
        N = len(lines)

        for idx, (line, example) in enumerate(zip(lines, ref_data)):
            omissions = 0
            hallucinations = 0
            ref = line
            hyps = example["text"]
            hyp_sent = " ".join(hyps)

            for hyp in hyps:
                prediction = self.predict(ref, hyp)

                if self.nli_map[prediction] != "ENTAILMENT":
                    omissions += 1

            prediction = self.predict(hyp_sent, ref)

            if self.nli_map[prediction] != "ENTAILMENT":
                hallucinations += 1

            print(f"{idx}/{N} {len(hyps)} {omissions} {hallucinations}")
            self.acc_file.write(f"{len(hyps)} {omissions} {hallucinations}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref_file", type=str, required=True,
        help="File with references.")
    parser.add_argument("--hyp_file", type=str, default=None,
        help="File with output from the model.")
    parser.add_argument("--gpu", action="store_true", 
        help="Use gpu.")
    args = parser.parse_args()

    
    am = AccuracyMetric(args)
    am.run()