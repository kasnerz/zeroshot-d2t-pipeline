#!/usr/bin/env python3
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import json
import torch
import sys
import os
import argparse
import logging
import numpy as np

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO, datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

nli_map = {
    0 : "CONTRADICTION",
    1 : "NEUTRAL",
    2 : "ENTAILMENT"
}

def roberta_classify(a, b, model, tokenizer, use_gpu):
    with torch.no_grad():
        inputs = tokenizer("%s </s></s> %s" % (a, b), return_tensors="pt")
        labels = torch.tensor([1]).unsqueeze(0)  # batch size = 1
        if use_gpu:
            inputs = inputs.to('cuda')
            labels = labels.to('cuda')
        output = model(**inputs, labels=labels)["logits"]
        if use_gpu:
            output = output.cpu()

        output = np.argmax(torch.nn.Softmax(dim=1)(output).detach().numpy()[0])
        return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_dir", type=str, default="chunks",
        help="Input jsonl file.")
    parser.add_argument("--filename", type=str, required=True,
        help="Input jsonl file.")
    parser.add_argument("--out_dir", type=str, default="chunks_filtered",
        help="Output directory with processed jsonl files.")
    parser.add_argument("--use_gpu", action="store_true",
        help="Use GPU for the NLI model.")
    args = parser.parse_args()

    logger.info(args)
    os.makedirs(args.out_dir, exist_ok=True)

    use_gpu = args.use_gpu

    tokenizer = RobertaTokenizer.from_pretrained('roberta-large-mnli')
    model = RobertaForSequenceClassification.from_pretrained('roberta-large-mnli')

    if use_gpu:
        model.to('cuda')

    accepted = 0
    total = 0

    print(f"Reading {args.filename}")

    with open(os.path.join(args.in_dir, args.filename), "r") as in_file, \
         open(os.path.join(args.out_dir, args.filename), "w") as out_file:

        for line in in_file.readlines():
            total += 1
            j = json.loads(line)

            in_sent = j["in"]
            out_sents = j["out"]

            for out_sent in out_sents:
                prediction = roberta_classify(in_sent, out_sent, model, tokenizer, use_gpu)
                # print(in_sent, "->", out_sent)
                # print(nli_map[prediction])
                # print("--------------")

                if nli_map[prediction] != "ENTAILMENT":
                    # print("REJECTED")
                    # print("================")
                    continue

            prediction = roberta_classify(" ".join(out_sents), in_sent, model, tokenizer, use_gpu)

            # print(" ".join(out_sents), "->", in_sent)
            # print(nli_map[prediction])
            # print("--------------")

            if nli_map[prediction] != "ENTAILMENT":
                # print("REJECTED")
                # print("================")
                continue

            # print("ACCEPTED")
            # print("================")
            accepted += 1
            out_file.write(line)

            if total % 100 == 0:
                print(f"Accepted {accepted}/{total}")