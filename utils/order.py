#!/usr/bin/env python3

import os
import json
import argparse
from use import OrderingModel
from training.scripts.models.bart_simple import BartForSequenceOrdering
import numpy as np

class D2TOrderingModule:
    def __init__(self, ckpt_dir, model_name):
        self.model = OrderingModel(BartForSequenceOrdering, ckpt_dir, model_name)

    def order_dataset(self, in_filename, out_filename, join_sents):
        output = {
            "data" : []
        }
        with open(in_filename) as in_file:
            j = json.load(in_file)

            for i, example in enumerate(j["data"]):
                passages = example["sents"]
                passages_ordered = self.model.order(passages)

                print(i)
                print(passages)
                print(passages_ordered)
                print("================")

                if join_sents:
                    passages_ordered = " ".join(passages_ordered)

                example_sorted = {
                    "sents" : passages_ordered,
                    "text" : example["text"]
                }
                output["data"].append(example_sorted)

        with open(os.path.join(out_filename), "w") as f:
            json.dump(output, f, indent=4, ensure_ascii=False)

    def order_dataset_indices(self, in_filename, out_filename):
        with open(in_filename) as in_file, open(os.path.join(out_filename), "w") as f:
            j = json.load(in_file)

            for i, example in enumerate(j["data"]):
                passages = example["sents"]

                if len(passages) == 1:
                    # skip trivial examples
                    continue

                # indices = np.random.permutation(len(passages))
                indices = self.model.order_indices(passages)

                print(i)
                print(passages)
                print(indices)
                print("================")

                f.write(" ".join([str(x) for x in indices]) + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="facebook/bart-base",
        help="Name of the pretrained model.")
    parser.add_argument("--model_dir", type=str, default="models/bart-base-simple-wikifp",
        help="Directory with the model checkpoints")
    parser.add_argument("--in_dir", type=str, required=True,
        help="Directory with the dataset to sort.")
    parser.add_argument("--out_dir", type=str, default=None,
        help="Output directory")
    parser.add_argument('--splits', type=str, nargs='+', default=["test"],
                    help='Dataset splits (e.g. train dev test)')
    parser.add_argument('--indices_only', action="store_true",
                    help='Output only permutation indices')
    parser.add_argument('--join_sents', action="store_true",
                    help='Join sentences to a single string on the output.')
    parser.add_argument('--last_ckpt', action="store_true",
                    help='Use the directory with last checkpoint.')
    parser.add_argument("--seed", type=str, default=42,
        help="Random seed")
    args = parser.parse_args()

    model_dir = args.model_dir

    if args.last_ckpt:
        last_ckpt_dir = list(sorted(os.listdir(model_dir)))[-1]
        model_dir = os.path.join(model_dir, last_ckpt_dir)

    dom = D2TOrderingModule(
        ckpt_dir=model_dir, 
        model_name=args.model_name
    )
    out_dir = args.out_dir

    os.makedirs(out_dir, exist_ok=True)

    for split in args.splits:

        if args.indices_only:
            dom.order_dataset_indices(
                in_filename=os.path.join(args.in_dir, f"{split}.json"),
                out_filename=os.path.join(out_dir, f"{split}.out")
            )
        else:
            dom.order_dataset(
                in_filename=os.path.join(args.in_dir, f"{split}.json"),
                out_filename=os.path.join(out_dir, f"{split}.json"),
                join_sents=args.join_sents
            )

