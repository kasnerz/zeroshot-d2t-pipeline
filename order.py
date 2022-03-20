#!/usr/bin/env python3

import os
import json
import argparse
import logging
import numpy as np
from model import OrdTrainingModule
from inference import OrdInferenceModule

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO, datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)



class D2TOrderingModule:
    def __init__(self, args, model_path):
        self.model = OrdInferenceModule(args, model_path=model_path)

    def order_dataset(self, in_filename, out_filename, join_sents, shuffle=False):
        output = {
            "data" : []
        }
        with open(in_filename) as in_file:
            j = json.load(in_file)

            for i, example in enumerate(j["data"]):
                passages = example["sents"]
                if len(passages) == 1:
                    passages_ordered = passages
                else:
                    if shuffle:
                        np.random.shuffle(passages)

                    passages_ordered = self.model.order(passages)

                logger.info(i)
                logger.info(passages)
                logger.info(passages_ordered)
                logger.info("================")

                if join_sents:
                    passages_ordered = " ".join(passages_ordered)

                example_sorted = {
                    "sents" : passages_ordered,
                    "text" : example["text"]
                }
                output["data"].append(example_sorted)

        with open(os.path.join(out_filename), "w") as f:
            json.dump(output, f, indent=4, ensure_ascii=False)

    def order_dataset_indices(self, in_filename, out_filename, shuffle=False):
        with open(in_filename) as in_file, open(os.path.join(out_filename), "w") as f:
            j = json.load(in_file)

            for i, example in enumerate(j["data"]):
                passages = example["sents"]

                if len(passages) == 1:
                    # skip trivial examples
                    continue

                if shuffle:
                    np.random.shuffle(passages)

                # indices = np.random.permutation(len(passages))
                indices = self.model.order_indices(passages)

                logger.info(i)
                logger.info(passages)
                logger.info(indices)
                logger.info("================")

                f.write(" ".join([str(x) for x in indices]) + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--exp_dir", default="experiments", type=str,
        help="Base directory of the experiment.")
    parser.add_argument("--experiment", type=str, required=True,
        help="Experiment name.")
    parser.add_argument("--checkpoint", type=str, default="model.ckpt",
        help="Override the default checkpoint name 'model.ckpt'.")
    parser.add_argument("--model_name", type=str, default="facebook/bart-base",
        help="Name of the pretrained model.")
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
    parser.add_argument('--shuffle', action="store_true",
                    help='Shuffle the sentences before ordering.')
    parser.add_argument("--seed", type=int, default=42,
        help="Random seed")
    parser.add_argument("--gpus", type=int, default=1,
        help="GPU count")
    # parser.add_argument("--device", type=str, default="cuda",
    #     help="Random seed")
    parser.add_argument("--max_length", type=int, default=1024,
        help="Maximum number of tokens per example")
    args = parser.parse_args()


    np.random.seed(args.seed)

    model_path = os.path.join(args.exp_dir, args.experiment, args.checkpoint)
    dom = D2TOrderingModule(args, model_path=model_path)
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    for split in args.splits:
        if args.indices_only:
            dom.order_dataset_indices(
                in_filename=os.path.join(args.in_dir, f"{split}.json"),
                out_filename=os.path.join(out_dir, f"{split}.out"),
                shuffle=args.shuffle
            )
        else:
            dom.order_dataset(
                in_filename=os.path.join(args.in_dir, f"{split}.json"),
                out_filename=os.path.join(out_dir, f"{split}.json"),
                join_sents=args.join_sents,
                shuffle=args.shuffle
            )