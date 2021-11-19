#!/usr/bin/env python3

import os
import json
import logging
import argparse
import random
import numpy as np

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO, datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)


def create_example_comp_ordered_separated(example, sep):
    prev_agg = 0
    out = []

    for agg, sent in zip(example["aggregation"], example["out"]):
        if agg != prev_agg:
            out.append(sep)
        out.append(sent)
        prev_agg = agg

    new_example = {
        "text" : " ".join(out),
        "labels" : example["in"]
    }
    return new_example

def create_example_comp_ordered(example):
    new_example = {
        "text" : " ".join(example["out"]),
        "labels" : example["in"]
    }
    return new_example


def create_example_comp_full(example):
    sentences = np.array(example["out"])
    permutation = np.random.permutation(len(sentences))

    new_example = {
        "text" : " ".join(sentences[permutation].tolist()),
        "labels" : example["in"]
    }
    return new_example


def create_example_ordering(example):
    sentences = np.array(example["out"])
    permutation = np.random.permutation(len(sentences))

    new_example = {
        "text" : sentences[permutation].tolist(),
        "labels" : np.argsort(permutation).tolist(),
    }
    return new_example


def create_example_aggregation(example):
    separators = []
    prev_agg = 0

    # skip examples with a single sentence
    if len(example["out"]) == 1:
        return None

    for agg, sent in list(zip(example["aggregation"], example["out"]))[1:]:
        if agg != prev_agg:
            separators.append(1)
        else:
            separators.append(0)
        prev_agg = agg

    new_example = {
        "text" : example["out"],
        "labels" : separators,
    }
    return new_example


def collate(in_dir, out_dir, mode, sep, val_test_ratio):
    idx = 0
    data = {
        "train" : {"data" : []},
        "dev" : {"data" : []},
        "test" : {"data" : []}
    }
    for filename in sorted(os.listdir(in_dir)):
        logger.info(f"Processing {filename}")

        with open(os.path.join(in_dir, filename)) as file:
            for line in file.readlines():
                example = json.loads(line)

                if mode == "ord":
                    res = create_example_ordering(example)
                if mode == "agg":
                    res = create_example_aggregation(example)
                elif mode == "comp_full":
                    res = create_example_comp_full(example)
                elif mode == "comp_ord":
                    res = create_example_comp_ordered(example)
                elif mode == "comp_ord_sep":
                    res = create_example_comp_ordered_separated(example, sep)
                else:
                    raise ValueError(f"Unknown mode: {mode}")

                if res is None:
                    continue

                if type(res) is not list:
                    res = [res]

                if idx % val_test_ratio == 0:
                    data["dev"]["data"] += res
                elif idx % val_test_ratio == 1:
                    data["test"]["data"] += res
                else:
                    data["train"]["data"] += res

                idx += 1

    return data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_dir", type=str, default="wikifluent-parts",
        help="Path to the directory with individual dataset parts.")
    parser.add_argument("--out_dir", type=str, required=True,
        help="Path to the output directory.")
    parser.add_argument("--mode", type=str, required=True,
        help="Which model to process the dataset for:\
                ord = ordering model, \
                agg = aggregation model, \
                comp_full = paragraph compression model including ordering and aggregation, \
                comp_ord = paragraph compression model on ordered sentences, \
                comp_ord_sep = paragraph compression model on ordered and separated sentences")
    parser.add_argument("--sep", type=str, default="<sep>",
        help="A token to be used as a separator between non-aggregated sentences.")
    parser.add_argument("--seed", type=int, default=42,
        help="Random seed.")
    parser.add_argument("--val_test_ratio", type=int, default=100,
        help="Every n-th example will be chosen for validation set, every n+1-th example for test set.")
    args = parser.parse_args()
    logger.info(args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    output = collate(in_dir=args.in_dir, 
        out_dir=args.out_dir, 
        mode=args.mode, 
        sep=args.sep,
        val_test_ratio=args.val_test_ratio
    )

    for split in ["train", "dev", "test"]:
        with open(os.path.join(args.out_dir, f"{split}.json"), "w") as f:
            json.dump(output[split], f, indent=4, ensure_ascii=False)