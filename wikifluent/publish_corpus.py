#!/usr/bin/env python3

import os
import json
import logging
import argparse
import random
import numpy as np

in_dir = "wikifluent-parts"
out_dir = "full"

idx = 0
val_test_ratio = 100

data = {
    "train" : {"data" : []},
    "dev" : {"data" : []},
    "test" : {"data" : []}
}

os.makedirs(out_dir, exist_ok=True)

for filename in sorted(os.listdir(in_dir)):
    print(f"Processing {filename}")

    with open(os.path.join(in_dir, filename)) as file:
        for line in file.readlines():
            idx += 1
            example = json.loads(line)
            prev_agg = 0
            separators = []

            for agg, sent in list(zip(example["aggregation"], example["out"]))[1:]:
                if agg != prev_agg:
                    separators.append(1)
                else:
                    separators.append(0)
                prev_agg = agg

            out = example["out"]

            new_example = {
                "sents" : out,
                "sep" : separators,
                "text" : example["in"]
            }

            if idx % val_test_ratio == 0:
                data["dev"]["data"].append(new_example)
            elif idx % val_test_ratio == 1:
                data["test"]["data"].append(new_example)
            else:
                data["train"]["data"].append(new_example)

for split in ["train", "dev", "test"]:
    with open(os.path.join(out_dir, f"{split}.json"), "w") as f:
        json.dump(data[split], f, indent=4, ensure_ascii=False)