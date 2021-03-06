#!/usr/bin/env python

import argparse
import logging
import numpy as np
import os
import re
import sys
import torch
import pytorch_lightning as pl

from pprint import pprint as pp
from utils.tokenizer import Tokenizer
from inference import (
    D2TInferenceModule, 
    OrdInferenceModule, 
    AggInferenceModule,
    PCInferenceModule
)

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO, datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", default="experiments", type=str,
        help="Base directory of the experiment.")
    parser.add_argument("--experiment", type=str, required=True,
        help="Experiment name. Refers to a folder in `exp_dir`.")
    parser.add_argument("--module", required=True,
        help="Pipeline module: pc (applies to all variants) / agg / ord")
    parser.add_argument("--seed", default=42, type=int,
        help="Random seed.")
    parser.add_argument("--max_threads", default=8, type=int,
        help="Maximum number of threads.")
    parser.add_argument("--beam_size", default=1, type=int,
        help="Beam size.")
    parser.add_argument("--gpus", default=1, type=int,
        help="Number of GPUs.")
    parser.add_argument("--max_length", type=int, default=1024,
        help="Maximum number of tokens per example")
    parser.add_argument("--checkpoint", type=str, default="model.ckpt",
        help="Override the default checkpoint name 'model.ckpt'.")
    args = parser.parse_args()

    logger.info(args)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(args.max_threads)

    model_path = os.path.join(args.exp_dir, args.experiment, args.checkpoint)

    # determine the PL module to be used for inference
    # works like this for convenience, feel free to override
    if args.module == "ord":
        inference_module_cls = OrdInferenceModule
    elif args.module == "agg":
        inference_module_cls = AggInferenceModule
    elif args.module == "pc":
        inference_module_cls = PCInferenceModule
    else:
        logger.error(f"Module not recognized: {args.module}. Use one of: {{pc,agg,ord}}.")

    dm = inference_module_cls(args, model_path=model_path)
    logger.info(f"Using {inference_module_cls}")

    while True:
        s = input("[In]: ")
        out = dm.predict(s, beam_size=args.beam_size)
        print("[Out]:")
        pp(out)
        print("============")
