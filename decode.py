#!/usr/bin/env python3

import argparse
import logging
import numpy as np
import os
import re
import torch

import pytorch_lightning as pl

from utils.tokenizer import Tokenizer
from inference import PCInferenceModule
from dataloader import D2TDataModule, PCDataModule

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO, datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = D2TDataModule.add_argparse_args(parser)

    parser.add_argument("--exp_dir", default="experiments", type=str,
        help="Base directory of the experiment.")
    parser.add_argument("--experiment", type=str, required=True,
        help="Experiment name.")
    parser.add_argument("--seed", default=42, type=int,
        help="Random seed.")
    parser.add_argument("--batch_size", default=1, type=int,
        help="Batch size used for decoding.")
    parser.add_argument("--dataset", type=str, required=True,
        help="Dataset name (webnlg / e2e / ...).")
    parser.add_argument("--split", type=str, required=True,
        help="Split to decode (dev / test).")
    parser.add_argument("--out_filename", type=str, default=None,
        help="Override the default output filename <split>.out.")
    parser.add_argument("--checkpoint", type=str, default="model.ckpt",
        help="Override the default checkpoint name 'model.ckpt'.")
    parser.add_argument("--max_threads", default=8, type=int,
        help="Maximum number of threads.")
    parser.add_argument("--beam_size", default=1, type=int,
        help="Beam size used for decoding.")
    parser.add_argument("--max_length", type=int, default=1024,
        help="Maximum number of tokens per example")
    parser.add_argument("--prefix", type=str, default=None,
        help="Prefix to be appended to input (after prompt)")
    parser.add_argument("--test_suffix", type=str, default="",
        help="Test file suffix (e.g. _seen)")


    return parser.parse_args(args)



if __name__ == "__main__":
    args = parse_args()

    logger.info(args)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(args.max_threads)

    model_path = os.path.join(args.exp_dir, args.experiment, args.checkpoint)
    out_path = os.path.join(args.exp_dir, args.experiment, f"{args.split}.out")
    di = PCInferenceModule(args, model_path=model_path)
    dm = PCDataModule(args, model_name=di.model_name)
    dm.setup('predict')

    trainer = pl.Trainer.from_argparse_args(args)

    out_filename = args.out_filename or f"{args.split}.out"
    out_file_handle = open(os.path.join(args.exp_dir, args.experiment, out_filename), "w")

    di.model.out_file_handle = out_file_handle
    di.model.tokenizer = dm.tokenizer
    di.model.beam_size_decode = args.beam_size

    dataloader_map = {
        "dev" : dm.val_dataloader,
        "test" : dm.test_dataloader
    }

    trainer.test(test_dataloaders=dataloader_map[args.split](), model=di.model)

    out_file_handle.close()
