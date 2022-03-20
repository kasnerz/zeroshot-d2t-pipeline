#!/usr/bin/env python3

from model import (
    D2TTrainingModule,
    OrdTrainingModule,
    AggTrainingModule,
    PCTrainingModule
)
from dataloader import (
    D2TDataModule,
    OrdDataModule,
    AggDataModule,
    PCDataModule,
    PCAggDataModule, 
    PCOrdAggDataModule, 
)

import logging
import argparse
import os
import warnings

import pytorch_lightning as pl

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO, datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = D2TDataModule.add_argparse_args(parser)
    parser = D2TTrainingModule.add_model_specific_args(parser)

    parser.add_argument("--module", type=str, required=True,
        help="Name of the pipeline module to be trained:\
            ord = ordering \
            agg = aggregation \
            pc = paragraph compression \
            pc_agg = paragraph compression + aggregation \
            pc_ord_agg = paragraph compression + ordering + aggregation."
        )
    parser.add_argument("--model_name", type=str, default="facebook/bart-base",
        help="Name of the model from the Huggingface Transformers library.")
    parser.add_argument("--model_path", type=str, default=None,
        help="Path to the saved checkpoint to be loaded.")
    parser.add_argument("--in_dir", type=str, required=True,
        help="Input directory with the data.")
    parser.add_argument("--batch_size", type=int, default=8,
        help="Batch size for finetuning the model")
    parser.add_argument("--out_dir", type=str, default="experiments",
        help="Output directory")
    parser.add_argument("--checkpoint_name", type=str, default="model",
        help="Name of the checkpoint (default='model')")
    parser.add_argument("--experiment", type=str, required=True,
        help="Experiment name used for naming the experiment directory")
    parser.add_argument("--max_length", type=int, default=512,
        help="Maximum number of tokens per example")
    parser.add_argument("--seed", default=42, type=int,
        help="Random seed.")
    parser.add_argument("--max_threads", default=8, type=int,
        help="Maximum number of CPU threads.")
    parser.add_argument("--resume_training", action="store_true",
        help="Resume training from the loaded checkpoint (useful if training was interrupted).")
    
    return parser.parse_args(args)


if __name__ == '__main__':
    args = parse_args()
    logger.info("Initializing...")
    logger.info(args)

    training_module = {
        "ord" : OrdTrainingModule,
        "agg" : AggTrainingModule,
        "pc" : PCTrainingModule,    # training module is the same for PC* modules
        "pc_agg" : PCTrainingModule,
        "pc_ord_agg" : PCTrainingModule
    }[args.module]

    data_module = {
        "ord" : OrdDataModule,
        "agg" : AggDataModule,
        "pc" : PCDataModule,
        "pc_agg" : PCAggDataModule,
        "pc_ord_agg" : PCOrdAggDataModule,
    }[args.module]

    pl.seed_everything(args.seed)
    dm = data_module(args)
    dm.prepare_data()
    dm.setup('fit')

    resume_from_checkpoint = None

    if args.model_path:
        model = training_module.load_from_checkpoint(
            args.model_path
        )
        if args.resume_training:
            resume_from_checkpoint = args.model_path
    else:
        model = training_module(args)

        if args.resume_training:
            logger.error("Model path not specified, training not resumed.")
        
    ckpt_out_dir = os.path.join(args.out_dir,
        args.experiment
    )
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=ckpt_out_dir,
        filename=args.checkpoint_name,
        save_top_k=1,
        verbose=True,
        monitor="loss/val",
        mode="min"
    )
    trainer = pl.Trainer.from_argparse_args(args, 
        callbacks=[checkpoint_callback], 
        strategy='dp',
        resume_from_checkpoint=resume_from_checkpoint
    )
    trainer.fit(model, dm)
