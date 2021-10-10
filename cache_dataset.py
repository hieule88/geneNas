import argparse

import torch
import pytorch_lightning as pl

from problem import NLPProblemRWE, DataModule, LightningRecurrentRWE
from evolution import Optimizer

import logging

logging.disable(logging.CRITICAL)


def parse_args():
    parser = argparse.ArgumentParser()
    parser = NLPProblemRWE.add_arguments(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    parser = DataModule.add_argparse_args(parser)
    parser = DataModule.add_cache_arguments(parser)
    parser = LightningRecurrentRWE.add_model_specific_args(parser)
    parser = LightningRecurrentRWE.add_learning_specific_args(parser)
    parser = Optimizer.add_optimizer_specific_args(parser)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    args.num_terminal = args.num_main + 1
    args.l_main = args.h_main * (args.max_arity - 1) + 1
    args.l_adf = args.h_adf * (args.max_arity - 1) + 1
    args.main_length = args.h_main + args.l_main
    args.adf_length = args.h_adf + args.l_adf
    args.chromosome_length = (
        args.num_main * args.main_length + args.num_adf * args.adf_length
    )
    args.D = args.chromosome_length
    args.mutation_rate = args.adf_length / args.chromosome_length

    return args


def save_cache_dataset(cached_dataset, filepath):
    torch.save(cached_dataset, filepath)


def main():
    args = parse_args()
    dm = DataModule.from_argparse_args(args)
    dm.prepare_data()
    dm.setup("fit")

    save_cache_dataset(dm.dataset, f"{args.task_name}.cached.dataset.pt")


if __name__ == "__main__":
    main()
