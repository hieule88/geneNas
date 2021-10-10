import argparse
import pickle
from tqdm.auto import tqdm

import torch
import pytorch_lightning as pl
from transformers import AutoModel

from problem import GLUEProblemRWE, GLUEDataModule, GLUERecurrentRWE
from evolution import Optimizer

import logging

logging.disable(logging.CRITICAL)


def parse_args():
    parser = argparse.ArgumentParser()
    parser = GLUEProblemRWE.add_arguments(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    parser = GLUEDataModule.add_argparse_args(parser)
    parser = GLUEDataModule.add_cache_arguments(parser)
    parser = GLUERecurrentRWE.add_model_specific_args(parser)
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


def cache_dataset(model, dataloader, prefix="dataset", max_size=100):
    cached_embeds = []
    cached_labels = []
    file_counter = 1
    for i, batch in enumerate(tqdm(dataloader, desc="Cache Dataset")):
        if "labels" in batch:
            labels = batch.pop("labels")
            cached_labels += torch.split(labels, 1)
        batch = {k: v.cuda() for k, v in batch.items()}
        embeds = model(**batch)[0].detach().cpu()
        cached_embeds += torch.split(embeds, 1)
        del batch
        if (i + 1) % max_size == 0:
            cached_data = [
                {
                    "embeds": cached_embeds[i].squeeze(),
                    "labels": cached_labels[i].squeeze(),
                }
                for i in range(len(cached_embeds))
            ]
            save_cache_dataset(cached_data, f"{prefix}.{file_counter}.pt")
            file_counter += 1
            del cached_data, cached_embeds, cached_labels
            cached_embeds = []
            cached_labels = []

    cached_data = [
        {
            "embeds": cached_embeds[i].squeeze(),
            "labels": cached_labels[i].squeeze(),
        }
        for i in range(len(cached_embeds))
    ]
    save_cache_dataset(cached_data, f"{prefix}.{file_counter}.pt")

    return


def save_cache_dataset(cached_dataset, filepath):
    torch.save(cached_dataset, filepath)


def main():
    args = parse_args()
    dm = GLUEDataModule.from_argparse_args(args)
    dm.prepare_data()
    dm.setup("fit")

    model = AutoModel.from_pretrained(args.model_name_or_path)
    model.cuda()

    cache_dataset(
        model,
        dm.train_dataloader(),
        f"{args.model_name_or_path}.{args.task_name}.cached.train",
    )

    if len(dm.eval_splits) > 1:
        for val_dataloader in dm.val_dataloader():
            cache_dataset(
                model,
                val_dataloader,
                f"{args.model_name_or_path}.{args.task_name}.cached.valid",
            )
    elif len(dm.eval_splits) == 1:
        cache_dataset(
            model,
            dm.val_dataloader(),
            f"{args.model_name_or_path}.{args.task_name}.cached.valid",
        )


if __name__ == "__main__":
    main()
