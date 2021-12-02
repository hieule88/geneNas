import time

import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from transformers import AutoModel, AutoConfig
import torch
import torch.nn as nn

from .abstract_problem import Problem
from .function_set import NLPFunctionSet
from .data_module import DataModule
from .lit_recurrent_ner import LightningRecurrent_NER

from network import RecurrentNet
from util.logger import ChromosomeLogger
from util.exception import NanException

class NERProblem(Problem):
    def __init__(self, args):
        super().__init__(args)
        self.function_set = NLPFunctionSet.return_func_name()
        self.dm = DataModule.from_argparse_args(self.hparams)
        self.dm.setup("fit")

        self.chromsome_logger = ChromosomeLogger()
        self.metric_name = self.dm.metrics_names[self.hparams.task_name]

        self.progress_bar = 0
        self.weights_summary = None
        self.early_stop = None

    def parse_chromosome(
        self, chromosome: np.array, function_set=NLPFunctionSet, return_adf=False
    ):
        return super().parse_chromosome(
            chromosome, function_set=function_set, return_adf=return_adf
        )

    @staticmethod
    def total_params(model):
        return sum(p.numel() for p in model.parameters())

    def lr_finder(self, model, trainer, train_dataloader, val_dataloaders):
        lr_finder = trainer.tuner.lr_find(
            model, train_dataloader=train_dataloader, val_dataloaders=val_dataloaders
        )
        new_lr = lr_finder.suggestion()
        model.hparams.lr = new_lr
        print(f"New optimal lr: {new_lr}")

    def setup_model_trainer(self, chromosome: np.array):
        glue_pl = self.setup_model(chromosome)
        trainer = self.setup_trainer()
        return glue_pl, trainer

    def setup_trainer(self):
        if type(self.early_stop) == int:
            early_stop = EarlyStopping(
                monitor=self.metric_name,
                min_delta=0.00,
                patience=self.early_stop,
                verbose=False,
                mode="max",
            )
            early_stop = [early_stop]
        else:
            early_stop = None

        trainer = pl.Trainer.from_argparse_args(
            self.hparams,
            progress_bar_refresh_rate=self.progress_bar,
            # automatic_optimization=False,
            weights_summary=self.weights_summary,
            checkpoint_callback=False,
            callbacks=early_stop,
        )
        return trainer

    def setup_model(self, chromosome):
        self.chromsome_logger.log_chromosome(chromosome)
        mains, adfs = self.parse_chromosome(chromosome, return_adf=True)

        glue_pl = LightningRecurrent_NER(
            vocab= self.dm.vocabulary,
            num_labels=self.dm.num_labels,
            eval_splits=self.dm.eval_splits,
            **vars(self.hparams),
        )
        glue_pl.init_metric(self.dm.metric)
        glue_pl.init_model(mains, adfs)
        glue_pl.init_chromosome_logger(self.chromsome_logger)
        return glue_pl

    def evaluate(self, chromosome: np.array):
        glue_pl, trainer = self.setup_model_trainer(chromosome)
        self.lr_finder(
            glue_pl, trainer, self.dm.train_dataloader(), self.dm.val_dataloader()
        )
        try:
            trainer.fit(glue_pl, self.dm)
            trainer.test(glue_pl, test_dataloaders=self.dm.test_dataloader())
        except NanException as e:
            print(e)
            log_data = {
                f"val_loss": 0.0,
                "metrics": {"accuracy": 0.0, "f1": 0.0},
                "epoch": -1,
            }
            self.chromsome_logger.log_epoch(log_data)

        # result = trainer.test()
        print(self.chromsome_logger.logs[-1]["data"][-1])
        return self.chromsome_logger.logs[-1]["data"][-1]["metrics"][self.metric_name]
