import time

import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks.base import Callback
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from transformers import AutoModel, AutoConfig
import torch
import torch.nn as nn

from .abstract_problem import Problem
from .function_set import NLPFunctionSet
from .data_module import DataModule
from .lit_recurrent_ner_train import LightningRecurrent_NERTrain

from network import RecurrentNet
from util.logger import ChromosomeLogger
from util.exception import NanException

from typing import List, Tuple
from evolution import GeneType

from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt

class NERProblemTrain(Problem):
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
        self.save_path = args.save_path
        self.baseline = True

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
            callbacks= early_stop,
            max_epochs = self.hparams.max_epochs,
        )
        return trainer

    def setup_model(self, chromosome):
        if not self.baseline:
            self.chromsome_logger.log_chromosome(chromosome)
            mains, adfs = self.parse_chromosome(chromosome, return_adf=True)

        glue_pl = LightningRecurrent_NERTrain(
            max_sequence_length= self.dm.max_seq_length,
            vocab= self.dm.vocabulary,
            num_labels=self.dm.num_labels,
            eval_splits=self.dm.eval_splits,
            **vars(self.hparams),
        )
        glue_pl.init_metric(self.dm.metric)
        if not self.baseline:
            glue_pl.init_model(mains, adfs)
            glue_pl.init_chromosome_logger(self.chromsome_logger)
        else: 
            recurrent_model = torch.nn.LSTM(input_size = glue_pl.embed.embed_dim,hidden_size= glue_pl.hidden_size, bidirectional=glue_pl.hparams.bidirection)
            glue_pl.add_module("recurrent_model", recurrent_model) 
        return glue_pl

    def train(self, model):
        
        trainer = self.setup_trainer()
        train_dataloader = DataLoader(self.dm.dataset['train'], batch_size= self.hparams.train_batch_size, shuffle= True, num_workers= self.hparams.num_workers)
        val_dataloader = DataLoader(self.dm.dataset['validation'], batch_size= self.hparams.eval_batch_size, num_workers= self.hparams.num_workers)
        # self.lr_finder(model, trainer, train_dataloader, val_dataloader)
        trainer.fit(
            model, 
            train_dataloaders= train_dataloader,
            val_dataloaders= val_dataloader,
        )
        num_epoch = len(model.callbacks)
        fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize= (10, 12), dpi=120)
        ax1.plot([i for i in range(1, num_epoch+1)], [i['accuracy'] for i in model.callbacks], color= 'g')
        ax2.plot([i for i in range(1, num_epoch+1)], [i['f1'] for i in model.callbacks], color= 'b')
        ax3.plot([i for i in range(1, num_epoch+1)], [i['val_loss'] for i in model.callbacks], color= 'r')

        ax1.set(title='Accuracy', xlabel='Epochs', ylabel='Accuracy')
        ax1.set(title='F1', xlabel='Epochs', ylabel='F1')
        ax1.set(title='Val Loss', xlabel='Epochs', ylabel='Loss')

        plt.show()
        # print(model.callbacks)

    def evaluate(self, chromosome=False):
        if not self.baseline:
            print(chromosome)
            symbols, _, _ = self.replace_value_with_symbol(chromosome)
            print(f"CHROMOSOME: {symbols}")
            print('Set up model')
        glue_pl = self.setup_model(chromosome)

        self.train(glue_pl)
        
        glue_pl.trainer.save_checkpoint(self.save_path)
