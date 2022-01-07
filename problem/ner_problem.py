import time

import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from transformers import AutoModel, AutoConfig
import torch
import torch.nn as nn
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from .abstract_problem import Problem
from .function_set import NLPFunctionSet
from .data_module import DataModule
from .lit_recurrent_ner import LightningRecurrent_NER

from network import RecurrentNet
from util.logger import ChromosomeLogger
from util.exception import NanException

from typing import List, Tuple
from evolution import GeneType

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
            limit_train_batches=0, 
            limit_val_batches=0,
        )
        return trainer

    def setup_model(self, chromosome):
        self.chromsome_logger.log_chromosome(chromosome)
        mains, adfs = self.parse_chromosome(chromosome, return_adf=True)

        glue_pl = LightningRecurrent_NER(
            max_sequence_length= self.dm.max_seq_length,
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
        try:
            trainer.fit(glue_pl, self.dm)
            trainer.test(glue_pl, test_dataloaders=self.dm.test_dataloader())

        except NanException as e:
            # print(e)
            log_data = {
                f"val_loss": 0.0,
                "metrics": {"accuracy": 0.0, "f1": 0.0},
                "epoch": -1,
            }
            self.chromsome_logger.log_epoch(log_data)

        # result = trainer.test()
        print(self.chromsome_logger.logs[-1]["data"][-1])
        return self.chromsome_logger.logs[-1]["data"][-1]["metrics"][self.metric_name]

class NERProblemMultiObj(NERProblem):
    def __init__(self, args):
        super().__init__(args)
        self.k_folds = self.hparams.k_folds
        self.weight_values = [0.5, 1, 2, 3]

    def apply_weight(self, model, value):
        sampler = torch.distributions.uniform.Uniform(low=-value, high=value)
        with torch.no_grad():
            for name, param in model.named_parameters():
                new_param = sampler.sample(param.shape)
                param.copy_(new_param)

    def run_inference(self, model, weight_value, val_dataloader):
        self.apply_weight(model, weight_value)
        
        outputs = []
        encounter_nan = False
        for batch in val_dataloader:
            labels = batch["labels"]

            with torch.cuda.amp.autocast():
                logits = model(None, **batch)[1]
                if logits.isnan().any():
                    print(f"NaN after NasgepNet")
                    encounter_nan = True
                    break

                if logits.isnan().any():
                    print(f"NaN after CLS head")
                    encounter_nan = True
                    break

            if self.dm.num_labels > 1:
                preds = logits
            else:
                preds = logits.squeeze()
            preds = preds.detach().cpu()
            # batch = {k: v.detach().cpu() for k, v in batch.items()}

            outputs.append({"preds": preds, "labels": labels})

        if not encounter_nan:
            preds = torch.cat([x["preds"] for x in outputs]).detach().cpu().numpy()
            labels = torch.cat([x["labels"] for x in outputs]).detach().cpu().numpy()
            if np.all(preds == preds[0]):
                metrics = 0
            else:
                preds = [i for j in range(len(preds)) for i in preds[j][:labels[j][-1]] ]
                labels = [i for j in range(len(labels)) for i in labels[j][:labels[j][-1]] ]
                metrics = {}
                metrics['accuracy'] = accuracy_score(labels, preds)
                metrics['f1'] = f1_score(labels, preds, average='macro')
                metrics['recall'] = recall_score(labels, preds, average='macro')
                metrics['precision'] = precision_score(labels, preds, average='macro')
                metrics = metrics[self.metric_name]
        else:
            metrics = 0
        return metrics

    def perform_kfold(self, model):
        avg_metrics = 0
        avg_max_metrics = 0
        total_time = 0

        for fold, _, val_dataloader in self.dm.kfold(self.k_folds, None):
            start = time.time()
            metrics = [
                self.run_inference(model, wval, val_dataloader)
                for wval in self.weight_values
            ]
            end = time.time()
            avg_metrics += np.mean(metrics)
            avg_max_metrics += np.max(metrics)
            total_time += end - start
            print(
                f"FOLD {fold}: {self.metric_name} {np.mean(metrics)} {np.max(metrics)} ; Time {end - start}"
            )

        # result = trainer.test()
        avg_metrics = avg_metrics / self.k_folds
        avg_max_metrics = avg_max_metrics / self.k_folds
        print(
            f"FOLD AVG: {self.metric_name} {avg_metrics} {avg_max_metrics} ; Time {total_time}"
        )
        return avg_metrics, avg_max_metrics

    def evaluate(self, chromosome: np.array):
        symbols, _, _ = self.replace_value_with_symbol(chromosome)
        print(f"CHROMOSOME: {symbols}")
        RNN_model = self.setup_model(chromosome)
        avg_metrics, avg_max_metrics = self.perform_kfold(RNN_model)
        return avg_metrics, avg_max_metrics, NERProblemMultiObj.total_params(RNN_model)