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
from .lit_recurrent import LightningRecurrent, LightningRecurrentRWE

from network import RecurrentNet
from util.logger import ChromosomeLogger
from util.exception import NanException


class NLPProblem(Problem):
    def __init__(self, args):
        super().__init__(args)
        self.function_set = NLPFunctionSet.return_func_name()
        self.dm = DataModule.from_argparse_args(self.hparams)
        self.dm.prepare_data()
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

        glue_pl = LightningRecurrent(
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


class NLPProblemMultiObj(NLPProblem):
    def evaluate(self, chromosome: np.array):
        glue_pl, trainer = self.setup_model_trainer(chromosome)
        self.lr_finder(
            glue_pl, trainer, self.dm.train_dataloader(), self.dm.val_dataloader()
        )
        try:
            trainer.fit(glue_pl, self.dm)
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
        return (
            self.chromsome_logger.logs[-1]["data"][-1]["metrics"][self.metric_name],
            NLPProblem.total_params(glue_pl),
        )


class NLPProblemRWE(NLPProblem):
    def __init__(self, args):
        super().__init__(args)
        self.k_folds = self.hparams.k_folds

    def setup_model(self, chromosome):
        self.chromsome_logger.log_chromosome(chromosome)
        mains, adfs = self.parse_chromosome(chromosome, return_adf=True)

        glue_pl = LightningRecurrentRWE(
            num_labels=self.dm.num_labels,
            eval_splits=self.dm.eval_splits,
            **vars(self.hparams),
        )
        glue_pl.init_metric(self.dm.metric)
        glue_pl.init_model(mains, adfs)
        glue_pl.init_chromosome_logger(self.chromsome_logger)
        return glue_pl

    def perform_kfold(self, model):
        avg_metrics = 0
        total_time = 0

        trainer = self.setup_trainer()
        model.reset_weights()
        _, train_dataloader, val_dataloader = next(self.dm.kfold(self.k_folds, None))
        self.lr_finder(model, trainer, train_dataloader, val_dataloader)

        for fold, train_dataloader, val_dataloader in self.dm.kfold(self.k_folds, None):
            start = time.time()
            try:
                model.reset_weights()
                trainer = self.setup_trainer()
                trainer.fit(
                    model,
                    train_dataloader=train_dataloader,
                    val_dataloaders=val_dataloader,
                )
                metrics = self.chromsome_logger.logs[-1]["data"][-1]["metrics"][
                    self.metric_name
                ]
            except NanException as e:
                print(e)
                log_data = {
                    f"val_loss": 0.0,
                    "metrics": {self.metric_name: 0.0},
                    "epoch": -1,
                }
                metrics = log_data["metrics"][self.metric_name]
            end = time.time()
            avg_metrics += metrics
            total_time += end - start
            print(f"FOLD {fold}: {self.metric_name} {metrics} ; Time {end - start}")

        # result = trainer.test()
        avg_metrics = avg_metrics / self.k_folds
        print(f"FOLD AVG: {self.metric_name} {avg_metrics} ; Time {total_time}")
        return avg_metrics

    def evaluate(self, chromosome: np.array):
        symbols, _, _ = self.replace_value_with_symbol(chromosome)
        print(f"CHROMOSOME: {symbols}")
        glue_pl = self.setup_model(chromosome)
        return self.perform_kfold(glue_pl)


class NLPProblemRWEMultiObj(NLPProblemRWE):
    def evaluate(self, chromosome: np.array):
        symbols, _, _ = self.replace_value_with_symbol(chromosome)
        print(f"CHROMOSOME: {symbols}")
        glue_pl = self.setup_model(chromosome)
        return self.perform_kfold(glue_pl), glue_pl.total_params()


class NLPProblemRWEMultiObjNoTrain(NLPProblemRWEMultiObj):
    def __init__(self, args):
        super().__init__(args)
        self.metric = self.dm.metric
        # self.weight_values = [-2, -1, -0.5, 0.5, 1, 2]
        self.weight_values = [0.5, 1, 2, 3]

        from .lit_recurrent import ClsHead, SimpleClsHead

        self.embed_config = AutoConfig.from_pretrained(args.model_name_or_path)
        self.embed = AutoModel.from_pretrained(args.model_name_or_path)
        self.cls_head = ClsHead(args.hidden_size, args.dropout, self.dm.num_labels)
        # self.cls_head = SimpleClsHead(
        #     args.hidden_size, args.dropout, self.dm.num_labels
        # )

        self.embed.cuda()
        self.cls_head.cuda()

        self.embed.eval()
        self.cls_head.eval()

    def apply_weight(self, model, value):
        sampler = torch.distributions.uniform.Uniform(low=-value, high=value)
        with torch.no_grad():
            for name, param in model.named_parameters():
                new_param = sampler.sample(param.shape)
                param.copy_(new_param)
        return

    def run_inference(self, model, weight_value, val_dataloader):
        self.apply_weight(model, weight_value)
        self.apply_weight(self.cls_head, weight_value)
        outputs = []
        encounter_nan = False
        for batch in val_dataloader:

            labels = batch.pop("labels")
            batch = {k: v.cuda() for k, v in batch.items()}

            with torch.cuda.amp.autocast():
                x = self.embed(**batch)[0]
                if x.isnan().any():
                    print(f"NaN after embeds")
                    encounter_nan = True
                    break

                x, _ = model(x)
                if x.isnan().any():
                    print(f"NaN after recurrent")
                    encounter_nan = True
                    break

                if self.hparams.batch_first:
                    x = x[:, 0, :]  # CLS token
                else:
                    x = x[0, :, :]

                logits = self.cls_head(x)
                if logits.isnan().any():
                    print(f"NaN after CLS head")
                    encounter_nan = True
                    break

            if self.dm.num_labels > 1:
                preds = torch.argmax(logits, dim=1)
            else:
                preds = logits.squeeze()
            preds = preds.detach().cpu()
            batch = {k: v.detach().cpu() for k, v in batch.items()}

            outputs.append({"preds": preds, "labels": labels})

        if not encounter_nan:
            preds = torch.cat([x["preds"] for x in outputs]).detach().cpu().numpy()
            labels = torch.cat([x["labels"] for x in outputs]).detach().cpu().numpy()
            if np.all(preds == preds[0]):
                metrics = 0
            else:
                print(np.unique(preds, return_counts=True))
                print(np.unique(labels, return_counts=True))
                metrics = self.metric.compute(predictions=preds, references=labels)[
                    self.metric_name
                ]
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

    def setup_model(self, chromosome):
        mains, adfs = self.parse_chromosome(chromosome, return_adf=True)

        rnn = RecurrentNet(
            mains,
            adfs,
            self.embed_config.hidden_size,
            self.hparams.hidden_size,
            num_layers=self.hparams.num_layers,
            batch_first=self.hparams.batch_first,
            bidirection=self.hparams.bidirection,
        )
        return rnn

    def evaluate(self, chromosome: np.array):
        symbols, _, _ = self.replace_value_with_symbol(chromosome)
        print(f"CHROMOSOME: {symbols}")
        rnn = self.setup_model(chromosome)
        rnn.cuda()
        rnn.eval()
        avg_metrics, avg_max_metrics = self.perform_kfold(rnn)
        return avg_metrics, avg_max_metrics, NLPProblem.total_params(rnn)
