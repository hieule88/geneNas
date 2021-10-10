from argparse import ArgumentParser

import pytorch_lightning as pl
from transformers import AutoModel, AutoConfig, AdamW, get_linear_schedule_with_warmup
import torch
import torch.nn as nn

from . import NLPProblem, DataModule
from .lit_recurrent import SimpleClsHead, ClsHead
from util.exception import NanException

from typing import Optional, Union, Dict


class LightningBERTSeqCls(pl.LightningModule):
    def __init__(
        self,
        model_name_or_path: str,
        num_labels: int,
        dropout: float = 0.1,
        learning_rate: float = 2e-5,
        epsilon: float = 1e-8,
        warmup_steps: int = 0,
        weight_decay: float = 0.0,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        eval_splits: Optional[list] = None,
        num_val_dataloader: int = 1,
        use_simple_cls: bool = False,
        freeze_embed: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.num_labels = num_labels
        self.num_val_dataloader = num_val_dataloader

        self.config = AutoConfig.from_pretrained(
            model_name_or_path, num_labels=num_labels
        )
        self.bert = AutoModel.from_pretrained(model_name_or_path, config=self.config)
        if freeze_embed:
            for param in self.bert.parameters():
                param.requires_grad = False

        if not use_simple_cls:
            self.cls_head = ClsHead(self.config.hidden_size // 2, dropout, num_labels)
        else:
            self.cls_head = SimpleClsHead(
                self.config.hidden_size // 2, dropout, num_labels
            )

        self.metric = None

    def init_metric(self, metric):
        self.metric = metric

    def forward(self, **inputs):
        labels = None
        if "labels" in inputs:
            labels = inputs.pop("labels")
        x = self.bert(**inputs)[0]
        x = x[:, 0, :]  # CLS token
        # if x.isnan().any():
        #     raise NanException(f"NaN after BERT")
        logits = self.cls_head(x)
        # if logits.isnan().any():
        #     raise NanException(f"NaN after CLS head")
        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        return loss, logits

    def training_step(self, batch, batch_idx):
        loss, _ = self(**batch)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        val_loss, logits = self(**batch)

        if self.hparams.num_labels >= 1:
            preds = torch.argmax(logits, dim=1)
        elif self.hparams.num_labels == 1:
            preds = logits.squeeze()
        labels = batch["labels"]

        return {"loss": val_loss, "preds": preds, "labels": labels}

    def validation_epoch_end(self, outputs):
        if self.num_val_dataloader > 1:
            for i, output in enumerate(outputs):
                # matched or mismatched
                split = self.hparams.eval_splits[i].split("_")[-1]
                preds = torch.cat([x["preds"] for x in output]).detach().cpu().numpy()
                labels = torch.cat([x["labels"] for x in output]).detach().cpu().numpy()
                loss = torch.stack([x["loss"] for x in output]).mean()
                self.log(f"val_loss_{split}", loss, prog_bar=True)
                split_metrics = {
                    f"{k}_{split}": v
                    for k, v in self.metric.compute(
                        predictions=preds, references=labels
                    ).items()
                }
                self.log_dict(split_metrics, prog_bar=True)
                log_data = {
                    f"val_loss_{split}": loss.item(),
                    "metrics": split_metrics,
                    "epoch": self.current_epoch,
                }
            return

        preds = torch.cat([x["preds"] for x in outputs]).detach().cpu().numpy()
        labels = torch.cat([x["labels"] for x in outputs]).detach().cpu().numpy()
        loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.log("val_loss", loss, prog_bar=True)
        metrics = self.metric.compute(predictions=preds, references=labels)
        self.log_dict(metrics, prog_bar=True)
        log_data = {
            f"val_loss": loss.item(),
            "metrics": metrics,
            "epoch": self.current_epoch,
        }
        return

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def test_epoch_end(self, outputs):
        return self.validation_epoch_end(outputs)

    def setup(self, stage):
        if stage == "fit":
            # Get dataloader by calling it - train_dataloader() is called after setup() by default
            train_loader = self.train_dataloader()

            # Calculate total steps
            self.total_steps = (
                (
                    len(train_loader.dataset)
                    // (self.hparams.train_batch_size * max(1, self.hparams.gpus))
                )
                // self.hparams.accumulate_grad_batches
                * float(self.hparams.max_epochs)
            )

    def configure_optimizers(self):
        "Prepare optimizer and schedule (linear warmup and decay)"
        bert = self.bert
        fc = self.cls_head
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in fc.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ]
                + [
                    p
                    for n, p in bert.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in bert.named_parameters()
                    if any(nd in n for nd in no_decay)
                ]
                + [
                    p
                    for n, p in fc.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.hparams.learning_rate,
            eps=self.hparams.epsilon,
        )

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.total_steps,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]

    @staticmethod
    def add_learning_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--learning_rate", default=2e-5, type=float)
        parser.add_argument("--epsilon", default=1e-8, type=float)
        parser.add_argument("--warmup_steps", default=0, type=int)
        parser.add_argument("--weight_decay", default=0.0, type=float)
        parser.add_argument("--freeze_embed", action="store_true")
        return parser


class LightningBERTLSTMSeqCls(LightningBERTSeqCls):
    def __init__(
        self,
        model_name_or_path: str,
        num_labels: int,
        hidden_size: int = 128,
        dropout: float = 0.1,
        learning_rate: float = 2e-5,
        epsilon: float = 1e-8,
        warmup_steps: int = 0,
        weight_decay: float = 0.0,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        eval_splits: Optional[list] = None,
        num_val_dataloader: int = 1,
        batch_first: bool = True,
        bidirection: bool = True,
        use_simple_cls: bool = False,
        freeze_embed: bool = False,
        **kwargs,
    ):
        super().__init__(
            model_name_or_path,
            num_labels,
            dropout=dropout,
            num_val_dataloader=num_val_dataloader,
            freeze_embed=freeze_embed,
        )

        self.save_hyperparameters()

        self.lstm = nn.LSTM(
            self.config.hidden_size,
            hidden_size,
            batch_first=self.hparams.batch_first,
            dropout=dropout,
            bidirectional=self.hparams.bidirection,
            num_layers=self.hparams.num_layers,
        )
        if not use_simple_cls:
            self.cls_head = ClsHead(hidden_size, dropout, num_labels)
        else:
            self.cls_head = SimpleClsHead(hidden_size, dropout, num_labels)

    def forward(self, **inputs):
        labels = None
        if "labels" in inputs:
            labels = inputs.pop("labels")
        x = self.bert(**inputs)[0]
        # if x.isnan().any():
        #     raise NanException(f"NaN after embeds")
        x, _ = self.lstm(x)
        # if x.isnan().any():
        #     raise NanException(f"NaN after LSTM")
        if self.hparams.batch_first:
            x = x[:, 0, :]  # CLS token
        else:
            x = x[0, :, :]
        logits = self.cls_head(x)
        # if logits.isnan().any():
        #     raise NanException(f"NaN after CLS head")
        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        return loss, logits

    def configure_optimizers(self):
        "Prepare optimizer and schedule (linear warmup and decay)"
        bert = self.bert
        model = self.lstm
        fc = self.cls_head
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ]
                + [
                    p
                    for n, p in fc.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ]
                + [
                    p
                    for n, p in bert.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ]
                + [
                    p
                    for n, p in fc.named_parameters()
                    if any(nd in n for nd in no_decay)
                ]
                + [
                    p
                    for n, p in bert.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.hparams.learning_rate,
            eps=self.hparams.epsilon,
        )

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.total_steps,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--num_layers", default=1, type=int)
        parser.add_argument("--batch_first", default=True, type=bool)
        parser.add_argument("--bidirection", default=True, type=bool)
        parser.add_argument("--hidden_size", default=128, type=int)
        parser.add_argument("--dropout", default=0.1, type=float)
        parser.add_argument("--use_simple_cls", action="store_true")
        return parser


class BaselineProblem(NLPProblem):
    def __init__(self, args, use_lstm: bool = False):
        super().__init__(args)
        self.use_lstm = use_lstm

    def evaluate(self, chromosome):
        trainer = self.setup_trainer()
        if self.use_lstm:
            model = LightningBERTLSTMSeqCls(
                num_labels=self.dm.num_labels,
                eval_splits=self.dm.eval_splits,
                **vars(self.hparams),
            )
        else:
            model = LightningBERTSeqCls(
                num_labels=self.dm.num_labels,
                eval_splits=self.dm.eval_splits,
                **vars(self.hparams),
            )
        model.init_metric(self.dm.metric)
        self.lr_finder(
            model, trainer, self.dm.train_dataloader(), self.dm.val_dataloader()
        )
        try:
            trainer.fit(model, self.dm)
            trainer.test(model, test_dataloaders=self.dm.test_dataloader())
        except NanException as e:
            print(e)
