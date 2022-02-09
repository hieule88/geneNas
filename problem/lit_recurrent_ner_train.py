from argparse import ArgumentParser
from datetime import datetime
from typing import Optional, Union, Dict

from network.recurrent_net import RecurrentNet

import numpy as np
import datasets
import pytorch_lightning as pl
import torch
import torch.nn as nn
from transformers import AdamW, get_linear_schedule_with_warmup
from util.logger import ChromosomeLogger
from util.exception import NanException

from collections import Counter
import os
from pytorch_forecasting.models.temporal_fusion_transformer.sub_modules import TimeDistributed

from torchcrf import CRF
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import warnings
warnings.filterwarnings("ignore")
class LightningRecurrent_NERTrain(pl.LightningModule):
    def __init__(
        self,
        model_name_or_path: str,
        vocab,
        max_sequence_length,
        num_labels: int,
        hidden_size: int = 128,
        dropout: float = 0.1,
        learning_rate: float = 2e-3,
        epsilon: float = 1e-8,
        warmup_steps: int = 0,
        weight_decay: float = 0.0,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        eval_splits: Optional[list] = None,
        num_val_dataloader: int = 1,
        freeze_embed: bool = True,
        use_simple_cls: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.save_hyperparameters()
        self.max_sequence_length = max_sequence_length
        self.num_labels = num_labels
        self.num_val_dataloader = num_val_dataloader

        self.embed = GloveEmbedding(glove_dir = model_name_or_path, vocab = vocab)
        if freeze_embed:
            for param in self.embed.parameters():
                param.requires_grad = False

        self.rnn_dropout = nn.Dropout(p=dropout)

        self.cls_head = ClsHead(hidden_size, dropout, num_labels)
        # self.cls_head = SimpleClsHead(hidden_size,num_labels)

        self.chromosome_logger: Optional[ChromosomeLogger] = None
        self.metric = None
        self.crf = CRF(self.num_labels, batch_first=self.hparams.batch_first)
        self.callbacks = []
        self.hidden_size = hidden_size

    def init_metric(self, metric):
        self.metric = metric

    def init_model(self, cells, adfs):
        recurrent_model = RecurrentNet(
            cells,
            adfs,
            self.embed.embed_dim,
            self.hparams.hidden_size,
            num_layers=self.hparams.num_layers,
            batch_first=self.hparams.batch_first,
            bidirection=self.hparams.bidirection,
        )
        self.add_module("recurrent_model", recurrent_model)

    def init_chromosome_logger(self, logger: ChromosomeLogger):
        self.chromosome_logger = logger

    def forward(self, hiddens, **inputs):
        labels = None
        if "labels" in inputs:
            labels = inputs.pop("labels")[:,:self.max_sequence_length]

        x = self.embed(**inputs)

        # if x.isnan().any():
        #     raise NanException(f"NaN after embeds")
        x, hiddens = self.recurrent_model(x, hiddens)
        x = self.rnn_dropout(x)
        # if x.isnan().any():
        #     raise NanException(f"NaN after RNN")

        logits = self.cls_head(x)
        # print('X after CLS: ',logits)
        
        # if logits.isnan().any():
        #     raise NanException(f"NaN after CLS head")

        # logits = torch.Tensor(self.crf.decode(after_lstm))
        # mask = torch.tensor([[1 if labels[j][i] != -2 else 0 \
        #                         for i in range(len(labels[j]))] \
        #                         for j in range(len(labels))], dtype=torch.uint8).cuda()

        # loss = (-1.0)*self.crf(after_lstm, labels, mask=mask, reduction='mean')
        
        loss = None
        if labels is not None:
            # labels = nn.functional.one_hot(labels.to(torch.int64),self.num_labels).to(torch.float32)
            # labels = torch.Tensor(labels)

            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = nn.CrossEntropyLoss(ignore_index= -2)

                # loss = loss_fct(logits.reshape((logits.shape[0]*logits.shape[1])),\
                                # labels.reshape((labels.shape[0]*labels.shape[1])))
                loss = loss_fct(logits.reshape((logits.shape[0]*logits.shape[1], logits.shape[2])),\
                                labels.reshape((labels.shape[0]*labels.shape[1])))


        return loss, logits, hiddens

    def training_step(self, batch, batch_idx, hiddens=None):
        loss, _, hiddens = self(hiddens, **batch)
        return {"loss": loss, "hiddens": hiddens}

    def validation_step(self, batch, batch_idx):
        val_loss, logits, _ = self(None, **batch)

        if self.hparams.num_labels >= 1:
            preds = torch.argmax(logits, dim=-1)
            # preds = logits
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
                self.chromosome_logger.log_epoch(log_data)
            return

        preds = torch.cat([x["preds"] for x in outputs]).detach().cpu().numpy()
        labels = torch.cat([x["labels"] for x in outputs]).detach().cpu().numpy()
        loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.log("val_loss", loss, prog_bar=True)

        # if np.all(preds == preds[0]):
        #     metrics = {self.metric.name: 0}
        # else:
        preds = [i for j in range(len(preds)) for i in preds[j][:labels[j][-1]] ]
        labels = [i for j in range(len(labels)) for i in labels[j][:labels[j][-1]] ]

        metrics = {}
        metrics['accuracy'] = accuracy_score(labels, preds)
        metrics['f1'] = f1_score(labels, preds, average='macro')
        metrics['recall'] = recall_score(labels, preds, average='macro')
        metrics['precision'] = precision_score(labels, preds, average='macro')

        self.log_dict(metrics, prog_bar=True)
        log_data = {
            f"val_loss": loss.item(),
            "metrics": metrics,
            "epoch": self.current_epoch,
        }
        
        # self.chromosome_logger.log_epoch(log_data)
        callbacks = metrics
        callbacks['val_loss'] = loss.item()
        self.callbacks.append(callbacks)
        print(f'epoch: {self.current_epoch}, val_loss: {loss}, accuracy: {metrics} ')
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
        embed = self.embed
        model = self.recurrent_model
        fc = self.cls_head
        # crf = self.crf
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
                    for n, p in embed.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                # + [
                #     p
                #     for n, p in crf.named_parameters()
                #     if not any(nd in n for nd in no_decay)
                # ],
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
                    for n, p in embed.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                # + [
                #     p
                #     for n, p in crf.named_parameters()
                #     if any(nd in n for nd in no_decay)
                # ],
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

    def total_params(self):
        return sum(p.numel() for p in self.recurrent_model.parameters())

    def reset_weights(self):
        self.cls_head.reset_parameters()
        self.recurrent_model.reset_parameters()

    @staticmethod
    def add_learning_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--learning_rate", default=2e-5, type=float)
        parser.add_argument("--epsilon", default=1e-8, type=float)
        parser.add_argument("--warmup_steps", default=0, type=int)
        parser.add_argument("--weight_decay", default=0.0, type=float)
        return parser

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--num_layers", default=1, type=int)
        parser.add_argument("--batch_first", default=True, type=bool)
        parser.add_argument("--bidirection", default=True, type=bool)
        parser.add_argument("--hidden_size", default=128, type=int)
        parser.add_argument("--dropout", default=0.1, type=float)
        parser.add_argument("--freeze_embed", action="store_true")
        parser.add_argument("--use_simple_cls", action="store_true")
        return parser

class ClsHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, hidden_size, dropout, num_labels):
        super().__init__()
        self.dense = nn.Linear(hidden_size * 2, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(hidden_size, num_labels)

    def forward(self, x, **kwargs):
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        x = nn.functional.softmax(x, dim=-1)
        return x

    def reset_parameters(self):
        self.dense.reset_parameters()
        self.out_proj.reset_parameters()

class SimpleClsHead(nn.Module):

    def __init__(self, hidden_size, num_labels):
        super().__init__()
        self.dense = nn.Linear(hidden_size * 2, num_labels)

    def forward(self, x, **kwargs):
        # x = torch.tanh(x)
        x = self.dense(x)
        return x

    def reset_parameters(self):
        self.dense.reset_parameters()

class GloveEmbedding(nn.Module):
    def __init__(self, glove_dir, vocab):
        super().__init__()

        self.vocab = vocab
        self.glove_dir = glove_dir
        self.embed_dim = int(glove_dir.split('.')[-2][:-1]) 
        self.token_emb = self.init_token_emb()

    def init_token_emb(self):
        vocabulary = self.vocab
        embeddings_index = {} # empty dictionary of GloVe
        f = open(self.glove_dir, encoding="utf-8")

        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        f.close()

        num_tokens = len(vocabulary) + 2
        embedding_dim = self.embed_dim

        word_index = dict(zip(vocabulary, range(1, len(vocabulary)+1))) # dict of Vocab Conll
        # Prepare embedding matrix
        embedding_matrix = np.zeros((num_tokens, embedding_dim))

        # Index of dict GloVe fit index of Vocab Conll + 1  
        for word, i in word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector            

        # Calculate unk_emb
        unk_embed = np.mean(embedding_matrix,axis=0,keepdims=True) 
        embedding_matrix[-1] = unk_embed

        for word, i in word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is None:
                embedding_matrix[i] = unk_embed
                # [PAD] = embedding_matrix[0] = [0,0,...0]
                # <unk> = embedding_matrix[-1] = unk_embed

        token_emb = nn.Embedding.from_pretrained(torch.from_numpy(embedding_matrix).float())
        return token_emb

    def forward(self, input_ids):
        # INPUT : torch tensor OF INDEXES OF SENTENCE
        # OUTPUT : torch tensor shape = (sentence_max_length, embed_dim) 
        
        x = self.token_emb(input_ids)
        return x 