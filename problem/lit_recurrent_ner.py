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

class LightningRecurrent_NER(pl.LightningModule):
    def __init__(
        self,
        model_name_or_path: str,
        vocab,
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
        unfreeze_embed: bool = True,
        use_simple_cls: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.num_labels = num_labels
        self.num_val_dataloader = num_val_dataloader

        # self.config = AutoConfig.from_pretrained(
        #     model_name_or_path, num_labels=num_labels
        # )
        # self.embed = AutoModel.from_pretrained(model_name_or_path, config=self.config)
        self.embed = GloveEmbedding(glove_dir = model_name_or_path, vocab = vocab)
        if unfreeze_embed:
            for param in self.embed.parameters():
                param.requires_grad = False

        self.rnn_dropout = nn.Dropout(p=dropout)

        if not use_simple_cls:
            self.cls_head = ClsHead(hidden_size, dropout, num_labels)
        else:
            self.cls_head = SimpleClsHead(hidden_size, dropout, num_labels)

        self.chromosome_logger: Optional[ChromosomeLogger] = None
        self.metric = None

    def init_metric(self, metric):
        self.metric = metric

    def init_model(self, cells, adfs):
        recurrent_model = RecurrentNet(
            cells,
            adfs,
            128,
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
            labels = inputs.pop("labels")

        x = self.embed(**inputs)[0]

        x = nn.Linear(50,1)(x)

        # if x.isnan().any():
        #     raise NanException(f"NaN after embeds")
        x, hiddens = self.recurrent_model(x, hiddens)
        x = self.rnn_dropout(x)
        # if x.isnan().any():
        #     raise NanException(f"NaN after recurrent")

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
        return loss, logits, hiddens

    def training_step(self, batch, batch_idx, hiddens=None):
        loss, _, hiddens = self(hiddens, **batch)
        return {"loss": loss, "hiddens": hiddens}

    def tbptt_split_batch(self, batch, split_size):
        num_splits = None
        split_dict = {}
        for k, v in batch.items():
            if k == "labels":
                split_dict[k] = v
                continue
            else:
                split_dict[k] = torch.split(
                    v, split_size, int(self.hparams.batch_first)
                )
                assert (
                    num_splits == len(split_dict[k]) or num_splits is None
                ), "mismatched splits"
                num_splits = len(split_dict[k])

        new_batch = []
        for i in range(num_splits):
            batch_dict = {}
            for k, v in split_dict.items():
                if k == "labels":
                    batch_dict[k] = v
                else:
                    batch_dict[k] = v[i]
            new_batch.append(batch_dict)

        return new_batch

    def validation_step(self, batch, batch_idx):
        val_loss, logits, _ = self(None, **batch)

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
                self.chromosome_logger.log_epoch(log_data)
            return

        preds = torch.cat([x["preds"] for x in outputs]).detach().cpu().numpy()
        labels = torch.cat([x["labels"] for x in outputs]).detach().cpu().numpy()
        loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.log("val_loss", loss, prog_bar=True)
        if np.all(preds == preds[0]):
            metrics = {self.metric.name: 0}
        else:
            metrics = self.metric.compute(predictions=preds, references=labels)
        self.log_dict(metrics, prog_bar=True)
        log_data = {
            f"val_loss": loss.item(),
            "metrics": metrics,
            "epoch": self.current_epoch,
        }
        self.chromosome_logger.log_epoch(log_data)
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
        model = self.recurrent_model
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
        parser.add_argument("--unfreeze_embed", action="store_false")
        parser.add_argument("--use_simple_cls", action="store_true")
        return parser


# class CustomNonPaddingTokenLoss(nn.CrossEntropyLoss()):
#     def __init__(self):
#         super().__init__()

#     def call(self, y_true, y_pred):
#         loss = self(y_true, y_pred)
#         mask = tf.cast((y_true > 0), dtype=tf.float32)
#         loss = loss * mask
#         return tf.reduce_sum(loss) / tf.reduce_sum(mask)

class ClsHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, hidden_size, dropout, num_labels):
        super().__init__()
        self.dense = nn.Linear(hidden_size * 2, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(hidden_size, num_labels)
        self.timeDistributed = TimeDistributed(self.dense)

    def forward(self, x, **kwargs):
        x = self.timeDistributed(x)
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

    def reset_parameters(self):
        self.dense.reset_parameters()
        self.out_proj.reset_parameters()

class SimpleClsHead(nn.Module):

    def __init__(self, hidden_size, dropout, num_labels):
        super().__init__()
        self.dense = nn.Linear(hidden_size * 2, num_labels)

    def forward(self, x, **kwargs):
        x = torch.tanh(x)
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

# from datasets import load_dataset
# import tensorflow as tf
# from collections import OrderedDict


# def get_vocab(conll_data):
#     all_tokens = sum(conll_data["train"]["tokens"], [])
#     all_tokens_array = np.array(list(map(str.lower, all_tokens)))

#     counter = Counter(all_tokens_array)

#     vocab_size = len(counter)

#     vocabulary = [token for token, count in counter.most_common(vocab_size - 2)]
#     return vocabulary

# import os
# par_root = os.path.abspath('..')
# root = os.getcwd()
# glove_dir = os.path.join(par_root, 'glove/glove.6B.200d.txt')
# conll_data = load_dataset("conll2003")

# glove_test = GloveEmbedding(glove_dir = glove_dir, vocab = get_vocab(conll_data))

# # print(glove_test.embedding_matrix.shape)
# input = torch.LongTensor([1,0,1])
# print(input)
# with torch.no_grad():
#     print(glove_test.forward(input))
