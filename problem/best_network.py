from argparse import ArgumentParser
import math

import pytorch_lightning as pl
from transformers import AutoModel, AutoConfig, AdamW, get_linear_schedule_with_warmup
import torch
import torch.nn as nn

from . import NLPProblem, DataModule
from .lit_recurrent import SimpleClsHead, ClsHead
from util.exception import NanException
from network import RecurrentNet
from .function_set import NLPFunctionSet
from .baseline import LightningBERTSeqCls

from typing import Optional, Union, Dict


class BestCell(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.blending1 = NLPFunctionSet.blending(hidden_size)
        self.blending2 = NLPFunctionSet.blending(hidden_size)
        self.ele_prod = NLPFunctionSet.element_wise_product(hidden_size)
        self.tanh = NLPFunctionSet.tanh(hidden_size)

    def forward(self, x, h):
        left = self.blending1(x, x, x)
        right = self.blending2(h, x, x)
        out = self.ele_prod(left, right)
        out = self.tanh(out)
        return out


class BestNetwork(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers: int = 1,
        batch_first: bool = False,
        bidirection: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.bidirection = bidirection

        self.fc = nn.Linear(input_size, hidden_size)

        self.num_layers = num_layers
        self.num_mains = 1
        self.layers = nn.ModuleList()
        for _ in range(self.num_layers):
            self.layers.append(BestCell(hidden_size))

        self.init_weights()

    def init_weights(self):
        for layer in self.layers:
            std = 1.0 / math.sqrt(self.hidden_size)
            for weight in layer.parameters():
                weight.data.uniform_(-std, std)

    # this code dumb, need more optimize
    def forward_unidirection(self, layer, x, hidden_states=None):
        if self.batch_first:
            _, seq_sz, _ = x.size()
        else:
            seq_sz, _, _ = x.size()

        hidden_seq = []
        # x = x.clone()
        # if hidden_states is not None:
        #     hidden_states = [states.clone() for states in hidden_states]
        for t in range(seq_sz):
            x_t = x[:, t, :].unsqueeze(0)
            h_t = hidden_states[0]

            new_hidden_states = []
            cell_output = layer(x_t, h_t)
            new_hidden_states.append(cell_output)
            hidden_states = new_hidden_states
            # hidden_states = [cell(input_dict) for cell in layer]

            hidden_seq.append(hidden_states[0])

        hidden_seq = torch.cat(hidden_seq, dim=0)  # S x B x H
        if self.batch_first:
            hidden_seq = hidden_seq.transpose(0, 1).contiguous()  # B x S x H

        return hidden_seq, hidden_states

    def forward_bidirection(self, layer, x, hidden_states=None):

        left_to_right_hidden = [
            states[0, :, :].unsqueeze(0) for states in hidden_states
        ]  # 1 x B x H
        right_to_left_hidden = [
            states[1, :, :].unsqueeze(0) for states in hidden_states
        ]  # 1 x B x H

        _, _, hidden = x.size()
        assert hidden % 2 == 0, f"Sequence size not divided by 2: {hidden}"
        left_to_right_x = x[:, :, : hidden // 2]
        right_to_left_x = torch.flip(
            x[:, :, hidden // 2 :], dims=[int(self.batch_first)]
        )

        left_to_right_output, left_to_right_hidden = self.forward_unidirection(
            layer, left_to_right_x, left_to_right_hidden
        )
        right_to_left_output, right_to_left_hidden = self.forward_unidirection(
            layer, right_to_left_x, right_to_left_hidden
        )
        right_to_left_output = torch.flip(
            right_to_left_output, dims=[int(self.batch_first)]
        )
        # right_to_left_hidden = torch.flip(right_to_left_hidden, dims=[self.batch_first])

        output = torch.cat([left_to_right_output, right_to_left_output], dim=2)
        hidden_states = []
        for i in range(self.num_mains):
            left_to_right_states = left_to_right_hidden[i]
            right_to_left_states = right_to_left_hidden[i]
            h = [left_to_right_states, right_to_left_states]
            hidden_states.append(
                torch.cat(
                    h,
                    dim=0,
                )
            )

        return output, hidden_states

    def forward(self, x, hidden_states=None):

        if self.batch_first:  # B x S x H
            # left_to_right_x = x
            # right_to_left_x = torch.flip(x, dims=[1])
            # x = torch.cat([left_to_right_x, right_to_left_x], dim=1)
            bs, seq_sz, _ = x.size()
        else:  # S x B x H
            # left_to_right_x = x
            # right_to_left_x = torch.flip(x, dims=[0])
            # x = torch.cat([left_to_right_x, right_to_left_x], dim=0)
            seq_sz, bs, _ = x.size()

        if hidden_states is None:
            hidden_states = [
                torch.zeros(
                    self.num_layers * (self.bidirection + 1), bs, self.hidden_size
                )
                for _ in range(self.num_mains)
            ]
            hidden_states = [states.type_as(x) for states in hidden_states]

        new_hidden_states = [[] for _ in range(self.num_mains)]
        hidden_states = [
            states.view(self.num_layers, (1 + self.bidirection), bs, self.hidden_size)
            for states in hidden_states
        ]
        for i, layer in enumerate(self.layers):
            if i == 0:
                seq_x = [self.fc(x[:, t, :].unsqueeze(0)) for t in range(seq_sz)]
                x = torch.cat(seq_x, dim=0)  # S x B x H
                if self.batch_first:
                    x = x.transpose(0, 1).contiguous()  # B x S x H
                x = torch.cat([x, x], dim=2)

            tmp_hidden_states = [states[i, :, :, :] for states in hidden_states]
            if self.bidirection:
                x, tmp_hidden_states = self.forward_bidirection(
                    layer, x, hidden_states=tmp_hidden_states
                )  # 2 x B x S
            else:
                x, tmp_hidden_states = self.forward_unidirection(
                    layer, x, hidden_states=tmp_hidden_states
                )  # 1 x B x S
            for main_id in range(self.num_mains):
                new_hidden_states[main_id].append(tmp_hidden_states[main_id])

        hidden_states = [torch.cat(states, dim=0) for states in new_hidden_states]

        return x, hidden_states


class BestModel(LightningBERTSeqCls):
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

        self.recurrent_net = BestNetwork(
            self.config.hidden_size,
            hidden_size,
            batch_first=self.hparams.batch_first,
            bidirection=self.hparams.bidirection,
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
        x, _ = self.recurrent_net(x)
        if self.hparams.batch_first:
            x = x[:, 0, :]  # CLS token
        else:
            x = x[0, :, :]
        logits = self.cls_head(x)
        loss = None
        if labels is not None:
            if self.num_labels == 1:
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        return loss, logits

    def configure_optimizers(self):
        "Prepare optimizer and schedule (linear warmup and decay)"
        bert = self.bert
        model = self.recurrent_net
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


class EvalBestModel(NLPProblem):
    def __init__(self, args):
        super().__init__(args)

    def evaluate(self, chromosome):
        trainer = self.setup_trainer()
        model = BestModel(
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
            trainer.save_checkpoint("best_model.ckpt")
            trainer.test(model, test_dataloaders=self.dm.test_dataloader())
        except NanException as e:
            print(e)
