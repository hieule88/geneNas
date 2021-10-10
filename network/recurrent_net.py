from copy import deepcopy
import math
import torch
import torch.nn as nn

from .module_tree import ModuleTree

from typing import List, Dict


class RecurrentNet(nn.Module):
    def __init__(
        self,
        cells: List[ModuleTree],
        adfs: Dict,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        batch_first: bool = False,
        bidirection: bool = False,
    ):
        # Need to mimic Pytorch RNN as much as possible
        super().__init__()
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.bidirection = bidirection

        self.fc = nn.Linear(input_size, hidden_size)

        self.num_mains = len(cells)
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        # self.layers = []
        for _ in range(num_layers):
            cell_list = nn.ModuleList()
            adfs_copy = deepcopy(adfs)
            for k, adf in adfs_copy.items():
                adf.init_tree(self.hidden_size)
                adfs_copy[k] = adf
            for cell in cells:
                new_cell = deepcopy(cell)
                new_cell.init_tree(self.hidden_size)
                new_cell.assign_adfs(new_cell.root, adfs_copy)
                cell_list.append(new_cell)
            self.layers.append(cell_list)
            # print(self.layers[-1]._modules)
        # print(self.layers)
        # self.hidden_size = self

        self.init_weights()

    def init_weights(self):
        for layer in self.layers:
            for cell in layer:
                std = 1.0 / math.sqrt(self.hidden_size)
                for weight in cell.parameters():
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

            input_dict = {"x1": x_t}
            for i, states in enumerate(hidden_states):
                input_dict[f"x{i + 2}"] = states

            new_hidden_states = []
            for cell in layer:
                cell_output = cell(input_dict)
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
