from copy import deepcopy

import torch.nn as nn

from evolution import GeneType
from .tree import Node, Tree

from typing import Optional, Dict, List


class ModuleNode(nn.Module):
    def __init__(
        self,
        node,
        function_set,
    ):
        super().__init__()
        self.node = node
        self.function_set = function_set

    def init_node_module(self, dim):
        if (
            self.node.node_type == GeneType.TERMINAL
            or self.node.node_type == GeneType.ADF_TERMINAL
            or self.node.node_type == GeneType.ADF
        ):
            return
        # elif self.node.node_type == GeneType.ADF:
        #     self.node_module.init_tree_module(self.node_module.root, dim)
        #     self.node_module = self.node_module.root
        else:
            self.add_module(
                "node_module", getattr(self.function_set, self.node.value)(dim)
            )

    def init_child_list(self, child_list):
        self.add_module("child_list", nn.ModuleList(child_list))

    def assign_adf(self, adf_dict):
        self.add_module("node_module", adf_dict[self.node.value])

    def forward(self, input_dict):
        if (
            self.node.node_type == GeneType.TERMINAL
            or self.node.node_type == GeneType.ADF_TERMINAL
        ):
            return input_dict[self.node.value]
        elif self.node.node_type == GeneType.ADF:
            for i, child in enumerate(self.child_list):
                copy_input_dict = {}
                for k, v in input_dict.items():
                    copy_input_dict[k] = v.detach().clone()
                input_dict[f"t{i + 1}"] = child(copy_input_dict)
            return self.node_module(input_dict)
        return self.node_module(*[child(input_dict) for child in self.child_list])


class ModuleTree(nn.Module):
    def __init__(self, symbols: List, arity: List, gene_types: List, function_set):
        super().__init__()
        self.symbols = symbols
        self.arity = arity
        self.gene_types = gene_types
        self.function_set = function_set
        # self.root: Optional[ModuleNode] = None
        self.tree_structure = Tree(symbols, arity, gene_types)

    def init_tree(self, default_dim):
        root = self.init_tree_module(self.tree_structure.root, default_dim)
        self.add_module("root", root)
        # self.init_tree_module_list(self.root)

    def init_tree_module(self, node, default_dim: int):
        # Postorder
        module_child_list = []
        for child in node.child_list:
            child_node = self.init_tree_module(child, default_dim)
            module_child_list.append(child_node)
        module_node = ModuleNode(node, self.function_set)
        module_node.init_node_module(default_dim)
        module_node.init_child_list(module_child_list)
        return module_node

    def assign_adfs(self, node: ModuleNode, adf_dict: Dict):
        for child in node.child_list:
            if child.node.node_type == GeneType.ADF:
                child.assign_adf(adf_dict)
                # print(self.chromosome)
                # child.module = adf_dict[child.value]
            self.assign_adfs(child, adf_dict)
        if node.node.node_type == GeneType.ADF:
            node.assign_adf(adf_dict)

    def forward(self, input_dict: Dict):
        return self.root(input_dict)
