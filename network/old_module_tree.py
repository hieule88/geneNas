from copy import deepcopy

import torch.nn as nn

from evolution import GeneType

from typing import Optional, Dict, List


class ModuleNode(nn.Module):
    def __init__(
        self,
        parent,
        child_list,
        value,
        node_type,
        function_set,
    ):
        super().__init__()
        self.parent = parent
        self.child_list = child_list
        self.value = value
        self.node_type = node_type
        self.node_module = None
        self.function_set = function_set

    def init_node_module(self, dim):
        if (
            self.node_type == GeneType.TERMINAL
            or self.node_type == GeneType.ADF_TERMINAL
        ):
            self.child_list = []
            return
        elif self.node_type == GeneType.ADF:
            self.node_module.init_tree_module(self.node_module.root, dim)
            # self.node_module = self.node_module.root
        else:
            self.node_module = getattr(self.function_set, self.value)(dim)

    def init_module_list(self):
        self.child_list = nn.ModuleList(self.child_list)

    def forward(self, input_dict):
        if (
            self.node_type == GeneType.TERMINAL
            or self.node_type == GeneType.ADF_TERMINAL
        ):
            return input_dict[self.value]
        elif self.node_type == GeneType.ADF:
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
        self.root: Optional[ModuleNode] = None
        self._init_tree_structure()

    def _init_tree_structure(self):
        symbols = deepcopy(self.symbols)
        arity = deepcopy(self.arity)
        gene_types = deepcopy(self.gene_types)

        # Create tree by BFS
        self.root = ModuleNode(
            parent=None,
            # child_list=[],
            value=symbols.pop(0),
            node_type=gene_types.pop(0),
            function_set=self.function_set,
        )

        queue = [self.root]
        while len(queue) and len(symbols):
            parent = queue.pop(0)
            parent_arity = arity.pop(0)
            for _ in range(parent_arity):
                gene = symbols.pop(0)
                node_type = gene_types.pop(0)
                node = ModuleNode(
                    parent=parent,
                    # child_list=[],
                    value=gene,
                    node_type=node_type,
                    function_set=self.function_set,
                )
                parent.child_list.append(node)
                queue.append(node)

    def init_tree(self, default_dim):
        self.init_tree_module(self.root, default_dim)
        self.init_tree_module_list(self.root)

    def init_tree_module(self, node: ModuleNode, default_dim: int):
        # Postorder
        for child in node.child_list:
            self.init_tree_module(child, default_dim)
        node.init_node_module(default_dim)

    def init_tree_module_list(self, node: ModuleNode):
        # Postorder
        if (
            node.node_type == GeneType.TERMINAL
            or node.node_type == GeneType.ADF_TERMINAL
        ):
            return
        for child in node.child_list:
            self.init_tree_module_list(child)
        node.init_module_list()

    def assign_adfs(self, node: ModuleNode, adf_dict: Dict):
        for child in node.child_list:
            if child.node_type == GeneType.ADF:
                # print(self.chromosome)
                child.module = adf_dict[child.value]
            self.assign_adfs(child, adf_dict)
        if node.node_type == GeneType.ADF:
            node.node_module = adf_dict[node.value]

    def forward(self, input_dict: Dict):
        return self.root(input_dict)
