from copy import deepcopy
from typing import List


class Node:
    def __init__(
        self,
        parent,
        child_list,
        value,
        node_type,
    ):
        self.parent = parent
        self.child_list = child_list
        self.value = value
        self.node_type = node_type


class Tree:
    def __init__(self, symbols: List, arity: List, gene_types: List):
        self.symbols = symbols
        self.arity = arity
        self.gene_types = gene_types
        self._init_tree_structure()

    def _init_tree_structure(self):
        symbols = deepcopy(self.symbols)
        arity = deepcopy(self.arity)
        gene_types = deepcopy(self.gene_types)

        # Create tree by BFS
        self.root = Node(
            parent=None,
            child_list=[],
            value=symbols.pop(0),
            node_type=gene_types.pop(0),
        )

        queue = [self.root]
        while len(queue) and len(symbols):
            parent = queue.pop(0)
            parent_arity = arity.pop(0)
            for _ in range(parent_arity):
                gene = symbols.pop(0)
                node_type = gene_types.pop(0)
                node = Node(
                    parent=parent,
                    child_list=[],
                    value=gene,
                    node_type=node_type,
                )
                parent.child_list.append(node)
                queue.append(node)
