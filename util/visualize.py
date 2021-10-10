import graphviz
from typing import List

from network.tree import Tree


def make_graph(symbols: List, arity: List, gene_types: List, filename="solution.gv"):
    tree = Tree(symbols, arity, gene_types)
    graph = graphviz.Digraph(comment="Best solution", filename=filename, format="png")
    node_counter = 1

    queue = [tree.root]
    while len(queue):
        parent = queue.pop(0)
        parent_id = f"node_{node_counter}"
        graph.node(parent_id, parent.value)
        node_counter += 1

        for child in parent.child_list:
            child_id = f"node_{node_counter}"
            graph.node(child_id, child.value)
            graph.edge(parent_id, child_id)
            node_counter += 1

    graph.engine = "dot"
    graph.save()
    graph.render()
