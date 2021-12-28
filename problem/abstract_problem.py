from abc import ABC, abstractmethod
from argparse import ArgumentParser
import numpy as np

from evolution import GeneType
from network import ModuleTree
from util.visualize import make_graph

from typing import List, Tuple


class Problem(ABC):
    def __init__(self, args):
        # self._hparams = AttributeDict()
        # self.parse_args(args)
        # self.hparams = self.parse_args(args)
        self.hparams = args
        self.function_set = None
        self.adf_name = self._create_adf_names()
        self.adf_terminal_name = self._create_adf_terminal_names()
        self.terminal_name = self._create_terminal_set_names()

    def _get_chromosome_range(self) -> Tuple[int, int, int, int]:
        R1 = len(self.function_set)
        R2 = R1 + self.hparams.num_adf
        R3 = R2 + self.hparams.max_arity
        R4 = R3 + self.hparams.num_terminal
        return R1, R2, R3, R4

    def get_feasible_range(self, idx) -> Tuple[int, int]:
        # Generate lower_bound, upper_bound for gene at given index of chromosome
        R1, R2, R3, R4 = self._get_chromosome_range()
        # gene at index idx belong to one of the given mains
        total_main_length = self.hparams.num_main * self.hparams.main_length
        if idx < total_main_length:
            if idx % self.hparams.main_length < self.hparams.h_main:
                # Head of main: adf_set and function_set
                return 0, R2
            else:
                # Tail of main: terminal_set
                return R3, R4
        if (idx - total_main_length) % self.hparams.adf_length < self.hparams.h_adf:
            # Head of ADF: function_set
            return 0, R1
        else:
            # Tail of ADF: adf_terminal_set
            return R2, R3

    def parse_chromosome(self, chromosome: np.array, function_set, return_adf=True):
        # self.replace_value_with_symbol(individual)

        total_main_length = self.hparams.num_main * self.hparams.main_length
        all_main_func = []
        adf_func = {}

        for i in range(self.hparams.num_adf):
            start_idx = total_main_length + i * self.hparams.adf_length
            end_idx = start_idx + self.hparams.adf_length
            sub_chromosome = chromosome[start_idx:end_idx]
            adf = self.parse_tree(sub_chromosome, function_set)
            adf_func[f"a{i + 1}"] = adf

        for i in range(self.hparams.num_main):
            start_idx = i * self.hparams.main_length
            end_idx = start_idx + self.hparams.main_length
            sub_chromosome = chromosome[start_idx:end_idx]
            main_func = self.parse_tree(sub_chromosome, function_set)
            # main_func.assign_adfs(main_func.root, adf_func)
            all_main_func.append(main_func)

        if return_adf:
            return all_main_func, adf_func
        else:
            return all_main_func

    def parse_tree(self, sub_chromosome, function_set):
        symbols, arity, gene_types = self.replace_value_with_symbol(sub_chromosome)
        return ModuleTree(symbols, arity, gene_types, function_set)

    def replace_value_with_symbol(
        self, chromosome: np.array
    ) -> Tuple[List, List, List]:
        # create GEP symbols from integer chromosome
        symbols = []
        arity = []
        gene_types = []
        R1, R2, R3, R4 = self._get_chromosome_range()
        for i, value in enumerate(chromosome):
            value = int(value)
            if value >= R3:
                symbols.append(self.terminal_name[value - R3])
                arity.append(0)
                gene_types.append(GeneType.TERMINAL)
            elif value >= R2:
                symbols.append(self.adf_terminal_name[value - R2])
                arity.append(0)
                gene_types.append(GeneType.ADF_TERMINAL)
            elif value >= R1:
                symbols.append(self.adf_name[value - R1])
                arity.append(self.hparams.max_arity)
                gene_types.append(GeneType.ADF)
            else:
                symbols.append(self.function_set[value]["name"])
                arity.append(self.function_set[value]["arity"])
                gene_types.append(GeneType.FUNCTION)
        return symbols, arity, gene_types

    @abstractmethod
    def evaluate(self, chromosome: np.array):
        pass

    def _create_adf_names(self):
        return [f"a{i + 1}" for i in range(self.hparams.num_adf)]

    def _create_adf_terminal_names(self):
        return [f"t{i + 1}" for i in range(self.hparams.max_arity)]

    def _create_terminal_set_names(self):
        return [f"x{i + 1}" for i in range(self.hparams.num_terminal)]

    @staticmethod
    def add_arguments(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--num_main", default=2, type=int)
        parser.add_argument("--num_adf", default=2, type=int)
        parser.add_argument("--h_main", default=4, type=int)
        parser.add_argument("--h_adf", default=3, type=int)

        # parser.add_argument("--num_terminal", default=2, type=int)
        parser.add_argument("--max_arity", default=3, type=int)

        parser.add_argument("--pop_size", default=10, type=int)

        return parser

    def get_bounds(self):
        lb, ub = list(zip(*[self.get_feasible_range(i) for i in range(self.hparams.D)]))
        return lb, ub

    def make_graph(self, chromosome, prefix=""):
        total_main_length = self.hparams.num_main * self.hparams.main_length

        for i in range(self.hparams.num_adf):
            start_idx = total_main_length + i * self.hparams.adf_length
            end_idx = start_idx + self.hparams.adf_length
            sub_chromosome = chromosome[start_idx:end_idx]
            symbols, arity, gene_types = self.replace_value_with_symbol(sub_chromosome)
            make_graph(symbols, arity, gene_types, filename=f"{prefix}.ADF_{i}.gv")

        for i in range(self.hparams.num_main):
            start_idx = i * self.hparams.main_length
            end_idx = start_idx + self.hparams.main_length
            sub_chromosome = chromosome[start_idx:end_idx]
            symbols, arity, gene_types = self.replace_value_with_symbol(sub_chromosome)
            make_graph(symbols, arity, gene_types, filename=f"{prefix}.main_{i}.gv")


class DataProblem(Problem):
    def evaluate(self, chromosome: np.array):
        pass
