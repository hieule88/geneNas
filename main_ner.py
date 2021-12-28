import argparse

import pytorch_lightning as pl

from problem import DataModule
from problem.lit_recurrent_ner import LightningRecurrent_NER
from problem.ner_problem import NERProblem
from evolution import Optimizer
import warnings
warnings.filterwarnings("ignore")

def parse_args():
    parser = argparse.ArgumentParser()
    parser = NERProblem.add_arguments(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    parser = DataModule.add_argparse_args(parser)
    parser = DataModule.add_cache_arguments(parser)
    parser = LightningRecurrent_NER.add_model_specific_args(parser)
    parser = LightningRecurrent_NER.add_learning_specific_args(parser)
    parser = Optimizer.add_optimizer_specific_args(parser)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    args.num_terminal = args.num_main + 1
    args.l_main = args.h_main * (args.max_arity - 1) + 1
    args.l_adf = args.h_adf * (args.max_arity - 1) + 1
    args.main_length = args.h_main + args.l_main
    args.adf_length = args.h_adf + args.l_adf
    args.chromosome_length = (
        args.num_main * args.main_length + args.num_adf * args.adf_length
    )
    args.D = args.chromosome_length
    args.mutation_rate = args.adf_length / args.chromosome_length

    return args


def main():
    # get args
    args = parse_args()

    # solve problems
    problem = NERProblem(args)
    # problem.progress_bar = 10
    problem.weights_summary = "top"

    optimizer = Optimizer(args)

    print('Run')

    population, objs = optimizer.ga(problem)

    # chromosome = ast.literal_eval(population)
    symbols, _, _ = problem.replace_value_with_symbol(population)
    print(f"Best Individual: {objs}, chromosome: {symbols}")
    problem.make_graph(population, prefix=f"{args.task_name}.idv_0")
    
    # chromosome = [int(x) for x in chromosome]
    problem.evaluate(population)

if __name__ == "__main__":
    main()
