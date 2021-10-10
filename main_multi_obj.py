import argparse
import pytorch_lightning as pl

from problem import NLPProblemMultiObj, DataModule, LightningRecurrent
from evolution import MultiObjectiveOptimizer

import logging

logging.disable(logging.CRITICAL)


def parse_args():
    parser = argparse.ArgumentParser()
    parser = NLPProblemMultiObj.add_arguments(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    parser = DataModule.add_argparse_args(parser)
    parser = DataModule.add_cache_arguments(parser)
    parser = LightningRecurrent.add_model_specific_args(parser)
    parser = LightningRecurrent.add_learning_specific_args(parser)
    parser = MultiObjectiveOptimizer.add_optimizer_specific_args(parser)
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

    # solve source problems
    problem = NLPProblemMultiObj(args)

    # create optimizer
    optimizer = MultiObjectiveOptimizer(args)

    # Optimize architecture
    population, objs = optimizer.ga(problem)

    for i, idv in enumerate(population):
        symbols, _, _ = problem.replace_value_with_symbol(population[i])
        print(f"Individual {i + 1}: {objs[i]}, chromosome: {symbols}")
        problem.make_graph(idv, prefix=f"{args.task_name}.idv_{i+1}")

    # build and save model
    # lb, ub = problem.get_bounds()
    # model = amt.MultinomialModel(population, lb, ub)
    # amt.util.save_model(model, args.task_name)


if __name__ == "__main__":
    main()
