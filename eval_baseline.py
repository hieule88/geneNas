import argparse

import pytorch_lightning as pl

from problem import (
    DataModule,
    BaselineProblem,
    LightningBERTSeqCls,
    LightningBERTLSTMSeqCls,
)
from evolution import Optimizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser = BaselineProblem.add_arguments(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    parser = DataModule.add_argparse_args(parser)
    parser = DataModule.add_cache_arguments(parser)
    parser = Optimizer.add_optimizer_specific_args(parser)
    parser = LightningBERTLSTMSeqCls.add_model_specific_args(parser)
    parser = LightningBERTSeqCls.add_learning_specific_args(parser)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--early_stop", type=int, default=0)
    parser.add_argument("--use_lstm", action="store_true")

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


def main_source():
    # get args
    args = parse_args()

    # solve source problems
    problem = BaselineProblem(args, use_lstm=args.use_lstm)
    problem.progress_bar = 10
    problem.weights_summary = "top"
    if args.early_stop > 0:
        problem.early_stop = args.early_stop
    problem.evaluate(None)


# def main_target():
#     # get args
#     args = parse_args()

#     # load source models
#     names, models = amt.util.load_models()

#     # solve source problems
#     problem = GLUEProblem(args)

#     # create optimizer
#     optimizer = Optimizer(args)

#     # Optimize architecture
#     population, fitness = optimizer.transfer_ga(problem, models)

#     # build and save model
#     lb, ub = problem.get_bounds()
#     model = amt.MultinomialModel(population, lb, ub)
#     amt.util.save_model(model, args.task_name)


def main():
    main_source()
    # main_target()


if __name__ == "__main__":
    main()
