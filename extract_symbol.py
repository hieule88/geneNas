import argparse
import ast
import numpy as np
from problem.abstract_problem import DataProblem
from problem import NLPFunctionSet

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Extract symbol from chromosome")
    parser = DataProblem.add_arguments(parser)
    parser.add_argument("--chromosome", default="", help="Input chromosome")
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

    if args.chromosome == "":
        chromosome = input("Input chromosome: ")
    else:
        chromosome = args.chromosome

    chromosome = ast.literal_eval(chromosome)
    chromosome = [int(x) for x in chromosome]
    chromosome = np.array(chromosome)
    problem = DataProblem(args)
    problem.function_set = NLPFunctionSet.return_func_name()
    symbols = problem.replace_value_with_symbol(chromosome)[0]
    print(symbols)
