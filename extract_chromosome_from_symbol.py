import argparse
import ast
from problem.abstract_problem import DataProblem
from problem import NLPFunctionSet

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Extract chromosome from symbol")
    parser = DataProblem.add_arguments(parser)
    parser.add_argument("--symbol", default="", help="Input symbols")
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

    if args.symbol == "":
        symbols = input("Input symbols: ")
    else:
        symbols = args.symbol

    symbols = ast.literal_eval(symbols)
    problem = DataProblem(args)
    problem.function_set = NLPFunctionSet.return_func_name()
    R1, R2, R3, R4 = problem._get_chromosome_range()
    terminal_names = problem.terminal_name
    adf_terminal_names = problem.adf_terminal_name
    adf_names = problem.adf_name
    function_names = [entry["name"] for entry in NLPFunctionSet.return_func_name()]

    chromosome = []
    for name in symbols:
        if name in function_names:
            index = function_names.index(name)
        elif name in adf_names:
            index = adf_names.index(name) + R1
        elif name in adf_terminal_names:
            index = adf_terminal_names.index(name) + R2
        elif name in terminal_names:
            index = terminal_names.index(name) + R3
        else:
            raise Exception(f"{name} not in any name set")
        chromosome.append(index)
    print(chromosome)
