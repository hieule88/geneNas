import argparse
import pickle

from evolution.operator import MultiObjectiveOperator

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Extract Chromosome")
    parser.add_argument("filepath", type=str, help="Filepath")
    args = parser.parse_args()

    with open(args.filepath, "rb") as f:
        data = pickle.load(f)

    selector = MultiObjectiveOperator(0, 1)
    best_chromosomes = selector.best_front(data["population"], data["fitness"])[
        0
    ].tolist()
    for chromosome in best_chromosomes:
        print(chromosome)
