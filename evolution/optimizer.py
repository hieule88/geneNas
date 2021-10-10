from argparse import ArgumentParser
import numpy as np
import pickle
from datetime import datetime
import os

from .operator import Operator, MultiObjectiveOperator


class Optimizer:
    def __init__(self, args):
        self.D = args.chromosome_length
        self.N = args.popsize
        self.T = args.num_iter
        self.operator = Operator(None, None)
        self.save_dict_path = args.save_dict_path

    def ga(self, problem, return_best=True):

        taskname = problem.hparams.task_name
        lb, ub = list(zip(*[problem.get_feasible_range(i) for i in range(self.D)]))
        self.operator.lb = lb
        self.operator.ub = ub

        if self.save_dict_path is None:
            today = datetime.today().strftime("%Y-%m-%d")
            self.save_dict_path = f"{taskname}.gene_nas.{today}.pkl"

        if not os.path.exists(self.save_dict_path):
            # initialization
            population = np.random.randint(low=lb, high=ub, size=(self.N, self.D))
            # first evaluation
            fitness = [problem.evaluate(population[i, :]) for i in range(self.N)]
            start_generation = 1
        else:
            save_dict = Optimizer.load_checkpoint(self.save_dict_path)
            population = save_dict["population"]
            fitness = save_dict["fitness"]
            start_generation = save_dict["num_generation"] + 1
            print(
                f"LOAD FROM CHECKPOINT {self.save_dict_path} AT GENERATION {start_generation}"
            )

        for t in range(start_generation, self.T):
            print(f"\nGENERATION: {t}\n")
            # reproduction
            offspring = self.operator.uniform_crossover(population)
            offspring = self.operator.mutate(offspring)

            # evaluation on offspring
            offspring_fitness = [
                problem.evaluate(offspring[i, :]) for i in range(self.N)
            ]

            # selection
            population, fitness = self.operator.select(
                population, fitness, offspring, offspring_fitness
            )

            # save checkpoint
            Optimizer.save_checkpoint(t, population, fitness, self.save_dict_path)

        # output
        if return_best:
            return self.best_population(population, fitness)
        else:
            return population, fitness

    @staticmethod
    def save_checkpoint(num_generation, population, fitness, save_dict_path):
        save_dict = {
            "num_generation": num_generation,
            "population": population,
            "fitness": fitness,
        }
        with open(save_dict_path, "wb") as f:
            pickle.dump(save_dict, f)

    @staticmethod
    def load_checkpoint(save_dict_path):
        with open(save_dict_path, "rb") as f:
            save_dict = pickle.load(f)
        return save_dict

    def best_population(self, population, fitness):
        return population[0], fitness[0]

    @staticmethod
    def add_optimizer_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--popsize", default=20, type=int)
        parser.add_argument("--num_iter", default=100, type=int)
        parser.add_argument("--save_dict_path", default=None)
        return parser


class MultiObjectiveOptimizer(Optimizer):
    def __init__(self, args):
        super().__init__(args)
        self.operator = MultiObjectiveOperator(None, None)

    def best_population(self, population, fitness):
        return self.operator.best_front(population, fitness)
