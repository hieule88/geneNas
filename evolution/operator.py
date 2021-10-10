import numpy as np

from typing import List, Optional


class Operator:
    def __init__(
        self, lower_bound: Optional[List[int]], upper_bound: Optional[List[int]]
    ):
        self.lb = lower_bound
        self.ub = upper_bound

    def uniform_crossover(self, population):

        # extract parameters
        N, D = population.shape

        # select for crossover
        parent1 = population[np.random.permutation(N), :]
        parent2 = population[np.random.permutation(N), :]
        offspring = np.zeros([N, D])

        # create random variable
        r = np.random.rand(N, D)

        # uniform crossover
        index = r >= 0.5
        offspring[index] = parent1[index]
        index = r < 0.5
        offspring[index] = parent2[index]

        return offspring.astype(np.int32)

    def mutate(self, offspring):

        # extract parameters
        N, D = offspring.shape

        # create random variable
        r = np.random.rand(N, D)

        # mutate with p=1/D
        index = r < 1.0 / float(D)
        offspring[index] = np.random.randint(
            low=self.lb, high=self.ub, size=offspring.shape
        )[index]

        return offspring.astype(np.int32)

    def select(self, population, fitness, offspring, offspring_fitness):

        # extract parameters
        N, D = population.shape

        # concat
        inter_population = np.concatenate([population, offspring], axis=0)
        inter_fitness = np.concatenate([fitness, offspring_fitness], axis=0)

        # sort
        index = np.argsort(-inter_fitness)

        # select
        population = inter_population[index[:N], :]
        fitness = inter_fitness[index[:N]]

        return population, fitness


def dominance_test(solution1_objs, solution2_objs) -> int:
    best_is_one = 0
    best_is_two = 0
    assert len(solution1_objs) == len(
        solution2_objs
    ), "Number of objectives does not match"
    num_objs = len(solution1_objs)

    for i in range(num_objs):
        value1 = solution1_objs[i]
        value2 = solution2_objs[i]
        if value1 != value2:
            if value1 < value2:
                best_is_one = 1
            if value1 > value2:
                best_is_two = 1

    if best_is_one > best_is_two:
        result = -1
    elif best_is_two > best_is_one:
        result = 1
    else:
        result = 0

    return result


class MultiObjectiveOperator(Operator):
    @staticmethod
    def compute_ranking(population, objs):
        N, D = population.shape

        # number of solutions dominating solution ith
        dominating_ith = [0 for _ in range(N)]

        # list of solutions dominated by solution ith
        ith_dominated = [[] for _ in range(N)]

        # front[i] contains the list of indices of solutions belonging to front i
        front = [[] for _ in range(N + 1)]

        for p in range(N - 1):
            for q in range(p + 1, N):
                dominance_test_result = dominance_test(objs[p], objs[q])

                if dominance_test_result == -1:
                    ith_dominated[p].append(q)
                    dominating_ith[q] += 1
                elif dominance_test_result == 1:
                    ith_dominated[q].append(p)
                    dominating_ith[p] += 1

        dominance_ranking = np.array([-1] * N, dtype=np.int32)
        for i in range(N):
            if dominating_ith[i] == 0:
                front[0].append(i)
                dominance_ranking[i] = 0

        i = 0
        while len(front[i]) != 0:
            i += 1
            for p in front[i - 1]:
                if p <= len(ith_dominated):
                    for q in ith_dominated[p]:
                        dominating_ith[q] -= 1
                        if dominating_ith[q] == 0:
                            front[i].append(q)
                            dominance_ranking[q] = i
        return dominance_ranking

    @staticmethod
    def compute_density(front, objs):
        size, _ = front.shape
        crowding_density = np.zeros(size)
        if size == 0:
            return crowding_density
        elif size == 1:
            crowding_density[0] = float("inf")
            return crowding_density
        elif size == 2:
            crowding_density[0] = float("inf")
            crowding_density[1] = float("inf")
            return crowding_density

        number_of_objectives = len(objs[0])
        for i in range(number_of_objectives):
            # Sort the population by Obj n
            sorted_objs = np.sort(objs, order="f1")
            sorted_indices = np.argsort(objs, order="f1")
            objective_minn = sorted_objs[0][i]
            objective_maxn = sorted_objs[-1][i]

            # Set the crowding distance
            crowding_density[sorted_indices[0]] = float("inf")
            crowding_density[sorted_indices[-1]] = float("inf")

            for j in range(1, size - 1):
                distance = sorted_objs[j + 1][i] - sorted_objs[j - 1][i]

                # Check if minimum and maximum are the same (in which case do nothing)
                if objective_maxn - objective_minn == 0:
                    pass
                    # LOGGER.warning('Minimum and maximum are the same!')
                else:
                    distance = distance / (objective_maxn - objective_minn)

                crowding_density[sorted_indices[j]] += distance
        return crowding_density

    def sort_population(self, population, objs, num_pop):
        N = num_pop

        ranking = MultiObjectiveOperator.compute_ranking(population, objs)
        acc_zero_indices = objs["f1"] == 0
        ranking[acc_zero_indices] = N + 1  # always sure rank individual <= N
        sorted_ranking_indices = np.argsort(ranking)
        last_rank = ranking[sorted_ranking_indices[N - 1]]
        last_rank_indices = ranking[ranking == last_rank]

        crowding_density = np.zeros(len(population))
        crowding_density[ranking < last_rank] = float("inf")
        crowding_density[ranking > last_rank] = -1
        crowding_density[last_rank_indices] = MultiObjectiveOperator.compute_density(
            population[last_rank_indices], objs[last_rank_indices]
        )

        sorted_pop_indices = np.argsort(crowding_density)[::-1][:N]
        return population[sorted_pop_indices, :], objs[sorted_pop_indices]

    def select(self, population, objs, offspring, offspring_objs):

        # extract parameters
        N, D = population.shape

        # concat
        inter_population = np.concatenate([population, offspring], axis=0)
        inter_objs = np.array(
            objs + offspring_objs,
            dtype=[(f"f{i + 1}", np.float32) for i in range(len(objs[0]))],
        )
        for i in range(len(objs[0]) - 1):
            inter_objs[f"f{i + 1}"] *= -1
        # inter_objs["f1"] *= -1
        # inter_objs["f2"] *= -1
        print(inter_objs)
        # inter_fitness = fitness + offspring_fitness

        # select
        population, objs = self.sort_population(inter_population, inter_objs, num_pop=N)
        for i in range(len(objs[0]) - 1):
            objs[f"f{i + 1}"] *= -1
        # objs["f1"] *= -1
        # objs["f2"] *= -1
        print(objs)
        objs = objs.tolist()

        return population, objs

    def best_front(self, population, objs):
        np_objs = np.array(
            objs, dtype=[(f"f{i+1}", np.float32) for i in range(len(objs[0]))]
        )
        for i in range(len(objs[0]) - 1):
            np_objs[f"f{i + 1}"] *= -1
        # np_objs["f1"] *= -1
        # np_objs["f2"] *= -1
        ranking = MultiObjectiveOperator.compute_ranking(population, np_objs)
        best_ranking_indces = ranking == 0
        for i in range(len(objs[0]) - 1):
            np_objs[f"f{i + 1}"] *= -1
        # np_objs["f1"] *= -1
        # np_objs["f2"] *= -1
        return population[best_ranking_indces], np_objs[best_ranking_indces].tolist()
