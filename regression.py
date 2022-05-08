import random
import sys

import cgp
import numpy as np
import warnings

regression_data = [None, None, None] # initial X, np.array of A values, np.array of X values


def logistic_map(a, x):
    return a * x * (1 - x)


def system2(a, x):
    return 2 * x - x * (a + x)


def system3(a, x):
    return x * (a - x - a * x)


def sample_data(f, file_name, a=np.linspace(3.4, 4, 10), iterations=100, start=0.5):
    x = np.zeros((len(a), iterations))
    x[:, 0] = 0.5

    for i in range(1, iterations):
        x[:, i] = f(a, x[:, i - 1])

    np.savetxt(f"{file_name}_{start}.txt", x)
    np.savetxt(f"{file_name}_space_{start}.txt", a)


def fitness_regression_mse(individual):
    global regression_data
    f = individual.to_func()
    a = regression_data[1]
    x = np.zeros(regression_data[2].shape)
    x[:, 0] = regression_data[0]

    try:
        for i in range(1, regression_data[2].shape[-1]):
            x[:, i] = f([a, x[:, i - 1]])[0]

        individual.fitness = -1.0 * np.sum(np.square(x - regression_data[2]))
    except:
        individual.fitness = -np.inf

    return individual


def evolve_known(data, mutpb=0.5, n_columns=6, n_rows=3, lback=6, turn_size=3,
                 fitness=fitness_regression_mse, popsize=100, ngen=1000,
                 primitives=(cgp.Add, cgp.Sub, cgp.Div, cgp.Mul, cgp.ConstantFloat)):
    global regression_data
    regression_data = data
    seed = random.randint(0, 2 ** 32 - 1)

    genome_params = {
        "n_inputs": 2,
        "n_outputs": 1,
        "n_columns": n_columns,
        "n_rows": n_rows,
        "levels_back": lback,
        "primitives": primitives,
    }

    evolve_params = {"max_generations": ngen, "min_fitness": 0.0}

    pop = cgp.Population(n_parents=popsize // 10, mutation_rate=mutpb, seed=seed, genome_params=genome_params)
    ea = cgp.ea.MuPlusLambda(n_offsprings=popsize, n_breeding=2*popsize, tournament_size=turn_size, n_processes=1)

    cgp.evolve(pop, fitness, ea, **evolve_params, print_progress=True)
    print(f"seed was: {seed}")
    return pop


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    c = 0.5
    a = np.loadtxt(f"system3_{c}.txt")
    x = np.loadtxt(f"system3_{c}.txt")
    pop = evolve_known((c, a, x), popsize=100, mutpb=0.5, ngen=np.inf, n_columns=8,
                       primitives=(cgp.Sub, cgp.Mul))
    print(pop.champion.to_sympy())
    # sample_data(system3, "system3", np.linspace(3.5, 4, 10))
