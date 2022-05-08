import os
import random
import uuid

import cgp
import nolds
import numpy as np
import warnings
import matplotlib.pyplot as plt

chaotic_parameter_space = np.arange(3.6, 4, 0.1)
x_range = (0, 1)
initial_value = 0.5


def fitness_chaotic(individual):
    m = 10
    f = individual.to_func()
    invalid_fitness = -1 * (len(chaotic_parameter_space) * 500.0 + 99999.0)
    sequences = np.zeros((len(chaotic_parameter_space), 500), "float")
    sequences[:, 0] = 0.5
    # enforce both inputs being used
    expr = str(individual.to_sympy())
    if "x_0" not in expr or "x_1" not in expr:
        individual.fitness = invalid_fitness
        return individual

    for param_index, A in enumerate(chaotic_parameter_space):
        for i in range(1, 500):
            try:
                sequences[param_index, i] = f([A, sequences[param_index, i - 1]])[0]
            except ZeroDivisionError:
                individual.fitness = invalid_fitness
                return individual
    duplicates = 0
    out_of_bounds = 0
    for sequence in sequences:
        duplicates += (len(sequence) - len(np.unique(sequence))) * m
        if x_range:
            out_of_bounds += np.sum([sequence < x_range[0]])
            out_of_bounds += np.sum([sequence > x_range[1]])

    lyapunov_cumulative = 0
    if duplicates == 0 and out_of_bounds == 0:
        for sequence in sequences:
            # calculate Lyapunov exponent of the series
            try:
                lyapunov = nolds.lyap_r(sequence)
            except:
                individual.fitness = invalid_fitness
                return individual

            if lyapunov < 0:
                lyapunov_cumulative += lyapunov

        if lyapunov_cumulative < 0:
            individual.fitness = lyapunov_cumulative
        if lyapunov_cumulative == 0:
            individual.fitness = 0.0
    else:
        individual.fitness = -1.0 * duplicates - 1.0 * out_of_bounds

    return individual


def test_expr(i):
    A = i[0]
    x = i[1]
    return [A*x*(1 - x)]


def bifurcation(f, name, directory, lower, upper, iterations=1000, count=1000, extend=True):
    if extend:
        a = np.linspace(lower * 0.9, upper * 1.1, count)
    else:
        a = np.linspace(lower, upper, 1000)

    try:
        fig, ax = plt.subplots(1, 1)
        x = np.repeat(initial_value, len(a))

        for _ in range(iterations):
            x = f([a, x])[0]
            ax.plot(a, x, ',k', alpha=0.25)
    except:
        a = np.linspace(lower * 0.9, upper * 1.1, 1000)
        fig, ax = plt.subplots(1, 1)
        x = np.repeat(initial_value, len(a))

        for _ in range(iterations):
            x = f([a, x])[0]
            ax.plot(a, x, ',k', alpha=0.25)

    plt.title(name)
    plt.xlabel('A')
    plt.ylabel('x')
    plt.savefig(os.path.join(directory, str(uuid.uuid4()) + ".png"))


def evolve_new_chaos(mutation_rate=0.5, n_columns=4, n_rows=3, lback=4, turn_size=3,
                 primitives=(cgp.Add, cgp.Sub, cgp.Mul)):
    seed = random.randint(0, 2 ** 32 - 1)
    genome_params = {
        "n_inputs": 2,
        "n_outputs": 1,
        "n_columns": n_columns,
        "n_rows": n_rows,
        "levels_back": lback,
        "primitives": primitives,
    }

    evolve_params = {"max_generations": 500, "min_fitness": 0.0}

    print(f"seed: {seed}")
    pop = cgp.Population(n_parents=10, mutation_rate=mutation_rate, seed=seed, genome_params=genome_params)
    ea = cgp.ea.MuPlusLambda(n_offsprings=100, n_breeding=200, tournament_size=turn_size, n_processes=1)

    cgp.evolve(pop, fitness_chaotic, ea, **evolve_params, print_progress=True)
    name = str(pop.champion.to_sympy()).replace("x_0", "A").replace("x_1", "x")
    print(f"Result {name}")

    os.mkdir(str(seed))
    bifurcation(pop.champion.to_func(), name, str(seed), chaotic_parameter_space[0], chaotic_parameter_space[-1], extend=False)
    with open(f"{seed}/setting.txt", "w+") as f:
        f.write(f"chaotic parameter space: {chaotic_parameter_space}")
        f.write(f"x range: {x_range}")
        f.write(f"initla value {initial_value}")


if __name__ == '__main__':
    bifurcation(test_expr, "[A*x*(1 - x)]", "", 2.9, 4, extend=False)

    # warnings.filterwarnings("ignore")
    # evolve_new_chaos()
