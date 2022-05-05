import cgp
import nolds
import numpy as np
import warnings
import matplotlib.pyplot as plt

from PIL import Image

chaotic_parameter_space = np.arange(3.6, 4.0, 0.1)


def objective(individual):
    m = 10
    f = individual.to_func()
    invalid_fitness = len(chaotic_parameter_space) * 500.0 + 99999.0

    sequences = np.zeros((len(chaotic_parameter_space), 500), "float")
    sequences[:, 0] = 0.5
    # todo rewrite to use numpy better

    for param_index, A in enumerate(chaotic_parameter_space):
        for i in range(1, 500):
            try:
                sequences[param_index, i] = f([A, sequences[param_index, i - 1]])[0]
            except ZeroDivisionError:
                individual.fitness = invalid_fitness
                return individual

    duplicates = 0
    for sequence in sequences:
        duplicates += (len(sequence) - len(np.unique(sequence))) * m

    lyapunov_cumulative = 0
    if duplicates == 0:
        for sequence in sequences:
            # calculate Lyapunov exponent of the series
            try:
                lyapunov = nolds.lyap_r(sequence)
            except (ZeroDivisionError, ValueError):
                individual.fitness = invalid_fitness
                return individual

            if lyapunov < 0:
                lyapunov_cumulative += lyapunov

        if lyapunov_cumulative < 0:
            individual.fitness = -lyapunov_cumulative
        if lyapunov_cumulative == 0:
            print(f"Good I found one {individual.to_sympy()}")
            individual.fitness = 0.0
    else:
        individual.fitness = float(duplicates)

    return individual


def test_expr(A, x):
    # x_1*(-x_0*x_1 + x_0 - x_1)
    return x * (-A * x + A - x)


def test():
    results = np.zeros(1000)
    results[0] = 0.6
    for i in range(1, 1000):
        results[i] = test_expr(3.6, results[i - 1])

    plt.plot(results, 'bo')
    plt.show()


def bifurcation(f, a=np.linspace(2, 4, 1000), iterations=1000):
    fig, ax = plt.subplots(1, 1)
    x = 0.5

    for _ in range(iterations):
        x = f(a, x)
        ax.plot(a, x, ',k', alpha=0.25)

    plt.title("x * (-A * x + A - x)")
    plt.xlabel('A')
    plt.ylabel('x')
    plt.savefig("bifurcation.png")


if __name__ == '__main__':
    bifurcation(test_expr)
    exit()

    warnings.filterwarnings("ignore")
    population_params = {"n_parents": 10, "mutation_rate": 0.5, "seed": 10}

    genome_params = {
        "n_inputs": 2,
        "n_outputs": 1,
        "n_columns": 10,
        "n_rows": 3,
        "levels_back": 3,
        "primitives": (cgp.Add, cgp.Sub, cgp.Div, cgp.Mul, cgp.ConstantFloat),
    }

    ea_params = {"n_offsprings": 100, "n_breeding": 200, "tournament_size": 2, "n_processes": 4}

    evolve_params = {"max_generations": 100000000, "min_fitness": 0.0}

    pop = cgp.Population(**population_params, genome_params=genome_params)
    ea = cgp.ea.MuPlusLambda(**ea_params)

    cgp.evolve(pop, objective, ea, **evolve_params, print_progress=True)
