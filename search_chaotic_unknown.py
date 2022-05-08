import cgp
import nolds
import numpy as np
import warnings
import matplotlib.pyplot as plt

chaotic_parameter_space = np.arange(3.6, 4.0, 0.1)
seed = 10
history = {}
history["fitness_parents"] = []
regression_data = (None, None) # initial X, 2D np array of A and values


def recording_callback(pop):
    history["fitness_parents"].append(pop.fitness_parents())


def fitness_chaotic(individual):
    m = 10
    f = individual.to_func()
    invalid_fitness = -1 * (len(chaotic_parameter_space) * 500.0 + 99999.0)

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
            except:
                individual.fitness = invalid_fitness
                return individual

            if lyapunov < 0:
                lyapunov_cumulative += lyapunov

        if lyapunov_cumulative < 0:
            individual.fitness = lyapunov_cumulative
        if lyapunov_cumulative == 0:
            print(f"Good I found one {individual.to_sympy()}")
            individual.fitness = 0.0
    else:
        individual.fitness = -1.0 * duplicates

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


def evolve_new_chaos(mutation_rate=0.5, n_columns=10, n_rows=3, lback=3, turn_size=2,
                 primitives=(cgp.Add, cgp.Sub, cgp.Div, cgp.Mul, cgp.ConstantFloat)):
    genome_params = {
        "n_inputs": 2,
        "n_outputs": 1,
        "n_columns": n_columns,
        "n_rows": n_rows,
        "levels_back": lback,
        "primitives": primitives,
    }

    evolve_params = {"max_generations": 1000, "min_fitness": 1.0}

    pop = cgp.Population(n_parents=10, mutation_rate=mutation_rate, seed=seed, genome_params=genome_params)
    ea = cgp.ea.MuPlusLambda(n_offsprings=100, n_breeding=200, tournament_size=turn_size, n_processes=4)

    cgp.evolve(pop, fitness_chaotic, ea, **evolve_params, print_progress=True, callback=recording_callback)


if __name__ == '__main__':
    # bifurcation(test_expr)

    warnings.filterwarnings("ignore")

    evolve_new_chaos()
