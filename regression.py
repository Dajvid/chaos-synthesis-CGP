import cgp
import numpy as np
import warnings

seed = 10
regression_data = [None, None, None] # initial X, np.array of A values, np.array of X values


def logistic_map(a, x):
    return a * x * (1 - x)


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
        individual.fitness = -99999999999999.0

    return individual


def evolve_known(data, mutation_rate=0.5, n_columns=10, n_rows=3, lback=3, turn_size=2,
                 primitives=(cgp.Add, cgp.Sub, cgp.Div, cgp.Mul, cgp.ConstantFloat)):
    global regression_data
    regression_data = data

    genome_params = {
        "n_inputs": 2,
        "n_outputs": 1,
        "n_columns": n_columns,
        "n_rows": n_rows,
        "levels_back": lback,
        "primitives": primitives,
    }

    evolve_params = {"max_generations": 1000, "min_fitness": 0.0}

    pop = cgp.Population(n_parents=10, mutation_rate=mutation_rate, seed=seed, genome_params=genome_params)
    ea = cgp.ea.MuPlusLambda(n_offsprings=100, n_breeding=200, tournament_size=turn_size, n_processes=4)

    cgp.evolve(pop, fitness_regression_mse, ea, **evolve_params, print_progress=True)
    print(pop.champion.to_sympy())


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    c = 0.5
    a = np.loadtxt(f"logistic_map_space_{c}.txt")
    x = np.loadtxt(f"logistic_map_{c}.txt")
    evolve_known((c, a, x))
