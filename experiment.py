import datetime
import os
import warnings
import time

import cgp
import numpy as np
import math
from matplotlib import pyplot as plt

from regression import fitness_regression_mse, evolve_known

default_kwargs = {
    "fitness": fitness_regression_mse,
    "popsize": 100,
    "mutpb": 0.5,
    "ngen": 500,
    "n_columns": 8,
    "n_rows": 3,
    "lback": 3,
    "turn_size": 2,
    "primitives": (cgp.Add, cgp.Sub, cgp.Mul, cgp.Div, cgp.ConstantFloat)
}


def benchmark_popsize(kwargs, data, output_path="o", popsizes=[20, 100, 250, 500, 1000],
                      evaluations=30, visualize=False):
    max_evals = 50000
    kwargs = kwargs.copy()
    results = np.zeros((evaluations, len(popsizes)))
    for i, popsize in enumerate(popsizes):
        kwargs["popsize"] = popsize
        kwargs["ngen"] = max_evals // popsize
        for j in range(evaluations):
            print(f"Population size benchmark {i * evaluations + j + 1} / {evaluations * len(popsizes)}.")
            pop = evolve_known(data, **kwargs)
            results[j][i] = pop.champion.fitness
            print(f"found solution: {pop.champion.to_sympy()}")

    plt.figure()
    plt.boxplot(results, labels=popsizes, notch=False)
    plt.xlabel = "population size"
    plt.ylabel = "fitness"
    plt.savefig(output_path + ".svg")
    if visualize:
        plt.show()
    np.savetxt(output_path + ".out", results)
    with open(output_path + ".meta", "w+") as f:
        f.write(f"popsizes: {popsizes}")
        f.write(f"kwargs: {kwargs}")


def benchmark_param(kwargs,  data, name, values, evaluations=30, visualize=False, output_path="o",
                    xlabel="variants", ylabel="values", labels=None):
    labels = labels if labels else values
    kwargs = kwargs.copy()
    results = np.zeros((evaluations, len(values)))
    for i, value in enumerate(values):
        kwargs[name] = value
        for j in range(evaluations):
            print(f"Param benchmark {i * evaluations + j + 1} / {evaluations * len(values)}.")
            pop = evolve_known(data, **kwargs)
            results[j][i] = pop.champion.fitness

    plt.figure()
    plt.boxplot(results, labels=labels, notch=False)
    plt.xlabel = xlabel
    plt.ylabel = ylabel
    plt.savefig(output_path + ".svg")
    if visualize:
        plt.show()

    np.savetxt(f"{output_path}-{name}.out", results)
    with open(output_path + ".meta", "w+") as f:
        f.write(f"{name}: {values}")
        f.write(f"kwargs: {kwargs}")


def experiment_find_logistic_map():



if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    out_dir = f"experiments-output-" + datetime.datetime.now().strftime("%d-%m-%Y-(%H:%M:%S)")
    os.mkdir(out_dir)

    c = 0.5
    a = np.loadtxt(f"logistic_map_space_{c}.txt")
    x = np.loadtxt(f"logistic_map_{c}.txt")
    # benchmark_popsize(default_kwargs, data=(c, a, x), output_path=os.path.join(out_dir, "population-size"),
    #                   visualize=True)
    # benchmark_param(default_kwargs, (c, a, x), "mutpb", [0.2, 0.4, 0.5, 0.6, 0.8],
    #                 output_path=os.path.join(out_dir, "mutpb"), xlabel="mutation probability",
    #                 ylabel="fitness")

    # benchmark_param(default_kwargs, (c, a, x), "n_columns", [3, 5, 7, 9, 11],
    #                 output_path=os.path.join(out_dir, "n_columns"), xlabel="number of collumns",
    #                 ylabel="fitness")

    benchmark_param(default_kwargs, (c, a, x), "lback", [1, 2, 3, 4, 5, 6, 7, 8],
                    output_path=os.path.join(out_dir, "lback"), xlabel="lback",
                    ylabel="fitness"
