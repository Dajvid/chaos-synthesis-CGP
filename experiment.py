import datetime
import os
import warnings

import cgp
import numpy as np
import math
from matplotlib import pyplot as plt

from regression import fitness_regression_mse, evolve_known

default_kwargs = {
    "fitness": fitness_regression_mse,
    "popsize": 50,
    "mutpb": 0.2,
    "ngen": 200,
    "n_columns": 10,
    "n_rows": 3,
    "lback": 3,
    "turn_size": 2,
    "primitives": (cgp.Add, cgp.Sub, cgp.Mul, cgp.ConstantFloat)
}


def benchmark_popsize(kwargs, data, output_path="o", popsizes=[20, 50, 100, 500, 1000],
                      evaluations=30, visualize=False):
    eval_count = 100000
    kwargs = kwargs.copy()
    results = np.zeros((evaluations, len(popsizes)))
    for i, popsize in enumerate(popsizes):
        kwargs["popsize"] = popsize
        kwargs["ngen"] = eval_count // popsize
        for j in range(evaluations):
            print(f"Population size benchmark {i * evaluations + j + 1} / {evaluations * len(popsizes)}.")
            pop = evolve_known(data, **kwargs)
            if math.isclose(pop.champion.fitness, 0.0, rel_tol=1e-09, abs_tol=0.0):
                results[j][i] = pop.generation
            else:
                print(f"Solution for popsize {popsize} not found in given time")
                popsizes.pop(i)
                break
    plt.figure()
    plt.boxplot(results, labels=popsizes, notch=False)
    plt.xlabel = "popuation size"
    plt.ylabel = "found in population"
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
            if math.isclose(pop.champion.fitness, 0.0, rel_tol=1e-09, abs_tol=0.0):
                results[j][i] = pop.generation
            else:
                print(f"Solution for popsize {value} not found in given time")
                values.pop(i)
                break
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


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    out_dir = f"experiments-output-" + datetime.datetime.now().strftime("%d-%m-%Y-(%H:%M:%S)")
    os.mkdir(out_dir)

    c = 0.5
    a = np.loadtxt(f"logistic_map_space_{c}.txt")
    x = np.loadtxt(f"logistic_map_{c}.txt")
    benchmark_popsize(default_kwargs, data=(c, a, x), output_path=os.path.join(out_dir, "population-size"),
                      visualize=True)
