import os
import matplotlib.pyplot as plt
import numpy as np


def draw_variants(input_dir, output_dir, name, xlabels):
    plt.figure(figsize=(10, 7))

    data = np.loadtxt(os.path.join(input_dir,  f"{name}.out"))
    plt.boxplot(data, labels=xlabels)
    plt.xlabel(name)
    plt.ylabel("fitness")
    plt.savefig(os.path.join(output_dir, f"{name}.png"))


if __name__ == '__main__':
    input_dir = "experiments-output-07-05-2022-(11:01:06)"
    out_dir_name = "plots"
    try:
        os.mkdir(out_dir_name)
    except FileExistsError:
        pass

    draw_variants(input_dir, out_dir_name, "lback", xlabels=[1, 2, 3, 4, 5, 6, 7, 8])
    draw_variants(input_dir, out_dir_name, "mutpb", xlabels=[0.2, 0.4, 0.5, 0.6, 0.8])
    draw_variants(input_dir, out_dir_name, "n-columns", xlabels=[3, 5, 7, 9, 11])
    draw_variants(input_dir, out_dir_name, "population-size", xlabels=[20, 100, 250, 500, 1000])
