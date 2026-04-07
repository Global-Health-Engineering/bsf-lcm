import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt


def set_mpl():
    """ set matplotlib parameters """
    fontsize = 16
    mpl.rcParams.update({
        "font.family": "Times New Roman",
        "mathtext.fontset": "dejavuserif",
        "font.size": fontsize,
        "axes.labelsize": fontsize,
        "axes.titlesize": fontsize,
        "legend.fontsize": fontsize,
        "xtick.top": True,
        "xtick.bottom": True,
        "ytick.left": True,
        "ytick.right": True,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.pad": 5,
        "ytick.major.pad": 5
    })


def plot_throughput(data_file: str, save_path: str):
    df = pd.read_csv(data_file)
    df.weight_g.replace("", np.nan, inplace=True)
    df.dropna(subset=["weight_g"], inplace=True)
    df.drop(labels="manual_count", axis=1, inplace=True)

    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    ax.plot(df.time_min, df.weight_g,
            marker="o",
            linestyle="-",
            color="b")
    ax.set_xlabel(r"$t$ / min")
    ax.set_ylabel("throughput / g")
    ax.set_xlim((0, 40))
    ax.set_ylim((0, 2.5))
    plt.tight_layout(pad=0.2)
    plt.savefig(save_path)


def main():
    set_mpl()
    plot_throughput(data_file="../../data/lcm_mk2-throughput.csv",
                    save_path="../../img/mk2/throughput.pdf")
    plt.show()


if __name__ == "__main__":
    main()