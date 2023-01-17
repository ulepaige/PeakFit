import argparse
import pathlib
import re

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages


def get_args():
    parser = argparse.ArgumentParser(description="Plot intensity profiles.")
    parser.add_argument("-f", "--files", nargs="+", type=pathlib.Path)
    return parser.parse_args()


def make_fig(name, data):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.errorbar(data["xlabel"], data["intensity"], yerr=data["error"], fmt=".")
    ax.set_title(name)
    ax.set_ylabel(r"Intensities")
    plt.close()

    return fig


def plot(args):

    figs = {}

    print()
    print("Reading files...")

    files_ordered = sorted(args.files, key=lambda x: int(re.sub(r"\D", "", str(x))))

    for a_file in files_ordered:

        print(f"  * {a_file.name}")

        data = np.genfromtxt(a_file, dtype=None, names=("xlabel", "intensity", "error"))
        figs[a_file.name] = make_fig(a_file.name, data)

    print()
    print("Plotting...")

    with PdfPages("profiles.pdf") as pdf:
        for fig in figs.values():
            pdf.savefig(fig)

    print("  * profiles.pdf")


def main():

    args = get_args()
    plot(args)

    return
