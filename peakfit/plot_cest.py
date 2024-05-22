import argparse
import pathlib
import re

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from peakfit.messages import print_filename, print_plotting, print_reading_files


def get_args():
    parser = argparse.ArgumentParser(description="Plot CEST profiles.")
    parser.add_argument("-f", "--files", nargs="+", type=pathlib.Path)
    parser.add_argument("--ref", nargs="+", type=int, default=-1)
    return parser.parse_args()


def make_fig(name, offset, intensity, error):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.errorbar(offset, intensity, yerr=error, fmt=".")
    ax.set_title(name)
    ax.set_xlabel(r"$B_1$ offset (Hz)")
    ax.set_ylabel(r"$I/I_0$")
    plt.close()

    return fig


def plot(args) -> None:
    figs = {}

    print_reading_files()

    files_ordered = sorted(args.files, key=lambda x: int(re.sub(r"\D", "", str(x))))

    for a_file in files_ordered:
        print_filename(a_file)
        offset, intensity, error = np.loadtxt(a_file, unpack=True)
        if args.ref == -1:
            ref = abs(offset) >= 1e4
        else:
            ref = np.full_like(offset, fill_value=False, dtype=bool)
            ref[args.ref] = True

        intensity_ref = np.mean(intensity[ref])

        offset = offset[~ref]
        intensity = intensity[~ref] / intensity_ref
        error = error[~ref] / abs(intensity_ref)

        figs[a_file.name] = make_fig(a_file.name, offset, intensity, error)

    print_plotting()

    with PdfPages("profiles.pdf") as pdf:
        for fig in figs.values():
            pdf.savefig(fig)


def main() -> None:
    args = get_args()
    plot(args)
