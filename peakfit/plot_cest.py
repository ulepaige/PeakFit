import argparse
import pathlib
import re

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages


def main():

    args = get_args()
    plot(args)

    return


def get_args():
    parser = argparse.ArgumentParser(description="Plot CEST profiles.")
    parser.add_argument("-f", "--files", nargs="+", type=pathlib.Path)
    parser.add_argument("--ref", nargs="+", type=int, default=-1)
    return parser.parse_args()


def plot(args):

    figs = {}

    print()
    print("Reading files...")

    files_ordered = sorted(args.files, key=lambda x: int(re.sub(r"\D", "", str(x))))

    for a_file in files_ordered:

        print(f"  * {a_file.name}")

        data = np.genfromtxt(a_file, dtype=None, names=("offset", "intensity", "error"))

        if args.ref == -1:
            ref = abs(data["offset"]) >= 1e4
        else:
            ref = np.full_like(data["offset"], False, dtype=bool)
            ref[args.ref] = True
        data_ref = data[ref]
        data_cest = data[~ref]

        figs[a_file.name] = make_fig(a_file.name, data_cest, data_ref)

    print()
    print("Plotting...")

    with PdfPages("profiles.pdf") as pdf:
        for fig in figs.values():
            pdf.savefig(fig)

    print("  * profiles.pdf")


def offset_to_nu_cpmg(offset, time_t2):

    nu_cpmg = []

    for a_offset in offset:
        if a_offset > 0.0:
            nu_cpmg.append(a_offset / time_t2)
        else:
            nu_cpmg.append(0.5 / time_t2)

    return nu_cpmg


def make_fig(name, data_cest, data_ref):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.errorbar(
        data_cest["offset"],
        data_cest["intensity"] / np.mean(data_ref["intensity"]),
        yerr=data_cest["error"] / abs(np.mean(data_ref["intensity"])),
        fmt=".",
    )
    ax.set_title(name)
    ax.set_xlabel(r"$B_1$ offset (Hz)")
    ax.set_ylabel(r"$I/I_0$")
    plt.close()

    return fig
