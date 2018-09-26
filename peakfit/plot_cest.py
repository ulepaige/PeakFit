import argparse
import pathlib

import matplotlib.pyplot as plt
from natsort import natsorted
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages


def main():

    args = get_args()
    plot(args.files)

    return


def get_args():
    parser = argparse.ArgumentParser(description="Plot CEST profiles.")
    parser.add_argument("-f", "--files", nargs="+", type=pathlib.Path)
    return parser.parse_args()


def plot(files):

    figs = {}

    print()
    print("Reading files...")

    for a_file in files:

        print("  * {}".format(a_file.name))

        data = np.loadtxt(
            a_file,
            dtype={
                "names": ("offset", "intensity", "error"),
                "formats": ("i4", "f8", "f8"),
            },
        )

        data_ref = data[abs(data["offset"]) >= 1e4]
        data_cest = data[abs(data["offset"]) < 1e4]

        figs[a_file.name] = make_fig(a_file.name, data_cest, data_ref)

    print()
    print("Plotting...")

    with PdfPages("profiles.pdf") as pdf:
        for name in natsorted(figs):
            pdf.savefig(figs[name])

    print("  * profiles.pdf")


def offset_to_nu_cpmg(offset, time_t2):

    nu_cpmg = []

    for a_offset in offset:
        if a_offset > 0.0:
            nu_cpmg.append(a_offset / time_t2)
        else:
            nu_cpmg.append(0.5 / time_t2)

    return nu_cpmg


def intensity_to_r2eff(intensity, intensity_ref, time_t2):

    if np.any(intensity / intensity_ref <= 0.0):
        print(intensity / intensity_ref)
        print()

    return -np.log(intensity / intensity_ref) / time_t2


def make_ens(data, size=1000):

    return data["intensity"] + data["error"] * np.random.randn(
        size, len(data["intensity"])
    )


def make_fig(name, data_cest, data_ref):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.errorbar(
        data_cest["offset"],
        data_cest["intensity"] / np.mean(data_ref["intensity"]),
        yerr=data_cest["error"] / np.mean(data_ref["intensity"]),
        fmt="o",
    )
    ax.set_title(name)
    ax.set_xlabel(r"$B_1$ offset (Hz)")
    ax.set_ylabel(r"$I/I_0$")
    plt.close()

    return fig
