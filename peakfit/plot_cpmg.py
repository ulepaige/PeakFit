import argparse
import pathlib
import re

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages


def main():

    args = get_args()
    plot(args.files, args.time_t2)

    return


def get_args():
    parser = argparse.ArgumentParser(description="Plot CPMG R2eff profiles.")
    parser.add_argument("-f", "--files", nargs="+", type=pathlib.Path)
    parser.add_argument("-t", "--time_t2", type=float)
    return parser.parse_args()


def plot(files, time_t2):

    figs = {}

    print()
    print("Reading files...")

    files_ordered = sorted(files, key=lambda x: int(re.sub(r"\D", "", str(x))))

    for a_file in files_ordered:

        print(f"  * {a_file.name}")

        data = np.loadtxt(
            a_file,
            dtype={
                "names": ("ncyc", "intensity", "error"),
                "formats": ("i4", "f8", "f8"),
            },
        )

        data_ref = data[data["ncyc"] == 0]
        data_cpmg = data[data["ncyc"] != 0]

        intensity_ref = np.mean(data_ref["intensity"])
        error_ref = np.mean(data_ref["error"]) / np.sqrt(len(data_ref))

        nu_cpmg = ncyc_to_nu_cpmg(data_cpmg["ncyc"], time_t2)

        r2_exp = intensity_to_r2eff(data_cpmg["intensity"], intensity_ref, time_t2)
        r2_ens = intensity_to_r2eff(
            make_ens(data_cpmg),
            make_ens(
                {"intensity": np.array([intensity_ref]), "error": np.array([error_ref])}
            ),
            time_t2,
        )

        r2_erd, r2_eru = abs(np.percentile(r2_ens, [15.9, 84.1], axis=0) - r2_exp)

        figs[a_file.name] = make_fig(a_file.name, nu_cpmg, r2_exp, r2_erd, r2_eru)

    print()
    print("Plotting...")

    with PdfPages("profiles.pdf") as pdf:
        for fig in figs.values():
            pdf.savefig(fig)

    print("  * profiles.pdf")


def ncyc_to_nu_cpmg(ncyc, time_t2):

    nu_cpmg = []

    for a_ncyc in ncyc:
        if a_ncyc > 0.0:
            nu_cpmg.append(a_ncyc / time_t2)
        else:
            nu_cpmg.append(0.5 / time_t2)

    return nu_cpmg


def intensity_to_r2eff(intensity, intensity_ref, time_t2):
    return -np.log(intensity / intensity_ref) / time_t2


def make_ens(data, size=1000):

    return data["intensity"] + data["error"] * np.random.randn(
        size, len(data["intensity"])
    )


def make_fig(name, nu_cpmg, r2_exp, r2_erd, r2_eru):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.errorbar(nu_cpmg, r2_exp, yerr=(r2_erd, r2_eru), fmt="o")
    ax.set_title(name)
    ax.set_xlabel(r"$\nu_{CPMG}$ (Hz)")
    ax.set_ylabel(r"$R_{2,eff}$ (s$^{-1}$)")
    plt.close()

    return fig
