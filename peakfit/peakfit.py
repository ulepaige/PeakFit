import argparse
import pathlib
import random
import sys

import lmfit as lf
import nmrglue as ng
import numpy as np
import numpy.random as nr

import peakfit.clustering as pcl
import peakfit.computing as pco
import peakfit.shapes as ps
import peakfit.util as pu
from peakfit import __version__


LOGO = r"""

* * * * * * * * * * * * * * * * * * * * * *
*    ____            _    _____ _ _       *
*   |  _ \ ___  __ _| | _|  ___(_) |_     *
*   | |_) / _ \/ _` | |/ / |_  | | __|    *
*   |  __/  __/ (_| |   <|  _| | | |_     *
*   |_|   \___|\__,_|_|\_\_|   |_|\__|    *
*                                         *
*  Perform peak integration in pseudo-3D  *
*  spectra                                *
*                                         *
*  Version: {:<29s} *
*                                         *
* * * * * * * * * * * * * * * * * * * * * *
""".format(
    __version__
)


def parse_command_line() -> argparse.Namespace:
    description = "Perform peak integration in pseudo-3D."

    parser = argparse.ArgumentParser(description=description)

    parser.add_argument(
        "--spectra",
        "-s",
        dest="path_spectra",
        required=True,
        type=pathlib.Path,
        nargs="+",
    )
    parser.add_argument(
        "--list", "-l", dest="path_list", required=True, type=pathlib.Path
    )
    parser.add_argument(
        "--zvalues", "-z", dest="path_z_values", required=True, nargs="+"
    )
    parser.add_argument("--ct", "-t", dest="contour_level", required=True, type=float)
    parser.add_argument(
        "--out", "-o", dest="path_output", default="Fits", type=pathlib.Path
    )
    parser.add_argument("--noise", "-n", dest="noise", default=1.0, type=float)
    parser.add_argument(
        "--mc",
        nargs=5,
        metavar=("X_MIN", "X_MAX", "Y_MIN", "Y_MAX", "N"),
    )
    parser.add_argument("--fixed", dest="fixed", action="store_true")
    parser.add_argument("--pvoigt", dest="pvoigt", action="store_true")
    parser.add_argument("--lorentzian", dest="lorentzian", action="store_true")

    return parser.parse_args()


def main() -> None:

    print(LOGO)

    clargs = parse_command_line()

    # Read spectra
    dic_list = []
    data_list = []
    for path in clargs.path_spectra:
        dic, data = ng.fileio.pipe.read(str(path))
        dic_list.append(dic)
        data_list.append(data)
    data = np.concatenate(data_list, axis=0)
    dic = dic_list[0]

    # Normalize spectra related to noise
    if clargs.noise < 0.0:
        clargs.noise = 1.0
        print("Warning: `noise` option is < 0.0 (set to 1.0)")

    # Create unit conversion object for indirect and direct dimension
    ucy = ng.pipe.make_uc(dic, data, dim=1)
    ucx = ng.pipe.make_uc(dic, data, dim=2)

    # Create 'Spectra' object
    spectra = pcl.Spectra(data, ucx, ucy)

    # Read peak list
    lines = clargs.path_list.read_text().replace("Ass", "#").splitlines()
    peaks = np.genfromtxt(
        lines, dtype=None, encoding="utf-8", names=("names", "y", "x")
    )

    # Read z values
    z_list = [np.genfromtxt(path, dtype=None) for path in clargs.path_z_values]
    z_values = np.concatenate(z_list)

    # Create the output directory
    clargs.path_output.mkdir(parents=True, exist_ok=True)

    # Cluster peaks
    clusters = pcl.cluster_peaks(spectra, peaks, clargs.contour_level)

    with (clargs.path_output / "logs.out").open("w") as file_logs:
        shifts = run_fit(clargs, spectra, z_values, clusters, file_logs)

    with (clargs.path_output / "shifts.out").open("w") as file_shifts:
        write_shifts(peaks["names"], shifts, file_shifts)


def run_fit(clargs, spectra, z_values, clusters, file_logs):

    print("- Lineshape fitting...", end="\n\n\n")

    shifts = {}

    for cluster in clusters:

        pu.print_peaks(cluster.peaks, files=(sys.stdout, file_logs))

        params = ps.create_params(cluster.peaks, spectra, clargs)

        out = lf.minimize(
            pco.residuals,
            params,
            args=(cluster, clargs.noise),
            method="least_squares",
            verbose=2,
        )

        _, heights = pco.calculate_shape_heights(out.params, cluster)

        out.init_vals.extend(heights.ravel())
        out._calculate_statistics()

        print(f"\nReduced Chi2 = {out.redchi}\n\n")
        print(lf.fit_report(out, min_correl=0.5), end="\n\n\n", file=file_logs)

        params_err = {param.name: param.stderr for param in out.params.values()}
        height_err = np.full_like(heights, clargs.noise)

        shifts.update(
            {
                peak.name: (out.params[f"p{i}_x0"].value, out.params[f"p{i}_y0"].value)
                for i, peak in enumerate(cluster.peaks)
            }
        )

        # Monte-Carlo
        if clargs.mc and int(clargs.mc[4]) > 1:
            params_err, height_err = monte_carlo(
                clargs.mc, spectra, cluster, out, clargs.noise
            )

        write_profiles(
            clargs.path_output,
            z_values,
            cluster,
            out.params,
            heights,
            params_err,
            height_err,
        )

    return shifts


def write_profiles(path, z_values, cluster, params, heights, params_err, height_err):
    for i, peak in enumerate(cluster.peaks):
        vals = params.valuesdict()
        errs = {k: v if v is not None else 0.0 for k, v in params_err.items()}

        x_ppm = vals[f"p{i}_x0"]
        y_ppm = vals[f"p{i}_y0"]
        x_ppm_e = errs[f"p{i}_x0"]
        y_ppm_e = errs[f"p{i}_y0"]

        xw_hz = vals[f"p{i}_x_fwhm"]
        yw_hz = vals[f"p{i}_y_fwhm"]
        xw_hz_e = errs[f"p{i}_x_fwhm"]
        yw_hz_e = errs[f"p{i}_y_fwhm"]

        x_eta = vals[f"p{i}_x_eta"]
        y_eta = vals[f"p{i}_y_eta"]
        x_eta_e = errs[f"p{i}_x_eta"]
        y_eta_e = errs[f"p{i}_y_eta"]

        ampl_e = np.mean(height_err[i])

        with (path / f"{peak.name}.out").open("w") as f:
            f.write(f"# Name: {peak.name}\n")
            f.write(f"# y_ppm: {y_ppm:10.5f} {y_ppm_e:10.5f}\n")
            f.write(f"# yw_hz: {yw_hz:10.5f} {yw_hz_e:10.5f}\n")
            f.write(f"# y_eta: {y_eta:10.5f} {y_eta_e:10.5f}\n")
            f.write(f"# x_ppm: {x_ppm:10.5f} {x_ppm_e:10.5f}\n")
            f.write(f"# xw_hz: {xw_hz:10.5f} {xw_hz_e:10.5f}\n")
            f.write(f"# x_eta: {x_eta:10.5f} {x_eta_e:10.5f}\n")
            f.write("#---------------------------------------------\n")
            f.write(f"# {'Z':>10s}  {'I':>14s}  {'I_err':>14s}\n")
            f.write(
                "\n".join(
                    f"  {str(z):>10s}  {ampl:14.6e}  {ampl_e:14.6e}"
                    for z, ampl in zip(z_values, heights[i])
                )
            )


def monte_carlo(mc, spectra, cluster, result, noise):

    n_iter, spectra_noise = extract_noise_spectra(mc, spectra)

    x_pt = np.rint(spectra.ucx.f(cluster.x, "ppm")).astype(int)
    y_pt = np.rint(spectra.ucy.f(cluster.y, "ppm")).astype(int)

    data_sim = pco.simulate_data(result.params, cluster)

    mc_list = []

    for _ in range(int(n_iter)):

        data_noise = get_some_noise(spectra, spectra_noise, x_pt, y_pt)
        data_mc = data_sim + data_noise
        cluster_mc = pcl.Cluster(cluster.peaks, cluster.x, cluster.y, data_mc)

        params_mc = lf.minimize(
            pco.residuals,
            result.params,
            args=(cluster_mc, noise),
            method="least_squares",
        ).params

        _shapes, heights = pco.calculate_shape_heights(params_mc, cluster_mc)

        mc_list.append((params_mc, heights))

    params_list, heights_list = zip(*mc_list)

    params_err, height_err = calc_err_from_mc(params_list, heights_list)

    return params_err, height_err


def get_some_noise(spectra, noise, x_pt, y_pt):
    nr.shuffle(noise)

    nz_noise, ny_noise, nx_noise = noise.shape

    x_off = random.randint(0, nx_noise - 1)
    y_off = random.randint(0, ny_noise - 1)

    x_noise = (x_pt + x_off) % nx_noise
    y_noise = (y_pt + y_off) % ny_noise

    return spectra.data[:, y_noise, x_noise].reshape((nz_noise, x_noise.size)).T


def extract_noise_spectra(mc, spectra):
    x1, x2, y1, y2, n_iter = mc

    x1_pt, x2_pt = sorted((spectra.ucx.i(x1), spectra.ucx.i(x2)))
    y1_pt, y2_pt = sorted((spectra.ucy.i(y1), spectra.ucy.i(y2)))

    noise = spectra.data[:, y1_pt:y2_pt, x1_pt:x2_pt]
    return n_iter, noise


def calc_err_from_mc(params_list, heights_list):

    params_dict = {}

    for params_mc in params_list:
        for name, param in params_mc.items():
            params_dict.setdefault(name, []).append(param.value)

    params_err = {name: np.std(values) for name, values in params_dict.items()}
    height_err = np.std(heights_list, axis=0, ddof=1)

    return params_err, height_err


def write_shifts(names, shifts, file_shifts):
    for name in names:
        file_shifts.write(
            f"{name:>15s} {shifts[name][1]:10.5f} {shifts[name][0]:10.5f}\n"
        )


if __name__ == "__main__":
    main()
