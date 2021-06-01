import argparse
import pathlib
import sys

import lmfit as lf
import nmrglue as ng
import numpy as np

from peakfit import __version__
from peakfit import clustering
from peakfit import computing
from peakfit import monte_carlo
from peakfit import shapes
from peakfit.util import print_peaks


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


def parse_command_line():
    description = "Perform peak integration in pseudo-3D."

    parser = argparse.ArgumentParser(description=description)

    parser.add_argument(
        "--spectra", "-s", dest="path_spectra", required=True, type=pathlib.Path
    )
    parser.add_argument(
        "--list", "-l", dest="path_list", required=True, type=pathlib.Path
    )
    parser.add_argument("--zvalues", "-z", dest="path_list_z", required=True)
    parser.add_argument("--ct", "-t", dest="contour_level", required=True, type=float)
    parser.add_argument(
        "--out", "-o", dest="path_output", default="Fits", type=pathlib.Path
    )
    parser.add_argument("--noise", "-n", dest="noise", default=1.0, type=float)
    parser.add_argument(
        "--mc",
        dest="noise_box",
        nargs=5,
        metavar=("X_MIN", "X_MAX", "Y_MIN", "Y_MAX", "N"),
    )
    parser.add_argument("--fixed", dest="fixed", action="store_true")
    parser.add_argument("--pvoigt", dest="pvoigt", action="store_true")
    parser.add_argument("--lorentzian", dest="lorentzian", action="store_true")

    return parser.parse_args()


def main():

    print(LOGO)

    args = parse_command_line()

    # Read spectra
    dic, spectra = ng.fileio.pipe.read(str(args.path_spectra))

    # Read peak list
    lines = args.path_list.read_text().replace("Ass", "#").splitlines()
    peak_list = np.genfromtxt(lines, dtype=None, encoding="utf-8")

    # Read z values
    list_z = np.genfromtxt(args.path_list_z, dtype=None)

    # Create the output directory
    args.path_output.mkdir(parents=True, exist_ok=True)

    # Create unit conversion object for indirect and direct dimension
    ucy = ng.pipe.make_uc(dic, spectra, dim=1)
    ucx = ng.pipe.make_uc(dic, spectra, dim=2)

    # Define the active spectral regions for the fit
    mask = np.zeros_like(spectra)
    mask = clustering.mark_peaks(mask, peak_list, ucx, ucy)
    mask = clustering.find_independent_regions(mask, spectra, args.contour_level)

    # Cluster peaks
    clusters = clustering.cluster_peaks(spectra, mask, peak_list, ucx, ucy)

    #
    # Lineshape Fitting
    #

    print("- Lineshape fitting...", end="\n\n\n")

    string_shifts = []
    string_amplitudes = []

    file_logs = (args.path_output / "logs.out").open("w")

    for peaks, x, y, data in clusters:

        print_peaks(peaks, files=(sys.stdout, file_logs))

        params = shapes.create_params(peaks, ucx, ucy, args.pvoigt, args.lorentzian)
        args_func = (x, y, data, peaks, ucx, ucy, args.noise)
        kwargs = {"args": args_func, "method": "least_squares"}

        # out = lf.minimize(computing.residuals, params, **kwargs)
        if not args.fixed:
            for name, param in params.items():
                if "x0" in name or "y0" in name:
                    param.set(min=param.min, max=param.max, vary=True)
            # out = lf.minimize(computing.residuals, out.params, **kwargs)
        if args.pvoigt:
            for name, param in params.items():
                if "eta" in name:
                    param.set(min=param.min, max=param.max, vary=True)
        out = lf.minimize(computing.residuals, params, **kwargs)

        message = lf.fit_report(out, min_correl=0.5)
        print(message, end="\n\n\n")
        print(message, end="\n\n\n", file=file_logs)

        args_cal = args_func[:6]
        _, amplitudes = computing.calculate_shape_amplitudes(out.params, *args_cal)

        #
        # Monte-Carlo
        #

        if args.noise_box and int(args.noise_box[4]) > 1:

            from random import sample

            noise_box = args.noise_box[:4]
            n_iter = int(args.noise_box[4])

            bounds_x = sorted(ucx.i(bound_x) for bound_x in noise_box[0:2])
            bounds_y = sorted(ucy.i(bound_y) for bound_y in noise_box[2:4])

            grid_noise = monte_carlo.get_noise_grid(bounds_x, bounds_y, x, y)

            data_sim = computing.simulate_data(out.params, *args_cal)

            params_mc_list = []
            amplitudes_mc_list = []

            for xn, yn in sample(grid_noise, n_iter):
                x_noise = x - min(x) + xn
                y_noise = y - min(y) + yn

                nz = spectra.shape[0]
                data_noise = spectra[:, y_noise, x_noise].reshape((nz, x.size)).T

                data_mc = data_sim + data_noise
                args_mc = (x, y, data_mc, peaks, ucx, ucy)
                kwargs_mc = {"args": args_mc, "method": "least_squares"}
                out_mc = lf.minimize(computing.residuals, out.params, **kwargs_mc)

                _, amplitudes_mc = computing.calculate_shape_amplitudes(
                    out_mc.params, *args_mc
                )

                params_mc_list.append(out_mc.params.valuesdict())
                amplitudes_mc_list.append(amplitudes_mc)

            parerrs = calc_err_from_mc(params_mc_list)
            amplitudes_err = np.std(amplitudes_mc_list, axis=0, ddof=1)

        else:
            parerrs = {param.name: param.stderr for param in out.params.values()}
            amplitudes_err = np.zeros_like(amplitudes) + args.noise

        for index, peak in enumerate(peaks):
            name, _, _, _, _ = peak

            pre = f"p{index}_"
            parvals = out.params.valuesdict()

            parerrs_ = {k: v if v is not None else 0.0 for k, v in parerrs.items()}
            x_ppm = parvals[f"{pre}x0"]
            y_ppm = parvals[f"{pre}y0"]
            x_ppm_e = parerrs_[f"{pre}x0"]
            y_ppm_e = parerrs_[f"{pre}y0"]

            xw_hz = parvals[f"{pre}x_fwhm"]
            yw_hz = parvals[f"{pre}y_fwhm"]
            xw_hz_e = parerrs_[f"{pre}x_fwhm"]
            yw_hz_e = parerrs_[f"{pre}y_fwhm"]

            x_eta = parvals[f"{pre}x_eta"]
            y_eta = parvals[f"{pre}y_eta"]
            x_eta_e = parerrs_[f"{pre}x_eta"]
            y_eta_e = parerrs_[f"{pre}y_eta"]

            ampl_e = np.mean(amplitudes_err[index])

            filename = args.path_output / "".join([name, ".out"])

            with filename.open("w") as f:

                f.write(f"# Name: {name:>10s}\n")
                f.write(f"# y_ppm: {y_ppm:10.5f} {y_ppm_e:10.5f}\n")
                f.write(f"# yw_hz: {yw_hz:10.5f} {yw_hz_e:10.5f}\n")
                f.write(f"# y_eta: {y_eta:10.5f} {y_eta_e:10.5f}\n")
                f.write(f"# x_ppm: {x_ppm:10.5f} {x_ppm_e:10.5f}\n")
                f.write(f"# xw_hz: {xw_hz:10.5f} {xw_hz_e:10.5f}\n")
                f.write(f"# x_eta: {x_eta:10.5f} {x_eta_e:10.5f}\n")
                f.write("#---------------------------------------------\n")
                f.write(f"# {'Z':>10s}  {'I':>14s}  {'I_err':>14s}\n")

                string_shifts.append(f"{name:>15s} {y_ppm:10.5f} {x_ppm:10.5f}\n")

                string_amplitudes.append(f"{name:>15s}")

                for z, ampl in zip(list_z, amplitudes[index]):
                    f.write(f"  {str(z):>10s}  {ampl:14.6e}  {ampl_e:14.6e}\n")
                    string_amplitudes.append(f"{ampl:14.6e}")

                string_amplitudes.append("\n")

    file_logs.close()

    file_amplitudes = args.path_output / "amplitudes.out"
    file_amplitudes.write_text(" ".join(string_amplitudes))

    file_shifts = args.path_output / "shifts.out"
    file_shifts.write_text(" ".join(string_shifts))


def calc_err_from_mc(params_mc_list):
    params_list = {}

    for params_mc in params_mc_list:

        for key, val in params_mc.items():
            params_list.setdefault(key, []).append(val)

    params_err = {key: np.std(val_list) for key, val_list in params_list.items()}

    return params_err


if __name__ == "__main__":
    main()
