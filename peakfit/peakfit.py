import argparse
import pathlib
import sys

import lmfit as lf
import nmrglue as ng
import numpy as np

from peakfit import clustering, computing, monte_carlo, shapes, __version__
from peakfit.util import print_peaks


LOGO = """

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

    parser.add_argument("--spectra", "-s", dest="path to spectra", required=True)
    parser.add_argument("--lists", "-l", dest="path_list_peak", required=True)
    parser.add_argument("--zvalues", "-z", dest="path_list_z", required=True)
    parser.add_argument("--ct", "-t", dest="contour_level", required=True, type=float)
    parser.add_argument("--out", "-o", dest="path_output", default="Fits", type=pathlib.Path)
    parser.add_argument("--noise", "-n", dest="noise", default=1.0, type=float)
    parser.add_argument(
        "--mc",
        dest="noise_box",
        nargs=5,
        metavar=("X_MIN", "X_MAX", "Y_MIN", "Y_MAX", "N"),
    )
    parser.add_argument("--pvoigt", dest="pvoigt", action="store_true")

    return parser.parse_args()


def main():

    print(LOGO)

    args = parse_command_line()

    # Read spectra
    dic, spectra = ng.fileio.pipe.read(args.path_spectra)

    # Read peak list
    peak_list = np.genfromtxt(args.path_list_peak, dtype=None, encoding="utf-8")

    # Read z values
    list_z = np.genfromtxt(args.path_list_z, dtype=None)

    # Create the output directory
    args.path_output.mkdir(parents=True, exist_ok=True)

    # Create unit conversion object for indirect and direct dimension
    ucy = ng.pipe.make_uc(dic, spectra, dim=1)
    ucx = ng.pipe.make_uc(dic, spectra, dim=2)

    len_z, len_y, len_x = spectra.shape

    # Define the active spectral regions for the fit
    mask = np.zeros_like(spectra)
    mask = clustering.mark_peaks(mask, peak_list, ucx, ucy)
    mask = clustering.find_independent_regions(spectra, mask, args.contour_level)

    # Cluster peaks
    clusters = clustering.cluster_peaks(spectra, mask, peak_list, ucx, ucy)

    #
    # Lineshape Fitting
    #

    print("- Lineshape fitting...", end="\n\n\n")

    string_amplitudes = []

    file_logs = (args.path_output / "logs.out").open("w")

    for peaks, grid_x, grid_y, data_to_fit in clusters:

        print_peaks(peaks, files=(sys.stdout, file_logs))

        n_peaks = len(peaks)
        params = shapes.create_params(peaks, args.pvoigt)
        args_fit = (grid_x, grid_y, data_to_fit, n_peaks, args.noise)

        out = lf.minimize(computing.residuals, params, args=args_fit)

        if args.pvoigt:
            for name, param in out.params.items():
                if "eta" in name:
                    param.set(min=param.min, max=param.max, vary=True)

            out = lf.minimize(computing.residuals, out.params, args=args_fit)

        message = lf.fit_report(out, show_correl=0.5)
        print(message, end="\n\n\n")
        print(message, end="\n\n\n", file=file_logs)

        _shapes, amplitudes = computing.calculate_shape_amplitudes(
            out.params, *(args_fit[:4])
        )

        data_sim = computing.simulate_data(
            out.params, grid_x, grid_y, data_to_fit, n_peaks
        )

        #
        # Monte-Carlo
        #

        if args.noise_box and int(args.noise_box[4]) > 1:

            from random import sample

            noise_box = args.noise_box[:4]
            n_iter = int(args.noise_box[4])

            bounds_x = sorted(ucx.i(bound_x) for bound_x in noise_box[0:2])
            bounds_y = sorted(ucy.i(bound_y) for bound_y in noise_box[2:4])

            grid_for_noise = monte_carlo.get_noise_grid(
                bounds_x, bounds_y, grid_x, grid_y
            )

            params_mc_list = []
            amplitudes_mc_list = []

            for x, y in sample(grid_for_noise, n_iter):
                x_noise_list = grid_x - min(grid_x) + x
                y_noise_list = grid_y - min(grid_y) + y

                data_noise = (
                    spectra[:, y_noise_list, x_noise_list]
                    .reshape((len_z, grid_x.size))
                    .T
                )

                data_mc = data_sim + data_noise
                args_mc = (grid_x, grid_y, data_mc, n_peaks)
                out_mc = lf.minimize(computing.residuals, out.params, args=args_mc)

                _shapes_mc, amplitudes_mc = computing.calculate_shape_amplitudes(
                    out_mc.params, *args_mc
                )

                params_mc_list.append(out_mc.params.valuesdict())
                amplitudes_mc_list.append(amplitudes_mc)

            params_err = calc_err_from_mc(params_mc_list)
            amplitudes_err = np.std(amplitudes_mc_list, axis=0, ddof=1)

        else:

            params_err = {param.name: param.stderr for param in out.params.values()}
            amplitudes_err = np.zeros_like(amplitudes) + args.noise

        for index, peak in enumerate(peaks):
            name, _, _, x_alias, y_alias = peak

            string_amplitudes.append("{:>15s}".format(name))

            x_scale_ppm = abs(ucx.ppm(1.0) - ucx.ppm(0.0))
            y_scale_ppm = abs(ucx.ppm(1.0) - ucx.ppm(0.0))
            x_scale_hz = abs(ucx.hz(1.0) - ucx.hz(0.0))
            y_scale_hz = abs(ucy.hz(1.0) - ucy.hz(0.0))

            prefix = "p{}_".format(index)

            x0 = out.params["".join([prefix, "x0"])].value
            y0 = out.params["".join([prefix, "y0"])].value
            x0_err = params_err["".join([prefix, "x0"])]
            y0_err = params_err["".join([prefix, "y0"])]

            x_fwhm = out.params["".join([prefix, "x_fwhm"])].value
            y_fwhm = out.params["".join([prefix, "y_fwhm"])].value
            x_fwhm_err = params_err["".join([prefix, "x_fwhm"])]
            y_fwhm_err = params_err["".join([prefix, "y_fwhm"])]

            x_eta = out.params["".join([prefix, "x_eta"])].value
            y_eta = out.params["".join([prefix, "y_eta"])].value
            x_eta_err = params_err["".join([prefix, "x_eta"])]
            y_eta_err = params_err["".join([prefix, "y_eta"])]

            amplitude_err = np.mean(amplitudes_err[index])

            filename = args.path_output / "".join([name, ".out"])

            with filename.open("w") as f:

                f.write("# Name: {:>10s}\n".format(name))
                f.write(
                    "# y_ppm: {:10.5f} {:10.5f}\n".format(
                        ucy.ppm(y0 + y_alias * len_y), y0_err * y_scale_ppm
                    )
                )
                f.write(
                    "# yw_hz: {:10.5f} {:10.5f}\n".format(
                        y_fwhm * y_scale_hz, y_fwhm_err * y_scale_hz
                    )
                )
                f.write("# y_eta: {:10.5f} {:10.5f}\n".format(y_eta, y_eta_err))
                f.write(
                    "# x_ppm: {:10.5f} {:10.5f}\n".format(
                        ucx.ppm(x0 + x_alias * len_x), x0_err * x_scale_ppm
                    )
                )
                f.write(
                    "# xw_hz: {:10.5f} {:10.5f}\n".format(
                        x_fwhm * x_scale_hz, x_fwhm_err * x_scale_hz
                    )
                )
                f.write("# x_eta: {:10.5f} {:10.5f}\n".format(x_eta, x_eta_err))

                f.write("#---------------------------------------------\n")
                f.write("# {:>10s}  {:>14s}  {:>14s}\n".format("Z", "I", "I_err"))

                for z, amplitude in zip(list_z, amplitudes[index]):
                    f.write(
                        "  {:>10s}  {:14.6e}  {:14.6e}\n".format(
                            str(z), amplitude, amplitude_err
                        )
                    )
                    string_amplitudes.append("{:14.6e}".format(amplitude))

                string_amplitudes.append("\n")

    file_logs.close()

    filename = args.path_output / "amplitudes.out"
    filename.write_text(" ".join(string_amplitudes))


def calc_err_from_mc(params_mc_list):
    params_list = {}

    for params_mc in params_mc_list:

        for key, val in params_mc.items():
            params_list.setdefault(key, []).append(val)

    params_err = {key: np.std(val_list) for key, val_list in params_list.items()}

    return params_err


if __name__ == "__main__":
    main()
