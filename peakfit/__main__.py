import argparse
import os
import os.path
import sys

import lmfit as lf
import nmrglue as ng
import numpy as np

from peakfit import clustering, computing, monte_carlo, shapes, util


def print_peaks(peaks, files=None):
    if files is None:
        files = (sys.stdout, )

    message = "*  Peak(s): "
    message += ", ".join(['{:s}'.format(peak[0].decode("utf-8")) for peak in peaks])
    message += "  *"
    stars = "*" * len(message)
    message = '\n'.join([stars, message, stars])

    for file in files:
        print(message, end='\n\n', file=file)


def calc_err(params_mc_list):
    params_list = {}

    for params_mc in params_mc_list:

        for key, val in params_mc.items():
            params_list.setdefault(key, []).append(val)

    params_err = {key: np.std(val_list) for key, val_list in params_list.items()}

    return params_err


def main():
    description = 'Perform peak integration in pseudo-3D.'

    parser = argparse.ArgumentParser(description=description)

    parser.add_argument('-s', dest='path_spectra', required=True)
    parser.add_argument('-l', dest='path_list_peak', required=True)
    parser.add_argument('-z', dest='path_list_z', required=True)
    parser.add_argument('-t', dest='contour_level', required=True, type=float)
    parser.add_argument('-o', dest='path_output', default='Fits')
    parser.add_argument('-n', dest='noise', default=1.0, type=float)
    parser.add_argument(
        '--mc',
        dest='noise_box',
        default=None,
        nargs=5,
        metavar=('X_MIN', 'X_MAX', 'Y_MIN', 'Y_MAX', 'N'))

    arguments = parser.parse_args()

    dic, spectra = ng.fileio.pipe.read(arguments.path_spectra)
    peak_list = np.genfromtxt(arguments.path_list_peak, dtype=None).reshape(-1)
    list_z = np.genfromtxt(arguments.path_list_z, dtype=None)
    contour_level = arguments.contour_level
    path_output = arguments.path_output
    noise = arguments.noise

    util.make_dirs(path_output)

    ucy = ng.pipe.make_uc(dic, spectra, dim=1)
    ucx = ng.pipe.make_uc(dic, spectra, dim=2)

    nz, ny, nx = spectra.shape

    mask = np.zeros_like(spectra)

    mask = clustering.mark_peaks(mask, peak_list, ucx, ucy)
    mask = clustering.find_disjoint_regions(spectra, mask, contour_level)
    clusters = clustering.make_peak_clusters(spectra, mask, peak_list, ucx, ucy)

    #
    # Lineshape Fitting
    #

    print('- Lineshape fitting...', end='\n\n\n')

    string_amplitudes = []

    file_logs = open(os.path.join(path_output, 'logs.out'), 'w')

    for peaks, x_grid, y_grid, data_to_fit in clusters:

        print_peaks(peaks, files=(sys.stdout, file_logs))

        n_peaks = len(peaks)
        params = shapes.create_params(peaks)
        args = (x_grid, y_grid, data_to_fit, n_peaks, noise)

        out = lf.minimize(computing.residuals, params, args=args)

        for name, param in out.params.items():
            if 'eta' in name:
                param.set(min=param.min, max=param.max, vary=True)

        out = lf.minimize(computing.residuals, out.params, args=args)

        message = lf.fit_report(out, min_correl=0.8)
        print(message, end='\n\n\n')
        print(message, end='\n\n\n', file=file_logs)

        _shapes, amplitudes = computing.calculate_shape_amplitudes(out.params, *(args[:4]))

        data_sim = computing.simulate_data(out.params, x_grid, y_grid, data_to_fit, n_peaks)

        #
        # Monte-Carlo
        #

        if arguments.noise_box and int(arguments.noise_box[4]) > 1:

            from random import sample

            noise_box = arguments.noise_box[:4]
            n_iter = int(arguments.noise_box[4])

            x_box_min = int(ucx.f(noise_box[0]))
            x_box_max = int(ucx.f(noise_box[1]))
            y_box_min = int(ucy.f(noise_box[2]))
            y_box_max = int(ucy.f(noise_box[3]))

            grid_for_noise = monte_carlo.get_noise_grid(x_box_min, x_box_max, y_box_min, y_box_max,
                                                        x_grid, y_grid)

            params_mc_list = []
            amplitudes_mc_list = []

            for x, y in sample(grid_for_noise, n_iter):
                x_noise_list = x_grid - min(x_grid) + x
                y_noise_list = y_grid - min(y_grid) + y

                n_fit = len(x_noise_list)

                data_noise = spectra[:, y_noise_list, x_noise_list].reshape((nz, n_fit)).T

                data_mc = data_sim + data_noise
                args_mc = (x_grid, y_grid, data_mc, n_peaks)
                out_mc = lf.minimize(computing.residuals, out.params, args=args_mc)

                _shapes_mc, amplitudes_mc = computing.calculate_shape_amplitudes(out_mc.params,
                                                                                 *args_mc)

                params_mc_list.append(out_mc.params.valuesdict())
                amplitudes_mc_list.append(amplitudes_mc)

            params_err = calc_err(params_mc_list)
            amplitudes_err = np.std(amplitudes_mc_list, axis=0, ddof=1)

        else:

            params_err = {param.name: param.stderr for param in out.params.values()}
            amplitudes_err = np.zeros_like(amplitudes) + arguments.noise

        for index, peak in enumerate(peaks):
            name, _, _, x_alias, y_alias = peak

            name = name.decode("utf-8")

            string_amplitudes.append('{:>15s}'.format(name))

            x_scale_ppm = abs(ucx.ppm(1.0) - ucx.ppm(0.0))
            y_scale_ppm = abs(ucx.ppm(1.0) - ucx.ppm(0.0))
            x_scale_hz = abs(ucx.hz(1.0) - ucx.hz(0.0))
            y_scale_hz = abs(ucy.hz(1.0) - ucy.hz(0.0))

            prefix = 'p{}_'.format(index)

            x0 = out.params[''.join([prefix, 'x0'])].value
            y0 = out.params[''.join([prefix, 'y0'])].value
            x0_err = params_err[''.join([prefix, 'x0'])]
            y0_err = params_err[''.join([prefix, 'y0'])]

            x_fwhm = out.params[''.join([prefix, 'x_fwhm'])].value
            y_fwhm = out.params[''.join([prefix, 'y_fwhm'])].value
            x_fwhm_err = params_err[''.join([prefix, 'x_fwhm'])]
            y_fwhm_err = params_err[''.join([prefix, 'y_fwhm'])]

            x_eta = out.params[''.join([prefix, 'x_eta'])].value
            y_eta = out.params[''.join([prefix, 'y_eta'])].value
            x_eta_err = params_err[''.join([prefix, 'x_eta'])]
            y_eta_err = params_err[''.join([prefix, 'y_eta'])]

            amplitude_err = np.mean(amplitudes_err[index])

            filename = os.path.join(path_output, ''.join([name, '.out']))

            with open(filename, 'w') as f:

                f.write("# Name: {:>10s}\n".format(name))
                f.write("# y_ppm: {:10.5f} {:10.5f}\n".format(
                    ucy.ppm(y0 + y_alias * ny), y0_err * y_scale_ppm))
                f.write("# yw_hz: {:10.5f} {:10.5f}\n".format(y_fwhm * y_scale_hz, y_fwhm_err *
                                                              y_scale_hz))
                f.write("# y_eta: {:10.5f} {:10.5f}\n".format(y_eta, y_eta_err))
                f.write("# x_ppm: {:10.5f} {:10.5f}\n".format(
                    ucx.ppm(x0 + x_alias * nx), x0_err * x_scale_ppm))
                f.write("# xw_hz: {:10.5f} {:10.5f}\n".format(x_fwhm * x_scale_hz, x_fwhm_err *
                                                              x_scale_hz))
                f.write("# x_eta: {:10.5f} {:10.5f}\n".format(x_eta, x_eta_err))

                f.write("#---------------------------------------------\n")
                f.write("# {:>10s}  {:>14s}  {:>14s}\n".format("Z", "I", "I_err"))

                for z, amplitude in zip(list_z, amplitudes[index]):
                    f.write("  {:>10s}  {:14.6e}  {:14.6e}\n".format(
                        str(z), amplitude, amplitude_err))
                    string_amplitudes.append('{:14.6e}'.format(amplitude))

                string_amplitudes.append('\n')

    file_logs.close()

    filename = os.path.join(path_output, 'amplitudes.out')

    with open(filename, 'w') as f:
        f.write(' '.join(string_amplitudes))


if __name__ == '__main__':
    main()
