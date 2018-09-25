import numpy as np
from scipy import linalg

from peakfit.shapes import pvoigt2d


def calculate_shape_amplitudes(
    params, x_list, y_list, data, n_peaks, amplitude_params=False
):
    shapes = []

    for index in range(n_peaks):
        prefix = "p{:d}_".format(index)
        x0 = params["".join([prefix, "x0"])]
        y0 = params["".join([prefix, "y0"])]
        x_fwhm = params["".join([prefix, "x_fwhm"])]
        y_fwhm = params["".join([prefix, "y_fwhm"])]
        x_eta = params["".join([prefix, "x_eta"])]
        y_eta = params["".join([prefix, "y_eta"])]

        shapes.append(pvoigt2d(x_list, y_list, x0, y0, x_fwhm, y_fwhm, x_eta, y_eta))

    shapes = np.asarray(shapes).T

    if amplitude_params:

        amplitudes = []

        for index in range(n_peaks):
            prefix = "p{:d}_".format(index)
            amplitudes.append(
                [params["".join([prefix, "i", str(nb)])] for nb in range(data.shape[1])]
            )

        amplitudes = np.asarray(amplitudes)

    else:

        amplitudes = linalg.lstsq(shapes, data)[0]

    return shapes, amplitudes


def residuals(params, x_list, y_list, data, n_peaks, noise=1.0, amplitude_params=False):
    shapes, amplitudes = calculate_shape_amplitudes(
        params, x_list, y_list, data, n_peaks, amplitude_params
    )

    return np.ravel((data - shapes.dot(amplitudes)) / noise)


def simulate_data(params, x_list, y_list, data, n_peaks):
    shapes, amplitudes = calculate_shape_amplitudes(
        params, x_list, y_list, data, n_peaks
    )

    return shapes.dot(amplitudes)
