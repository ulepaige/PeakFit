import numpy as np
from scipy import linalg

from peakfit.shapes import pvoigt2d


def calculate_shape_amplitudes(
    params, x_list, y_list, data, peaks, ucx, ucy, amplitudes=False
):
    shapes = []

    for index, (_, _, x_alias, _, y_alias) in enumerate(peaks):
        pre = f"p{index}_"
        x0 = ucx.f(params[f"{pre}x0"].value - x_alias, "PPM")
        y0 = ucy.f(params[f"{pre}y0"].value - y_alias, "PPM")
        x_fwhm = ucx.f(params[f"{pre}x_fwhm"].value, "HZ") - ucx.f(0.0, "HZ")
        y_fwhm = ucy.f(params[f"{pre}y_fwhm"].value, "HZ") - ucy.f(0.0, "HZ")
        x_eta = params["".join([pre, "x_eta"])].value
        y_eta = params["".join([pre, "y_eta"])].value

        shapes.append(
            pvoigt2d(x_list, y_list, x0, y0, abs(x_fwhm), abs(y_fwhm), x_eta, y_eta)
        )

    shapes = np.asarray(shapes).T

    if amplitudes:

        amp_values = []

        for index in range(len(peaks)):
            pre = f"p{index}_"
            amp_values.append(
                [params["".join([pre, "i", str(nb)])] for nb in range(data.shape[1])]
            )

        amp_values = np.asarray(amp_values)

    else:

        amp_values = linalg.lstsq(shapes, data)[0]

    return shapes, amp_values


def residuals(
    params, x_list, y_list, data, peaks, ucx, ucy, noise=1.0, amplitude_params=False
):
    shapes, amplitudes = calculate_shape_amplitudes(
        params, x_list, y_list, data, peaks, ucx, ucy, amplitude_params
    )
    return np.ravel((data - shapes.dot(amplitudes)) / noise)


def simulate_data(params, x_list, y_list, data, peaks, ucx, ucy):
    shapes, amplitudes = calculate_shape_amplitudes(
        params, x_list, y_list, data, peaks, ucx, ucy
    )

    return shapes.dot(amplitudes)
