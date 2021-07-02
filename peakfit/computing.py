import numpy as np
import scipy.linalg as sl

import peakfit.shapes as ps


def calculate_shape_heights(params, cluster):
    shapes = []

    for index in range(len(cluster.peaks)):
        pre = f"p{index}_"
        shapes.append(
            ps.pvoigt2d(
                cluster.x,
                cluster.y,
                params[f"{pre}x0"],
                params[f"{pre}y0"],
                params[f"{pre}x_fwhm_ppm"],
                params[f"{pre}y_fwhm_ppm"],
                params[f"{pre}x_eta"],
                params[f"{pre}y_eta"],
            )
        )

    shapes = np.asarray(shapes).T
    amp_values = sl.lstsq(shapes, cluster.data)[0]

    return shapes, amp_values


def residuals(params, cluster, noise):
    shapes, amplitudes = calculate_shape_heights(params, cluster)
    return np.ravel((cluster.data - shapes.dot(amplitudes)) / noise)


def simulate_data(params, cluster):
    shapes, amplitudes = calculate_shape_heights(params, cluster)
    return shapes.dot(amplitudes)
