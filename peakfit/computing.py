import lmfit as lf
import numpy as np

import peakfit.shapes as ps
from peakfit.clustering import Cluster


def calculate_shape_heights(
    params: lf.Parameters, cluster: Cluster
) -> tuple[np.ndarray, np.ndarray]:
    shapes = []

    for index in range(len(cluster.peaks)):
        pre = f"p{index}_"
        shapes.append(
            ps.pvoigt2d(
                cluster.x,
                cluster.y,
                params[f"{pre}x0_pt"].value,
                params[f"{pre}y0_pt"].value,
                params[f"{pre}x_fwhm_pt"].value,
                params[f"{pre}y_fwhm_pt"].value,
                params[f"{pre}x_eta"].value,
                params[f"{pre}y_eta"].value,
            ),
        )

    shapes = np.asarray(shapes).T
    amp_values = np.linalg.lstsq(shapes, cluster.data, rcond=None)[0]

    return shapes, amp_values


def residuals(params: lf.Parameters, cluster: Cluster, noise: float) -> np.ndarray:
    shapes, amplitudes = calculate_shape_heights(params, cluster)
    return np.ravel((cluster.data - shapes.dot(amplitudes)) / noise)


def simulate_data(params: lf.Parameters, cluster: Cluster) -> np.ndarray:
    shapes, amplitudes = calculate_shape_heights(params, cluster)
    return shapes.dot(amplitudes)
