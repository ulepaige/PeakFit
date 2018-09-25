import collections
import itertools

import numpy as np

from peakfit import util

Peak = collections.namedtuple("Peak", ["name", "x_pt", "y_pt", "x_alias", "y_alias"])


def flood_fill(data, mask, threshold, cluster_id, x_start, y_start):
    ny, nx = data.shape

    stack = [(x_start, y_start)]

    while stack:

        x, y = stack.pop()

        if (abs(data[y, x]) >= threshold and mask[0, y, x] == 0.0) or mask[
            0, y, x
        ] < 0.0:

            mask[:, y, x] = cluster_id

            if x > 0:
                stack.append((x - 1, y))

            if x < (nx - 1):
                stack.append((x + 1, y))

            if y > 0:
                stack.append((x, y - 1))

            if y < (ny - 1):
                stack.append((x, y + 1))


def mark_peaks(mask, peak_list, ucx, ucy):
    """Marks peaks in mask"""

    print("- Marking peaks...")

    nz, ny, nx = mask.shape

    for peak in peak_list:

        x_index = ucx.i(peak[2], "ppm") % nx
        y_index = ucy.i(peak[1], "ppm") % ny

        x_index = util.clamp(x_index, 2, nx - 2)
        y_index = util.clamp(y_index, 2, ny - 2)

        mask[:, y_index - 2 : y_index + 3, x_index - 2 : x_index + 3] = -1.0

    return mask


def find_independent_regions(data, mask, contour_level):
    """Finds regions of the spectra above the contour level threshold."""

    print("- Segmenting the spectra according to the threshold level...")

    nz, ny, nx = data.shape

    cluster_index = 0

    for y, x in itertools.product(range(ny), range(nx)):

        if mask[0, y, x] <= 0.0:
            cluster_index += 1
            flood_fill(data[0], mask, contour_level, cluster_index, x, y)

    return mask


def cluster_peaks(spectra, mask, peak_list, ucx, ucy):
    """Identifies overlapping peaks."""

    print("- Clustering of peaks...")

    nz, ny, nx = mask.shape

    names = peak_list["f0"]
    x_pts = ucx.f(peak_list["f2"], "ppm")
    y_pts = ucy.f(peak_list["f1"], "ppm")
    x_aliases = x_pts // nx
    y_aliases = y_pts // ny
    x_pts %= nx
    y_pts %= ny
    peaks = zip(names, x_pts, y_pts, x_aliases, y_aliases)

    cluster_ids = list(mask[0, np.rint(y_pts).astype(int), np.rint(x_pts).astype(int)])

    peak_clusters = dict()
    for cluster_id, peak in zip(cluster_ids, peaks):
        peak_clusters.setdefault(cluster_id, []).append(peak)

    clusters = []

    for cluster_id, peak_cluster in peak_clusters.items():
        grid_y, grid_x = np.where(mask[0] == cluster_id)
        data_to_fit = spectra[:, grid_y, grid_x].reshape((nz, grid_x.size)).T
        clusters.append((peak_cluster, grid_x, grid_y, data_to_fit))

    return clusters
