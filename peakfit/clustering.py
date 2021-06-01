import itertools as it

import numpy as np

from peakfit import util


def mark_peaks(mask, peak_list, ucx, ucy):
    """Marks peaks in mask"""

    print("- Marking peaks...")

    ny, nx = mask.shape[1:]

    for peak in peak_list:
        x_index = ucx.i(peak[2], "ppm") % nx
        y_index = ucy.i(peak[1], "ppm") % ny

        x_index = util.clamp(x_index, 2, nx - 2)
        y_index = util.clamp(y_index, 2, ny - 2)

        ymin, ymax = y_index - 2, y_index + 3
        xmin, xmax = x_index - 2, x_index + 3
        mask[:, ymin:ymax, xmin:xmax] = -1.0

    return mask


def find_independent_regions(mask, data, contour_level):
    """Finds regions of the spectra above the contour level threshold."""

    print("- Segmenting the spectra according to the threshold level...")

    # ref = np.argmax([np.linalg.norm(datum) for datum in data])
    plane = np.linalg.norm(data, ord=np.inf, axis=0)
    ny, nx = data.shape[1:]
    cluster_index = 0
    for y, x in it.product(range(ny), range(nx)):
        if mask[0, y, x] <= 0.0:
            cluster_index += 1
            # flood_fill(data[ref], mask, contour_level, cluster_index, x, y)
            _flood_fill(mask, plane, contour_level, cluster_index, x, y)
    return mask


def _flood_fill(mask, data, threshold, cluster_id, x_start, y_start):
    ny, nx = data.shape
    stack = [(x_start, y_start)]
    while stack:
        x, y = stack.pop()
        above = abs(data[y, x]) >= threshold
        unvisited = mask[0, y, x] == 0.0
        is_peak = mask[0, y, x] < 0.0
        if (above and unvisited) or is_peak:
            mask[:, y, x] = cluster_id
            if x > 0:
                stack.append((x - 1, y))
            if x < (nx - 1):
                stack.append((x + 1, y))
            if y > 0:
                stack.append((x, y - 1))
            if y < (ny - 1):
                stack.append((x, y + 1))


def cluster_peaks(spectra, mask, peak_list, ucx, ucy):
    """Identifies overlapping peaks."""

    print("- Clustering of peaks...")

    nz, ny, nx = spectra.shape

    names = peak_list["f0"]
    x_ppm = peak_list["f2"]
    y_ppm = peak_list["f1"]
    x0 = ucx.unit(0, "ppm")
    x1 = ucx.unit(nx, "ppm")
    y0 = ucy.unit(0, "ppm")
    y1 = ucy.unit(ny, "ppm")
    x_aliases = ((x_ppm - x0) // (x1 - x0)) * (x1 - x0)
    y_aliases = ((y_ppm - y0) // (y1 - y0)) * (y1 - y0)

    peak_list = list(zip(names, x_ppm, x_aliases, y_ppm, y_aliases))

    x_indexes = [ucx.i(x, "ppm") for x in x_ppm - x_aliases]
    y_indexes = [ucy.i(y, "ppm") for y in y_ppm - y_aliases]

    cluster_ids = list(mask[0, y_indexes, x_indexes])

    peak_clusters = {}
    for cluster_id, peak in zip(cluster_ids, peak_list):
        peak_clusters.setdefault(cluster_id, []).append(peak)

    clusters = []
    nz = spectra.shape[0]
    for cluster_id, peak_cluster in peak_clusters.items():
        y, x = np.where(mask[0] == cluster_id)
        data = spectra[:, y, x].reshape((nz, x.size)).T
        clusters.append((peak_cluster, x, y, data))

    return sorted(clusters, key=lambda x: len(x[0]))
