from itertools import product

import numpy as np


def flood_fill(data, mask, threshold, cluster_id, x_start, y_start):
    ny, nx = data.shape

    stack = [(x_start, y_start)]

    while stack:

        x, y = stack.pop()

        if (abs(data[y, x]) >= threshold and mask[0, y, x] == 0.0) or mask[0, y, x] < 0.0:

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

    print('- Marking peaks...')

    nz, ny, nx = mask.shape

    for peak in peak_list:

        x_pt = ucx.f(peak[2], 'ppm')
        y_pt = ucy.f(peak[1], 'ppm')

        x_pt %= nx
        y_pt %= ny

        x_index = int(x_pt + 0.5)
        y_index = int(y_pt + 0.5)

        if x_index == 0:
            x_index += 1

        if y_index == 0:
            y_index += 1

        if x_index == nx - 1:
            x_index -= 1

        if y_index == ny - 1:
            y_index -= 1

        mask[:, y_index - 2:y_index + 3, x_index - 2:x_index + 3] = -1.0

    return mask


def find_disjoint_regions(data, mask, contour_level):
    """Finds regions of the spectra above the contour level threshold."""

    print('- Segmenting the spectra according to the threshold level...')

    nz, ny, nx = data.shape

    cluster_index = 0

    for y, x in product(range(ny), range(nx)):

        if mask[0, y, x] <= 0.0:
            cluster_index += 1
            flood_fill(data[0], mask, contour_level, cluster_index, x, y)

    return mask


def make_peak_clusters(spectra, mask, peak_list, ucx, ucy):
    """Identifies overlapping peaks."""

    print('- Clustering of peaks...')

    nz, ny, nx = mask.shape

    peak_list_pt = []

    for peak in peak_list:
        x_pt = ucx.f(peak[2], 'ppm')
        y_pt = ucy.f(peak[1], 'ppm')

        x_alias = x_pt // nx
        y_alias = y_pt // ny

        x_pt %= nx
        y_pt %= ny

        peak_list_pt.append((peak[0], x_pt, y_pt, x_alias, y_alias))

    peak_clusters = {}

    for peak in peak_list_pt:
        x, y = peak[1:3]
        x = int(round(x))
        y = int(round(y))
        cluster_id = mask[0, y, x]
        peak_clusters.setdefault(cluster_id, []).append(peak)

    clusters = []

    for index, (cluster_id, peak_cluster) in enumerate(peak_clusters.items()):
        y_grid, x_grid = np.where(mask[0, :, :] == cluster_id)
        n_fit = len(x_grid)
        data_to_fit = spectra[:, y_grid, x_grid].reshape((nz, n_fit)).T
        clusters.append((peak_cluster, x_grid, y_grid, data_to_fit))

    return clusters
