import collections as cl
import itertools as it

import numpy as np


Spectra = cl.namedtuple("Spectra", "data ucx ucy")
Peak = cl.namedtuple("Peak", "name x0 y0")
Cluster = cl.namedtuple("Cluster", "peaks x y data")


def cluster_peaks(spectra: Spectra, peaks: np.ndarray, contour_level: float) -> list:

    print("- Segmenting the spectra and clustering the peaks...")

    nz, ny, nx = spectra.data.shape

    peaks_ppm = {name: Peak(name, x, y) for name, y, x in peaks}

    x_pts = np.rint(spectra.ucx.f(peaks["x"], "ppm")).astype(int)
    y_pts = np.rint(spectra.ucy.f(peaks["y"], "ppm")).astype(int)
    peaks_pt = {name: (x, y) for name, x, y in zip(peaks["names"], x_pts, y_pts)}

    names = set(peaks_pt)

    clusters = []
    for name, (x_pt, y_pt) in peaks_pt.items():
        if name not in names:
            continue

        region = _flood(x_pt, y_pt, spectra, contour_level)

        peak_names = {name for name in names if peaks_pt[name] in region}

        x_reg, y_reg = np.asarray(region).T

        data_cluster = (
            spectra.data[:, y_reg % ny, x_reg % nx].reshape((nz, len(region))).T
        )

        x_cluster = spectra.ucx.unit(x_reg, "ppm")
        y_cluster = spectra.ucy.unit(y_reg, "ppm")

        peak_cluster = [peaks_ppm[name] for name in peak_names]

        clusters.append(Cluster(peak_cluster, x_cluster, y_cluster, data_cluster))

        names = names - peak_names

    clusters = _merge_close_clusters(clusters, spectra, 20.0)

    return sorted(clusters, key=lambda x: len(x[0]))


def _merge_close_clusters(clusters, spectra, cutoff):
    grouped = True
    while grouped:
        grouped = False
        for cluster1, cluster2 in it.combinations(clusters, 2):
            if _distance(cluster1, cluster2, spectra) <= cutoff:
                merged = _merge_clusters(cluster1, cluster2)
                clusters.remove(cluster1)
                clusters.remove(cluster2)
                clusters.append(merged)
                grouped = True
                break
    return clusters


def _merge_clusters(cluster1, cluster2):
    peak_cluster = [*cluster1.peaks, *cluster2.peaks]
    x_cluster = np.hstack((cluster1.x, cluster2.x))
    y_cluster = np.hstack((cluster1.y, cluster2.y))
    data_cluster = np.vstack((cluster1.data, cluster2.data))
    return Cluster(peak_cluster, x_cluster, y_cluster, data_cluster)


def _distance(cluster1, cluster2, spectra):
    ucx, ucy = spectra.ucx, spectra.ucy
    distances2 = [
        (ucx.hz(ucx.f(peak1.x0, "ppm")) - ucx.hz(ucx.f(peak2.x0, "ppm"))) ** 2
        + (ucy.hz(ucy.f(peak1.y0, "ppm")) - ucy.hz(ucy.f(peak2.y0, "ppm"))) ** 2
        for peak1, peak2 in it.product(cluster1.peaks, cluster2.peaks)
    ]

    return np.sqrt(min(distances2))


def _flood(x_start: int, y_start: int, spectra: Spectra, threshold: float) -> list:
    ny, nx = spectra.data.shape[1:]
    stack = [(x_start, y_start)]
    points = set()
    while stack:
        x, y = stack.pop()
        above = abs(spectra.data[0, y % ny, x % nx]) >= threshold
        unvisited = (x, y) not in points
        if above and unvisited:
            points.add((x, y))
            stack.extend([(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)])
    if len(points) < 9:
        points = zip(range(x_start - 1, x_start + 2), range(y_start - 1, y_start + 2))
    return list(points)
