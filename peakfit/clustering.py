import itertools as it
from itertools import product
from typing import NamedTuple

import numpy as np
import pandas as pd
from nmrglue.fileio.fileiobase import unit_conversion
from scipy import spatial
from scipy.cluster.hierarchy import fcluster, single

from peakfit.messages import print_segmenting


class Spectra(NamedTuple):
    data: np.ndarray
    ucx: unit_conversion
    ucy: unit_conversion
    z_values: np.ndarray


class Cluster(NamedTuple):
    peaks: pd.DataFrame
    x: np.ndarray
    y: np.ndarray
    data: np.ndarray


def _merge_clusters(clusters: list[Cluster]) -> Cluster:
    cluster_peaks = pd.concat([cluster.peaks for cluster in clusters]).reset_index(
        drop=True
    )
    cluster_x = np.hstack([cluster.x for cluster in clusters])
    cluster_y = np.hstack([cluster.y for cluster in clusters])
    cluster_data = np.vstack([cluster.data for cluster in clusters])
    return Cluster(cluster_peaks, cluster_x, cluster_y, cluster_data)


def _distance(cluster1: Cluster, cluster2: Cluster) -> float:
    points1 = cluster1.peaks[["x0_hz", "y0_hz"]]
    points2 = cluster2.peaks[["x0_hz", "y0_hz"]]
    distances = spatial.distance.cdist(points1, points2)
    return np.min(distances)


def _merge_close_clusters2(
    clusters: list[Cluster],
    cutoff: float,
) -> list[Cluster]:
    pdist = np.array(
        [
            _distance(cluster1, cluster2)
            for cluster1, cluster2 in it.combinations(clusters, 2)
        ]
    )
    dendrogram = single(pdist)
    clusters_ids = fcluster(dendrogram, t=cutoff, criterion="distance")

    if len(clusters) == len(np.unique(clusters_ids)):
        return clusters

    clusters_dict: dict[int, list[Cluster]] = {}
    for i, cluster in zip(clusters_ids, clusters, strict=False):
        clusters_dict.setdefault(i, []).append(cluster)

    return [_merge_clusters(cluster_list) for cluster_list in clusters_dict.values()]


def _select_square(x: int, y: int, size: int) -> set[tuple[int, int]]:
    size = size // 2
    return set(product(range(x - size, x + size + 1), range(y - size, y + size + 1)))


def _flood(
    x_start: int,
    y_start: int,
    spectra: Spectra,
    threshold: float,
) -> pd.DataFrame:
    data = spectra.data
    data_above = np.any(np.absolute(data) >= threshold, axis=0)
    _nz, ny, nx = data.shape
    stack = [(x_start, y_start)]
    points: set[tuple[int, int]] = set()
    while stack:
        x, y = stack.pop()
        x %= nx
        y %= ny
        above = data_above[y, x]
        unvisited = (x, y) not in points
        if above and unvisited:
            points.add((x, y))
            stack.extend(_select_square(x, y, 3) - {(x, y)})
    if len(points) < 9:
        points = _select_square(x, y, 3)
    return pd.DataFrame(points, columns=["x0_int", "y0_int"])


def get_cluster_data(data: np.ndarray, region: pd.DataFrame) -> np.ndarray:
    nz, ny, nx = data.shape
    region_x, region_y = region["x0_int"], region["y0_int"]
    region_data = data[:, region_y, region_x]
    return region_data.reshape((nz, len(region))).T


def cluster_peaks(
    spectra: Spectra,
    peaks: pd.DataFrame,
    contour_level: float,
    merge_cluster_threshold_hz: float | None = None,
) -> list[Cluster]:
    print_segmenting()

    clusters = []
    selected_peaks: set[str] = set()

    for peak in peaks.itertuples(index=False, name="Peak"):
        if peak.name in selected_peaks:
            continue
        region = _flood(peak.x0_int, peak.y0_int, spectra, contour_level)
        cluster_peaks = peaks.merge(region, how="inner", on=["x0_int", "y0_int"])
        cluster_x = region["x0_int"].to_numpy()
        cluster_y = region["y0_int"].to_numpy()
        cluster_data = get_cluster_data(spectra.data, region)
        cluster = Cluster(cluster_peaks, cluster_x, cluster_y, cluster_data)
        clusters.append(cluster)
        selected_peaks = {*selected_peaks, *cluster_peaks["name"]}

    if merge_cluster_threshold_hz is not None:
        clusters = _merge_close_clusters2(clusters, merge_cluster_threshold_hz)

    return sorted(clusters, key=lambda x: len(x[0]))
