"""
Author: Philip Ciunkiewicz
"""
import inspect

import numpy as np
from numpy.linalg import norm
from scipy.spatial import ConvexHull
from scipy.stats import gaussian_kde
from scipy.signal import argrelmin, argrelmax

from .utils import custom_ravel


def cluster_stats(hdb, X, n, p=0.0):
    """Compute individual cluster statistics.

    Parameters:
    -----------
    hdb : hdbscan.HDBSCAN object
        Trained clustering model with labels.
    X : array_like
        Input data for clustering.
    n : int
        Cluster index (must be < len(hdb.labels_)).
    p : float
        Probability threshold on half-open interval [0,1).

    Returns:
    --------
    stats : dict
        Cluster statistics dictionary.
    cluster_info : str
        Formatter docstring with cluster stats printed.
    perimeter : array_like
        Array of perimeter vertex coordinates.
    """
    clust = hdb.labels_ == n
    mask = clust & (hdb.probabilities_ > p)
    xy = X[mask]

    stats = {}
    stats['total'] = np.sum(clust)
    stats['threshold'] = np.sum(mask)/np.sum(clust)*100
    stats['gy_radius'] = gy_radius(xy)
    stats['density'], vertices, unit = cluster_density(xy)
    perimeter = xy[vertices]

    cluster_info = inspect.cleandoc(
        f"""Cluster {n}
        Total points -- {stats['total']}
        Points in threshold -- {stats['threshold']:.2f}%
        Radius of gyration (nm) -- {stats['gy_radius']:.2f}
        Density (per {unit}^2) -- {stats['density']:.2f}
        """)

    return stats, cluster_info, perimeter


def gy_radius(X):
    """Compute the radius of gyration of
    a set of points in N dimensions.
    """
    cm = np.mean(X, axis=0)
    gy_radius = np.sqrt(np.mean(norm(X-cm, axis=1)**2))

    return gy_radius


def cluster_density(X):
    """Compute the convex-hull-density
    of a set of points in N dimensions.
    """
    total_points = X.shape[0]
    X_ch = ConvexHull(X)
    area = X_ch.volume # 2D volume is area for ConvexHull
    density = total_points / area * 1e6 # density per square micrometer
    vertices = np.append(X_ch.vertices, X_ch.vertices[0]) # first entry at end
    unit = ("micron" if density < 1e5 else "nm")
    if density >= 1e5:
        density /= 1e6

    return density, vertices, unit


def count_multi_clusters(hdb):
    """Estimate the number of multi-clusters
    based on probability distribution for a
    trained hdbscan.HDBSCAN model.
    """
    count = 0
    for n in set(hdb.labels_):
        try:
            clust = hdb.labels_ == n
            if clust.sum() > 0:
                prob = hdb.probabilities_[clust]

                kernel = gaussian_kde(prob)
                X = np.linspace(0, 1, 100)
                Z = kernel(X)

                mins = argrelmin(Z)[0]
                maxs = argrelmax(Z)[0]
                normed_peaks = np.diff(custom_ravel(Z[maxs], Z[mins])) / Z.max()

                if normed_peaks[normed_peaks < -0.25].size > 0:
                    count += 1
        except Exception:
            pass

    return count


def full_cluster_info(hdb):
    """Compute and format the statistics
    for the full clustering input dataset.
    Includes total number of points, estimated
    number of clusters, estimated number of
    outliers, and estimated number of multi-clusters.
    """
    labels = hdb.labels_
    n_clusters = len(set(labels[labels != -1]))
    n_points = len(labels)
    n_outliers = np.sum(labels == -1)
    n_multi = count_multi_clusters(hdb)
    full_info = inspect.cleandoc(
        f"""Total number of points: {n_points}
        Estimated number of clusters: {n_clusters}
        Estimated number of outliers: {n_outliers}
        Estimated number of multi-clusters: {n_multi}
        """)

    return full_info
