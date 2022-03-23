"""
Author: Philip Ciunkiewicz
"""
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_samples
from yellowbrick.cluster import SilhouetteVisualizer

from .stats import cluster_stats
from .utils import resample_2d

sns.set()


def plot_clusters(X, labels, ax, resolution='auto'):
    """Plot color-coded clustering results.

    Parameters:
    -----------
    X : array_like
        Input data for clustering.
    labels : array_like
        Cluster labels from HDBSCAN results.
    ax : matplotlib axes object
        Axis for plotting clusters.
    resolution : int
        Number of "pixels" for 2d histogram downscaling.
        Default 'auto' downscales to 200x200 for >5000
        samples, and no downscaling for <=5000 samples.

    Returns:
    --------
    None
    """
    if resolution == 'auto':
        resolution = np.ptp(X, axis=0) / 200 if X.shape[0] > 5000 else None
    unique_labels = set(labels)
    colors = [plt.cm.tab20(each) for each in np.linspace(0, 1, len(unique_labels))]

    for k, col in zip(unique_labels, colors):
        cluster = (labels == k)
        col = None
        alpha, m, ms, zorder = 0.4, 'o', 6, 2
        if k == -1:
            col = (0, 0, 0, 1)
            alpha, m, ms, zorder = 1, '^', 3, 1

        xy = X[cluster]
        if resolution is not None:
            ax.plot(
                *resample_2d(xy, resolution), m,
                markerfacecolor=col,
                markeredgewidth=0.0,
                markersize=ms,
                alpha=alpha,
                zorder=zorder)
        else:
            ax.plot(
                *xy.T, m,
                markerfacecolor=col,
                markeredgewidth=0.0,
                markersize=ms,
                alpha=alpha,
                zorder=zorder)

    ax.set(xlabel='X [nm]', ylabel='Y [nm]')


def view_cluster(hdb, X, n, p=0.0, axes=None):
    """Draw cluster probability distribution
    and zoomed spatial coordinates on 2x1 axes.

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
    axes : list
        Minimum two matplotlib axes objects for plotting.

    Returns:
    --------
    None
    """
    clust = hdb.labels_ == n
    if clust.sum() > 0:
        prob = hdb.probabilities_[clust]

        if axes is None:
            fig, axes = plt.subplots(1, 2, figsize=[14,4])

        stats, cluster_info, perimeter = cluster_stats(hdb, X, n, p)

        try:
            sns.histplot(prob, ax=axes[0], bins=np.linspace(0, 1, 20), kde=True)
        except Exception:
            sns.histplot(prob, ax=axes[0], bins=np.linspace(0, 1, 20), kde=False)

        axes[0].set(xlim=(0,1), xlabel='Probability', ylabel='Frequency')
        axes[0].text(
            0.05, 0.95, cluster_info,
            transform=axes[0].transAxes,
            horizontalalignment='left',
            verticalalignment='top',)

        x, y = X[clust][prob<=p].T
        sns.scatterplot(x=x, y=y, alpha=0.5, color='r', ax=axes[1], label=f'$p\\leq{p}$')

        x, y = X[clust][prob>p].T
        sns.scatterplot(x=x, y=y, alpha=0.5, color='b', ax=axes[1], label=f'$p>{p}$')

        axes[1].plot(*perimeter.T, 'k:')
        axes[1].set(xlabel='X [nm]', ylabel='Y [nm]')


def view_silhouette(hdb, X, ax):
    """Generates a Silhouette plot for
    the clustering results.
    """
    nclusters = len(set(hdb.labels_[hdb.labels_ != -1]))
    setattr(hdb, 'n_clusters_', nclusters)

    sil_samples = silhouette_samples(X, hdb.labels_)
    setattr(hdb, 'silhouette_samples_', sil_samples)
    setattr(hdb, 'silhouette_score_', sil_samples.mean())
    setattr(hdb, 'n_samples_', X.shape[0])

    visualizer = SilhouetteVisualizer(hdb, ax=ax)

    visualizer.draw(hdb.labels_)
    ax.text(
        0.05, 0.95, f'Score: {sil_samples.mean():.3f}',
        transform=ax.transAxes,
        horizontalalignment='left',
        verticalalignment='top',)
    ax.set(ylabel='Cluster label', xlabel='Silhouette coefficient', yticks=[])
