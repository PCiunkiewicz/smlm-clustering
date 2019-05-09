
###############################################################################
################################### Imports ###################################
###############################################################################


import sklearn
import hdbscan
import inspect
import pandas as pd
import numpy as np
from numpy.linalg import norm
from tqdm import trange
from time import time
import matplotlib.pyplot as plt

from sklearn.model_selection import ParameterSampler
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabaz_score
from sklearn.metrics import davies_bouldin_score

from scipy.spatial import ConvexHull
from scipy.stats import gaussian_kde
from scipy.signal import argrelmin
from scipy.signal import argrelmax
from sklearn import datasets
import seaborn as sns
sns.set()


###############################################################################
################################## Functions ##################################
###############################################################################


def custom_round(x, base=5):
    return int(base * np.ceil(float(x)/base))

def random_search_custom_hdb(param_dist, X, n=1, allowsingle=False, 
                             silhouette=False, calinski=False, davies=False):
    df = pd.DataFrame(ParameterSampler(param_dist, n))
    for i in [0,1]:
        rounding = np.ceil((df.iloc[:,i].max() - df.iloc[:,i].min()) / 100)
        df.iloc[:,i] = df.iloc[:,i].apply(lambda x: custom_round(x, base=rounding))

    allparams = df.to_dict('records')
    unique = list(set(frozenset(x.items()) for x in allparams))
    params = [dict(x) for x in unique]
    results = []
    
    for i in trange(len(params)):
        hdb = hdbscan.HDBSCAN(core_dist_n_jobs=6, gen_min_span_tree=True,
                              allow_single_cluster=allowsingle)
        hdb.set_params(**params[i])
        hdb.fit(X)
        params[i]['score'] = hdb.relative_validity_
        if silhouette:
            params[i]['silhouette'] = silhouette_score(X, hdb.labels_)
        if calinski:
            params[i]['calinski_harabaz'] = calinski_harabaz_score(X, hdb.labels_)
        if davies:
            params[i]['davies_bouldin'] = davies_bouldin_score(X, hdb.labels_)
        results.append(params[i])
        
    results = pd.DataFrame(results).round(3)
    results.sort_values(by=['score'], ascending=False, inplace=True)
    
    return results


###############################################################################
############################### Helper Functions ##############################
###############################################################################


def plot_clusters_lite(X, labels, ax, resolution='auto'):
    if resolution == 'auto':
        resolution = np.ptp(X, axis=0) / 200 if X.shape[0] > 5000 else None
    unique_labels = set(labels)
    colors = [plt.cm.tab20(each)
              for each in np.linspace(0, 1, len(unique_labels))]

    for k, col in zip(unique_labels, colors):
        cluster = (labels == k)
        alpha, ms, zorder = 0.4, 6, 2
        if k == -1:
            col = [0, 0, 0, 1]
            alpha, ms, zorder = 1, 2, 1

        xy = X[cluster]
        if resolution is not None:
            ax.plot(*resample_2d(xy, resolution), 'o', markerfacecolor=tuple(col),
                    markeredgewidth=0.0, markersize=ms, alpha=alpha, zorder=zorder)
        else:
            ax.plot(*xy.T, 'o', markerfacecolor=tuple(col),markeredgewidth=0.0, 
                    markersize=ms, alpha=alpha, zorder=zorder)
        
    ax.set(xlabel='X [nm]', ylabel='Y [nm]')
        
def resample_2d(X, resolution):
    x, y = X[:,0], X[:,1]
    nbins = np.ptp(X, axis=0) / resolution
    
    hh, locx, locy = np.histogram2d(x, y, bins=np.ceil(nbins))
    xwidth, ywidth = np.diff(locx).mean(), np.diff(locy).mean()
    mask = hh != 0
    
    locx = locx[:-1] + xwidth
    locy = locy[:-1] + ywidth
    yy, xx = np.meshgrid(locy, locx)
    np.random.seed(0)
    yy += np.random.uniform(-xwidth/2, xwidth/2, size=hh.shape)
    xx += np.random.uniform(-ywidth/2, ywidth/2, size=hh.shape)
    
    return xx[mask], yy[mask]

def cluster_stats(hdb, X, n, p=0.0):
    clust = hdb.labels_ == n
    mask = clust & (hdb.probabilities_ > p)
    xy = X[mask]
    
    stats = {}
    stats['total'] = np.sum(clust)
    stats['threshold'] = np.sum(mask)/np.sum(clust)*100
    stats['gy_radius'] = gy_radius(xy)
    stats['density'], vertices, unit = cluster_density(xy)
    perimeter = xy[vertices]

    cluster_info = inspect.cleandoc(f"""Cluster {n}
    Total points -- {stats['total']}
    Points in threshold -- {stats['threshold']:.2f}%
    Radius of gyration (nm) -- {stats['gy_radius']:.2f}
    Density (per {unit}^2) -- {stats['density']:.2f}
    """)
    
    return stats, cluster_info, perimeter

def gy_radius(X):
    cm = np.mean(X, axis=0)
    gy_radius = np.sqrt(np.mean(norm(X-cm, axis=1)**2))
    
    return gy_radius

def cluster_density(X):
    total_points = X.shape[0]
    X_ch = ConvexHull(X)
    area = X_ch.volume # 2D volume is area for ConvexHull
    density = total_points / area * 1e6 # density per square micrometer
    vertices = np.append(X_ch.vertices, X_ch.vertices[0]) # first entry at end
    unit = ("micron" if density < 1e5 else "nm")
    if density >= 1e5:
        density /= 1e6
    
    return density, vertices, unit

def _custom_ravel(a, b):
    ab = []
    for i, x in enumerate(a):
        ab.append(x)
        try:
            ab.append(b[i])
        except:
            pass
    return ab
        
def count_multi_clusters(hdb):
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
                normed_peaks = np.diff(_custom_ravel(Z[maxs], Z[mins])) / Z.max()

                if normed_peaks[normed_peaks < -0.25].size > 0:
                    count += 1
        except:
            pass

    return count

def full_cluster_info(hdb):
    labels = hdb.labels_
    n_clusters = len(set(labels[labels != -1]))
    n_points = len(labels)
    n_outliers = np.sum(labels == -1)
    n_multi = count_multi_clusters(hdb)
    full_info = inspect.cleandoc(f"""Total number of points: {n_points}
    Estimated number of clusters: {n_clusters}
    Estimated number of outliers: {n_outliers}
    Estimated number of multi-clusters: {n_multi}
    """)

    return full_info

def make_training_dataset(size=1000, noise=0.03, std=0.1, state=8):
    noisy_circles = datasets.make_circles(n_samples=size, random_state=state, 
                                          factor=.3, noise=noise)
    noisy_moons = datasets.make_moons(n_samples=size, random_state=state, 
                                      noise=noise)
    blobs = datasets.make_blobs(n_samples=size, random_state=state, 
                                centers=15, cluster_std=std)
    np.random.seed(state)
    bkgnd = np.dstack((np.random.uniform(-12, 15, size=size//4),
                       np.random.uniform(-15, 12, size=size//4)))[0]
    
    moons = noisy_moons[0] * 4
    moons[:,0] += 5
    moons[:,1] -= 10

    circles = noisy_circles[0] * 2
    circles[:,0] -= 7
    circles[:,1] += 3

    XY = np.vstack((circles, moons, blobs[0], bkgnd)) * 1000
    
    return XY    

def view_cluster(hdb, X, n, p=0.0, axes=None):
    clust = hdb.labels_ == n
    if clust.sum() > 0:
        prob = hdb.probabilities_[clust]

        if axes is None:
            fig, axes = plt.subplots(1, 2, figsize=[14,4])

        stats, cluster_info, perimeter = cluster_stats(hdb, X, n, p)

        try:
            sns.distplot(prob, ax=axes[0], bins=np.linspace(0, 1, 20))
        except:
            sns.distplot(prob, ax=axes[0], bins=np.linspace(0, 1, 20), kde=False)

        axes[0].set(xlim=(0,1), xlabel='Probability', ylabel='Frequency')
        axes[0].text(0.05, 0.95, cluster_info, transform=axes[0].transAxes,
                     horizontalalignment='left', verticalalignment='top',)

        sns.scatterplot(*X[clust][prob<=p].T, alpha=0.5, color='r', ax=axes[1], 
                        label=f'$p\\leq{p}$')
        sns.scatterplot(*X[clust][prob>p].T, alpha=0.5, color='b', ax=axes[1], 
                        label=f'$p>{p}$')
        axes[1].plot(*perimeter.T, 'k:')
        axes[1].set(xlabel='X [nm]', ylabel='Y [nm]')
