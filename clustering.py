
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
import seaborn as sns
sns.set()


###############################################################################
################################## Functions ##################################
###############################################################################


def random_search_custom_hdb(param_dist,X,n=1):
    allparams = list(ParameterSampler(param_dist,n))
    unique = list(set(frozenset(x.items()) for x in allparams))
    params = [dict(x) for x in unique]
    results = []
    
    for i in trange(len(params)):
        hdb = hdbscan.HDBSCAN(core_dist_n_jobs=6,gen_min_span_tree=True)
        hdb.set_params(**params[i])
        hdb.fit(X)
        params[i]['score'] = np.round(hdb.relative_validity_, 3)
        results.append(params[i])
        
    results = pd.DataFrame(results)
    results.sort_values(by=['score'],ascending=False,inplace=True)
    results = results[np.roll(results.columns.values,1)]
    
    return results


###############################################################################
############################### Helper Functions ##############################
###############################################################################


def plot_clusters_lite(X, labels, ax, downscale=100):
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
        if downscale > 1:
            ax.plot(*resample_2d(xy, downscale), 'o', markerfacecolor=tuple(col),
                    markeredgewidth=0.0, markersize=ms, alpha=alpha, zorder=zorder)
        else:
            ax.plot(*xy.T, 'o', markerfacecolor=tuple(col),markeredgewidth=0.0, 
                    markersize=ms, alpha=alpha, zorder=zorder)
        
    ax.set(xlabel='X [nm]', ylabel='Y [nm]')
        
def resample_2d(X, downscale=100):
    x, y = X[:,0], X[:,1]
    nbins = max(X.shape[0] // downscale, 1)
    
    hh, locx, locy = np.histogram2d(x, y, bins=nbins)
    xwidth, ywidth = np.diff(locx).mean(), np.diff(locy).mean()
    mask = hh != 0
    
    locx = locx[:-1] + xwidth
    locy = locy[:-1] + ywidth
    yy, xx = np.meshgrid(locy, locx) + np.random.uniform(-np.mean([xwidth, ywidth])/2,
                                                         np.mean([xwidth, ywidth])/2,
                                                         size=hh.shape)
    
    return xx[mask], yy[mask]

def cluster_stats(hdb, X, n, p=0.0):
    clust = hdb.labels_ == n
    mask = clust & (hdb.probabilities_ > p)
    xy = X[mask]
    
    stats = {}
    stats['total'] = np.sum(clust)
    stats['threshold'] = np.round(np.sum(mask)/np.sum(clust)*100, 2)
    stats['gy_radius'] = np.round(gy_radius(xy), 2)
    stats['density'] = np.round(cluster_density(xy), 3)
    
    return stats

def gy_radius(X):
    cm = np.mean(X, axis=0)
    gy_radius = np.sqrt(np.mean(norm(X-cm, axis=1)**2))
    
    return gy_radius

def cluster_density(X):
    total_points = X.shape[0]
    area = np.pi * gy_radius(X)**2
    density = total_points / area
    
    return density

def full_cluster_info(hdb):
    labels = hdb.labels_
    n_clusters = len(set(labels[labels != -1]))
    n_points = len(labels)
    n_outliers = np.sum(labels == -1)
    full_info = inspect.cleandoc(f"""Total number of points: {n_points}
    Estimated number of clusters: {n_clusters}
    Estimated number of outliers: {n_outliers}
    """)

    return full_info
    

def view_cluster(hdb, X, n, p=0.0, axes=None):
    if not hasattr(n,'__iter__'):
        n = [n]
    for i in n:
        clust = hdb.labels_ == i
        prob = hdb.probabilities_[clust]

        if axes is None:
            fig, axes = plt.subplots(1,2,figsize=[14,4])

        stats = cluster_stats(hdb, X, n, p)
        cluster_info = inspect.cleandoc(f"""Cluster {i}
        Total points -- {stats['total']}
        Points in threshold -- {stats['threshold']}%
        Radius of gyration -- {stats['gy_radius']}nm
        Relative Density -- {stats['density']}
        """)
        try:
            sns.distplot(prob, ax=axes[0], bins=np.linspace(0, 1, 20))
        except:
            sns.distplot(prob, ax=axes[0], bins=np.linspace(0, 1, 20), kde=False)
        axes[0].set(xlim=(0,1), xlabel='Probability', ylabel='Frequency')
        axes[0].text(0.05, 0.95, cluster_info, transform=axes[0].transAxes,
                     horizontalalignment='left', verticalalignment='top',)

        sns.scatterplot(*X[clust][prob<=p].T,alpha=0.5,color='r',ax=axes[1],label=f'$p\\leq{p}$')
        sns.scatterplot(*X[clust][prob>p].T,alpha=0.5,color='b',ax=axes[1],label=f'$p>{p}$')
        axes[1].set(xlabel='X [nm]', ylabel='Y [nm]')
