
###############################################################################
################################### Imports ###################################
###############################################################################


import sklearn
from sklearn import metrics
import pandas as pd
import numpy as np
from tqdm import trange
from scipy.spatial.distance import cdist
from time import time
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.model_selection import ParameterSampler
import seaborn as sns
sns.set()


###############################################################################
################################## Functions ##################################
###############################################################################


def dbscan_verbose(db, X, plot=False,n=None,silent=False):
    plt.close()
    core = np.zeros_like(db.labels_, dtype=bool)
    core[db.core_sample_indices_] = True
    labels = db.labels_

    if n is None:
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise_ = list(labels).count(-1)

        print(f'Estimated number of clusters: {n_clusters_}')
        print(f'Estimated number of noise points: {n_noise_}')
        print(f'Total number of points: {len(labels)}')
    
    if plot:
        plot_clusters(X,labels,core,n,silent)
        
def custom_silhouette(dbscan,X,cv=1):
    scores = []
    for i in range(cv):
        samples = np.random.choice(np.arange(len(X)),10000)
        xy = X[samples]
        xydist = cdist(xy,xy)
        db = dbscan.fit(X)
        labels = db.labels_[samples]
        scores.append(silhouette_score(xydist, labels, metric="precomputed"))
    score = np.mean(scores)
    stdev = np.std(scores)
    if cv > 1:
        return score,stdev
    
    return score
         
def random_search_custom(dbscan,param_dist,X,n=1,cv=5):
    allparams = list(ParameterSampler(param_dist,n))
    unique = list(set(frozenset(x.items()) for x in allparams))
    params = [dict(x) for x in unique]
    results = []
    for i in trange(len(params)):
        dbscan.set_params(**params[i])
        params[i]['score'],params[i]['stdev'] = custom_silhouette(dbscan,X,cv=cv)
        results.append(params[i])
        
    results = pd.DataFrame(results)
    results.sort_values(by=['score'],ascending=False,inplace=True)
    results = results[['score','stdev','eps','min_samples']]
    
    return results


###############################################################################
############################### Helper Functions ##############################
###############################################################################


def plot_clusters(X,labels,core,n=None,silent=False):
    if n is not None:
        mask = np.isin(labels,n)
        X,labels,core = X[mask],labels[mask],core[mask]
        
    plotlabel = None
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each)
              for each in np.linspace(0, 1, len(unique_labels))]

    for k, col in zip(unique_labels, colors):
        cluster = (labels == k)
        alpha = 0.4
        ms = 6
        if k == -1:
            col = [0, 0, 0, 1]
            alpha = 1
            ms = 1
        if n is not None:
            plotlabel = str(k)
            if not silent:
                print(f'Cluster index: {k}')
                print(f'Estimated number of points: {np.sum(cluster)}')
                print(f'Estimated number of core points: {np.sum(cluster & core)}')
                print(f'Estimated number of edge points: {np.sum(cluster & ~core)}\n')

        xy = X[cluster & core]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgewidth=0.0, markersize=ms, alpha=alpha,
                 label=plotlabel)

        xy = X[cluster & ~core]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=ms, alpha=alpha)

    if n is not None:
        plt.legend()
#     plt.show()
