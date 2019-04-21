
###############################################################################
################################### Imports ###################################
###############################################################################


import sklearn
import hdbscan
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


def dbscan_verbose(db, X, plot=False,n=None,p=None,silent=False):
    plt.close()
    labels = np.copy(db.labels_)
    core = np.zeros_like(labels, dtype=bool)
    try:
        core[db.core_sample_indices_] = True
    except:
        core[:] = True
        
    if p is not None:
        lower_lim = (db.probabilities_ <= p)
        for i in list(set(labels))[:-1]:
            clust = (labels == i)
            labels[clust & lower_lim] = -1

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
        if cv == 1:
            params[i]['score'] = custom_silhouette(dbscan,X,cv=cv)
            roll = 1
        else:
            params[i]['score'],params[i]['stdev'] = custom_silhouette(dbscan,X,cv=cv)
            roll = 2
        results.append(params[i])
        
    results = pd.DataFrame(results)
    results.sort_values(by=['score'],ascending=False,inplace=True)
    results = results[np.roll(results.columns.values,roll)]
    
    return results

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
#         params[i]['score'] = hdbscan.validity.validity_index(X,hdb.labels_)
        results.append(params[i])
        
    results = pd.DataFrame(results)
    results.sort_values(by=['score'],ascending=False,inplace=True)
    results = results[np.roll(results.columns.values,1)]
    
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
    colors = [plt.cm.tab20(each)
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

def view_cluster(hdb,XY,n,p=0.0):
    if not hasattr(n,'__iter__'):
        n = [n]
    for i in n:
        clust = hdb.labels_ == i
        prob = hdb.probabilities_[clust]

        fig,axes = plt.subplots(1,2,figsize=[14,4])

        _ = sns.distplot(prob, norm_hist=True, ax=axes[0])
        _ = axes[0].set(xlim=(0,1), xlabel='Probability', ylabel='Frequency')

        _ = sns.scatterplot(*XY[clust][prob<=p].T,alpha=0.5,color='r',ax=axes[1],label=f'$p\\leq{p}$')
        _ = sns.scatterplot(*XY[clust][prob>p].T,alpha=0.5,color='b',ax=axes[1],label=f'$p>{p}$')
        _ = axes[1].set(xlabel='X [nm]', ylabel='Y [nm]')
