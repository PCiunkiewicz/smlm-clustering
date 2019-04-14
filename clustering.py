import sklearn
from sklearn import metrics
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
from time import time
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.model_selection import RandomizedSearchCV
import seaborn as sns
sns.set()


def dbscan_verbose(db, XY, plot=False):
    core = np.zeros_like(db.labels_, dtype=bool)
    core[db.core_sample_indices_] = True
    labels = db.labels_

    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print('Estimated number of clusters: %d' % n_clusters_)
    print('Estimated number of noise points: %d' % n_noise_)
    print('Total number of points: %d' % len(labels))
    
    if plot:
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

            xy = XY[cluster & core]
            plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                     markeredgewidth=0.0, markersize=ms, alpha=alpha)

            xy = XY[cluster & ~core]
            plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                     markeredgecolor='k', markersize=ms, alpha=alpha)

        plt.show()
        
def custom_silhouette(dbscan,X):
    samples = np.random.choice(np.arange(len(X)),10000)
    xy = X[samples]
    xydist = cdist(xy,xy)
    db = dbscan.fit(X)
    labels = db.labels_[samples]
    score = silhouette_score(xydist, labels, metric="precomputed")
    
    return score

def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")
            
def param_search(dbscan,param_dist,X,n=10,cv=3,showreport=True):
    random_search = RandomizedSearchCV(dbscan,param_distributions=param_dist,
                                       n_iter=n,cv=cv,iid=False,
                                       scoring=custom_silhouette)
    start = time()
    random_search.fit(X)
    print("RandomizedSearchCV took %.2f seconds for %d candidate"
          " parameter settings." % ((time() - start), n))
    
    if showreport:
        report(random_search.cv_results_)
        
    return random_search
