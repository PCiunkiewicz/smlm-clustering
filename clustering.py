import sklearn
from sklearn import metrics
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


def dbscan_verbose(db, XY, plot=False, S_score=False):
    core = np.zeros_like(db.labels_, dtype=bool)
    core[db.core_sample_indices_] = True
    labels = db.labels_

    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print('Estimated number of clusters: %d' % n_clusters_)
    print('Estimated number of noise points: %d' % n_noise_)
    print('Total number of points: %d' % len(labels))
    
    if S_score:
        print("Silhouette Coefficient: %0.3f"
              % metrics.silhouette_score(XY, labels))
    
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