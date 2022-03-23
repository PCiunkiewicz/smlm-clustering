"""
Author: Philip Ciunkiewicz
"""
import hdbscan
import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterSampler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from tqdm import trange

from .utils import custom_round


def random_search(
        param_dist, X, n=1, allowsingle=False, alpha=1.0,
        silhouette=False, calinski=False, davies=False, method='eom'):
    """Custom random parameter search function
    specifically for the hdbscan.HDBSCAN model.

    Parameters:
    -----------
    param_dist : dict
        Dict containing parameter distribution.
        See SKLearn GridSearchCV function.
    X : array_like
        Input data for clustering.
    n : int
        Number of random searches, dafault 1.
    allowsingle : bool
        Allow single cluster in HDBSCAN, default False.
    silhouette : bool
        Compute Silhouette coefficient, default False.
    calinski : bool
        Compute Calinski-Harabasz index, default False.
    davies : bool
        Compute Davies-Bouldin index, default False.

    Returns:
    --------
    results : pd.DataFrame
        Parameter search results table.
    """
    df = pd.DataFrame(ParameterSampler(param_dist, n))
    for i in [0,1]:
        rounding = np.ceil((df.iloc[:,i].max() - df.iloc[:,i].min()) / 100)
        df.iloc[:,i] = df.iloc[:,i].apply(lambda x: custom_round(x, base=rounding))

    allparams = df.to_dict('records')
    unique = list(set(frozenset(x.items()) for x in allparams))
    params = [dict(x) for x in unique]
    results = []

    for i in trange(len(params)):
        hdb = hdbscan.HDBSCAN(
            core_dist_n_jobs=6,
            gen_min_span_tree=True,
            allow_single_cluster=allowsingle,
            alpha=alpha,
            cluster_selection_method=method)
        hdb.set_params(**params[i])
        hdb.fit(X)
        params[i]['rel_validity'] = hdb.relative_validity_

        if silhouette:
            params[i]['silhouette'] = silhouette_score(X, hdb.labels_)
        if calinski:
            params[i]['calinski_harabasz'] = calinski_harabasz_score(X, hdb.labels_)
        if davies:
            params[i]['davies_bouldin'] = davies_bouldin_score(X, hdb.labels_)

        results.append(params[i])

    results = pd.DataFrame(results).round(3)
    results.sort_values(by=['rel_validity'], ascending=False, inplace=True)

    return results
