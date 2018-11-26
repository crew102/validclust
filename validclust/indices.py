import warnings

import numpy as np
from sklearn.metrics import davies_bouldin_score


def _get_clust_pairs(clusters):
    return [(i, j) for i in clusters for j in clusters if i > j]


def dunn(X, labels):
    clusters = set(labels)
    inter_dists = [
        X[np.ix_(labels == i, labels == j)].min()
        for i, j in _get_clust_pairs(clusters)
    ]
    intra_dists = [
        X[np.ix_(labels == i, labels == i)].max()
        for i in clusters
    ]
    return min(inter_dists) / max(intra_dists)


def davies_bouldin_score2(X, labels):
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', 'divide by zero')
        return davies_bouldin_score(X, labels)
