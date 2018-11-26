import warnings

import numpy as np
from sklearn.metrics import (
    davies_bouldin_score, silhouette_score, calinski_harabaz_score
)


def _get_clust_pairs(clusters):
    return [(i, j) for i in clusters for j in clusters if i > j]


def dunn(data, dist, labels):
    clusters = set(labels)
    inter_dists = [
        dist[np.ix_(labels == i, labels == j)].min()
        for i, j in _get_clust_pairs(clusters)
    ]
    intra_dists = [
        dist[np.ix_(labels == i, labels == i)].max()
        for i in clusters
    ]
    return min(inter_dists) / max(intra_dists)


def silhouette_score2(data, dist, labels):
    return silhouette_score(dist, labels, 'precomputed')


def davies_bouldin_score2(data, dist, labels):
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', 'divide by zero')
        return davies_bouldin_score(data, labels)


def calinski_harabaz_score2(data, dist, labels):
    return calinski_harabaz_score(data, labels)
