import warnings

import numpy as np
from sklearn.metrics import (
    davies_bouldin_score, silhouette_score, calinski_harabaz_score,
    pairwise_distances
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


def cop(data, dist, labels):
    clusters = set(labels)
    cpairs = _get_clust_pairs(clusters)
    prox_lst = [
        dist[np.ix_(labels == i[0], labels == i[1])].max()
        for i in cpairs
    ]

    out_l = []
    for c in clusters:
        c_data = data[labels == c]
        c_center = c_data.mean(axis=0, keepdims=True)
        c_intra = pairwise_distances(c_data, c_center).mean()

        c_prox = [prox for pair, prox in zip(cpairs, prox_lst) if c in pair]
        c_inter = min(c_prox)

        to_add = len(c_data) * c_intra / c_inter
        out_l.append(to_add)

    return sum(out_l) / len(labels)
