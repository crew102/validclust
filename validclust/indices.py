import numpy as np


def get_clust_pairs(clusters):
    return [(i, j) for i in clusters for j in clusters if i > j]


def dunn(dist, labels):
    clusters = set(labels)
    inter_dists = [
        dist[np.ix_(labels == i, labels == j)].min()
        for i, j in get_clust_pairs(clusters)
    ]
    intra_dists = [
        dist[np.ix_(labels == i, labels == i)].max()
        for i in clusters
    ]
    return min(inter_dists) / max(intra_dists)

