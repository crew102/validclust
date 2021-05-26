# coding: utf-8

import warnings

import numpy as np
from packaging import version
import sklearn
from sklearn.metrics import (
    davies_bouldin_score, silhouette_score, pairwise_distances
)

# They changed the name of calinski_harabaz_score in later version of sklearn:
# https://github.com/scikit-learn/scikit-learn/blob/c4733f4895c1becdf587b38970f6f7066656e3f9/doc/whats_new/v0.20.rst#id2012
sklearn_version = version.parse(sklearn.__version__)
nm_chg_ver = version.parse("0.23")
if sklearn_version >= nm_chg_ver:
    from sklearn.metrics import calinski_harabasz_score as _cal_score
else:
    from sklearn.metrics import calinski_harabaz_score as _cal_score


def _get_clust_pairs(clusters):
    return [(i, j) for i in clusters for j in clusters if i > j]


def _dunn(data=None, dist=None, labels=None):
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


def dunn(dist, labels):
    r"""Calculate the Dunn CVI

    See Dunn (2008) for details on how the index is calculated. [2]_

    Parameters
    ----------
    dist : array-like, shape = [n_samples, n_samples]
        A distance matrix containing the distances between each observation.
    labels : array [n_samples]
        The cluster labels for each observation.

    Returns
    -------
    float
        The Dunn index.

    References
    ----------
    .. [2] Dunn, J. C. (1973). Well-Separated Clusters and Optimal Fuzzy
       Partitions. Journal of Cybernetics, 4(1), 95-104. DOI:
       10.1080/01969727408546059.

    Examples
    --------
    >>> from sklearn.cluster import k_means
    >>> from sklearn.metrics import pairwise_distances
    >>> from sklearn.datasets import load_iris
    >>> from validclust import dunn
    >>> data = load_iris()['data']
    >>> _, labels, _ = k_means(data, n_clusters=3)
    >>> dist = pairwise_distances(data)
    >>> dunn(dist, labels)
    0.09880739332808611
    """
    return _dunn(data=None, dist=dist, labels=labels)


def cop(data, dist, labels):
    r"""Calculate the COP CVI

    See Gurrutxaga et al. (2010) for details on how the index is calculated. [1]_

    Parameters
    ----------
    data : array-like, shape = [n_samples, n_features]
        The data to cluster.
    dist : array-like, shape = [n_samples, n_samples]
        A distance matrix containing the distances between each observation.
    labels : array [n_samples]
        The cluster labels for each observation.

    Returns
    -------
    float
        The COP index.

    References
    ----------
    .. [1] Gurrutxaga, I., Albisua, I., Arbelaitz, O., Martín, J., Muguerza,
       J., Pérez, J., Perona, I. (2010). SEP/COP: An efficient method to find
       the best partition in hierarchical clustering based on a new cluster
       validity index. Pattern Recognition, 43(10), 3364-3373. DOI:
       10.1016/j.patcog.2010.04.021.

    Examples
    --------
    >>> from sklearn.cluster import k_means
    >>> from sklearn.metrics import pairwise_distances
    >>> from sklearn.datasets import load_iris
    >>> from validclust import cop
    >>> data = load_iris()['data']
    >>> _, labels, _ = k_means(data, n_clusters=3)
    >>> dist = pairwise_distances(data)
    >>> cop(data, dist, labels)
    0.133689170400615
    """
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


def _silhouette_score2(data=None, dist=None, labels=None):
    return silhouette_score(dist, labels, metric='precomputed')


def _davies_bouldin_score2(data=None, dist=None, labels=None):
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', 'divide by zero')
        return davies_bouldin_score(data, labels)


def _calinski_harabaz_score2(data=None, dist=None, labels=None):
    return _cal_score(data, labels)
