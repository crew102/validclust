import pytest
import pandas as pd
import numpy as np
from sklearn.datasets.samples_generator import make_blobs
from sklearn.datasets import load_iris
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import (
    silhouette_score, calinski_harabaz_score, pairwise_distances
)

from validclust.validclust import ValidClust
from validclust.indices import dunn

data, y = make_blobs(n_samples=500, centers=3, n_features=5, random_state=0)
iris = load_iris()['data']


def test_basic_run():
    vclust = ValidClust(data)
    vclust.validate()
    ser = vclust.score_df.loc[('hierarchical', ['silhouette', 'calinski']), 2]

    aclust = AgglomerativeClustering(n_clusters=2)
    y = aclust.fit_predict(data)
    actl = np.array(
        [silhouette_score(data, y), calinski_harabaz_score(data, y)]
    )

    assert np.allclose(actl, ser)


def test_index_aliases():
    vclust = ValidClust(data, indices=['sil', 'cal', 'dav'])
    assert ['silhouette', 'calinski', 'davies'] == vclust.indices
def test_dunn():
    kmeans = KMeans(n_clusters=2, random_state=0)
    labels = kmeans.fit_predict(iris)
    d_val = dunn(iris, pairwise_distances(iris), labels)
    assert .05 < d_val < .1

