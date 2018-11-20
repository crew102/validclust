import pytest
import pandas as pd
import numpy as np
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import (
    silhouette_score, calinski_harabaz_score, davies_bouldin_score
)

from validclust.validclust import ValidClust


data, y = make_blobs(n_samples=500, centers=3, n_features=5, random_state=0)


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

