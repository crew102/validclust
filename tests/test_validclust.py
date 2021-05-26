import pandas as pd
import numpy as np
from packaging import version
import sklearn
from sklearn.datasets import make_blobs
from sklearn.datasets import load_iris
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score, pairwise_distances

sklearn_version = version.parse(sklearn.__version__)
nm_chg_ver = version.parse("0.23")
if sklearn_version >= nm_chg_ver:
    from sklearn.metrics import calinski_harabasz_score as _cal_score
else:
    from sklearn.metrics import calinski_harabaz_score as _cal_score

from validclust import ValidClust
from validclust import dunn

data, y = make_blobs(n_samples=500, centers=3, n_features=5, random_state=0)
iris = load_iris()['data']


def test_basic_run():
    vclust = ValidClust(k=[2, 3, 4, 5])
    score_df = vclust.fit_predict(data)
    ser = score_df.loc[('hierarchical', ['silhouette', 'calinski']), 2]

    aclust = AgglomerativeClustering(n_clusters=2)
    y = aclust.fit_predict(data)
    actl = np.array(
        [silhouette_score(data, y), _cal_score(data, y)]
    )

    assert np.allclose(actl, ser)


def test_index_aliases():
    vclust = ValidClust(k=[2, 3, 4, 5], indices=['sil', 'cal', 'dav'])
    assert ['silhouette', 'calinski', 'davies'] == vclust.indices


def test_normalize():
    vclust = ValidClust(k=[2, 3, 4, 5])
    vclust.fit(data)
    df = vclust._normalize()
    assert all(df.apply(lambda col: all(col <= 1)))


def test_dunn():
    kmeans = KMeans(n_clusters=2, random_state=0)
    labels = kmeans.fit_predict(iris)
    d_val = dunn(pairwise_distances(iris), labels)
    assert .05 < d_val < .1


def test_non_zero_diag_edge_case():
    # https://github.com/crew102/validclust/issues/4
    df = pd.DataFrame(2 + 3 * np.random.randn(1000, 43))
    vclust = ValidClust(k=20, methods=['kmeans'])
    v_out = vclust.fit_predict(df)
    dev_null = v_out.head()
    assert True
