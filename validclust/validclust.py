import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans, SpectralClustering
from sklearn.metrics import (
    silhouette_score, calinski_harabaz_score, davies_bouldin_score
)
from sklearn.metrics.pairwise import pairwise_distances

from .indices import dunn


class ValidClust:

    def __init__(self, data, n_clusters=[2, 3, 4, 5],
                 indices=['silhouette', 'calinski', 'davies'],
                 methods=['hierarchical', 'kmeans'],
                 linkage='ward', metric='euclidean'):

        self.data = data
        self.n_clusters = n_clusters
        self.indices = indices
        self.methods = methods
        self.linkage = linkage
        self.metric = metric

        self.score_df = None

    def _get_method_objs(self):
        _method_switcher = {
            'hierarchical': AgglomerativeClustering(),
            'kmeans': KMeans(),
            'spectral': SpectralClustering()
        }
        objs = {i: _method_switcher[i] for i in self.methods}
        for key, value in objs.items():
            if isinstance(value, AgglomerativeClustering):
                affinity = 'euclidean' if self.linkage == 'ward' else 'precomputed'
                value.set_params(linkage=self.linkage, affinity=affinity)
        return objs

    def _get_index_funs(self):
        _index_switcher = {
            'silhouette': lambda X, labels: silhouette_score(
                X, labels, 'precomputed'
            ),
            'calinski': lambda X, labels: calinski_harabaz_score(X, labels),
            'davies': lambda X, labels: davies_bouldin_score(X, labels),
            'dunn': lambda X, labels: dunn(X, labels)
        }
        return {i: _index_switcher[i] for i in self.indices}

    def validate(self):
        method_objs = self._get_method_objs()
        index_funs = self._get_index_funs()
        dist_inds = ['silhouette', 'dunn']

        d_overlap = [i for i in self.indices if i in dist_inds]
        if d_overlap or 'hierarchical' in self.methods:
            dist = pairwise_distances(self.data)

        index = pd.MultiIndex.from_product(
            [self.methods, self.indices],
            names=['method', 'index']
        )
        output_df = pd.DataFrame(index=index, columns=self.n_clusters)

        for k in self.n_clusters:
            for alg_name, alg_obj in method_objs.items():
                alg_obj.set_params(n_clusters=k)
                if alg_name == 'hierarchical' and self.linkage != 'ward':
                    labels = alg_obj.fit_predict(dist)
                else:
                    labels = alg_obj.fit_predict(self.data)
                scores = [
                    fun(dist, labels) if key in dist_inds else fun(self.data, labels)
                    for key, fun in index_funs.items()
                ]
                output_df.loc[(alg_name, self.indices), k] = scores

        self.score_df = output_df

        return self
