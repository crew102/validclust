import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans, SpectralClustering
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import normalize

from .indices import (
    dunn, davies_bouldin_score2, silhouette_score2, calinski_harabaz_score2,
    cop
)


class ValidClust:

    def __init__(self, data, n_clusters=[2, 3, 4, 5],
                 indices=['silhouette', 'calinski', 'davies', 'dunn'],
                 methods=['hierarchical', 'kmeans'],
                 linkage='ward', metric='euclidean'):

        for i in ['n_clusters', 'indices', 'methods']:
            if type(locals()[i]) is not list:
                raise ValueError('{0} must be a list'.format(i))

        if 'hierarchical' in methods and linkage == 'ward' and metric != 'euclidean':
            raise ValueError(
                "You must specify `metric='euclidean'` if you use choose the "
                "ward linkage type"
            )

        ok_indices = ['silhouette', 'calinski', 'davies', 'dunn', 'cop']
        ind_aliases = {i[0:3]: i for i in ok_indices}
        indices = [
            ind_aliases[i] if i in ind_aliases else i
            for i in indices
        ]
        for i in indices:
            if i not in ok_indices:
                raise ValueError('{0} is not a valid index metric'.format(i))

        self.data = data
        self.n_clusters = n_clusters
        self.indices = indices
        self.methods = methods
        self.linkage = linkage
        self.metric = metric

        self.score_df = None

    def _get_method_objs(self):
        method_switcher = {
            'hierarchical': AgglomerativeClustering(),
            'kmeans': KMeans(),
            'spectral': SpectralClustering()
        }
        objs = {i: method_switcher[i] for i in self.methods}
        for key, value in objs.items():
            if key == 'hierarchical':
                affinity = 'euclidean' if self.linkage == 'ward' else 'precomputed'
                value.set_params(linkage=self.linkage, affinity=affinity)
        return objs

    def _get_index_funs(self):
        index_fun_switcher = {
            'silhouette': silhouette_score2,
            'calinski': calinski_harabaz_score2,
            'davies': davies_bouldin_score2,
            'dunn': dunn,
            'cop': cop
        }
        return {i: index_fun_switcher[i] for i in self.indices}

    def validate(self):
        method_objs = self._get_method_objs()
        index_funs = self._get_index_funs()
        dist_inds = ['silhouette', 'dunn']

        d_overlap = [i for i in self.indices if i in dist_inds]
        if d_overlap or 'hierarchical' in self.methods:
            dist = pairwise_distances(self.data)
        else:
            dist = None

        index = pd.MultiIndex.from_product(
            [self.methods, self.indices],
            names=['method', 'index']
        )
        output_df = pd.DataFrame(
            index=index, columns=self.n_clusters, dtype=np.float64
        )

        for k in self.n_clusters:
            for alg_name, alg_obj in method_objs.items():
                alg_obj.set_params(n_clusters=k)
                if alg_name == 'hierarchical' and self.linkage != 'ward':
                    labels = alg_obj.fit_predict(dist)
                else:
                    labels = alg_obj.fit_predict(self.data)
                # have to iterate over self.indices here so that ordering of
                # validity indices is same in scores list as it is in output_df
                scores = [
                    index_funs[key](self.data, dist, labels)
                    for key in self.indices
                ]
                output_df.loc[(alg_name, self.indices), k] = scores

        self.score_df = output_df

        return self

    def _normalize(self):
        score_df_norm = self.score_df.copy()
        normalize(score_df_norm, norm='max', copy=False)
        if 'davies' in self.indices:
            score_df_norm.loc[(slice(None), 'davies'), :] = \
                1 - score_df_norm.loc[(slice(None), 'davies'), :]
        return score_df_norm
