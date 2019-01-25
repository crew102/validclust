import re

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.cluster import AgglomerativeClustering, KMeans, SpectralClustering
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import normalize

from .indices import (
    _dunn, cop, _davies_bouldin_score2, _silhouette_score2,
    _calinski_harabaz_score2
)


class ValidClust:

    def __init__(self, n_clusters,
                 indices=['silhouette', 'calinski', 'davies', 'dunn'],
                 methods=['hierarchical', 'kmeans'],
                 linkage='ward', metric='euclidean'):

        for i in ['n_clusters', 'indices', 'methods']:
            if type(locals()[i]) is not list:
                raise ValueError('{0} must be a list'.format(i))

        if linkage == 'ward' and metric != 'euclidean':
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

        self.n_clusters = n_clusters
        self.indices = indices
        self.methods = methods
        self.linkage = linkage
        self.metric = metric

        self.score_df = None

    def __repr__(self):
        argspec = [
            '{}={}'.format('  ' + key, value)
            for key, value in self.__dict__.items() if key != 'score_df'
        ]
        argspec = ',\n'.join(argspec)
        argspec = re.sub('(linkage|metric)=(\w*)', "\\1='\\2'", argspec)
        return 'ValidClust(\n' + argspec + '\n)'

    def _get_method_objs(self):
        method_switcher = {
            'hierarchical': AgglomerativeClustering(),
            'kmeans': KMeans()
        }
        objs = {i: method_switcher[i] for i in self.methods}
        for key, value in objs.items():
            if key == 'hierarchical':
                value.set_params(linkage=self.linkage, affinity=self.metric)
        return objs

    def _get_index_funs(self):
        index_fun_switcher = {
            'silhouette': _silhouette_score2,
            'calinski': _calinski_harabaz_score2,
            'davies': _davies_bouldin_score2,
            'dunn': _dunn,
            'cop': cop
        }
        return {i: index_fun_switcher[i] for i in self.indices}

    def fit(self, data):
        method_objs = self._get_method_objs()
        index_funs = self._get_index_funs()
        dist_inds = ['silhouette', 'dunn']

        d_overlap = [i for i in self.indices if i in dist_inds]
        dist = pairwise_distances(data) if d_overlap else None

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
                labels = alg_obj.fit_predict(data)
                # have to iterate over self.indices here so that ordering of
                # validity indices is same in scores list as it is in output_df
                scores = [
                    index_funs[key](data, dist, labels)
                    for key in self.indices
                ]
                output_df.loc[(alg_name, self.indices), k] = scores

        self.score_df = output_df
        return self

    def fit_predict(self, data):
        return self.fit(data).score_df

    def _normalize(self):
        score_df_norm = self.score_df.copy()
        for i in ['davies', 'cop']:
            if i in self.indices:
                score_df_norm.loc[(slice(None), i), :] = \
                    1 - score_df_norm.loc[(slice(None), i), :]
        normalize(score_df_norm, norm='max', copy=False)
        return score_df_norm

    def plot(self):
        norm_df = self._normalize()

        yticklabels = [',\n'.join(i) for i in norm_df.index.values]
        hmap = sns.heatmap(
            norm_df, cmap='Blues', cbar=False, yticklabels=yticklabels
        )
        hmap.set_xlabel('\nNumber of clusters')
        hmap.set_ylabel('Method, index\n')
        plt.tight_layout()
