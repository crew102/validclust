import re

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import normalize

from validclust.indices import (
    _dunn, cop, _davies_bouldin_score2, _silhouette_score2,
    _calinski_harabaz_score2
)


class ValidClust:
    """Validate clustering results

    Parameters
    ----------
    k : int or list of int
        The number of clusters to partition your data into.
    indices : str or list of str, optional
        The cluster validity indices to calculate. Acceptable values include
        'silhouette', 'calinski', 'davies', 'dunn', and 'cop'. You can use
        a three-character abbreviation for these values as well. For example,
        you could specify ``indices=['cal', 'dav', 'dun']``.
    methods : str or list of str, optional
        The clustering algorithm(s) to use. Acceptable values are
        'hierarchical' and 'kmeans'.
    linkage : {'ward', 'complete', 'average', 'single'}, optional
        Which linkage criterion to use for hierarchical clustering. See the
        `sklean docs <https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html#sklearn.cluster.AgglomerativeClustering>`_
        for more details.
    affinity : {'euclidean', 'l1', 'l2', 'manhattan', 'cosine'}, optional
        The metric used to compute the linkage for hierarchical clustering.
        Note, you must specify ``affinity='euclidean'`` when
        ``linkage='ward'``. See the sklearn docs linked above for more details.

    Attributes
    ----------
    score_df : DataFrame
        A Pandas DataFrame with the computed cluster validity index values.
    """
    def __init__(self, k,
                 # No big deal that these are lists (i.e., mutable), given that
                 # we don't mutate them inside the class.
                 indices=['silhouette', 'calinski', 'davies', 'dunn'],
                 methods=['hierarchical', 'kmeans'],
                 linkage='ward', affinity='euclidean'):

        k, indices, methods = (
            [i] if type(i) in [int, str] else i
            for i in [k, indices, methods]
        )

        if linkage == 'ward' and affinity != 'euclidean':
            raise ValueError(
                "You must specify `affinity='euclidean'` when using the "
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
                raise ValueError('{0} is not a valid index value'.format(i))

        self.k = k
        self.indices = indices
        self.methods = methods
        self.linkage = linkage
        self.affinity = affinity

        self.score_df = None

    def __repr__(self):
        argspec = [
            '{}={}'.format('  ' + key, value)
            for key, value in self.__dict__.items() if key != 'score_df'
        ]
        argspec = ',\n'.join(argspec)
        argspec = re.sub('(linkage|affinity)=(\\w*)', "\\1='\\2'", argspec)
        return 'ValidClust(\n' + argspec + '\n)'

    def _get_method_objs(self):
        method_switcher = {
            'hierarchical': AgglomerativeClustering(),
            'kmeans': KMeans()
        }
        objs = {i: method_switcher[i] for i in self.methods}
        for key, value in objs.items():
            if key == 'hierarchical':
                value.set_params(linkage=self.linkage, affinity=self.affinity)
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
        """Fit the clustering algorithm(s) to the data and calculate the CVI
        scores

        Parameters
        ----------
        data : array-like, shape = [n_samples, n_features]
            The data to cluster.

        Returns
        -------
        self
            A ``ValidClust`` object whose ``score_df`` attribute contains the
            calculated CVI scores.
        """
        method_objs = self._get_method_objs()
        index_funs = self._get_index_funs()
        dist_inds = ['silhouette', 'dunn']

        d_overlap = [i for i in self.indices if i in dist_inds]
        if d_overlap:
            dist = pairwise_distances(data)
            np.fill_diagonal(dist, 0)
        else:
            dist = None

        index = pd.MultiIndex.from_product(
            [self.methods, self.indices],
            names=['method', 'index']
        )
        output_df = pd.DataFrame(
            index=index, columns=self.k, dtype=np.float64
        )

        for k in self.k:
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
        """Fit the clustering algorithm(s) to the data and calculate the CVI
        scores

        Parameters
        ----------
        data : array-like, shape = [n_samples, n_features]
            The data to cluster.

        Returns
        -------
        DataFrame
            A Pandas DataFrame with the computed cluster validity index values
            (``self.score_df``).
        """
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
        """Plot normalized CVI scores in a heatmap

        The CVI scores are normalized along each method/index pair using the
        `max norm <https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.normalize.html>`_.
        Note that, because the scores are normalized along each method/index
        pair, you should compare the colors of the cells in the heatmap only
        within a given row. You should not, for instance, compare the color of
        the cells in the "kmeans, dunn" row with those in the
        "kmeans, silhouette" row.

        Returns
        -------
        None
            Nothing is returned. Instead, a plot is rendered using a
            graphics backend.
        """
        norm_df = self._normalize()

        yticklabels = [',\n'.join(i) for i in norm_df.index.values]
        hmap = sns.heatmap(
            norm_df, cmap='Blues', cbar=False, yticklabels=yticklabels
        )
        hmap.set_xlabel('\nNumber of clusters')
        hmap.set_ylabel('Method, index\n')
        plt.tight_layout()
