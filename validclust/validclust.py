import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans, SpectralClustering
from sklearn.metrics import (
    silhouette_score, calinski_harabaz_score, davies_bouldin_score
)


def _get_method_objs(methods, linkage, metric):
    _method_switcher = {
        'hierarchical': AgglomerativeClustering(),
        'kmeans': KMeans(),
        'spectral': SpectralClustering()
    }
    objs = {i: _method_switcher[i] for i in methods}
    for key, value in objs.items():
        if isinstance(value, AgglomerativeClustering):
            value.set_params(linkage=linkage, affinity=metric)
    return objs


def _get_index_funs(indices, metric):
    _index_switcher = {
        'silhouette': lambda X, labels: silhouette_score(X, labels, metric),
        'calinski': lambda X, labels: calinski_harabaz_score(X, labels),
        'davies': lambda X, labels: davies_bouldin_score(X, labels)
    }
    return {i: _index_switcher[i] for i in indices}


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

    def validate(self):
        method_objs = _get_method_objs(self.methods, self.linkage, self.metric)
        index_funs = _get_index_funs(self.indices, self.metric)

        index = pd.MultiIndex.from_product(
            [self.methods, self.indices],
            names=['method', 'index']
        )
        output_df = pd.DataFrame(index=index, columns=self.n_clusters)

        for k in self.n_clusters:
            for alg_name, alg_obj in method_objs.items():
                alg_obj.set_params(n_clusters=k)
                labels = alg_obj.fit_predict(self.data)
                scores = [value(self.data, labels) for key, value in
                          index_funs.items()]
                output_df.loc[(alg_name, self.indices), k] = scores

        return output_df
