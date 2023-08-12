from yellowbrick.cluster import KElbowVisualizer

from sklearn.cluster import KMeans
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import MeanShift
from sklearn.cluster import SpectralClustering
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.cluster import HDBSCAN
from sklearn.cluster import OPTICS
from sklearn.cluster import Birch
from sklearn.cluster import BisectingKMeans
from sklearn.mixture import GaussianMixture
from Reducao_dimensionalidade import Tecnicas_reducao
class Clustering(Tecnicas_reducao):
    def __init__(self,data, target, num_components,range_test):
        super().__init__(data, target, num_components)
        self.data =data
        self.target = target
        self.num_components = num_components
        self.range_test = range_test

    def elbow_value(self,modelo):
        visualizer = KElbowVisualizer(modelo, k=(2,self.range_test))
        visualizer.fit(self.X)
        return visualizer.elbow_value_

    def k_means(self,tecnica_reducao):
        kmeans = KMeans(self.elbow_value(KMeans()), random_state=0)
        k_means = kmeans.fit(tecnica_reducao).predict(tecnica_reducao)
        return k_means
    def gaussian_mixture(self,tecnica_reducao):
        gmm = GaussianMixture(n_components=self.elbow_value(GaussianMixture()), random_state=0).fit(tecnica_reducao)
        gmm_predict = gmm.predict(tecnica_reducao)
        return gmm_predict
    def affinity_propagation(self,tecnica_reducao):
        clustering = AffinityPropagation(random_state=5).fit(tecnica_reducao)
        AffinityPropagation(random_state=5)
        return clustering.labels_

    def mean_shift(self,tecnica_reducao):
        clustering = MeanShift(bandwidth=2).fit(tecnica_reducao)
        return clustering.labels_

    def spectral_clustering(self,tecnica_reducao):
        clustering = SpectralClustering(n_clusters=self.elbow_value(SpectralClustering()),assign_labels='discretize',random_state=0).fit(tecnica_reducao)
        return clustering.labels_

    def agglomerative_clustering(self,tecnica_reducao):
        clustering = AgglomerativeClustering().fit(tecnica_reducao)
        return clustering.labels_

    def dbscan(self,tecnica_reducao):
        clustering = DBSCAN(eps=3, min_samples=2).fit(tecnica_reducao)
        return clustering.labels_

    def HDBSCAN(self,tecnica_reducao):
        hdb = HDBSCAN(min_cluster_size=20)
        hdb.fit(tecnica_reducao)
        return hdb.labels_
    def optics(self,tecnica_reducao):
        clustering = OPTICS(min_samples=2).fit(tecnica_reducao)
        return clustering.labels_
    def birch(self,tecnica_reducao):
        brc = Birch(n_clusters=None)
        brc.fit(tecnica_reducao)
        return brc.predict(tecnica_reducao)

    def bisect_means(self,tecnica_reducao):
        bisect_means = BisectingKMeans(n_clusters=3, random_state=0).fit(tecnica_reducao)
        return bisect_means.predict(tecnica_reducao)