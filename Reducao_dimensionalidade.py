from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import KernelPCA
from sklearn.manifold import Isomap
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.manifold import SpectralEmbedding
from sklearn.manifold import TSNE

from Tratamento_Dados import Tratamento_Dados


class Tecnicas_reducao(Tratamento_Dados):
    def __init__(self,data,target,num_components):
        self.data = data
        self.target = target
        self.num_components = num_components
        super().__init__(data, target)
        self.X, self.y = self.tratamento_dados
    @property
    def PCA(self):
        pca = PCA(n_components=self.num_components)
        PCA_values = pca.fit_transform(self.X)
        return PCA_values

    @property
    def LDA(self):
        clf = LinearDiscriminantAnalysis(n_components = self.num_components)
        clf.fit(X, y)
        Linear_Discriminant_Analysis = clf.fit_transform(self.X, self.y)
        return Linear_Discriminant_Analysis

    @property
    def ISOMAP(self):
        clf = Isomap(n_components=self.num_components)
        Isomap = clf.fit_transform(self.X)
        return Isomap

    @property
    def KernelPCA(self):
        X, y = self.tratamento_dados()
        transformer = KernelPCA(n_components=self.num_components, kernel='cosine')
        Kernel_PCA = transformer.fit_transform(self.X)
        return Kernel_PCA

    @property
    def locally_linear_embedding(self):
        X, y = self.tratamento_dados()
        clf = LocallyLinearEmbedding(n_components=self.num_components)
        Locally_Linear_Embedding = clf.fit_transform(self.X)
        return Locally_Linear_Embedding

    @property
    def spectral_smbedding(self):
        X, y = self.tratamento_dados()
        clf = SpectralEmbedding(n_components=self.num_components)
        Spectral_Embedding = clf.fit_transform(self.X)
        return Spectral_Embedding

    @property
    def tsne(self):
        X, y = self.tratamento_dados()
        tsne = TSNE(n_components=self.num_components, learning_rate='auto',init='random').fit_transform(self.X)
        return tsne