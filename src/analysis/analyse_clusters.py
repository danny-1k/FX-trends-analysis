import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


class ClusterAnalysis:
    def __init__(self):
        pass

    def _run_pca(self):
        pass

    def _standardise_features(self):
        pass


class KmeansRunner:
    def __init__(self, X, k_range=(4, 20)):
        self.X = X
        self.k_range = k_range

        self.inertias = []
        self.silhouettes = []
        self.ks = range(k_range[0], k_range[-1] + 1)

    def fit(self):
        for k in tqdm(self.ks):
            inertia, silhouette = self._fit_once(k=k)

            self.inertias.append(inertia)
            self.silhouettes.append(silhouette)

        return self.ks

    def _fit_once(self, k):
        km = KMeans(n_clusters=k)
        km.fit(self.X)
        cluster_labels = km.labels_
        silhouette = silhouette_score(self.X, cluster_labels)

        return km.inertia_, silhouette

    def show(self):
        plt.plot(self.ks, self.inertias, label="Inertias")
        plt.legend()
        plt.show()

        plt.plot(self.ks, self.silhouettes, label="Silhouettes")
        plt.legend()
        plt.show()