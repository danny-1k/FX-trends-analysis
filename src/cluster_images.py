import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from data import TrendDataset, MLPTrendEmbeddingDataset
from utils import create_image_from_time_series
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from tqdm import tqdm

import matplotlib.pyplot as plt


class Converter:
    def __init__(self, height, width):
        self.height = height
        self.width = width
    def __call__(self, chunk):
        image = create_image_from_time_series(window=chunk, height=self.height, width=self.width)

        return image


converter = Converter(height=64, width=128)
train_csv = pd.read_csv("../data/processed/splits/train.csv")
train_dataset = MLPTrendEmbeddingDataset(df=train_csv, converter=converter)#TrendDataset(df=train_csv, converter=converter)
train_dataset = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False)


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




X = None
for x in train_dataset:
    X = x


runner = KmeansRunner(X=X, k_range=(4, 30))

runner.fit()
runner.show()