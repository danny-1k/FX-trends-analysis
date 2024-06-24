import numpy as np

def euclidean_distance(x1, x2):
    return (((x1 - x2)**2)**.5).mean(axis=-1)


class KNN:
    def __init__(self, k):
        self.k = k

    def fit(self, x):
        self.X = x
        # Choose
        pass

    def _one_iter(self, x):
        random_k_clusters_indices = np.random.choice(
            np.arange(self.X.shape[0]), 
            size=(self.k), 
            replace=False
        )

        random_k_clusters_centriods = self.X[random_k_clusters_indices]
        distance_to_centroids = [euclidean_distance(self.X, random_k_clusters_centriods[i][np.newaxis])[np.newaxis] for i in range(self.k)] # distance between (N, E) and (k, E)
        distance_to_centroids = np.concatenate(distance_to_centroids, axis=0) # (k, N)
        closest_centroids = np.argmax(distance_to_centroids, axis=0)

        # take average of current centroids and continue
        new_clusters = [self.X[np.arange(self.X.shape[0])[closest_centroids == i]] for i in range(self.k)]

        new_clusters_mean = [cluster.mean(axis=0) for cluster in new_clusters] 
        new_clusters_std = [cluster.std() for cluster in new_clusters] # calculate density

        print(new_clusters_std)

        print(new_clusters[0].shape)

        print(closest_centroids)
        pass

    def predict(self, x):
        pass



if __name__ == "__main__":
    from sklearn.datasets import make_blobs

    X, _ = make_blobs(n_samples=10000, centers=3, n_features=2, random_state=0)

    # X = np.random.randn(10000, 512)
    
    knn = KNN(k=3)
    

    knn.fit(X)

    knn._one_iter(X)