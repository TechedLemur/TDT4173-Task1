import numpy as np
import pandas as pd
# IMPORTANT: DO NOT USE ANY OTHER 3RD PARTY PACKAGES
# (math, random, collections, functools, etc. are perfectly fine)


class KMeans:

    def __init__(self, k=2, iterations=50):
        # NOTE: Feel free add any hyperparameters
        # (with defaults) as you see fit
        self.k = k
        self.iterations = iterations

    def fit(self, X):
        """
        Estimates parameters for the classifier

        Args:
            X (array<m,n>): a matrix of floats with
                m rows (#samples) and n columns (#features)
        """
        print("Training model, this might take some time...")

        models = []
        scores = []

        for l in range(1, 11):
            indices = np.arange(len(X))
            # Pick k starting centroids with the maximin principle. Max the minimum distance between inital centroids.
            centroids = [X.to_numpy()[np.random.randint(0, len(X))]]

            for _ in range(self.k - 1):
                distances = np.zeros(len(X))
                for i, x in X.iterrows():
                    dist = np.inf
                    # Calculate minimum distance between point x and each centroid
                    for c in centroids:
                        distance = euclidean_distance(x, c)
                        dist = min(dist, distance)
                    distances[i] = dist
                # Add the furtest point as a new centroid
                centroids.append(X.to_numpy()[np.argmax(distances)])
            centroids = np.array(centroids)
            r = np.zeros((X.shape[0], self.k))

            for _ in range(self.iterations):
                for i, x in X.iterrows():  # Assign datapoints to clusters
                    argmin = 0
                    dist = np.inf
                    for k in range(self.k):
                        distance = euclidean_distance(x, centroids[k])
                        if distance < dist:
                            argmin = k
                            dist = distance
                    r[i] = 0
                    r[i][argmin] = 1

                denominator = np.sum(r, axis=0)
                for k in range(self.k):  # Update centroids

                    rx = np.tile(r[:, k], (len(X.columns), 1)).T * X
                    sum_rx = np.sum(rx, axis=0)
                    centroids[k] = sum_rx / denominator[k]

            models.append(centroids)
            scores.append(euclidean_distortion(
                X, self.predict(X, centroids=centroids)))
            print(f"{l} iterations done, {10-l} remaining")

        print("Training done, selecting best centroids")
        self.centroids = models[np.argmin(np.array(scores))]

    def predict(self, X, centroids=None):
        """
        Generates predictions

        Note: should be called after .fit()

        Args:
            X (array<m,n>): a matrix of floats with 
                m rows (#samples) and n columns (#features)

        Returns:
            A length m integer array with cluster assignments
            for each point. E.g., if X is a 10xn matrix and 
            there are 3 clusters, then a possible assignment
            could be: array([2, 0, 0, 1, 2, 1, 1, 0, 2, 2])
        """
        if centroids is not None:
            c = centroids
        else:
            c = self.centroids
        result = []
        for i, x in X.iterrows():  # Assign samples to neareast cluster
            argmin = 0
            dist = np.inf
            for k in range(self.k):
                distance = euclidean_distance(x, c[k])
                if distance < dist:
                    argmin = k
                    dist = distance
            result.append(argmin)
        return np.array(result)

    def get_centroids(self):
        """
        Returns the centroids found by the K-mean algorithm

        Example with m centroids in an n-dimensional space:
        >>> model.get_centroids()
        numpy.array([
            [x1_1, x1_2, ..., x1_n],
            [x2_1, x2_2, ..., x2_n],
                    .
                    .
                    .
            [xm_1, xm_2, ..., xm_n]
        ])
        """
        return self.centroids


# --- Some utility functions


def euclidean_distortion(X, z):
    """
    Computes the Euclidean K-means distortion

    Args:
        X (array<m,n>): m x n float matrix with datapoints 
        z (array<m>): m-length integer vector of cluster assignments

    Returns:
        A scalar float with the raw distortion measure 
    """
    X, z = np.asarray(X), np.asarray(z)
    assert len(X.shape) == 2
    assert len(z.shape) == 1
    assert X.shape[0] == z.shape[0]

    distortion = 0.0
    for c in np.unique(z):
        Xc = X[z == c]
        mu = Xc.mean(axis=0)
        distortion += ((Xc - mu) ** 2).sum()

    return distortion


def euclidean_distance(x, y):
    """
    Computes euclidean distance between two sets of points 

    Note: by passing "y=0.0", it will compute the euclidean norm

    Args:
        x, y (array<...,n>): float tensors with pairs of 
            n-dimensional points 

    Returns:
        A float array of shape <...> with the pairwise distances
        of each x and y point
    """
    return np.linalg.norm(x - y, ord=2, axis=-1)


def cross_euclidean_distance(x, y=None):
    """
    Compute Euclidean distance between two sets of points 

    Args:
        x (array<m,d>): float tensor with pairs of 
            n-dimensional points. 
        y (array<n,d>): float tensor with pairs of 
            n-dimensional points. Uses y=x if y is not given.

    Returns:
        A float array of shape <m,n> with the euclidean distances
        from all the points in x to all the points in y
    """
    y = x if y is None else y
    assert len(x.shape) >= 2
    assert len(y.shape) >= 2
    return euclidean_distance(x[..., :, None, :], y[..., None, :, :])


def euclidean_silhouette(X, z):
    """
    Computes the average Silhouette Coefficient with euclidean distance 

    More info:
        - https://www.sciencedirect.com/science/article/pii/0377042787901257
        - https://en.wikipedia.org/wiki/Silhouette_(clustering)

    Args:
        X (array<m,n>): m x n float matrix with datapoints 
        z (array<m>): m-length integer vector of cluster assignments

    Returns:
        A scalar float with the silhouette score
    """
    X, z = np.asarray(X), np.asarray(z)
    assert len(X.shape) == 2
    assert len(z.shape) == 1
    assert X.shape[0] == z.shape[0]

    # Compute average distances from each x to all other clusters
    clusters = np.unique(z)
    D = np.zeros((len(X), len(clusters)))
    for i, ca in enumerate(clusters):
        for j, cb in enumerate(clusters):
            in_cluster_a = z == ca
            in_cluster_b = z == cb
            d = cross_euclidean_distance(X[in_cluster_a], X[in_cluster_b])
            div = d.shape[1] - int(i == j)
            D[in_cluster_a, j] = d.sum(axis=1) / np.clip(div, 1, None)

    # Intra distance
    a = D[np.arange(len(X)), z]
    # Smallest inter distance
    inf_mask = np.where(z[:, None] == clusters[None], np.inf, 0)
    b = (D + inf_mask).min(axis=1)

    return np.mean((b - a) / np.maximum(a, b))
