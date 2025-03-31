import numpy as np

class KMeans:
    def __init__(self, n_clusters=3, max_iter=300, tol=1e-4, distance_func=None, random_state=None):
        self.labels = None
        self.centroids = None
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.distance_func = distance_func if distance_func else self._default_distance

    def fit(self, X):
        X = X.to_numpy(dtype=np.float64)
        i = 0
        change = np.inf
        self.centroids = self._initialize_centroids(X)
        while i < self.max_iter and change > self.tol:
            self.labels = self._assign_clusters(X)
            new_centroids = self._compute_centroids(X)
            change = np.sum((new_centroids - self.centroids)**2)
            self.centroids = new_centroids
            i += 1

    def _initialize_centroids(self, X):
        np.random.seed(self.random_state)
        centroids = [X[np.random.randint(X.shape[0])]]
        for _ in range(1, self.n_clusters):
            distances = np.min(np.linalg.norm(X[:, np.newaxis] - centroids, axis=2), axis=1)
            probabilities = distances / np.sum(distances)
            centroid = X[np.random.choice(X.shape[0], p=probabilities)]
            centroids.append(centroid)
        return np.array(centroids)

    def _assign_clusters(self, X):
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)

    def _compute_centroids(self, X):
        centroids = []
        for i in range(self.n_clusters):
            cluster_points = X[self.labels == i]
            if len(cluster_points) > 0:
                centroids.append(cluster_points.mean(axis=0))
            else:
                # Si el cluster está vacío, reposiciona el centroide al azar
                centroids.append(X[np.random.randint(X.shape[0])])
        return np.array(centroids)

    def predict(self, X):
        X = X.to_numpy(dtype=np.float64)
        return self._assign_clusters(X)

    def inertia(self, X):
        distancias = np.array([self.distance_func(X, centroid) for centroid in self.centroids])
        return np.sum(np.min(distancias, axis=0))

    def _default_distance(self, x, y):
        return np.linalg.norm(x - y)

