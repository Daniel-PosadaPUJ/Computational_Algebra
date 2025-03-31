from src.kmeans import KMeans
import numpy as np

def elbow_method(X, max_k=10, distance_func=None):
    inercia = []
    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k, distance_func=distance_func)
        kmeans.fit(X)
        inercia.append(kmeans.inertia(X))

    inercia = np.array(inercia)
    k_optimo = np.argmin(np.diff(inercia)) + 1
    inercia_minima = inercia[k_optimo]

    return k_optimo, inercia_minima



