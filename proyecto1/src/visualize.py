from matplotlib import pyplot as plt
from src.kmeans import KMeans
import numpy as np
import pandas as pd

def plot_elbow_method(X, max_k=10, distance_func=None):
    inercia = []
    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k, distance_func=distance_func)
        kmeans.fit(X)
        inercia.append(kmeans.inertia(X))

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, max_k + 1), inercia, marker='o')
    plt.xlabel("Número de Clusters (K)")
    plt.ylabel("Inercia")
    plt.title("Método del Codo")
    plt.show()


def plot_kmeans_clusters(X, kmeans):
    # Convertir a NumPy array si es un DataFrame de pandas
    if isinstance(X, pd.DataFrame):
        X = X.values

    if X.shape[1] == 2:
        plt.figure(figsize=(10, 6))
        plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels, cmap='viridis')
        plt.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], c='red', marker='x', s=100)
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.title("K-Means Clustering (2D)")
        plt.show()
    elif X.shape[1] == 3:
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=kmeans.labels, cmap='viridis')
        ax.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], kmeans.centroids[:, 2], c='red', marker='x', s=100)
        ax.set_xlabel("Feature 1")
        ax.set_ylabel("Feature 2")
        ax.set_zlabel("Feature 3")
        ax.set_title("K-Means Clustering (3D)")
        plt.show()
    else:
        raise ValueError("La función solo soporta datos en 2D o 3D")
