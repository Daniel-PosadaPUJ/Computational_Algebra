import numpy as np

def euclidean_norm(x, y):
    return np.linalg.norm(x - y)

def manhattan_norm(x, y):
    return np.sum(np.abs(x - y))

def chebyshev_norm(x, y):
    return np.max(np.abs(x - y))

def minkowski_norm(x, y, p=2):
    if p <= 0:
        raise ValueError("El parámetro p debe ser mayor que 0.")
    return np.sum(np.abs(x - y) ** p) ** (1 / p)

def mahalanobis_norm(x, y, VI):
    x, y = np.array(x), np.array(y)
    if x.shape != y.shape:
        raise ValueError("x e y deben tener las mismas dimensiones.")
    if VI.shape[0] != VI.shape[1] or VI.shape[0] != x.shape[0]:
        raise ValueError("VI debe ser una matriz cuadrada con dimensiones consistentes con x e y.")
    delta = x - y
    return np.sqrt(np.dot(np.dot(delta.T, VI), delta))