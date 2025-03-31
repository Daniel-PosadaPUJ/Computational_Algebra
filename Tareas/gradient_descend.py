import numpy as np
import matplotlib.pyplot as plt

# Normalizar datos
def normalize_data(data):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    return (data - mean) / std

# Método 1: Ecuación normal
def normal_equation(X, y):
    X_transpose = X.T
    return np.linalg.inv(X_transpose.dot(X)).dot(X_transpose).dot(y)

# Método 2: Descenso de gradiente
def gradient_descent(X, y, learning_rate=0.01, max_iter=10000, tolerance=1e-6):
    m, n = X.shape
    theta = np.zeros(n)
    for _ in range(max_iter):
        gradient = -2/m * X.T.dot(y - X.dot(theta))
        new_theta = theta - learning_rate * gradient
        if np.linalg.norm(new_theta - theta) < tolerance:
            break
        theta = new_theta
    return theta

