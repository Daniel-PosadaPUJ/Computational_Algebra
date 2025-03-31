'''
Exercise 1: Linear Regresión and Gradient Descent
1. Ajusta una regresión a los datos dados, seleccionando 18 puntos aleatorios.
2. Ajusta un modelo usando Gradiente Descendente.
3. Grafica los resultados.
4. Calcula el error RMSE para ambos modelos sobre los 4 puntos restantes.
5. ¿Cuál modelo es mejor? ¿Por qué?
6. Repetimos el proceso 5 veces y calcula el promedio de los errores RMSE.
'''

import numpy as np
import matplotlib.pyplot as plt

# Datos
data = np.array([
    [0.59, 3980], [0.80, 2200], [0.95, 1850], [0.45, 6100], [0.79, 2100],
    [0.99, 1700], [0.90, 2000], [0.65, 4200], [0.79, 2440], [0.69, 3300],
    [0.79, 2300], [0.49, 6000], [1.09, 1190], [0.95, 1960], [0.79, 2760],
    [0.65, 4330], [0.45, 6960], [0.60, 4160], [0.89, 1990], [0.79, 2860],
    [0.99, 1920], [0.85, 2160]
])

# Normalización de datos para que los valores estén en una escala similar
def normalize_data(data):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    return (data - mean) / std

data = normalize_data(data)

# Separar datos en precio (P) y ventas (S)
P = data[:, 0]  # Variable independiente (Precio)
S = data[:, 1]  # Variable dependiente (Ventas)

# Crear matriz de diseño para la ecuación cuadrática
X = np.column_stack((np.ones(len(P)), P, P**2))

# Método 1: Ecuación normal (solución cerrada)
def normal_equation(X, y):
    X_transpose = X.T
    return np.linalg.solve(X_transpose.dot(X), X_transpose.dot(y))

# Método 2: Descenso de gradiente (optimización iterativa)
def gradient_descent(X, y, learning_rate=0.01, max_iter=10000, tolerance=1e-6):
    m, n = X.shape
    i, stop = 0, False
    theta = np.zeros(n)  # Inicialización de parámetros en ceros
    while i < max_iter and not stop:
        E = abs(y - X.dot(theta))
        gradient = 2 * X.T.dot(E)
        new_theta = theta - learning_rate * gradient
        if np.linalg.norm(new_theta - theta) < tolerance:
            stop = True # Condición de convergencia
        theta = new_theta
    return theta

# Predicciones y RMSE (Error Cuadrático Medio)
def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

# Generar gráficos
plt.figure(figsize=(12, 10))
semillas = [42, 7, 15, 99, 123]  # Semillas fijas

for i in range(len(semillas)):
    np.random.seed(semillas[i])
    indices = np.random.permutation(len(P))
    train_idx, test_idx = indices[:18], indices[18:]

    # División de datos
    X_train, X_test = X[train_idx], X[test_idx]
    S_train, S_test = S[train_idx], S[test_idx]

    # Calcular Coeficientes
    theta_normal = normal_equation(X_train, S_train)
    theta_gd = gradient_descent(X_train, S_train)

    print(f"Semilla: {semillas[i]}, iteración: {i + 1}")
    print(f"Coeficientes (Ecuación Normal): c1 = {theta_normal[0]:.6f}, c2 = {theta_normal[1]:.6f}, c3 = {theta_normal[2]:.6f}")
    print(f"Coeficientes (Gradiente Descendente): c1 = {theta_gd[0]:.6f}, c2 = {theta_gd[1]:.6f}, c3 = {theta_gd[2]:.6f}")

    # Predicciones
    S_pred_normal = X_test.dot(theta_normal)
    S_pred_gd = X_test.dot(theta_gd)

    # Error RMSE
    error_normal = rmse(S_test, S_pred_normal)
    error_gd = rmse(S_test, S_pred_gd)

    print(f"RMSE (Ecuación Normal): {error_normal:.6f}")
    print(f"RMSE (Gradiente Descendente): {error_gd:.6f}")

    # Predicciones en todo el conjunto de datos
    S_pred_normal = X.dot(theta_normal)
    S_pred_gd = X.dot(theta_gd)

    # Ordenar todos los datos por el precio para una mejor visualización
    datos_ordenados = np.argsort(P)  # Índices ordenados para todos los datos
    P_sorted = P[datos_ordenados]    # Precio ordenado

    # Ordenar las predicciones correspondientes a todos los datos
    S_sorted_normal = S_pred_normal[datos_ordenados]
    S_sorted_gd = S_pred_gd[datos_ordenados]


    # Gráfico
    plt.subplot(3, 2, i + 1)
    plt.scatter(P, S, color='blue', label='Datos reales')
    plt.plot(P_sorted, S_sorted_normal, color='red', label='Ajuste (Normal Eq)')
    plt.plot(P_sorted, S_sorted_gd, color='green', label='Ajuste (Gradiente Desc.)')
    plt.title(f'Semilla: {semillas[i]}\nCoef: {theta_normal[0]:.2f}, {theta_normal[1]:.2f}, {theta_normal[2]:.2f}\nRMSE: {error_normal:.4f}')
    plt.xlabel('Precio ($)')
    plt.ylabel('Ventas')
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.show()

