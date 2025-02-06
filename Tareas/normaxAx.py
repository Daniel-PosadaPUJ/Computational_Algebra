'''
Dibujo de una "circunferencia" de radio 1, bajo la norma X
'''
import numpy as np
import matplotlib.pyplot as plt

def norm(A, X):
    return A[0, 0] * X[0]**2 + 2 * A[0, 1] * X[0] * X[1] + A[1, 1] * X[1]**2 - 1

def plot_eigenvectors(A):
    # Calcular los valores propios y vectores propios de A
    eigenvalues, eigenvectors = np.linalg.eig(A)

    # Colores para los vectores propios
    colors = ['red', 'green', 'blue', 'orange', 'purple']

    # Dibujar los vectores propios en la gráfica
    for i in range(len(eigenvalues)):
        # Cada eigenvector se multiplica por un factor de escala para hacerlo visible
        plt.quiver(0, 0, eigenvectors[0, i], eigenvectors[1, i], angles='xy', scale_units='xy', scale=1, 
                   color=colors[i % len(colors)], label=f'Vector propio {i+1}: $\lambda={eigenvalues[i]:.2f}$')

def plotCircunference(norm, A, radius):
    # Generamos una malla de puntos en el espacio
    x_vals = np.linspace(-2, 2, 1000)
    y_vals = np.linspace(-2, 2, 1000)
    X, Y = np.meshgrid(x_vals, y_vals)

    # Evaluamos la función f(x, y) = A_00 * X^2 + 2 * A_01 * X * Y + A_11 * Y^2 - 1
    Z = norm(A, [X, Y])

    # Graficamos el contorno donde f(x, y) = 0
    plt.contour(X, Y, Z, levels=[0], colors='blue')  # Dibujamos la curva donde Z = 0

    # Dibujar los vectores propios de A
    plot_eigenvectors(A)

    # Agregar líneas de referencia en el eje X e Y
    plt.axhline(0, color='gray', linewidth=0.5)
    plt.axvline(0, color='gray', linewidth=0.5)
    plt.title(r'Curva de la norma inducida, $f(x, y) = 0$')

    # Mostrar leyenda
    plt.legend(loc='best')
    plt.show()

def is_symmetric_and_positive_definite(A):
    # Verificar si la matriz es simétrica y si todos los eigenvalores son positivos
    return np.allclose(A, A.T) and np.all(np.linalg.eigvals(A) > 0)

def main():
    # Permitir al usuario ingresar la matriz A de una manera más legible
    print("Ingrese los valores de la matriz A (2x2):")
    a00 = float(input("A[0,0]: "))
    a01 = float(input("A[0,1]: "))
    a10 = float(input("A[1,0]: "))
    a11 = float(input("A[1,1]: "))
    A = np.array([[a00, a01], [a10, a11]])
    
    if is_symmetric_and_positive_definite(A):
        plotCircunference(norm, A, 1)
    else:
        print("La matriz A no es simétrica y definida positiva.")

if __name__ == "__main__":
    main()
