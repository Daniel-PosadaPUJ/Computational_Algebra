'''
Ejercicio 1 Producto Punto 
Daniel Alejandro Posada Noguera
'''

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

def integrand(x, f, g):
    return f(x) * g(x)

def inner_product(f, g):
    x = sp.Symbol('x')
    integrand_expr = integrand(x, f, g)
    result = sp.integrate(integrand_expr, (x, -sp.pi, sp.pi))
    return result

def are_all_ortogonals(set_vectors):
    ans = True
    i = 0
    while i < len(set_vectors) and ans:
        j = i + 1
        while j < len(set_vectors) and ans:
            if inner_product(set_vectors[i], set_vectors[j]) != 0:
                ans = False
            j += 1
        i += 1
    return ans  

def rewrite_as_linear_combination(vector, set_vectors):
    coefficients = []
    for w_k in set_vectors:
        numerator = inner_product(vector, w_k)
        denominator = inner_product(w_k, w_k)
        coefficient = numerator / denominator
        coefficients.append(coefficient)
    return coefficients

def plot_funct_and_linear_combination(vec, set_vectors, coefficients):
    x_vals = np.linspace(-np.pi, np.pi, 400)
    y_original = np.array([float(vec(x_val).evalf()) for x_val in x_vals])

    y_combination = np.zeros_like(x_vals, dtype=float)
    for i in range(len(set_vectors)):
        y_component = np.array([float(set_vectors[i](x_val).evalf()) for x_val in x_vals])
        y_combination += float(coefficients[i]) * y_component

    plt.figure(figsize=(8, 5))
    plt.plot(x_vals, y_original, label='Original function $x^3$', color='black', linestyle='dashed')
    plt.plot(x_vals, y_combination, label='Linear combination', color='red', alpha=0.7)
    
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    plt.title('Original function and its Linear Combination Approximation')
    plt.show()


def main():
    x = sp.Symbol('x')
    f1 = sp.Lambda(x, 1)
    f2 = sp.Lambda(x, sp.cos(x))
    f3 = sp.Lambda(x, sp.sin(x))
    f4 = sp.Lambda(x, sp.cos(2*x))
    f5 = sp.Lambda(x, sp.sin(2*x))
    set_vectors = [f1, f2, f3, f4, f5]
    
    if are_all_ortogonals(set_vectors):
        print("All vectors are orthogonal to each other.")
        vec = sp.Lambda(x, x**3)
        coefficients = rewrite_as_linear_combination(vec, set_vectors)
        print(f"The coefficients to rewrite x^3 are: {coefficients}")

        plot_funct_and_linear_combination(vec, set_vectors, coefficients)
        
    else:
        print("Not all vectors are orthogonal to each other.")

if __name__ == "__main__":
    main()