# Implementación de K-means desde Cero en Python

## Descripción  
Este proyecto consiste en implementar el algoritmo **K-means** desde cero en Python, sin utilizar librerías especializadas de clustering, como parte de un taller para el curso con el profesor Abel Alvarez. Se trabaja con datos sintéticos en espacios de **2D y 3D**, permitiendo comparar los resultados del algoritmo con etiquetas verdaderas. Además, se analiza la influencia de diferentes métricas de distancia en la asignación de clusters.

---

## Objetivos  
1. **Cargar y analizar datos**: Utilizando archivos CSV con datos en 2D y 3D.  
2. **Estudio estadístico**: Obtener estadísticas descriptivas de las variables.  
3. **Visualización de datos**: Graficar los datos en 2D y 3D.  
4. **Implementación del algoritmo K-means**:
   - Inicialización aleatoria de centroides con una semilla fija.
   - Asignación de puntos al centroide más cercano.
   - Actualización de centroides hasta converger o alcanzar un número máximo de iteraciones.
5. **Evaluación de rendimiento**: Medir la inercia del algoritmo y experimentar con diferentes métricas de distancia, tales como:  
   - **Euclidiana**  
   - **Manhattan**  
   - **Chebyshev**  
   - **Mahalanobis**  
6. **Análisis de resultados**: Comparar la calidad del clustering obtenido con las etiquetas verdaderas y analizar el impacto de cambiar las semillas de los centroides.

---

## Punto 2: Análisis de Clustering con Datos Reales  
En la segunda parte del proyecto, se realiza un análisis de clustering utilizando el algoritmo **K-means** en un conjunto de datos reales (`Mall_Customers.csv`), que incluye información de clientes de un centro comercial. Se utilizan las variables:  
- **Edad**  
- **Ingresos Anuales (k$)**  
- **Puntuación de Gastos (1-100)**  

### Pasos a seguir:  
1. **Cargar y preprocesar los datos**: Se ignoran las variables categóricas (`CustomerID` y `Género`) y se normalizan las variables numéricas.  
2. **Aplicar K-means**: Se aplica el algoritmo para diferentes números de clusters (de 2 a 10).  
3. **Selección del número óptimo de clusters**: Utilizando la **regla del codo** al graficar la inercia en función del número de clusters.  
4. **Análisis de resultados**: Concluir sobre la calidad del clustering obtenido en base a los datos reales.

---

## Requisitos  
- **Python 3.x**  
- **Librerías**: 
  - `pandas` para el manejo de datos.  
  - `numpy` para cálculos numéricos.  
  - `matplotlib` para visualización de datos en 2D y 3D.

---

## Instalación  
Clona este repositorio e instala las dependencias:  
```bash
git clone https://github.com/Daniel-PosadaPUJ/Proyecto1_ALC.git
cd Proyecto1_ALC
pip install -r requirements.txt
