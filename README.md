### Autores: Valentina Barbetty Arango | Lenin Esteban Carabalí Moreno | Juan José Cortés Rodríguez
#### Fecha: Febrero-Julio 2025

# Descripción
En este repositorio se aloja el codigo desarrollado para apoyar el proyecto de la materia *SIMULACIÓN Y COMPUTACIÓN NUMERICA* relacionado con la realizacion de la simulación numérica de flujo incompresible utilizando las ecuaciones de Navier-Stokes, desde discretizar las ecuaciones para poder trabajarlas con codigo, hasta solucionarlas con metodos numericos para sistemas de ecuaciones no lineales y lineales, por ultimo mejorar la visualización interpolando los resultados con splines.

# Información
Este repositorio está dividido en 3 secciones, cada una en una carpeta:

# Sección 1: Primer Avance de solucion del sistema no lineal (carpeta "No lineal method") 
En esta sección se desarrolló la solución del problema utilizando el metodo de Newton Raphson para resolver el sistema no lineal y de forma interna se utilizaron dos opciones de metodos lineales (no iterativos), Eliminación de gauss y spsolve propio de python. esto ultimo con el objetivo de poder visualizar resultados ya que para ese entonces no teníamos conocimiento de los metodos lineales iterativos que se verían mas adelante en el curso y que son utilizados en la siguiente sección.

En esta parte del proyecto se implementa la simulación de flujo de fluidos en una malla bidimensional utilizando el método de Newton-Raphson para resolver sistemas de ecuaciones no lineales.

Para visualizar los resultados de esta sección se siguen los pasos:
## Instalación necesaria:
```bash
pip install numpy scipy matplotlib
```

## Ejecución:
```bash
cd "No lineal method"
python main_spsolve.py     # Método con spsolve
python main_gauss.py       # Método con eliminación de Gauss
```

## ¿Qué se visualiza?
- **Consola**: Información del Jacobiano (dimensiones, elementos) y progreso de convergencia
- **Gráfica 1**: Malla de velocidades con cuadrícula
- **Gráfica 2**: Matriz Jacobiana 

---

# Sección 2: Métodos Lineales Iterativos (carpeta "Lineal methods")
Implementación de métodos iterativos (Jacobi, Gauss-Seidel, Richardson, LU) con análisis de convergencia.

## Instalaciones necesarias:
```bash
pip install numpy scipy matplotlib pandas
```

## Ejecución:
```bash
cd "Lineal methods"

# PASO 1 - Análisis inicial 
python main_analysis.py

# PASO 2 - Ejecutar métodos individuales
python main_lu.py
python main_jacobi.py
python main_gauss_seidel.py
python main_richardson.py
```

## ¿Qué se visualiza?

### main_analysis.py:
- **Consola**: 
  - Tabla de velocidades inicial
  - Información del Jacobiano (dimensiones, número de condición)
  - Análisis de supuestos de convergencia para cada método
  - Verificación de diagonal dominante y normas

### Métodos individuales (main_*.py):
- **Consola**: Progreso iterativo con normas de error y convergencia
- **Gráfica**: Campo de velocidades final del método correspondiente

---

# Sección 3: Visualización Mejorada (carpeta "Improve Visualization")
Mejora de visualización usando interpolación con splines cúbicos bidimensionales.

## Instalaciones necesarias:
```bash
pip install numpy scipy matplotlib
```

## Ejecución:
```bash
cd "Improve Visualization"
python cubic_splines.py
```

## ¿Qué se visualiza?
- **Gráfica**: Comparación lado a lado del campo original vs suavizado con splines
- **Mejora**: Resolución aumentada (7x52 → 50x200) con transiciones suaves

---

## Anexo de ejcución en estorno virtual (notebook)
https://colab.research.google.com/drive/1t9WZjB2AtkLwX0Lvln5KLzFgRprM7IUF?usp=sharing