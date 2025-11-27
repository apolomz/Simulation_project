### Autores: Juan Sebastian Sierra | Joan Sebastian Fernandez | Luis Gabriel Rodriguez
#### Fecha: Noviembre de 2025

# Simulación Numérica de Flujo Incompresible (Navier-Stokes)

- **Asignatura:** Simulación y Computación Numérica  
- **Proyecto:** Discretización y solución de ecuaciones de Navier-Stokes mediante Métodos Numéricos.

## Descripción

Este repositorio aloja el desarrollo computacional para la simulación de flujo de fluidos incompresibles. El proyecto abarca el ciclo completo de la simulación numérica:

1.  **Discretización:** Transformación de las ecuaciones diferenciales de Navier-Stokes para su tratamiento computacional.
2.  **Solución No Lineal:** Implementación del método de Newton-Raphson.
3.  **Solución de Sistemas Lineales:** Comparativa entre solvers directos y métodos iterativos (Jacobi, Gauss-Seidel, Richardson).
4.  **Post-procesamiento:** Mejora de la visualización de resultados mediante interpolación con Splines Cúbicos.

---

## Requisitos e Instalación

Este proyecto utiliza **Python**. Para asegurar el correcto funcionamiento de todos los módulos, se recomienda instalar las siguientes librerías de computación científica y visualización:

`pip install numpy scipy matplotlib pandas`

## Estructura del Proyecto
El repositorio está organizado en tres fases lógicas de desarrollo:

```
├── linear_methods/        # Fase 1: Solución del sistema no lineal (Newton-Raphson)
├── non_linear_methods/          # Fase 2: Análisis y aplicación de métodos iterativos
└── splines/   # Fase 3: Post-procesamiento y suavizado de gráficos
```

### Sección 1: Solución de Sistema No Lineal
Carpeta: `non_linear_methods/`

En esta etapa inicial se implementó la física del problema utilizando el Método de Newton-Raphson para resolver el sistema de ecuaciones no lineales resultante de la discretización.

Para resolver los sistemas lineales internos que genera cada iteración de Newton-Raphson, se utilizaron inicialmente solvers directos (no iterativos) para validar el modelo físico antes de proceder a métodos más complejos.

Ejecución:

`non_linear_methods/`

- **Opción A:** Solución usando el solver optimizado de SciPy
`python main_spsolve.py`

- **Opción B:** Solución implementando Eliminación de Gauss manual
`python main_gauss.py`

##### Resultados Esperados
Consola: Detalles de la Matriz Jacobiana (dimensiones y elementos) y el error en cada iteración de convergencia.

#### Visualización:

Gráfica de la malla de velocidades.

Representación gráfica de la estructura de la Matriz Jacobiana (dispersión).

### Sección 2: Métodos Lineales Iterativos
Carpeta: `linear_methods/`

Esta sección constituye el núcleo del análisis numérico. Se reemplazan los solvers directos por métodos iterativos, realizando un estudio profundo sobre su estabilidad y convergencia aplicada al problema de fluidos.

Métodos implementados:

- Jacobi

- Gauss-Seidel

- Richardson

- Descomposición LU (como referencia directa)

Ejecución

Paso 1: Análisis de Convergencia Antes de ejecutar los métodos, se recomienda correr el script de análisis para verificar si el sistema cumple con las condiciones necesarias (como ser diagonal dominante) y calcular el número de condición.

```
cd "linear_methods"
python main_analysis.py
```

Paso 2: Ejecución de Métodos
```
python main_lu.py            # Método LU
python main_jacobi.py        # Método de Jacobi
python main_gauss_seidel.py  # Método de Gauss-Seidel
python main_richardson.py    # Método de Richardson
```
#### Resultados Esperados:

Análisis (`main_analysis.py`): Tabla de velocidades iniciales, propiedades del Jacobiano (número de condición) y validación teórica de convergencia para cada método.

Métodos (`main_*.py`): Log de evolución del error por iteración y gráfica final del campo de velocidades generado por el método específico.

### Sección 3: Visualización Mejorada
Carpeta: `splines/`

Fase de post-procesamiento. Dado que las mallas computacionales pueden ser gruesas para ahorrar coste de cálculo, se utilizan técnicas de interpolación para generar visualizaciones de alta calidad y transiciones suaves.

Técnica: Interpolación con Splines Cúbicos Bidimensionales.

Ejecución:
```
cd "splines/"
python cubic_splines.py
```

##### Resultados Esperados
Comparativa: Se genera una gráfica que muestra lado a lado el campo de velocidades original (baja resolución, ej: 7x52) contra el campo suavizado e interpolado (alta resolución, ej: 50x200).

