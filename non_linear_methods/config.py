"""Parámetros de configuración para la simulación"""

# Parámetros de la malla
FILAS = 7
COLUMNAS = 52

# Parámetros del método Newton-Raphson
TOLERANCIA = 1e-7
MAX_ITERACIONES = 40

# Parámetros físicos
VY_CONSTANTE = 0.5

# Parámetros de visualización
PAUSA_ANIMACION = 0.3
COLORMAP = 'plasma'

# Parámetros para debugging del Jacobiano
MOSTRAR_JACOBIANO = True           # Si mostrar información del Jacobiano
GRAFICAR_JACOBIANO = True          # Si graficar el Jacobiano