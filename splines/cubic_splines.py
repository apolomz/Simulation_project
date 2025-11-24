import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import get_cmap
from scipy.sparse import lil_matrix
import pandas as pd
from scipy.sparse import lil_matrix
from scipy.interpolate import RectBivariateSpline

# Configuración de la malla
rows = 7  # Número de filas (incluyendo bordes)
cols = 52  # Número de columnas (incluyendo bordes)

# Crear malla de coordenadas
X, Y = np.meshgrid(np.arange(cols), np.arange(rows))

# Inicialización de las velocidades
vx = np.zeros((rows, cols))  # Velocidad en dirección x
vy = np.zeros((rows, cols))  # Velocidad en dirección y

# Condiciones de frontera modificadas
vx[0, :] = 0; vx[-1, :] = 0
vy[0, :] = 0.1; vy[-1, :] = 0.1
vx[1:6, 0] = 1
vx[0, 0] = 0
vx[6, 0] = 0
vy[:, 0] = 0.1
vx[:, -1] = 0
vy[:, -1] = 0.1

# Condición inicial en puntos internos
for i in range(1, rows-1):
    for j in range(1, cols-1):
        # vx[i, j] = max(0, 1 - (j / (cols-2)))  # Disminución lineal
        vx[i, j] = 0.5
        vy[i, j] = 0.1


def calculate_F(vx, h=1.0):
    rows, cols = vx.shape
    F = np.zeros((rows-2, cols-2))
    vy_const = 0.1

    for i in range(1, rows-1):
        for j in range(1, cols-1):
            term1 = vx[i+1,j] + vx[i-1,j] + vx[i,j+1] + vx[i,j-1]
            term2 = (h/2) * vx[i,j] * (vx[i+1,j] - vx[i-1,j])
            term3 = (h/2) * vy_const * (vx[i,j+1] - vx[i,j-1])
            vx_calculated = vx[i,j] - (1/4) * (term1 - term2 - term3)
            F[i-1, j-1] = vx_calculated
    return F


def calculate_Jacobian_sparse(vx, h=1.0):
    rows, cols = vx.shape
    n = (rows-2) * (cols-2)
    J = lil_matrix((n, n))

    for i in range(1, rows-1):
        for j in range(1, cols-1):
            idx = (i-1)*(cols-2) + (j-1)
            J[idx, idx] = 1 + (h/8)*(vx[i+1,j] - vx[i-1,j]) + (h/8)*vy[i,j]*(vx[i,j+1] - vx[i,j-1])

            if i > 1:
                J[idx, idx - (cols-2)] = -0.25 + (h/8)*vx[i,j]
            if i < rows-2:
                J[idx, idx + (cols-2)] = -0.25 - (h/8)*vx[i,j]
            if j > 1:
                J[idx, idx - 1] = -0.25 + (h/8)*0.1
            if j < cols-2:
                J[idx, idx + 1] = -0.25 - (h/8)*0.1
    return J
vx_copy_richardson = vx.copy()
def Richardson(a, b, M):
    n = b.shape[0]
    x = np.zeros(n)  # Vector unidimensional para la solución
    tol=1e-7

    for k in range(M):
        r = b - np.dot(a, x)  # Calculamos el residuo
        x = x + r  # Método de Richardson: corregir el valor de x

        # Criterio de convergencia opcional (norma del residuo)
        if np.linalg.norm(np.dot(a, x) - b) < tol:

            break

    return x
max_iter = 100
tol = 1e-7


for it in range(max_iter):
    F = calculate_F(vx_copy_richardson).flatten()  # Vector de residuos
    J = calculate_Jacobian_sparse(vx_copy_richardson).toarray()  # Matriz Jacobiana

    # Resolver J·ΔX = -F(X) usando LU
    delta_X = Richardson(J, -F, 1000)



    # Actualizar la copia de vx
    vx_copy_richardson[1:-1, 1:-1] += delta_X.reshape((rows-2, cols-2))

    # Verificar convergencia
    if np.linalg.norm(delta_X) < tol:
        print(f"\nConvergencia alcanzada en la iteración {it+1}")
        break
else:
    print("\nNo se alcanzó la convergencia dentro del número máximo de iteraciones")

# Aplicar splines para suavizar el flujo
x = np.arange(cols)
y = np.arange(rows)

# Crear spline cúbico bidimensional
spline = RectBivariateSpline(y, x, vx_copy_richardson, kx=3, ky=3)

# Crear una malla más fina para visualización suave
x_fine = np.linspace(0, cols-1, 200)
y_fine = np.linspace(0, rows-1, 50)
X_fine, Y_fine = np.meshgrid(x_fine, y_fine)

# Evaluar el spline en la malla fina
vx_smooth = spline(y_fine, x_fine)

# Visualización
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Gráfico original
cmap = get_cmap("viridis")
norm = Normalize(vmin=np.min(vx_copy_richardson), vmax=np.max(vx_copy_richardson))
im1 = ax1.imshow(vx_copy_richardson, cmap=cmap, norm=norm, extent=[0, cols-1, 0, rows-1])
ax1.set_title("Campo de velocidad original")
plt.colorbar(im1, ax=ax1)

# Gráfico suavizado con splines
im2 = ax2.imshow(vx_smooth, cmap=cmap, norm=norm, extent=[0, cols-1, 0, rows-1])
ax2.set_title("Campo de velocidad suavizado con splines")
plt.colorbar(im2, ax=ax2)


plt.tight_layout()
plt.show()