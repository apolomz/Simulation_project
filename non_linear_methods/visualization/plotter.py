import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.interpolate import RectBivariateSpline
from config import COLORMAP, PAUSA_ANIMACION, FILAS, COLUMNAS

def setup_interactive_plot():
    """Configura el plot interactivo"""
    plt.ion()
    plt.figure(1, figsize=(10, 5))  # Especificar número de figura

def plot_iteration(malla, iteration, with_grid=False):
    """Grafica una iteración específica"""
    plt.figure(1)  # Asegurar que usamos la figura 1
    plt.clf()
    
    if with_grid:
        # Estilo con cuadrícula (como en Gauss)
        im = plt.imshow(malla, cmap=COLORMAP, aspect='equal', 
                       vmin=0, vmax=np.max(malla))
        cbar = plt.colorbar(im, orientation='horizontal', pad=0.15)
        
        # Agregar líneas de cuadrícula
        for x in range(1, COLUMNAS):
            plt.axvline(x - 0.5, color='gray', linestyle='-', 
                       linewidth=0.5, alpha=0.2, zorder=0, ymin=0, ymax=1)
        for y in range(1, FILAS):
            plt.axhline(y - 0.5, color='gray', linestyle='-', 
                       linewidth=0.5, alpha=0.2, zorder=0, xmin=0, xmax=1)
        
        plt.xticks(np.arange(0, COLUMNAS, 10))
        plt.yticks(np.arange(0, FILAS, 1))
    else:
        # Estilo simple (como en spsolve)
        im = plt.imshow(malla, cmap=COLORMAP, aspect='auto', 
                  vmin=0, vmax=np.max(malla))
        cbar = plt.colorbar(im, label='Velocidad vx')
    
    cbar.set_label('Velocidad vx')
    plt.title(f'Iteración {iteration}')
    plt.xlabel('Columnas')
    plt.ylabel('Filas')
    plt.pause(PAUSA_ANIMACION)

def plot_jacobiano(J, iteration, sparse=True):
    """Visualiza la matriz Jacobiana en una ventana separada"""
    if sparse:
        J_dense = J.toarray()
    else:
        J_dense = J
    
    # Crear una nueva figura específica para el Jacobiano
    plt.figure(2, figsize=(12, 5))  # Figura 2 para el Jacobiano
    plt.clf()  # Limpiar la figura por si ya existe
    
    # Crear subplots
    plt.subplot(1, 2, 1)
    im1 = plt.imshow(J_dense, cmap='RdBu_r', aspect='auto')
    plt.colorbar(im1, label='Valor')
    plt.title('Valores de la Matriz Jacobiana')
    plt.xlabel('Columna')
    plt.ylabel('Fila')
    
    plt.suptitle(f'Matriz Jacobiana - Iteración {iteration}', fontsize=14)
    plt.tight_layout()
    
    # Forzar la visualización
    plt.draw()
    plt.pause(0.1)  # Pequeña pausa para asegurar el renderizado
    plt.show(block=False)

def print_jacobiano_info(J, iteration, sparse=True):
    """Imprime información básica del Jacobiano"""
    if sparse:
        J_dense = J.toarray()
    else:
        J_dense = J
    
    print(f"\n{'='*50}")
    print(f"JACOBIANO")
    print(f"{'='*50}")
    print(f"Dimensiones: {J_dense.shape}")
    print(f"Elementos no cero: {np.count_nonzero(J_dense)}")
    print(f"{'='*50}\n")
    
    # Mostrar una muestra de la matriz (primeras 8x8)
    print("Muestra de la matriz (primeras 8x8 elementos):")
    print(J_dense[:min(8, J_dense.shape[0]), :min(8, J_dense.shape[1])])
    print()

def suavizar_con_splines(malla, x_fine_factor=4, y_fine_factor=4):
    """
    Suaviza una malla usando splines cúbicos bidimensionales
    
    Args:
        malla: Array 2D con los valores a suavizar
        x_fine_factor: Factor de refinamiento en dirección x (default: 4)
        y_fine_factor: Factor de refinamiento en dirección y (default: 4)
    
    Returns:
        malla_suavizada: Array 2D con valores suavizados en malla más fina
        x_fine: Coordenadas x de la malla fina
        y_fine: Coordenadas y de la malla fina
        spline: Objeto RectBivariateSpline creado
        X_fine: Malla X para visualización 3D
        Y_fine: Malla Y para visualización 3D
    """
    rows, cols = malla.shape
    
    # Crear coordenadas originales
    x = np.arange(cols)
    y = np.arange(rows)
    
    # Crear spline cúbico bidimensional (natural)
    spline = RectBivariateSpline(y, x, malla, kx=3, ky=3)
    
    # Crear una malla más fina para visualización suave
    x_fine = np.linspace(0, cols-1, cols * x_fine_factor)
    y_fine = np.linspace(0, rows-1, rows * y_fine_factor)
    X_fine, Y_fine = np.meshgrid(x_fine, y_fine)
    
    # Evaluar el spline en la malla fina
    malla_suavizada = spline(y_fine, x_fine)
    
    return malla_suavizada, x_fine, y_fine, spline, X_fine, Y_fine

def plot_final_with_splines(malla):
    """
    Muestra la visualización final con comparación entre original, suavizado y superficie 3D del spline
    """
    plt.ioff()  # Desactivar modo interactivo para la visualización final
    
    # Suavizar con splines
    malla_suavizada, x_fine, y_fine, spline, X_fine, Y_fine = suavizar_con_splines(malla)
    
    # Crear figura con tres subplots: 2D original, 2D suavizado, y 3D del spline
    fig = plt.figure(figsize=(20, 6))
    
    # Gráfico original (2D)
    ax1 = plt.subplot(1, 3, 1)
    im1 = ax1.imshow(malla, cmap=COLORMAP, aspect='equal', 
                     vmin=0, vmax=np.max(malla), origin='upper')
    ax1.set_title("Campo de velocidad original")
    ax1.set_xlabel('Columnas')
    ax1.set_ylabel('Filas')
    plt.colorbar(im1, ax=ax1, label='Velocidad vx')
    
    # Gráfico suavizado con splines (2D)
    ax2 = plt.subplot(1, 3, 2)
    im2 = ax2.imshow(malla_suavizada, cmap=COLORMAP, aspect='equal',
                     vmin=0, vmax=np.max(malla), origin='upper',
                     extent=[0, malla.shape[1]-1, 0, malla.shape[0]-1])
    ax2.set_title("Campo de velocidad suavizado con splines cúbicos")
    ax2.set_xlabel('Columnas')
    ax2.set_ylabel('Filas')
    plt.colorbar(im2, ax=ax2, label='Velocidad vx')
    
    # Gráfico 3D del spline
    ax3 = plt.subplot(1, 3, 3, projection='3d')
    surf = ax3.plot_surface(X_fine, Y_fine, malla_suavizada, cmap=COLORMAP,
                           linewidth=0, antialiased=True, alpha=0.9,
                           vmin=0, vmax=np.max(malla))
    ax3.set_title("Superficie 3D del spline cúbico")
    ax3.set_xlabel('Columnas')
    ax3.set_ylabel('Filas')
    ax3.set_zlabel('Velocidad vx')
    plt.colorbar(surf, ax=ax3, label='Velocidad vx', shrink=0.6)
    
    plt.tight_layout()
    plt.show()

def finalize_plot():
    """Finaliza el modo interactivo"""
    plt.ioff()
    plt.show()