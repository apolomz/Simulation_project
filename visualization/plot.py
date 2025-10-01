import matplotlib.pyplot as plt
import numpy as np
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

def finalize_plot():
    """Finaliza el modo interactivo"""
    plt.ioff()
    plt.show()