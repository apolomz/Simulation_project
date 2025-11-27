import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.cm import get_cmap
from config import COLORMAP

def plot_velocity_field(vx, title="Mapa de calor de vx", method_name=""):
    """Grafica el campo de velocidades"""
    fig, axes = plt.subplots(1, 1, figsize=(10, 5))
    cmap = get_cmap(COLORMAP)
    norm = Normalize(vmin=np.min(vx), vmax=np.max(vx))

    im1 = axes.imshow(vx, cmap=cmap, norm=norm)
    axes.set_title(f"{title} - {method_name}")
    plt.colorbar(im1, ax=axes)
    plt.tight_layout()
    plt.show()

def plot_convergence_comparison(methods_data):
    """Grafica comparación de convergencia entre métodos"""
    plt.figure(figsize=(12, 8))
    
    for method_name, data in methods_data.items():
        plt.plot(data['iterations'], data['errors'], 
                label=f"{method_name} ({data['final_iter']} iter)", 
                marker='o', markersize=3)
    
    plt.xlabel('Iteración')
    plt.ylabel('Error (norma)')
    plt.title('Comparación de Convergencia - Métodos Lineales Iterativos')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.tight_layout()
    plt.show()