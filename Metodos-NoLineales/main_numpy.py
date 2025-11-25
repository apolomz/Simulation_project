import numpy as np
from logic.mesh import inicializar_malla, obtener_dimensiones
from logic.newton_raphson import newton_raphson_step_numpy
from logic.equations import Jacobiano
from visualization.plot import (
    setup_interactive_plot, plot_iteration, 
    finalize_plot, plot_jacobiano, print_jacobiano_info
)
from config import (
    TOLERANCIA, MAX_ITERACIONES, MOSTRAR_JACOBIANO, GRAFICAR_JACOBIANO
)

def main():
    malla = inicializar_malla()
    nx, ny, N = obtener_dimensiones()
    
    X0 = np.full((nx, ny), 0.5)
    X = X0.flatten()
    
    # Mostrar Jacobiano inicial
    if MOSTRAR_JACOBIANO or GRAFICAR_JACOBIANO:
        J = Jacobiano(X)
        
        if MOSTRAR_JACOBIANO:
            print_jacobiano_info(J, 0, sparse=False)
        
        if GRAFICAR_JACOBIANO:
            plot_jacobiano(J, 0, sparse=False)

    setup_interactive_plot()
    
    # Método de Newton-Raphson
    for it in range(MAX_ITERACIONES):
        X, error = newton_raphson_step_numpy(X, malla)
        
        # Actualizar malla y graficar
        malla[1:6, 1:51] = X.reshape((nx, ny))
        plot_iteration(malla, it+1, with_grid=True)
        
        print(f"Iteración {it+1}, error: {error:.6e}")
        if error < TOLERANCIA:
            print("Convergencia alcanzada.")
            break
    
    finalize_plot()

if __name__ == "__main__":
    main()
