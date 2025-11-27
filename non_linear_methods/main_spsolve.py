import numpy as np
from core.mesh import inicializar_malla, obtener_dimensiones
from core.solvers import newton_raphson_step_spsolve
from core.equations import Jacobiano
from visualization.plotter import (setup_interactive_plot, plot_iteration, 
                                  finalize_plot, plot_jacobiano, print_jacobiano_info)
from config import (TOLERANCIA, MAX_ITERACIONES, MOSTRAR_JACOBIANO, 
                   GRAFICAR_JACOBIANO)

def main():
    # Inicialización
    malla, mascara_solidos = inicializar_malla()
    nx, ny, N = obtener_dimensiones()
    
    X0 = np.full((nx, ny), 0.5)
    
    # Aplicar condiciones de sólidos a la condición inicial
    for i in range(nx):
        for j in range(ny):
            if mascara_solidos[i+1, j+1]:  # +1 porque la malla incluye bordes
                X0[i, j] = 0.0
    
    X = X0.flatten()
    
    # Mostrar Jacobiano inicial
    if MOSTRAR_JACOBIANO or GRAFICAR_JACOBIANO:
        J = Jacobiano(X, mascara_solidos, sparse=True)
        
        if MOSTRAR_JACOBIANO:
            print_jacobiano_info(J, 0, sparse=True)
        
        if GRAFICAR_JACOBIANO:
            plot_jacobiano(J, 0, sparse=True)
    

    setup_interactive_plot()
    
    # Método de Newton-Raphson
    for it in range(MAX_ITERACIONES):

        X, error = newton_raphson_step_spsolve(X, malla, mascara_solidos)
        
        # Actualizar malla y graficar
        malla[1:6, 1:51] = X.reshape((nx, ny))
        
        # Asegurar que los puntos sólidos mantengan velocidad 0
        malla[mascara_solidos] = 0
        
        plot_iteration(malla, it+1, with_grid=True)
        
        print(f"Iteración {it+1}, error: {error:.6e}")
        if error < TOLERANCIA:
            print("Convergencia alcanzada.")
            break
    
    finalize_plot()

if __name__ == "__main__":
    main()