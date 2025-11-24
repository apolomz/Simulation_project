import numpy as np
from core.mesh import inicializar_malla, obtener_dimensiones
from core.equations import calculate_F, calculate_Jacobian_sparse
from core.solvers import solve_LU
from visualization.plotter import plot_velocity_field
from config import MAX_ITERACIONES_NEWTON, TOLERANCIA

def main():
    print("="*50)
    print("MÉTODO DE FACTORIZACIÓN LU")
    print("="*50)
    
    # Inicialización
    vx, vy, mascara_solidos = inicializar_malla()
    vx_copy = vx.copy()
    rows, cols = vx.shape
    
    print(f"Iniciando método LU...")
    print(f"Tolerancia: {TOLERANCIA}")
    print(f"Máximo de iteraciones: {MAX_ITERACIONES_NEWTON}")
    print(f"Obstáculos: {np.sum(mascara_solidos)} puntos sólidos")
    
    # Método de Newton con LU
    for it in range(MAX_ITERACIONES_NEWTON):
        F = calculate_F(vx_copy, mascara_solidos).flatten()
        J = calculate_Jacobian_sparse(vx_copy, mascara_solidos).toarray()
        
        # Resolver usando LU
        delta_X = solve_LU(J, -F)
        
        # Actualizar vx
        vx_copy[1:-1, 1:-1] += delta_X.reshape((rows-2, cols-2))
        
        # Asegurar que los puntos sólidos mantengan velocidad 0
        vx_copy[mascara_solidos] = 0
        
        # Verificar convergencia
        error = np.linalg.norm(delta_X)
        if it % 100 == 0 or error < TOLERANCIA:
            print(f"Iteración {it+1}, error: {error:.6e}")
        
        if error < TOLERANCIA:
            print(f"\n✅ Convergencia alcanzada en la iteración {it+1}")
            break
    else:
        print(f"\n❌ No se alcanzó convergencia en {MAX_ITERACIONES_NEWTON} iteraciones")
    
    # Mostrar resultado
    plot_velocity_field(vx_copy, method_name="Factorización LU")
    
    print("="*50)

if __name__ == "__main__":
    main()