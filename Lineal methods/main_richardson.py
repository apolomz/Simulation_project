import numpy as np
from core.mesh import inicializar_malla, obtener_dimensiones
from core.equations import calculate_F, calculate_Jacobian_sparse
from core.solvers import Richardson
from visualization.plotter import plot_velocity_field
from config import MAX_ITERACIONES_NEWTON, TOLERANCIA

def main():
    print("="*50)
    print("MÉTODO ITERATIVO DE RICHARDSON")
    print("="*50)
    
    # Inicialización
    vx, vy, mascara_solidos = inicializar_malla()
    vx_copy = vx.copy()
    rows, cols = vx.shape
    
    print(f"Iniciando método Richardson...")
    print(f"Tolerancia: {TOLERANCIA}")
    print(f"Máximo de iteraciones: {MAX_ITERACIONES_NEWTON}")
    print(f"Obstáculos: {np.sum(mascara_solidos)} puntos sólidos")
    
    # Método de Newton con Richardson
    for it in range(MAX_ITERACIONES_NEWTON):
        F = calculate_F(vx_copy, mascara_solidos).flatten()
        J = calculate_Jacobian_sparse(vx_copy, mascara_solidos).toarray()
        
        # Resolver usando Richardson
        delta_X = Richardson(J, -F)
        
        # Información de la iteración
        error = np.linalg.norm(delta_X)
        print(f"\nIteración {it + 1}:")
        print(f"  - Norma de delta_X: {error:.3e}")
        print(f"  - Valores extremos: Min = {np.min(delta_X):.3e}, Max = {np.max(delta_X):.3e}")
        
        # Actualizar vx
        vx_copy[1:-1, 1:-1] += delta_X.reshape((rows-2, cols-2))
        
        # Asegurar que los puntos sólidos mantengan velocidad 0
        vx_copy[mascara_solidos] = 0
        
        # Verificar convergencia
        if error < TOLERANCIA:
            print(f"\n✅ Convergencia alcanzada en la iteración {it+1}")
            break
    else:
        print(f"\n❌ No se alcanzó convergencia en {MAX_ITERACIONES_NEWTON} iteraciones")
    
    # Mostrar resultado
    plot_velocity_field(vx_copy, method_name="Método Richardson")
    
    print("="*50)

if __name__ == "__main__":
    main()