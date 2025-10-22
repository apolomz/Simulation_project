import numpy as np
from core.mesh import inicializar_malla
from core.equations import calculate_Jacobian_sparse
from core.analysis import (mostrar_tabla_velocidades, mostrar_jacobiano, 
                          verificar_convergencia_jacobi, verifica_convergencia_richardson,
                          analizar_matriz)
from config import MOSTRAR_JACOBIANO, MOSTRAR_SUPUESTOS, MOSTRAR_TABLA_INICIAL

def main():
    print("="*60)
    print("ANÁLISIS INICIAL - MÉTODOS LINEALES ITERATIVOS")
    print("="*60)
    
    # Inicializar malla
    vx, vy, mascara_solidos = inicializar_malla()
    print(f"Malla inicializada: {vx.shape}")
    
    # Mostrar tabla inicial
    if MOSTRAR_TABLA_INICIAL:
        print("\n" + "="*50)
        print("TABLA DE VELOCIDADES INICIAL")
        print("="*50)
        mostrar_tabla_velocidades(vx)
    
    # Calcular Jacobiano
    J = calculate_Jacobian_sparse(vx, mascara_solidos)
    J_dense = J.toarray()
    
    # Número de condición
    cond_number = np.linalg.cond(J_dense)
    print(f"\n" + "="*50)
    print("INFORMACIÓN DEL JACOBIANO")
    print("="*50)
    print(f"Dimensiones: {J_dense.shape}")
    print(f"Elementos no cero: {np.count_nonzero(J_dense)}")
    print(f"Densidad: {np.count_nonzero(J_dense) / J_dense.size:.4f}")
    print(f"Número de condición: {cond_number:.2e}")
    
    # Mostrar Jacobiano
    if MOSTRAR_JACOBIANO:
        mostrar_jacobiano(J)
    
    # Análisis de supuestos
    if MOSTRAR_SUPUESTOS:
        print(f"\n" + "="*50)
        print("ANÁLISIS DE SUPUESTOS PARA CONVERGENCIA")
        print("="*50)
        
        # Análisis general de la matriz
        propiedades = analizar_matriz(J_dense)
        print(f"\nPropiedades de la matriz:")
        for prop, valor in propiedades.items():
            print(f"  - {prop}: {valor}")
        
        # Verificar convergencia Jacobi
        verificar_convergencia_jacobi(J_dense)
        
        # Verificar convergencia Richardson
        verifica_convergencia_richardson(J_dense)
        
        # Información para Gauss-Seidel
        print(f"\nAnálisis de convergencia Gauss-Seidel:")
        if propiedades["Diagonal dominante"]:
            print("✅ Matriz es diagonalmente dominante - Gauss-Seidel converge")
        else:
            print("⚠️ Matriz no es diagonalmente dominante - convergencia no garantizada")
    
    print(f"\n" + "="*60)
    print("ANÁLISIS COMPLETADO")
    print("="*60)

if __name__ == "__main__":
    main()