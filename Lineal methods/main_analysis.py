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
            print("❌ Matriz no es diagonalmente dominante - Gauss-Seidel puede fallar")
    
    # --- NUEVO: repetir el mismo análisis con una malla inicial en gradiente 1->0 ---
    print("\n" + "="*60)
    print("ANÁLISIS ADICIONAL - MALLA INICIAL EN GRADIENTE 1 -> 0 (izq -> der)")
    print("="*60)
    
    # Construir malla con gradiente en la zona interna (sin borrar análisis anterior)
    vx_grad = vx.copy()
    rows, cols = vx_grad.shape
    internal_cols = cols - 2
    # gradiente de 1 a 0 de izquierda a derecha en la malla interna
    grad_row = np.linspace(1.0, 0.0, num=internal_cols)
    vx_grad[1:-1, 1:-1] = np.tile(grad_row, (rows-2, 1))
    # aplicar máscara de sólidos (velocidad cero donde haya sólidos)
    vx_grad[mascara_solidos] = 0.0

    if MOSTRAR_TABLA_INICIAL:
        print("\n" + "="*50)
        print("TABLA DE VELOCIDADES INICIAL (GRADIENTE)")
        print("="*50)
        mostrar_tabla_velocidades(vx_grad)
    
    # Calcular Jacobiano para la malla en gradiente
    Jg = calculate_Jacobian_sparse(vx_grad, mascara_solidos)
    Jg_dense = Jg.toarray()
    
    # Número de condición
    cond_number_g = np.linalg.cond(Jg_dense)
    print(f"\n" + "="*50)
    print("INFORMACIÓN DEL JACOBIANO (GRADIENTE)")
    print("="*50)
    print(f"Dimensiones: {Jg_dense.shape}")
    print(f"Elementos no cero: {np.count_nonzero(Jg_dense)}")
    print(f"Densidad: {np.count_nonzero(Jg_dense) / Jg_dense.size:.4f}")
    print(f"Número de condición: {cond_number_g:.2e}")
    
    # Mostrar Jacobiano
    if MOSTRAR_JACOBIANO:
        mostrar_jacobiano(Jg)
    
    # Análisis de supuestos para la malla en gradiente
    if MOSTRAR_SUPUESTOS:
        print(f"\n" + "="*50)
        print("ANÁLISIS DE SUPUESTOS PARA CONVERGENCIA (GRADIENTE)")
        print("="*50)
        
        propiedades_g = analizar_matriz(Jg_dense)
        print(f"\nPropiedades de la matriz (GRADIENTE):")
        for prop, valor in propiedades_g.items():
            print(f"  - {prop}: {valor}")
        
        verificar_convergencia_jacobi(Jg_dense)
        verifica_convergencia_richardson(Jg_dense)
        
        print(f"\nAnálisis de convergencia Gauss-Seidel (GRADIENTE):")
        if propiedades_g["Diagonal dominante"]:
            print("✅ Matriz es diagonalmente dominante - Gauss-Seidel converge")
        else:
            print("❌ Matriz no es diagonalmente dominante - Gauss-Seidel puede fallar")
    # --- FIN análisis adicional ---
    
    print(f"\n" + "="*60)
    print("ANÁLISIS COMPLETADO")
    print("="*60)

if __name__ == "__main__":
    main()