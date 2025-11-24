#!/usr/bin/env python3
"""
Script para analizar el radio espectral de diferentes matrices
"""

import numpy as np
from core.mesh import inicializar_malla
from core.equations import calculate_Jacobian_sparse
from core.analysis import analizar_radio_espectral, calcular_radio_espectral

def analizar_matrices_iterativas():
    """Analiza el radio espectral de matrices de iteración para diferentes métodos"""
    print("="*70)
    print("ANÁLISIS DEL RADIO ESPECTRAL DE MATRICES DE ITERACIÓN")
    print("="*70)
    
    # Inicializar malla
    vx, vy, mascara_solidos = inicializar_malla()
    J = calculate_Jacobian_sparse(vx, mascara_solidos)
    J_dense = J.toarray()
    
    print(f"Dimensiones de la matriz: {J_dense.shape}")
    print(f"Elementos no cero: {np.count_nonzero(J_dense)}")
    print(f"Densidad: {np.count_nonzero(J_dense) / J_dense.size:.4f}")
    
    # 1. Análisis del Jacobiano original
    analizar_radio_espectral(J_dense, "Jacobiano Original")
    
    # 2. Matriz de iteración para Jacobi: M = I - D^(-1) * A
    print(f"\n" + "="*50)
    print("MATRIZ DE ITERACIÓN JACOBI")
    print("="*50)
    
    D = np.diag(np.diag(J_dense))
    try:
        D_inv = np.linalg.inv(D) 
        M_jacobi = np.eye(J_dense.shape[0]) - D_inv @ J_dense
        analizar_radio_espectral(M_jacobi, "Matriz de Iteración Jacobi")
    except np.linalg.LinAlgError:
        print("❌ No se pudo invertir la matriz diagonal para Jacobi")
    
    # 3. Matriz de iteración para Gauss-Seidel: M = -(D+L)^(-1) * U
    print(f"\n" + "="*50)
    print("MATRIZ DE ITERACIÓN GAUSS-SEIDEL")
    print("="*50)
    
    D = np.diag(np.diag(J_dense))
    L = np.tril(J_dense, k=-1)
    U = np.triu(J_dense, k=1)
    
    try:
        D_L_inv = np.linalg.inv(D + L)
        M_gauss_seidel = -D_L_inv @ U
        analizar_radio_espectral(M_gauss_seidel, "Matriz de Iteración Gauss-Seidel")
    except np.linalg.LinAlgError:
        print("❌ No se pudo invertir la matriz (D+L) para Gauss-Seidel")
    
    # 4. Matriz de iteración para Richardson: M = I - α * A
    print(f"\n" + "="*50)
    print("MATRIZ DE ITERACIÓN RICHARDSON")
    print("="*50)
    
    alphas = [0.1, 0.2, 0.5, 1.0]
    for alpha in alphas:
        M_richardson = np.eye(J_dense.shape[0]) - alpha * J_dense
        analizar_radio_espectral(M_richardson, f"Matriz de Iteración Richardson (α={alpha})")
    
    # 5. Comparación de radios espectrales
    print(f"\n" + "="*70)
    print("COMPARACIÓN DE RADIOS ESPECTRALES")
    print("="*70)
    
    radios = {}
    
    # Jacobiano
    radio_jacobiano, _ = calcular_radio_espectral(J_dense)
    radios["Jacobiano"] = radio_jacobiano
    
    # Jacobi
    try:
        D = np.diag(np.diag(J_dense))
        D_inv = np.linalg.inv(D)
        M_jacobi = np.eye(J_dense.shape[0]) - D_inv @ J_dense
        radio_jacobi, _ = calcular_radio_espectral(M_jacobi)
        radios["Jacobi"] = radio_jacobi
    except:
        radios["Jacobi"] = None
    
    # Gauss-Seidel
    try:
        D = np.diag(np.diag(J_dense))
        L = np.tril(J_dense, k=-1)
        U = np.triu(J_dense, k=1)
        D_L_inv = np.linalg.inv(D + L)
        M_gauss_seidel = -D_L_inv @ U
        radio_gauss_seidel, _ = calcular_radio_espectral(M_gauss_seidel)
        radios["Gauss-Seidel"] = radio_gauss_seidel
    except:
        radios["Gauss-Seidel"] = None
    
    # Richardson (mejor alpha)
    mejor_alpha = None
    mejor_radio = float('inf')
    for alpha in np.linspace(0.01, 2.0, 20):
        M_richardson = np.eye(J_dense.shape[0]) - alpha * J_dense
        radio_richardson, _ = calcular_radio_espectral(M_richardson)
        if radio_richardson is not None and radio_richardson < mejor_radio:
            mejor_radio = radio_richardson
            mejor_alpha = alpha
    
    radios["Richardson (óptimo)"] = mejor_radio
    
    print(f"\nResumen de radios espectrales:")
    print(f"{'Método':<20} {'Radio Espectral':<15} {'Convergencia'}")
    print("-" * 50)
    
    for metodo, radio in radios.items():
        if radio is not None:
            convergencia = "✅" if radio < 1 else "❌" if radio > 1 else "⚠️"
            print(f"{metodo:<20} {radio:<15.6f} {convergencia}")
        else:
            print(f"{metodo:<20} {'Error':<15} ❌")
    
    if mejor_alpha is not None:
        print(f"\nMejor parámetro α para Richardson: {mejor_alpha:.3f}")

def main():
    analizar_matrices_iterativas()

if __name__ == "__main__":
    main()
