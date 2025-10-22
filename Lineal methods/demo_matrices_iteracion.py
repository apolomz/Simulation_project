#!/usr/bin/env python3
"""
Script para demostrar las diferencias entre matriz Jacobiana y matrices de iteración
"""

import numpy as np
from core.mesh import inicializar_malla
from core.equations import calculate_Jacobian_sparse
from core.solvers import comparar_matrices, crear_matrices_iteracion
from core.analysis import analizar_radio_espectral

def main():
    print("="*80)
    print("DEMOSTRACIÓN: MATRIZ JACOBIANA vs MATRICES DE ITERACIÓN")
    print("="*80)
    
    # Inicializar malla
    vx, vy, mascara_solidos = inicializar_malla()
    J = calculate_Jacobian_sparse(vx, mascara_solidos)
    J_dense = J.toarray()
    
    print(f"\n📊 INFORMACIÓN DEL SISTEMA:")
    print(f"   Dimensiones de la malla: {vx.shape}")
    print(f"   Puntos sólidos: {np.sum(mascara_solidos)}")
    print(f"   Dimensiones del Jacobiano: {J_dense.shape}")
    
    # Comparar matrices
    matrices = comparar_matrices(J_dense)
    
    # Análisis del radio espectral
    print(f"\n" + "="*70)
    print("ANÁLISIS DEL RADIO ESPECTRAL")
    print("="*70)
    
    # Radio espectral de la matriz original
    analizar_radio_espectral(J_dense, "Jacobiano Original")
    
    # Radio espectral de matrices de iteración
    if matrices['jacobi'] is not None:
        analizar_radio_espectral(matrices['jacobi'], "Matriz de Iteración Jacobi")
    
    if matrices['gauss_seidel'] is not None:
        analizar_radio_espectral(matrices['gauss_seidel'], "Matriz de Iteración Gauss-Seidel")
    
    analizar_radio_espectral(matrices['richardson'], "Matriz de Iteración Richardson")
    
    # Comparación visual de diferencias
    print(f"\n" + "="*70)
    print("DIFERENCIAS ESTRUCTURALES")
    print("="*70)
    
    print(f"\n🔍 ANÁLISIS DETALLADO:")
    
    # Mostrar submatrices pequeñas para comparación
    n_show = min(8, J_dense.shape[0])
    
    print(f"\n1. MATRIZ JACOBIANA (primeros {n_show}x{n_show}):")
    print(J_dense[:n_show, :n_show])
    
    if matrices['jacobi'] is not None:
        print(f"\n2. MATRIZ DE ITERACIÓN JACOBI (primeros {n_show}x{n_show}):")
        print(matrices['jacobi'][:n_show, :n_show])
    
    if matrices['gauss_seidel'] is not None:
        print(f"\n3. MATRIZ DE ITERACIÓN GAUSS-SEIDEL (primeros {n_show}x{n_show}):")
        print(matrices['gauss_seidel'][:n_show, :n_show])
    
    print(f"\n4. MATRIZ DE ITERACIÓN RICHARDSON (primeros {n_show}x{n_show}):")
    print(matrices['richardson'][:n_show, :n_show])
    
    # Resumen de diferencias
    print(f"\n" + "="*70)
    print("RESUMEN DE DIFERENCIAS")
    print("="*70)
    
    print(f"\n📋 CARACTERÍSTICAS CLAVE:")
    print(f"   • Matriz Jacobiana: Sistema completo A·x = b")
    print(f"   • Matriz Jacobi: Aproximación usando solo diagonal D")
    print(f"   • Matriz Gauss-Seidel: Aproximación usando D + L")
    print(f"   • Matriz Richardson: Aproximación usando α·A")
    
    print(f"\n🎯 PROPÓSITO:")
    print(f"   • Jacobiana: Resolver sistema exacto (costoso)")
    print(f"   • Matrices de iteración: Resolver sistema aproximado (económico)")
    
    print(f"\n⚡ CONVERGENCIA:")
    print(f"   • Radio espectral < 1: Convergencia garantizada")
    print(f"   • Radio espectral = 1: Convergencia marginal")
    print(f"   • Radio espectral > 1: No converge")

if __name__ == "__main__":
    main()
