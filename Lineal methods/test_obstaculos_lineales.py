#!/usr/bin/env python3
"""
Script para probar la malla con obstáculos en métodos lineales
"""

import numpy as np
from core.mesh import inicializar_malla, visualizar_malla_con_obstaculos
from core.equations import calculate_Jacobian_sparse, calculate_F

def main():
    print("="*60)
    print("PRUEBA DE MALLA CON OBSTÁCULOS - MÉTODOS LINEALES")
    print("="*60)
    
    # Inicializar malla con obstáculos
    vx, vy, mascara_solidos = inicializar_malla()
    
    print(f"\nMalla inicializada:")
    print(f"Dimensiones vx: {vx.shape}")
    print(f"Dimensiones vy: {vy.shape}")
    print(f"Tipo de datos: {vx.dtype}")
    
    # Mostrar información de los obstáculos
    print(f"\nInformación de obstáculos:")
    print(f"Total de puntos sólidos: {np.sum(mascara_solidos)}")
    print(f"Porcentaje de obstáculos: {100 * np.sum(mascara_solidos) / (vx.size):.1f}%")
    
    # Verificar que los obstáculos tienen velocidad 0
    velocidades_vx_obstaculos = vx[mascara_solidos]
    velocidades_vy_obstaculos = vy[mascara_solidos]
    print(f"\nVelocidades vx en obstáculos: {velocidades_vx_obstaculos}")
    print(f"Velocidades vy en obstáculos: {velocidades_vy_obstaculos}")
    print(f"Todas las velocidades vx en obstáculos son 0: {np.all(velocidades_vx_obstaculos == 0)}")
    print(f"Todas las velocidades vy en obstáculos son 0: {np.all(velocidades_vy_obstaculos == 0)}")
    
    # Probar cálculo de F con obstáculos
    print(f"\nProbando cálculo de F con obstáculos:")
    F = calculate_F(vx, mascara_solidos)
    print(f"Dimensiones de F: {F.shape}")
    print(f"Valores de F en obstáculos (deberían ser 0): {F[mascara_solidos[1:-1, 1:-1]]}")
    
    # Probar cálculo del Jacobiano con obstáculos
    print(f"\nProbando cálculo del Jacobiano con obstáculos:")
    J = calculate_Jacobian_sparse(vx, mascara_solidos)
    J_dense = J.toarray()
    print(f"Dimensiones del Jacobiano: {J_dense.shape}")
    print(f"Elementos no cero: {np.count_nonzero(J_dense)}")
    print(f"Densidad: {np.count_nonzero(J_dense) / J_dense.size:.4f}")
    
    # Verificar que los puntos sólidos tienen derivada = 1
    print(f"\nVerificando derivadas en puntos sólidos:")
    for i in range(1, vx.shape[0]-1):
        for j in range(1, vx.shape[1]-1):
            if mascara_solidos[i, j]:
                idx = (i-1)*(vx.shape[1]-2) + (j-1)
                print(f"Punto sólido ({i},{j}) -> índice {idx}, derivada: {J_dense[idx, idx]}")
    
    # Visualizar (comentado para evitar problemas de display)
    # visualizar_malla_con_obstaculos(vx, vy, mascara_solidos)
    
    print(f"\n" + "="*60)
    print("PRUEBA COMPLETADA")
    print("="*60)

if __name__ == "__main__":
    main()
