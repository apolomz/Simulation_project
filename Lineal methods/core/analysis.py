import numpy as np
import pandas as pd
from config import FILAS, COLUMNAS

def mostrar_tabla_velocidades(vx):
    """Muestra la tabla de velocidades en la malla interna"""
    tabla = vx[1:-1, 1:-1].reshape((FILAS-2, COLUMNAS-2))
    df = pd.DataFrame(tabla, 
                      index=[f"Fila {i+1}" for i in range(FILAS-2)],
                      columns=[f"Col {j+1}" for j in range(COLUMNAS-2)])
    print("\nTabla de velocidades en la malla interna:\n")
    print(df.round(4))
    return df

def mostrar_jacobiano(J, filas=20, columnas=20):
    """Muestra el Jacobiano de forma tabular"""
    J_dense = J.toarray()
    df = pd.DataFrame(J_dense)
    
    print("Jacobiano (parcial):")
    with pd.option_context('display.max_rows', 20, 'display.max_columns', 20):
        print(df.iloc[:filas, :columnas])

def es_diagonalmente_dominante(A):
    """Verifica si una matriz es diagonalmente dominante"""
    for i in range(len(A)):
        diagonal = abs(A[i, i])
        suma_no_diagonal = sum(abs(A[i, j]) for j in range(len(A)) if j != i)
        if diagonal < suma_no_diagonal:
            return False
    return True

def verificar_convergencia_jacobi(A):
    """Verifica condiciones de convergencia para Jacobi"""
    n = A.shape[0]
    D = np.diag(np.diag(A))
    
    try:
        D_inv = np.linalg.inv(D)
    except np.linalg.LinAlgError:
        print("❌ Matriz diagonal no es invertible. Jacobi no puede aplicarse.")
        return False, float('inf')
    
    M = np.eye(n) - D_inv @ A
    norma = np.linalg.norm(M, ord=np.inf)
    
    print(f"\nAnálisis de convergencia Jacobi:")
    print(f"Norma ‖M‖∞ = {norma:.6f}")
    if norma <= 1:
        print("✅ El método Jacobi puede converger (norma < 1)")
    else:
        print("❌ El método Jacobi no garantiza convergencia (norma >= 1)")
    
    return norma < 1, norma

def verifica_convergencia_richardson(A):
    """Verifica condiciones de convergencia para Richardson"""
    A = np.array(A, dtype=float)
    I = np.eye(A.shape[0])
    Q = np.eye(A.shape[0])
    
    try:
        Q_inv = np.linalg.inv(Q)
        B = I - Q_inv @ A
        norma_inf = np.linalg.norm(B, ord=np.inf)
        
        print(f"\nAnálisis de convergencia Richardson:")
        print(f"Norma infinito de (I - Q^(-1) * A): {norma_inf:.6f}")
        if norma_inf <= 1:
            print("✅ El método Richardson puede converger (norma < 1)")
        else:
            print("❌ El método Richardson no garantiza convergencia (norma >= 1)")
        
        return norma_inf < 1, norma_inf
    except np.linalg.LinAlgError:
        print("❌ No se pudo invertir la matriz Q")
        return False, float('inf')

def calcular_radio_espectral(A):
    """Calcula el radio espectral de la matriz A"""
    try:
        autovalores = np.linalg.eigvals(A)
        radio_espectral = np.max(np.abs(autovalores))
        return radio_espectral, autovalores
    except np.linalg.LinAlgError:
        print("❌ Error al calcular autovalores")
        return None, None

def analizar_radio_espectral(A, nombre_matriz="Matriz"):
    """Analiza el radio espectral de una matriz"""
    print(f"\n" + "="*50)
    print(f"ANÁLISIS DEL RADIO ESPECTRAL - {nombre_matriz.upper()}")
    print("="*50)
    
    radio, autovalores = calcular_radio_espectral(A)
    
    if radio is not None:
        print(f"Radio espectral: {radio:.6f}")
        print(f"Autovalor máximo (módulo): {np.max(np.abs(autovalores)):.6f}")
        print(f"Autovalor mínimo (módulo): {np.min(np.abs(autovalores)):.6f}")
        
        # Análisis de convergencia
        if radio < 1:
            print("✅ Radio espectral < 1 - Convergencia garantizada")
        elif radio == 1:
            print("⚠️ Radio espectral = 1 - Convergencia marginal")
        else:
            print("❌ Radio espectral > 1 - Convergencia no garantizada")
        
        # Información adicional
        autovalores_reales = np.real(autovalores)
        autovalores_imaginarios = np.imag(autovalores)
        
        print(f"\nInformación de autovalores:")
        print(f"  - Autovalores reales: {np.sum(np.abs(autovalores_imaginarios) < 1e-10)}")
        print(f"  - Autovalores complejos: {np.sum(np.abs(autovalores_imaginarios) >= 1e-10)}")
        print(f"  - Rango real: [{np.min(autovalores_reales):.3f}, {np.max(autovalores_reales):.3f}]")
        print(f"  - Rango imaginario: [{np.min(autovalores_imaginarios):.3f}, {np.max(autovalores_imaginarios):.3f}]")
        
        return radio, autovalores
    else:
        return None, None

def analizar_matriz(A):
    """Analiza propiedades de la matriz"""
    simetrica = np.allclose(A, A.T)
    autovalores = np.linalg.eigvals(A)
    
    if np.all(autovalores > 0):
        tipo = "Definida positiva"
    elif np.all(autovalores < 0):
        tipo = "Definida negativa"
    elif np.any(autovalores > 0) and np.any(autovalores < 0):
        tipo = "Indefinida (punto de silla)"
    else:
        tipo = "Semidefinida"
    
    # Calcular radio espectral
    radio_espectral = np.max(np.abs(autovalores))
    
    return {
        "Simetrica": simetrica,
        "Tipo de matriz": tipo,
        "Diagonal dominante": es_diagonalmente_dominante(A),
        "Radio espectral": f"{radio_espectral:.6f}"
    }