import numpy as np
import time
from config import TOLERANCIA, MAX_ITERACIONES_LINEALES

def LU_decomposition(A):
    """Factorización LU de una matriz A"""
    n = A.shape[0]
    L = np.zeros_like(A)
    U = np.zeros_like(A)

    for i in range(n):
        for j in range(i, n):
            U[i, j] = A[i, j] - np.dot(L[i, :i], U[:i, j])
        for j in range(i+1, n):
            L[j, i] = (A[j, i] - np.dot(L[j, :i], U[:i, i])) / U[i, i]
        L[i, i] = 1

    return L, U

def solve_LU(A, B):
    """Resuelve Ax = B usando factorización LU"""
    L, U = LU_decomposition(A)
    n = len(B)

    # Sustitución hacia adelante: L * Y = B
    Y = np.zeros(n)
    for i in range(n):
        Y[i] = B[i] - np.dot(L[i, :i], Y[:i])

    # Sustitución hacia atrás: U * X = Y
    X = np.zeros(n)
    for i in range(n-1, -1, -1):
        X[i] = (Y[i] - np.dot(U[i, i+1:], X[i+1:])) / U[i, i]

    return X

def Jacobi(A, b, max_iter=None):
    """Método de Jacobi para resolver Ax = b usando matriz de iteración"""
    start_time = time.perf_counter()
    
    if max_iter is None:
        max_iter = MAX_ITERACIONES_LINEALES
    
    n = b.shape[0]
    x = np.zeros(n)
    
    # Crear matriz de iteración de Jacobi: M = I - D^(-1) * A
    D = np.diag(np.diag(A))
    try:
        D_inv = np.linalg.inv(D)
        M = np.eye(n) - D_inv @ A
        c = D_inv @ b
    except np.linalg.LinAlgError:
        print("❌ No se puede invertir la matriz diagonal para Jacobi")
        return x

    for k in range(max_iter):
        x_new = M @ x + c
        
        if np.linalg.norm(x_new - x) < TOLERANCIA:
            end_time = time.perf_counter()
            print(f"Jacobi convergió en {k+1} iteraciones")
            print(f"Tiempo total: {end_time - start_time:.4f} segundos")
            break
            
        x = x_new

    return x

def GaussSeidel(A, b, x0=None, max_iter=None, tol=None):
    """Método de Gauss-Seidel para resolver Ax = b usando matriz de iteración"""
    start_time = time.perf_counter()
    
    if max_iter is None:
        max_iter = MAX_ITERACIONES_LINEALES
    if tol is None:
        tol = TOLERANCIA
    
    n = len(b)
    x = np.zeros(n) if x0 is None else x0.copy()
    
    # Crear matrices para Gauss-Seidel: A = D + L + U
    D = np.diag(np.diag(A))
    L = np.tril(A, k=-1)
    U = np.triu(A, k=1)
    
    try:
        D_L_inv = np.linalg.inv(D + L)
        M = -D_L_inv @ U
        c = D_L_inv @ b
    except np.linalg.LinAlgError:
        print("❌ No se puede invertir la matriz (D+L) para Gauss-Seidel")
        return x

    for k in range(max_iter):
        x_new = M @ x + c
        
        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            end_time = time.perf_counter()
            print(f"Gauss-Seidel convergió en {k+1} iteraciones")
            print(f"Tiempo total: {end_time - start_time:.4f} segundos")
            return x_new

        x = x_new

    end_time = time.perf_counter()
    print("Gauss-Seidel no convergió dentro del número máximo de iteraciones")
    print(f"Tiempo total: {end_time - start_time:.4f} segundos")
    return x

def Richardson(A, b, alpha=0.1, max_iter=None):
    """Método de Richardson para resolver Ax = b usando matriz de iteración"""
    start_time = time.perf_counter()
    
    if max_iter is None:
        max_iter = MAX_ITERACIONES_LINEALES
    
    n = b.shape[0]
    x = np.zeros(n)
    
    # Crear matriz de iteración de Richardson: M = I - α * A
    M = np.eye(n) - alpha * A
    c = alpha * b

    for k in range(max_iter):
        x_new = M @ x + c
        
        if np.linalg.norm(x_new - x) < TOLERANCIA:
            end_time = time.perf_counter()
            print(f"Richardson convergió en {k+1} iteraciones")
            print(f"Tiempo total: {end_time - start_time:.4f} segundos")
            break
            
        x = x_new

    return x

def crear_matrices_iteracion(A):
    """Crea las matrices de iteración para diferentes métodos"""
    n = A.shape[0]
    
    # Matriz de iteración Jacobi: M = I - D^(-1) * A
    D = np.diag(np.diag(A))
    try:
        D_inv = np.linalg.inv(D)
        M_jacobi = np.eye(n) - D_inv @ A
    except np.linalg.LinAlgError:
        M_jacobi = None
    
    # Matriz de iteración Gauss-Seidel: M = -(D+L)^(-1) * U
    D = np.diag(np.diag(A))
    L = np.tril(A, k=-1)
    U = np.triu(A, k=1)
    try:
        D_L_inv = np.linalg.inv(D + L)
        M_gauss_seidel = -D_L_inv @ U
    except np.linalg.LinAlgError:
        M_gauss_seidel = None
    
    # Matriz de iteración Richardson: M = I - α * A
    alpha = 0.1
    M_richardson = np.eye(n) - alpha * A
    
    return {
        'original': A,
        'jacobi': M_jacobi,
        'gauss_seidel': M_gauss_seidel,
        'richardson': M_richardson
    }

def comparar_matrices(A):
    """Compara la matriz original con las matrices de iteración"""
    matrices = crear_matrices_iteracion(A)
    
    print("="*70)
    print("COMPARACIÓN: MATRIZ ORIGINAL vs MATRICES DE ITERACIÓN")
    print("="*70)
    
    print(f"\n1. MATRIZ ORIGINAL (Jacobiano):")
    print(f"   Dimensiones: {matrices['original'].shape}")
    print(f"   Elementos no cero: {np.count_nonzero(matrices['original'])}")
    print(f"   Densidad: {np.count_nonzero(matrices['original']) / matrices['original'].size:.4f}")
    
    if matrices['jacobi'] is not None:
        print(f"\n2. MATRIZ DE ITERACIÓN JACOBI:")
        print(f"   Dimensiones: {matrices['jacobi'].shape}")
        print(f"   Elementos no cero: {np.count_nonzero(matrices['jacobi'])}")
        print(f"   Densidad: {np.count_nonzero(matrices['jacobi']) / matrices['jacobi'].size:.4f}")
        print(f"   Fórmula: M = I - D^(-1) * A")
    else:
        print(f"\n2. MATRIZ DE ITERACIÓN JACOBI: ❌ No se pudo calcular")
    
    if matrices['gauss_seidel'] is not None:
        print(f"\n3. MATRIZ DE ITERACIÓN GAUSS-SEIDEL:")
        print(f"   Dimensiones: {matrices['gauss_seidel'].shape}")
        print(f"   Elementos no cero: {np.count_nonzero(matrices['gauss_seidel'])}")
        print(f"   Densidad: {np.count_nonzero(matrices['gauss_seidel']) / matrices['gauss_seidel'].size:.4f}")
        print(f"   Fórmula: M = -(D+L)^(-1) * U")
    else:
        print(f"\n3. MATRIZ DE ITERACIÓN GAUSS-SEIDEL: ❌ No se pudo calcular")
    
    print(f"\n4. MATRIZ DE ITERACIÓN RICHARDSON:")
    print(f"   Dimensiones: {matrices['richardson'].shape}")
    print(f"   Elementos no cero: {np.count_nonzero(matrices['richardson'])}")
    print(f"   Densidad: {np.count_nonzero(matrices['richardson']) / matrices['richardson'].size:.4f}")
    print(f"   Fórmula: M = I - α * A (α=0.1)")
    
    return matrices

def gradient_descent(A, b, max_iter=None, alpha=0.01):
    """Método del gradiente descendiente para resolver Ax = b"""
    start_time = time.perf_counter()
    
    if max_iter is None:
        max_iter = MAX_ITERACIONES_LINEALES
    
    n = len(b)
    x = np.zeros(n)
    AT = A.T
    ATb = AT @ b
    
    for k in range(max_iter):
        grad = AT @ (A @ x - b)
        x_new = x - alpha * grad
        
        if np.linalg.norm(x_new - x) < TOLERANCIA:
            end_time = time.perf_counter()
            print(f"Gradiente Descendiente convergió en {k+1} iteraciones")
            print(f"Tiempo total: {end_time - start_time:.4f} segundos")
            return x_new
            
        x = x_new
    
    end_time = time.perf_counter()
    print("Gradiente Descendiente no convergió")
    print(f"Tiempo total: {end_time - start_time:.4f} segundos")
    return x

def conjugate_gradient(A, b, max_iter=None):
    """Método del gradiente conjugado para resolver Ax = b"""
    start_time = time.perf_counter()
    
    if max_iter is None:
        max_iter = MAX_ITERACIONES_LINEALES
    
    n = len(b)
    x = np.zeros(n)
    r = b - A @ x
    p = r.copy()
    
    for k in range(max_iter):
        Ap = A @ p
        alpha = np.dot(r, r) / np.dot(p, Ap)
        x_new = x + alpha * p
        r_new = r - alpha * Ap
        
        if np.linalg.norm(r_new) < TOLERANCIA:
            end_time = time.perf_counter()
            print(f"Gradiente Conjugado convergió en {k+1} iteraciones")
            print(f"Tiempo total: {end_time - start_time:.4f} segundos")
            return x_new
            
        beta = np.dot(r_new, r_new) / np.dot(r, r)
        p = r_new + beta * p
        r = r_new
        x = x_new
    
    end_time = time.perf_counter()
    print("Gradiente Conjugado no convergió")
    print(f"Tiempo total: {end_time - start_time:.4f} segundos")
    return x