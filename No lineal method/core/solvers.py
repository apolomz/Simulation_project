import numpy as np
from scipy.sparse.linalg import spsolve
from core.equations import F, Jacobiano  # Cambiar de .equations a core.equations

def gauss_elimination(A, b):
    """Eliminación de Gauss manual"""
    A = A.copy().astype(float)
    b = b.copy().astype(float)
    n = len(b)

    for i in range(n):
        for j in range(i+1, n):
            factor = A[j, i] / A[i, i]
            A[j, i:] -= factor * A[i, i:]
            b[j] -= factor * b[i]

    x = np.zeros(n)
    for i in reversed(range(n)):
        x[i] = (b[i] - np.dot(A[i, i+1:], x[i+1:])) / A[i, i]

    return x

def newton_raphson_step_spsolve(X, malla, mascara_solidos=None):
    """Un paso del método Newton-Raphson usando spsolve"""
    Fx = F(X, malla, mascara_solidos)
    Jx = Jacobiano(X, mascara_solidos, sparse=True)
    delta_X = spsolve(Jx, -Fx)
    return X + delta_X, np.linalg.norm(delta_X, np.inf)

def newton_raphson_step_gauss(X, malla, mascara_solidos=None):
    """Un paso del método Newton-Raphson usando eliminación de Gauss"""
    Fx = F(X, malla, mascara_solidos)
    Jx = Jacobiano(X, mascara_solidos, sparse=False)
    delta_X = gauss_elimination(Jx, -Fx)
    return X + delta_X, np.linalg.norm(delta_X, np.inf)