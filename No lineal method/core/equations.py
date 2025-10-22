import numpy as np
from scipy.sparse import lil_matrix
from .mesh import idx, obtener_dimensiones
from config import VY_CONSTANTE  # Cambiar de ..config a config

nx, ny, N = obtener_dimensiones()

def F(X, malla, mascara_solidos=None):
    """Función F para el sistema no lineal"""
    Fx = np.zeros(N)
    for i in range(nx):
        for j in range(ny):
            k = idx(i, j, ny)
            
            # Si es un punto sólido, la velocidad es 0
            if mascara_solidos is not None and mascara_solidos[i+1, j+1]:
                Fx[k] = X[k]  # F = 0 cuando v = 0
                continue
                
            v = X[k]
            
            # Valores vecinos
            vr = X[idx(i+1, j, ny)] if i+1 < nx else malla[i+2, j+1]
            vl = X[idx(i-1, j, ny)] if i-1 >= 0 else malla[i, j+1]
            vu = X[idx(i, j+1, ny)] if j+1 < ny else malla[i+1, j+2]
            vd = X[idx(i, j-1, ny)] if j-1 >= 0 else malla[i+1, j]

            Fx[k] = v - (1/4) * (vr + vl + vu + vd
                          - 0.5 * v * (vr - vl)
                          - 0.5 * VY_CONSTANTE * (vu - vd))
    return Fx

def Jacobiano(X, mascara_solidos=None, sparse=True):
    """Calcula el Jacobiano del sistema"""
    J = lil_matrix((N, N))
    
    for i in range(nx):
        for j in range(ny):
            k = idx(i, j, ny)
            
            # Si es un punto sólido, la derivada es 1 (v = 0)
            if mascara_solidos is not None and mascara_solidos[i+1, j+1]:
                J[k, k] = 1.0
                continue
                
            v = X[k]
            vr = X[idx(i+1, j, ny)] if i+1 < nx else 0
            vl = X[idx(i-1, j, ny)] if i-1 >= 0 else 0

            J[k, k] = 1 + (1/8) * (vr - vl)

            if i+1 < nx:
                J[k, idx(i+1, j, ny)] = -0.25 + (1/8) * v

            if i-1 >= 0:
                J[k, idx(i-1, j, ny)] = -0.25 - (1/8) * v

            if j+1 < ny:
                J[k, idx(i, j+1, ny)] = -0.25 + (1/8) * VY_CONSTANTE

            if j-1 >= 0:
                J[k, idx(i, j-1, ny)] = -0.25 - (1/8) * VY_CONSTANTE
    
    return J.tocsr() if sparse else J.toarray()