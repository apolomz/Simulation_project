import numpy as np
from scipy.sparse import lil_matrix
from config import VY_CONSTANTE

def calculate_F(vx, mascara_solidos=None, h=1.0):
    """Calcula la función F para el sistema no lineal"""
    rows, cols = vx.shape
    F = np.zeros((rows-2, cols-2))
    
    for i in range(1, rows-1):
        for j in range(1, cols-1):
            # Si es un punto sólido, la velocidad es 0
            if mascara_solidos is not None and mascara_solidos[i, j]:
                F[i-1, j-1] = vx[i, j]  # F = 0 cuando v = 0
                continue
                
            term1 = vx[i+1,j] + vx[i-1,j] + vx[i,j+1] + vx[i,j-1]
            term2 = (h/2) * vx[i,j] * (vx[i+1,j] - vx[i-1,j])
            term3 = (h/2) * VY_CONSTANTE * (vx[i,j+1] - vx[i,j-1])
            vx_calculated = vx[i,j] - (1/4) * (term1 - term2 - term3)
            F[i-1, j-1] = vx_calculated
    return F

def calculate_Jacobian_sparse(vx, mascara_solidos=None, h=1.0):
    """Calcula el Jacobiano del sistema"""
    rows, cols = vx.shape
    n = (rows-2) * (cols-2)
    J = lil_matrix((n, n))
    vy = np.full_like(vx, VY_CONSTANTE)

    for i in range(1, rows-1):
        for j in range(1, cols-1):
            idx = (i-1)*(cols-2) + (j-1)
            
            # Si es un punto sólido, la derivada es 1 (v = 0)
            if mascara_solidos is not None and mascara_solidos[i, j]:
                J[idx, idx] = 1.0
                continue
                
            J[idx, idx] = 1 + (h/8)*(vx[i+1,j] - vx[i-1,j]) + (h/8)*vy[i,j]*(vx[i,j+1] - vx[i,j-1])

            if i > 1:
                J[idx, idx - (cols-2)] = -0.25 + (h/8)*vx[i,j]
            if i < rows-2:
                J[idx, idx + (cols-2)] = -0.25 - (h/8)*vx[i,j]
            if j > 1:
                J[idx, idx - 1] = -0.25 + (h/8)* VY_CONSTANTE
            if j < cols-2:
                J[idx, idx + 1] = -0.25 - (h/8)* VY_CONSTANTE
    return J