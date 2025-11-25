import numpy as np
from logic.equations import F, Jacobiano

def newton_raphson_step_numpy(X, malla):
    """Una iteración del método Newton-Raphson usando numpy.linalg.solve"""
    Fx = F(X, malla)
    Jx = Jacobiano(X)
    delta_X = np.linalg.solve(Jx, -Fx)
    return X + delta_X, np.linalg.norm(delta_X, np.inf)
