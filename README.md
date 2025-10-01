### Método de Newton-Raphson para Sistemas No Lineales ###

En cada iteración se resuelve el sistema lineal:
Se usa la función: np.linalg.solve(A, b)
de la librería numPy

Sirve para resolver un sistema lineal de la forma: 𝐴⋅𝑥=𝑏
Donde:
A es la matriz del sistema (Jacobiano(x))
x es el vector incognita(H = K^x+1)
b es el vector de resultados = -F(x)

Este sistema se consigue a partir de eliminar la inversa del Jacobiano para evitar
errores de redondeo, usando Newton-Raphson.
Por lo que se hace: delta_X = np.linalg.solve(Jx, -Fx)

-- Ejecución
python main_numpy.py