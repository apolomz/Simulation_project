import numpy as np
from config import FILAS, COLUMNAS

def crear_mascara_solidos():
    """Crea una máscara de sólidos con dos bloques"""
    malla = np.zeros((FILAS, COLUMNAS), dtype=bool)
    
    # Bloque 1: Centro, pegado al borde inferior
    # Altura: 2, Ancho: 10, Posición: centro horizontal, pegado al borde inferior
    centro_x = COLUMNAS // 2
    inicio_x = centro_x - 5  # 10 de ancho centrado
    malla[FILAS-3:FILAS-1, inicio_x:inicio_x+10] = True
    
    # Bloque 2: Esquina superior derecha
    # Altura: 2, Ancho: 15, Posición: esquina superior derecha
    malla[1:3, COLUMNAS-16:COLUMNAS-1] = True
    
    return malla

def inicializar_malla():
    """Inicializa la malla física con condiciones de frontera y sólidos"""
    malla = np.empty((FILAS, COLUMNAS))
    
    # Condiciones de frontera básicas
    malla[0, :] = 0              # Borde superior
    malla[-1, :] = 0             # Borde inferior
    malla[:, -1] = 0             # Borde derecho
    malla[1:6, 0] = 1            # Entrada del fluido
    
    # Aplicar máscara de sólidos (velocidad = 0 en sólidos)
    mascara_solidos = crear_mascara_solidos()
    malla[mascara_solidos] = 0
    
    return malla

def obtener_dimensiones():
    """Retorna las dimensiones de la malla interna"""
    nx = FILAS - 2
    ny = COLUMNAS - 2
    N = nx * ny
    return nx, ny, N

def idx(i, j, ny):
    """Convierte índices 2D a índice 1D"""
    return i * ny + j

def es_solido(i, j):
    """Verifica si la posición (i,j) es un sólido"""
    mascara = crear_mascara_solidos()
    # Convertir índices de malla interna a índices de malla completa
    return mascara[i+1, j+1]  # +1 porque la malla interna empieza en [1,1]
