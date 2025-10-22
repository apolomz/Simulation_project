import numpy as np
from config import FILAS, COLUMNAS  # Cambiar de ..config a config

def crear_mascara_solidos():
    """Crea una máscara de sólidos rectangulares en la malla
    
    Obstáculos:
    1. Rectángulo 2x10 en la parte inferior, mitad izquierda-derecha
    2. Rectángulo 2x15 en la parte superior derecha, esquina
    """
    mascara = np.zeros((FILAS, COLUMNAS), dtype=bool)
    
    # Obstáculo 1: Rectángulo 2x10 en la parte inferior, centrado
    altura_obstaculo1 = 2
    ancho_obstaculo1 = 10
    fila_inicio1 = FILAS - altura_obstaculo1 - 1  # Una fila arriba del borde inferior
    col_inicio1 = (COLUMNAS - ancho_obstaculo1) // 2  # Centrado horizontalmente
    
    for i in range(fila_inicio1, fila_inicio1 + altura_obstaculo1):
        for j in range(col_inicio1, col_inicio1 + ancho_obstaculo1):
            if 0 <= i < FILAS and 0 <= j < COLUMNAS:
                mascara[i, j] = True
    
    # Obstáculo 2: Rectángulo 2x15 en la esquina superior derecha
    altura_obstaculo2 = 2
    ancho_obstaculo2 = 15
    fila_inicio2 = 1  # Una fila abajo del borde superior
    col_inicio2 = COLUMNAS - ancho_obstaculo2 - 1  # Una columna antes del borde derecho
    
    for i in range(fila_inicio2, fila_inicio2 + altura_obstaculo2):
        for j in range(col_inicio2, col_inicio2 + ancho_obstaculo2):
            if 0 <= i < FILAS and 0 <= j < COLUMNAS:
                mascara[i, j] = True
    
    return mascara

def inicializar_malla():
    """Inicializa la malla física con condiciones de frontera y obstáculos"""
    malla = np.empty((FILAS, COLUMNAS))
    
    # Condiciones de frontera
    malla[0, :] = 0              # Borde superior - velocidad inicial 1
    malla[-1, :] = 0             # Borde inferior
    malla[:, -1] = 0             # Borde derecho
    malla[1:6, 0] = 1            # Entrada del fluido
    
    # Crear máscara de sólidos
    mascara_solidos = crear_mascara_solidos()
    
    # Aplicar condiciones de sólidos (velocidad 0 en obstáculos)
    malla[mascara_solidos] = 0
    
    return malla, mascara_solidos

def obtener_dimensiones():
    """Retorna las dimensiones de la malla interna"""
    nx = FILAS - 2  # 5
    ny = COLUMNAS - 2  # 50
    N = nx * ny  # 250
    return nx, ny, N

def idx(i, j, ny):
    """Convierte índices 2D a índice 1D"""
    return i * ny + j

def visualizar_malla_con_obstaculos(malla, mascara_solidos):
    """Visualiza la malla con los obstáculos marcados"""
    import matplotlib.pyplot as plt
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Visualizar la malla de velocidades
    im1 = ax1.imshow(malla, cmap='viridis', aspect='auto')
    ax1.set_title('Malla de Velocidades')
    ax1.set_xlabel('Columna')
    ax1.set_ylabel('Fila')
    plt.colorbar(im1, ax=ax1)
    
    # Visualizar la máscara de obstáculos
    im2 = ax2.imshow(mascara_solidos.astype(int), cmap='Reds', aspect='auto')
    ax2.set_title('Máscara de Obstáculos')
    ax2.set_xlabel('Columna')
    ax2.set_ylabel('Fila')
    plt.colorbar(im2, ax=ax2)
    
    plt.tight_layout()
    plt.show()
    
    # Información de los obstáculos
    print(f"Dimensiones de la malla: {FILAS} x {COLUMNAS}")
    print(f"Obstáculo 1 (inferior): 2x10 centrado")
    print(f"Obstáculo 2 (superior derecho): 2x15 en esquina")
    print(f"Total de puntos sólidos: {np.sum(mascara_solidos)}")
    print(f"Porcentaje de obstáculos: {100 * np.sum(mascara_solidos) / (FILAS * COLUMNAS):.1f}%")