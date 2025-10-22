import numpy as np
from config import FILAS, COLUMNAS, VY_CONSTANTE

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
    """Inicializa la malla con condiciones de frontera y obstáculos"""
    vx = np.zeros((FILAS, COLUMNAS))
    vy = np.zeros((FILAS, COLUMNAS))
    
    # Condiciones de frontera
    vx[0, :] = 0; vx[-1, :] = 0
    vy[0, :] = VY_CONSTANTE; vy[-1, :] = VY_CONSTANTE
    vx[1:6, 0] = 1
    vx[0, 0] = 0
    vx[6, 0] = 0
    vy[:, 0] = VY_CONSTANTE
    vx[:, -1] = 0
    vy[:, -1] = VY_CONSTANTE
    
    # Crear máscara de sólidos
    mascara_solidos = crear_mascara_solidos()
    
    # Condición inicial en puntos internos
    for i in range(1, FILAS-1):
        for j in range(1, COLUMNAS-1):
            # vx[i, j] = max(0, 1 - (j / (cols-2)))  # Disminución lineal
            vx[i, j] = 0.5
            vy[i, j] = VY_CONSTANTE
    
    # Aplicar condiciones de sólidos (velocidad 0 en obstáculos)
    vx[mascara_solidos] = 0
    vy[mascara_solidos] = 0
    
    return vx, vy, mascara_solidos

def obtener_dimensiones():
    """Retorna las dimensiones de la malla interna"""
    nx = FILAS - 2
    ny = COLUMNAS - 2
    N = nx * ny
    return nx, ny, N

def visualizar_malla_con_obstaculos(vx, vy, mascara_solidos):
    """Visualiza la malla con los obstáculos marcados"""
    import matplotlib.pyplot as plt
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Visualizar vx
    im1 = ax1.imshow(vx, cmap='viridis', aspect='auto')
    ax1.set_title('Velocidad vx')
    ax1.set_xlabel('Columna')
    ax1.set_ylabel('Fila')
    plt.colorbar(im1, ax=ax1)
    
    # Visualizar vy
    im2 = ax2.imshow(vy, cmap='plasma', aspect='auto')
    ax2.set_title('Velocidad vy')
    ax2.set_xlabel('Columna')
    ax2.set_ylabel('Fila')
    plt.colorbar(im2, ax=ax2)
    
    # Visualizar la máscara de obstáculos
    im3 = ax3.imshow(mascara_solidos.astype(int), cmap='Reds', aspect='auto')
    ax3.set_title('Máscara de Obstáculos')
    ax3.set_xlabel('Columna')
    ax3.set_ylabel('Fila')
    plt.colorbar(im3, ax=ax3)
    
    # Visualizar magnitud de velocidad
    magnitud = np.sqrt(vx**2 + vy**2)
    im4 = ax4.imshow(magnitud, cmap='hot', aspect='auto')
    ax4.set_title('Magnitud de Velocidad')
    ax4.set_xlabel('Columna')
    ax4.set_ylabel('Fila')
    plt.colorbar(im4, ax=ax4)
    
    plt.tight_layout()
    plt.show()
    
    # Información de los obstáculos
    print(f"Dimensiones de la malla: {FILAS} x {COLUMNAS}")
    print(f"Obstáculo 1 (inferior): 2x10 centrado")
    print(f"Obstáculo 2 (superior derecho): 2x15 en esquina")
    print(f"Total de puntos sólidos: {np.sum(mascara_solidos)}")
    print(f"Porcentaje de obstáculos: {100 * np.sum(mascara_solidos) / (FILAS * COLUMNAS):.1f}%")
