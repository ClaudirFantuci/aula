"""
Conversões RGB <-> HSI

Por que HSI?
- H (Hue): representa a cor pura
- S (Saturation): intensidade da cor
- I (Intensity): brilho médio
- Separar cor de brilho torna o K-Means mais eficiente
  pois agrupa pixels pela COR REAL, ignorando iluminação
"""

import numpy as np


def rgb_to_hsi(image_rgb: np.ndarray) -> np.ndarray:
    """
    Converte imagem RGB para HSI.
    
    Entrada : array (H, W, 3) com valores em [0, 1]
    Saída   : array (H, W, 3) com H,S,I em [0, 1]
              (H é normalizado de 0-360° para 0-1)
    """

    img = image_rgb.astype(np.float64)

    # Normaliza para [0,1] caso venha em [0,255]
    if img.max() > 1.0:
        img = img / 255.0

    R = img[:, :, 0]
    G = img[:, :, 1]
    B = img[:, :, 2]

    # --- Intensidade ---
    # Média aritmética dos três canais
    I = (R + G + B) / 3.0

    # --- Saturação ---
    # Mede o quão "pura" é a cor (0 = cinza, 1 = cor pura)
    min_rgb  = np.minimum(np.minimum(R, G), B)
    soma_rgb = R + G + B

    # Onde soma == 0 (pixel preto), saturação = 0
    S = np.where(soma_rgb == 0, 0.0, 1.0 - (3.0 * min_rgb / soma_rgb))

    # --- Matiz ---
    # Ângulo no círculo de cores usando arco-cosseno
    num   = 0.5 * ((R - G) + (R - B))
    den   = np.sqrt((R - G)**2 + (R - B) * (G - B)) + 1e-10
    theta = np.degrees(np.arccos(np.clip(num / den, -1.0, 1.0)))

    # Se B > G o ângulo está na metade inferior do círculo
    H = np.where(B <= G, theta, 360.0 - theta)

    # Normaliza H para [0, 1]
    H = H / 360.0

    return np.stack([H, S, I], axis=-1)


def hsi_to_rgb(image_hsi: np.ndarray) -> np.ndarray:
    """
    Converte imagem HSI para RGB.
    
    A conversão inversa usa fórmulas diferentes para cada um
    dos 3 setores do círculo de cores (0-120°, 120-240°, 240-360°).
    
    Entrada : array (H, W, 3) com H,S,I em [0, 1]
    Saída   : array (H, W, 3) com RGB em [0, 1]
    """

    img = image_hsi.copy().astype(np.float64)

    H = img[:, :, 0] * 360.0   # de volta para graus
    S = img[:, :, 1]
    I = img[:, :, 2]

    R = np.zeros_like(H)
    G = np.zeros_like(H)
    B = np.zeros_like(H)

    # Setor 1: 0° <= H < 120°
    m1 = (H >= 0) & (H < 120)
    if np.any(m1):
        h = np.radians(H[m1])
        s, i = S[m1], I[m1]
        B[m1] = i * (1 - s)
        R[m1] = i * (1 + s * np.cos(h) / np.cos(np.radians(60) - h))
        G[m1] = 3 * i - (R[m1] + B[m1])

    # Setor 2: 120° <= H < 240°
    m2 = (H >= 120) & (H < 240)
    if np.any(m2):
        h = np.radians(H[m2] - 120)
        s, i = S[m2], I[m2]
        R[m2] = i * (1 - s)
        G[m2] = i * (1 + s * np.cos(h) / np.cos(np.radians(60) - h))
        B[m2] = 3 * i - (R[m2] + G[m2])

    # Setor 3: 240° <= H <= 360°
    m3 = (H >= 240) & (H <= 360)
    if np.any(m3):
        h = np.radians(H[m3] - 240)
        s, i = S[m3], I[m3]
        G[m3] = i * (1 - s)
        B[m3] = i * (1 + s * np.cos(h) / np.cos(np.radians(60) - h))
        R[m3] = 3 * i - (G[m3] + B[m3])

    # Limita ao intervalo válido (erros numéricos podem sair de [0,1])
    return np.clip(np.stack([R, G, B], axis=-1), 0.0, 1.0)