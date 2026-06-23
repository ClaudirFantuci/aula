"""
Pipeline principal:
1. Carrega imagem RGB
2. Converte para HSI
3. Aplica K-Means para o k especificado
4. Salva imagem reconstruída em RGB

Configure K como um número inteiro único: K = 5
"""

import os
import numpy as np
from PIL import Image

from src.color_conversion  import rgb_to_hsi, hsi_to_rgb
from src.kmeans_clustering import aplicar_kmeans, reconstruir_imagem

# ── Configurações ──────────────────────────────────────────────
IMAGE_PATH = 'images/imagem.jpg'
OUTPUT_DIR = 'results'

K = 5  # número inteiro único

SEED       = 42
MAX_PIXELS = 150_000
# ───────────────────────────────────────────────────────────────


def carregar_imagem(path: str) -> np.ndarray:
    """Carrega imagem como float64 em [0, 1]."""
    img = Image.open(path).convert('RGB')
    return np.array(img, dtype=np.float64) / 255.0


def redimensionar(img: np.ndarray, max_px: int) -> np.ndarray:
    """Reduz a imagem se tiver mais de max_px pixels."""
    h, w = img.shape[:2]
    if h * w <= max_px:
        return img

    fator   = np.sqrt(max_px / (h * w))
    novo_h  = int(h * fator)
    novo_w  = int(w * fator)
    img_pil = Image.fromarray((img * 255).astype(np.uint8))
    img_pil = img_pil.resize((novo_w, novo_h), Image.LANCZOS)

    return np.array(img_pil, dtype=np.float64) / 255.0


def salvar_imagem(img_float: np.ndarray, path: str) -> None:
    """Salva array float [0,1] como PNG."""
    arr = (img_float * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(arr).save(path)


def main():
    if not isinstance(K, int):
        raise ValueError(f"K deve ser um inteiro. Recebido: {type(K)}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"K = {K}")

    # 1. Carrega e prepara a imagem
    print(f"\nCarregando: {IMAGE_PATH}")
    img_rgb = carregar_imagem(IMAGE_PATH)
    img_rgb = redimensionar(img_rgb, MAX_PIXELS)
    print(f"Shape final: {img_rgb.shape}")

    # 2. RGB → HSI
    print("\nConvertendo RGB → HSI ...")
    img_hsi = rgb_to_hsi(img_rgb)

    # 3. Aplica K-Means
    pixels  = img_hsi.reshape(-1, img_hsi.shape[2])
    labels, centroids, inertia = aplicar_kmeans(pixels, K, SEED)

    # 4. Reconstrói a imagem HSI e converte para RGB
    img_hsi_rec = reconstruir_imagem(labels, centroids, img_hsi.shape)
    img_rgb_rec = hsi_to_rgb(img_hsi_rec)

    # 5. Salva resultado
    path_out = os.path.join(OUTPUT_DIR, f'k_{K:03d}.png')
    salvar_imagem(img_rgb_rec, path_out)


if __name__ == '__main__':
    main()