"""
Aplicação do K-Means nos pixels da imagem HSI.

Por que K-Means para imagens?
- Agrupa pixels similares em k clusters
- Cada pixel recebe a cor média do seu cluster
- Resultado: imagem com exatamente k cores distintas

Por que medir a Inércia (WCSS)?
- Soma das distâncias quadráticas de cada pixel ao centroide
- Diminui conforme k cresce
- O "cotovelo" indica onde adicionar mais clusters
  não traz ganho significativo -> k ótimo
"""

import numpy as np
from sklearn.cluster import KMeans


def pixels_para_matriz(image_hsi: np.ndarray) -> np.ndarray:
    """
    Transforma a imagem (H, W, 3) em matriz (N, 3).
    O K-Means espera uma linha por amostra.
    """
    h, w, c = image_hsi.shape
    return image_hsi.reshape(-1, c)   # (H*W, 3)


def aplicar_kmeans(pixels: np.ndarray, k: int, seed: int = 42):
    """
    Roda o K-Means para um valor de k.

    Parâmetros
    ----------
    pixels : matriz (N, canais)
    k      : número de clusters
    seed   : semente para reprodutibilidade

    Retorna
    -------
    labels    : rótulo (0..k-1) de cada pixel  -> shape (N,)
    centroids : centros dos clusters            -> shape (k, canais)
    inertia   : WCSS (menor = mais compacto)
    """
    if not isinstance(k, int) or k < 1:
        raise ValueError(f"k deve ser um inteiro positivo. Recebido: {k}")

    km = KMeans(
        n_clusters=k,
        init='k-means++',  # inicialização esperta: evita mínimos locais ruins
        n_init=10,          # tenta 10 inicializações e pega a melhor
        max_iter=300,
        random_state=seed
    )

    labels = km.fit_predict(pixels)

    return labels, km.cluster_centers_, km.inertia_


def reconstruir_imagem(labels: np.ndarray,
                       centroids: np.ndarray,
                       shape_original: tuple) -> np.ndarray:
    """
    Substitui cada pixel pelo centroide do seu cluster.
    Resultado: imagem com k cores.

    shape_original : (altura, largura, canais)
    """
    img_rec = centroids[labels]      # (N, canais) – cada pixel vira o centroide
    return img_rec.reshape(shape_original)