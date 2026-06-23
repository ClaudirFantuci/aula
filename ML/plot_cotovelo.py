"""
Gera o gráfico do cotovelo a partir do CSV de inércias.
"""

import matplotlib.pyplot as plt
import pandas as pd

# ── Configurações ──────────────────────────────────────────────
CSV_PATH = 'results/inercias.csv'
OUTPUT_PATH = 'results/grafico_cotovelo.png'
# ───────────────────────────────────────────────────────────────


def plot_elbow_curve(csv_path, output_path):
    """Plota o gráfico do cotovelo."""
    
    # Lê o CSV
    df = pd.read_csv(csv_path)
    k_values = df['k'].tolist()
    inertias = df['inertia'].tolist()
    
    # Cria a figura
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plota a curva
    ax.plot(k_values, inertias, 'o-', linewidth=2, markersize=8,
            color='steelblue')
    ax.fill_between(k_values, inertias, alpha=0.2, color='steelblue')
    
    ax.set_xlabel('Número de Clusters (k)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Inércia (WCSS)', fontsize=12, fontweight='bold')
    ax.set_title('Método do Cotovelo - K-Means em Espaço HSI',
                 fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Inverte o eixo Y (inércia maior embaixo, menor em cima)
    ax.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nGráfico salvo em: {output_path}\n")
    plt.show()


if __name__ == '__main__':
    plot_elbow_curve(CSV_PATH, OUTPUT_PATH)