"""
Wizualizacja wyuczonych funkcji przynależności ANFIS
Pokazuje gaussowskie funkcje przynależności dla wybranych cech
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Wyłącza wyświetlanie okien - tylko zapis do plików
import matplotlib.pyplot as plt
import tensorflow as tf
from anfis import ANFISModel
import pickle
import os


def plot_membership_function(x, c, sigma, feature_name, n_memb, ax):
    """Rysuje funkcję przynależności gaussowską"""
    colors = ['blue', 'red', 'green', 'orange', 'purple']

    for i in range(n_memb):
        mu = np.exp(-(x - c[i])**2 / (2 * sigma[i]**2))
        ax.plot(x, mu, color=colors[i % len(colors)], linewidth=2,
                label=f'MF {i+1} (c={c[i]:.2f}, σ={sigma[i]:.2f})')

    ax.set_xlabel('Znormalizowana wartość cechy', fontsize=11)
    ax.set_ylabel('Stopień przynależności μ(x)', fontsize=11)
    ax.set_title(f'Funkcje przynależności dla: {feature_name}', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([-0.05, 1.05])


def visualize_membership_functions(n_memb=3):
    """
    Wizualizuje funkcje przynależności dla modelu ANFIS

    Args:
        n_memb: liczba funkcji przynależności (2 lub 3)
    """
    print(f"\n{'='*80}")
    print(f"Wizualizacja funkcji przynależności ANFIS ({n_memb} funkcje)")
    print(f"{'='*80}\n")

    # Sprawdź czy model istnieje
    # Prefer per-dataset model folder for wine
    candidate = os.path.join('models', 'wine-quality', f'anfis_best_{n_memb}memb.weights.h5')
    if os.path.exists(candidate):
        model_path = candidate
    else:
        model_path = f'models/anfis_best_{n_memb}memb.weights.h5'

    if not os.path.exists(model_path):
        print(f"✗ Model {model_path} nie istnieje. Pomiń wizualizację.")
        return

    # Wczytaj nazwy cech
    feature_names = [
        'fixed acidity',
        'volatile acidity',
        'citric acid',
        'residual sugar',
        'chlorides',
        'free sulfur dioxide',
        'total sulfur dioxide',
        'density',
        'pH',
        'sulphates',
        'alcohol'
    ]

    # Wczytaj dane treningowe (dla wymiaru) - prefer data/wine-quality
    if os.path.exists(os.path.join('data', 'wine-quality', 'X_train.npy')):
        X_train = np.load(os.path.join('data', 'wine-quality', 'X_train.npy'))
    else:
        X_train = np.load('data/X_train.npy')
    n_features = X_train.shape[1]

    # Stwórz model ANFIS
    print("Ładuję wytrenowany model ANFIS...")
    anfis_model = ANFISModel(
        n_input=n_features,
        n_memb=n_memb,
        batch_size=32
    )

    # Załaduj wagi
    anfis_model.model.load_weights(model_path)
    anfis_model.update_weights()

    # Pobierz parametry funkcji przynależności
    centers, sigmas = anfis_model.get_membership_functions()

    print(f"Wymiar centers: {centers.shape}")  # (n_memb, n_features)
    print(f"Wymiar sigmas: {sigmas.shape}\n")

    # Wybierz 6 najważniejszych cech do wizualizacji
    important_features = [10, 1, 8, 9, 0, 7]  # alcohol, volatile acidity, pH, sulphates, fixed acidity, density

    # Stwórz wykres
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    x_range = np.linspace(-3, 3, 300)  # Zakres dla znormalizowanych danych

    for idx, feat_idx in enumerate(important_features):
        c = centers[:, feat_idx]
        sigma = sigmas[:, feat_idx]
        feature_name = feature_names[feat_idx]

        plot_membership_function(x_range, c, sigma, feature_name, n_memb, axes[idx])

    plt.suptitle(f'Wyuczone funkcje przynależności ANFIS ({n_memb} funkcje na cechę)',
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()

    # Zapisz wykres
    output_path = f'results/membership_functions_{n_memb}memb.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Wykres zapisany: {output_path}")

    plt.close()


if __name__ == "__main__":
    os.makedirs('results', exist_ok=True)

    # Wizualizuj dla obu modeli
    for n_memb in [2, 3]:
        try:
            visualize_membership_functions(n_memb)
        except Exception as e:
            print(f"✗ Błąd dla n_memb={n_memb}: {e}")

    print("\n" + "="*80)
    print("✓ Wizualizacja funkcji przynależności zakończona!")
    print("="*80 + "\n")
