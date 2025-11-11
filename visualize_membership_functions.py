import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from anfis import ANFISModel
import os
import argparse


def visualize_membership_functions(n_memb=2, dataset='all'):
    """Rysuje funkcje przynależności ANFIS dla wybranych cech."""
    print(f"\n📈 Wizualizacja MF: dataset={dataset}, n_memb={n_memb}")

    model_path = f"models/anfis_{dataset}_best_{n_memb}memb.weights.h5"
    if not os.path.exists(model_path):
        print(f"✗ Model {model_path} nie istnieje!")
        return

    # Ładowanie danych (dla zakresów)
    if dataset == "concrete":
        X_train = np.load("data/concrete-strength/X_train.npy")
        feature_names = [
            "Cement", "Blast furnace slag", "Fly ash", "Water",
            "Superplasticizer", "Coarse aggregate", "Fine aggregate", "Age"
        ]
        important_features = list(range(min(6, X_train.shape[1])))
    else:
        # wine datasets
        try:
            if dataset == "all":
                X_train = np.load("data/X_train.npy")
            else:
                X_train = np.load(f"data/X_train_{dataset}.npy")
        except Exception:
            print(f"⚠️ Nie znaleziono danych dla {dataset}, pomijam.")
            return

        feature_names = [
            "Fixed acidity", "Volatile acidity", "Citric acid", "Residual sugar",
            "Chlorides", "Free SO₂", "Total SO₂", "Density",
            "pH", "Sulphates", "Alcohol"
        ]
        important_features = [10, 1, 8, 9, 0, 7]

    n_features = X_train.shape[1]
    important_features = [f for f in important_features if f < n_features]

    # Inicjalizacja i pobranie parametrów MF
    anfis_model = ANFISModel(n_input=n_features, n_memb=n_memb, batch_size=32)
    anfis_model.model.load_weights(model_path)
    anfis_model.update_weights()
    centers, sigmas = anfis_model.get_membership_functions()

    # Zakres danych dynamicznie (±15% margines)
    mins, maxs = X_train.min(axis=0), X_train.max(axis=0)
    margins = (maxs - mins) * 0.15

    # Liczba subplotów dopasowana automatycznie
    n_cols = 3
    n_rows = int(np.ceil(len(important_features) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 5 * n_rows))
    axes = axes.flatten()

    colors = ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e', '#9467bd', '#8c564b']

    for idx, feat_idx in enumerate(important_features):
        ax = axes[idx]
        x_range = np.linspace(mins[feat_idx] - margins[feat_idx],
                              maxs[feat_idx] + margins[feat_idx], 400)
        c = centers[:, feat_idx]
        s = sigmas[:, feat_idx]

        for i in range(n_memb):
            mu = np.exp(-((x_range - c[i]) ** 2) / (2 * s[i] ** 2))
            ax.plot(x_range, mu, color=colors[i % len(colors)],
                    linewidth=2, label=f'MF {i+1}')

        fname = feature_names[feat_idx] if feat_idx < len(feature_names) else f"Feature {feat_idx}"
        ax.set_title(fname, fontsize=12)
        ax.set_xlabel('Feature value')
        ax.set_ylabel('Membership μ(x)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([-0.05, 1.05])

    # Układ i zapis
    for j in range(len(important_features), len(axes)):
        axes[j].axis('off')

    plt.suptitle(f"ANFIS Membership Functions ({dataset}, {n_memb} MF)", fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    os.makedirs("results", exist_ok=True)
    out_path = f"results/membership_functions_{dataset}_{n_memb}memb.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Zapisano wykres: {out_path}")


# ===========================================================
# GŁÓWNY BLOK
# ===========================================================
if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["all"], choices=["concrete", "all", "red", "white"])
    parser.add_argument("--memb", nargs="+", type=int, default=[2, 3])
    args = parser.parse_args()

    for dataset in args.datasets:
        for n_memb in args.memb:
            try:
                visualize_membership_functions(n_memb, dataset)
            except Exception as e:
                print(f"✗ Błąd dla dataset={dataset}, n_memb={n_memb}: {e}")

    print("\n✓ Wizualizacja MF zakończona!")
