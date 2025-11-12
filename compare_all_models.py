"""
Generuje wykresy porównawcze modeli dla OBU problemów:
- Wine Quality (dataset 'all')
- Concrete Strength

Zapisuje je jako:
  - results/model_comparison_bar_wine.png
  - results/overfitting_analysis_wine.png
  - results/model_comparison_bar_concrete.png
  - results/overfitting_analysis_concrete.png

Uruchamianie:
    python3 compare_all_models.py
"""

import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# ---------------------------------------------------------------------
# FUNKCJE POMOCNICZE
# ---------------------------------------------------------------------
def load_wine_results():
    """Wczytuje wyniki dla Wine Quality (dataset 'all')."""
    paths = {
        "ANFIS (2 MF)": "results/anfis_all_2memb_results.json",
        "ANFIS (3 MF)": "results/anfis_all_3memb_results.json",
        "Neural Network": "results/nn_wine_results.json",
        "SVM": "results/svm_wine_results.json",
        "Random Forest": "results/rf_wine_results.json",
    }
    results = {}
    for name, path in paths.items():
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if name in ["Neural Network", "SVM", "Random Forest"]:
                        if "test_accuracy" in data:
                            results[name] = data
                        else:
                            print(f"Plik {path} nie zawiera 'test_accuracy' — pomijam dla Wine.")
                    else:
                        results[name] = data
            except Exception as e:
                print(f"Błąd wczytywania {path}: {e}")
        else:
            print(f"Brak pliku: {path}")
    return results


def load_concrete_results():
    """Wczytuje wyniki dla Concrete Strength."""
    paths = {
        "ANFIS (2 MF)": "results/anfis_concrete_2memb_results.json",
        "ANFIS (3 MF)": "results/anfis_concrete_3memb_results.json",
        "Neural Network": "results/nn_concrete_results.json",
        "SVM": "results/svm_concrete_results.json",
        "Random Forest": "results/rf_concrete_results.json",
    }
    results = {}
    for name, path in paths.items():
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if name in ["Neural Network", "SVM", "Random Forest"]:
                        if "test_mae" in data:
                            results[name] = data
                        else:
                            print(f"Plik {path} nie zawiera 'test_mae' — pomijam dla Concrete.")
                    else:
                        results[name] = data
            except Exception as e:
                print(f"Błąd wczytywania {path}: {e}")
        else:
            print(f"Brak pliku: {path}")
    return results


def plot_comparison_bar_chart(results, is_regression, output_path, title_suffix):
    if not results:
        print(f"Pomijam generowanie {output_path} — brak wyników.")
        return

    models = list(results.keys())
    train_vals, test_vals = [], []
    for m in models:
        res = results[m]
        if is_regression:
            train_vals.append(res.get("train_mae", np.nan))
            test_vals.append(res.get("test_mae", np.nan))
        else:
            train_vals.append(res.get("train_accuracy", np.nan) * 100)
            test_vals.append(res.get("test_accuracy", np.nan) * 100)

    x = np.arange(len(models))
    width = 0.35
    fig, ax = plt.subplots(figsize=(12, 7))

    label_train = "Train MAE" if is_regression else "Train Accuracy (%)"
    label_test = "Test MAE" if is_regression else "Test Accuracy (%)"

    bars1 = ax.bar(x - width / 2, train_vals, width, label=label_train, color="steelblue", alpha=0.8, edgecolor="black")
    bars2 = ax.bar(x + width / 2, test_vals, width, label=label_test, color="coral", alpha=0.8, edgecolor="black")

    ax.set_xlabel("Model", fontsize=14, fontweight="bold")
    ylabel = "MAE (niżej = lepiej)" if is_regression else "Dokładność (%)"
    ax.set_ylabel(ylabel, fontsize=14, fontweight="bold")
    ax.set_title(f"Porównanie modeli — {title_suffix}", fontsize=16, fontweight="bold", pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.legend(fontsize=12)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if np.isnan(height):
                continue
            text = f"{height:.2f}" if is_regression else f"{height:.1f}%"
            offset = 0.02 if is_regression else 0.5
            ax.text(bar.get_x() + bar.get_width() / 2., height + offset,
                    text, ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Zapisano: {output_path}")


def plot_overfitting_analysis(results, is_regression, output_path):
    if not results:
        print(f"⚠️ Pomijam generowanie {output_path} — brak wyników.")
        return

    models = list(results.keys())
    train_vals, test_vals = [], []
    for m in models:
        res = results[m]
        if is_regression:
            train_vals.append(res.get("train_mae", np.nan))
            test_vals.append(res.get("test_mae", np.nan))
        else:
            train_vals.append(res.get("train_accuracy", np.nan) * 100)
            test_vals.append(res.get("test_accuracy", np.nan) * 100)

    overfit_gap = []
    for t, v in zip(train_vals, test_vals):
        if np.isnan(t) or np.isnan(v):
            overfit_gap.append(np.nan)
        else:
            gap = (t - v) if not is_regression else (v - t)
            overfit_gap.append(gap)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = []
    for gap in overfit_gap:
        if np.isnan(gap):
            colors.append("gray")
        elif abs(gap) < (1 if not is_regression else 2):
            colors.append("green")
        elif abs(gap) < (5 if not is_regression else 5):
            colors.append("orange")
        else:
            colors.append("red")

    bars = ax.barh(models, overfit_gap, color=colors, alpha=0.8, edgecolor="black")
    label_x = "Różnica (Train - Test) [%]" if not is_regression else "Różnica (Test - Train) [MAE]"
    ax.set_xlabel(label_x, fontsize=13, fontweight="bold")
    ax.set_title("Analiza Overfittingu (mniejsza różnica = lepiej)", fontsize=15, fontweight="bold", pad=15)
    ax.grid(axis="x", alpha=0.3, linestyle="--")

    # Ustaw granice osi X, aby obejmowały wszystkie wartości
    min_val = min([x for x in overfit_gap if not np.isnan(x)] + [0])
    max_val = max([x for x in overfit_gap if not np.isnan(x)] + [0])
    ax.set_xlim(left=min_val - 0.5, right=max_val + 0.5)

    for i, (bar, val) in enumerate(zip(bars, overfit_gap)):
        if np.isnan(val):
            continue
        text_x = bar.get_width() + 0.05
        ax.text(text_x, i, f"{val:.2f}", va="center", ha='left', fontsize=10, fontweight="bold", color="black")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Zapisano: {output_path}")


# ---------------------------------------------------------------------
# GŁÓWNY BLOK — generuje wszystko automatycznie
# ---------------------------------------------------------------------
if __name__ == "__main__":
    print("======================================")
    print("STEP 5: Model Comparison")
    print("======================================")

    # --- Wine Quality (all) ---
    print("\n🍷 Ładuję wyniki dla Wine Quality (dataset 'all')...")
    wine_results = load_wine_results()
    if wine_results:
        plot_comparison_bar_chart(wine_results, is_regression=False,
                                  output_path="results/model_comparison_bar_wine.png",
                                  title_suffix="Wine Quality (all)")
        plot_overfitting_analysis(wine_results, is_regression=False,
                                  output_path="results/overfitting_analysis_wine.png")
    else:
        print("Pomijam Wine — brak wyników.")

    # --- Concrete Strength ---
    print("\n🏗️ Ładuję wyniki dla Concrete Strength...")
    concrete_results = load_concrete_results()
    if concrete_results:
        plot_comparison_bar_chart(concrete_results, is_regression=True,
                                  output_path="results/model_comparison_bar_concrete.png",
                                  title_suffix="Concrete Strength")
        plot_overfitting_analysis(concrete_results, is_regression=True,
                                  output_path="results/overfitting_analysis_concrete.png")
    else:
        print("Pomijam Concrete — brak wyników.")

    print("\nPorównanie modeli zakończone!")