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
def load_all_results():
    """
    Wczytuje wyniki wszystkich modeli z katalogu /results.
    Obsługuje zarówno klasyfikację (accuracy), jak i regresję (MAE).
    """
    result_files = {
        "ANFIS (2 MF)": "results/anfis_2memb_results.json",
        "ANFIS (3 MF)": "results/anfis_3memb_results.json",
        "Neural Network": "results/nn_results.json",
        "SVM": "results/svm_results.json",
        "Random Forest": "results/rf_results.json",
    }

    results = {}
    for name, path in result_files.items():
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    results[name] = data
            except Exception as e:
                print(f"⚠ Nie można wczytać {path}: {e}")
        else:
            print(f"⚠ Brak pliku: {path}")

    if not results:
        raise FileNotFoundError("❌ Brak plików z wynikami. Uruchom train_anfis.py i train_comparison_models.py")

    return results


def get_metric_key(result_dict):
    """Zwraca nazwę metryki ('accuracy' lub 'mae') w zależności od danych."""
    if "test_accuracy" in result_dict:
        return "accuracy"
    elif "test_mae" in result_dict:
        return "mae"
    # fallback na accuracy
    return "accuracy"


# ---------------------------------------------------------------------
# WYKRESY
# ---------------------------------------------------------------------
def plot_comparison_bar_chart(results):
    """
    Tworzy wykres słupkowy porównujący wyniki testowe i treningowe wszystkich modeli.
    W przypadku regresji (MAE) używa mniejsza wartość = lepsza.
    """
    models = list(results.keys())

    metric_type = get_metric_key(next(iter(results.values())))
    is_regression = metric_type == "mae"

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
    ax.set_title("Porównanie skuteczności modeli", fontsize=16, fontweight="bold", pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.legend(fontsize=12)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    # Dodaj wartości
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            text = f"{height:.2f}" if is_regression else f"{height:.1f}%"
            ax.text(bar.get_x() + bar.get_width() / 2., height + (0.5 if not is_regression else 0.02),
                    text, ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig("results/model_comparison_bar.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("✓ Wykres zapisany: results/model_comparison_bar.png")


def plot_overfitting_analysis(results):
    """
    Analiza różnicy Train - Test (Overfitting).
    Dla accuracy → im mniejsza różnica, tym lepiej.
    Dla MAE → im większy wzrost testowego błędu, tym gorzej.
    """
    models = list(results.keys())
    metric_type = get_metric_key(next(iter(results.values())))
    is_regression = metric_type == "mae"

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
        gap = (t - v) if not is_regression else (v - t)
        overfit_gap.append(gap)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["green" if abs(x) < 5 else "orange" if abs(x) < 10 else "red" for x in overfit_gap]
    bars = ax.barh(models, overfit_gap, color=colors, alpha=0.8, edgecolor="black")

    label_x = "Różnica (Train - Test) [%]" if not is_regression else "Różnica (Test - Train) [MAE]"
    ax.set_xlabel(label_x, fontsize=13, fontweight="bold")
    title = "Analiza Overfittingu (mniejsza różnica = lepiej)"
    ax.set_title(title, fontsize=15, fontweight="bold", pad=15)
    ax.grid(axis="x", alpha=0.3, linestyle="--")

    for i, (bar, val) in enumerate(zip(bars, overfit_gap)):
        ax.text(val + 0.5 if val > 0 else val - 3, i, f"{val:.2f}", va="center", fontsize=10, fontweight="bold")

    plt.tight_layout()
    plt.savefig("results/overfitting_analysis.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("✓ Wykres zapisany: results/overfitting_analysis.png")


# ---------------------------------------------------------------------
# TABELA
# ---------------------------------------------------------------------
def create_summary_table(results):
    """Drukuje tabelę podsumowującą w konsoli."""
    print("\n" + "=" * 80)
    print("SZCZEGÓŁOWA TABELA PORÓWNAWCZA")
    print("=" * 80)

    metric_type = get_metric_key(next(iter(results.values())))
    is_regression = metric_type == "mae"

    header = (f"{'Model':<25} {'Train MAE':<12} {'Test MAE':<12} {'Gap':<10} Ranking"
              if is_regression
              else f"{'Model':<25} {'Train Acc':<12} {'Test Acc':<12} {'Gap':<10} Ranking")
    print(header)
    print("-" * 80)

    # Sortuj po test metric
    sorted_models = sorted(
        results.items(),
        key=lambda x: x[1].get("test_mae" if is_regression else "test_accuracy", 0),
        reverse=not is_regression,
    )

    for rank, (model, res) in enumerate(sorted_models, 1):
        if is_regression:
            train = res.get("train_mae", np.nan)
            test = res.get("test_mae", np.nan)
            gap = test - train
        else:
            train = res.get("train_accuracy", np.nan) * 100
            test = res.get("test_accuracy", np.nan) * 100
            gap = train - test

        medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else "  "
        print(f"{model:<25} {train:>8.3f}{'%' if not is_regression else ''}"
              f"   {test:>8.3f}{'%' if not is_regression else ''}"
              f"   {gap:>6.2f}{'%' if not is_regression else ''}    {medal} #{rank}")

    print("=" * 80)


# ---------------------------------------------------------------------
# GŁÓWNY BLOK
# ---------------------------------------------------------------------
if __name__ == "__main__":
    print("Ładuję wyniki wszystkich modeli...")
    results = load_all_results()

    print("\n1️⃣  Tworzę wykres porównawczy...")
    plot_comparison_bar_chart(results)

    print("\n2️⃣  Tworzę analizę overfittingu...")
    plot_overfitting_analysis(results)

    print("\n3️⃣  Generuję tabelę podsumowującą...")
    create_summary_table(results)

    print("\n" + "=" * 80)
    print("✓ RAPORT GOTOWY!")
    print("=" * 80)
