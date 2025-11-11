"""
Eksploracja danych dla projektu ANFIS vs ML Models
Autorzy: Zespół IV_ROK 2025
--------------------------------------------------
Generuje wykresy:
 - rozkłady jakości wina (Wine Quality)
 - macierze korelacji (Wine / Concrete)
 - rozkłady cech
 - pairploty
 - rozkład wytrzymałości betonu
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------------
# Ustawienia globalne
# ---------------------------------------------------------------
sns.set(style="whitegrid", font_scale=1.0)
os.makedirs("results", exist_ok=True)

# ---------------------------------------------------------------
# Pomocnicze funkcje
# ---------------------------------------------------------------
def safe_read_csv(paths, **kwargs):
    """Próbuje wczytać CSV z listy możliwych ścieżek."""
    for path in paths:
        if os.path.exists(path):
            try:
                return pd.read_csv(path, **kwargs)
            except Exception as e:
                print(f"⚠️ Błąd przy wczytywaniu {path}: {e}")
    print(f"❌ Nie znaleziono pliku CSV w ścieżkach: {paths}")
    return None


def save_plot(fig, filename):
    """Bezpieczny zapis wykresu do /results."""
    out_path = os.path.join("results", filename)
    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Zapisano: {out_path}")


# ---------------------------------------------------------------
# 1️⃣  EKSPLORACJA DANYCH — WINE QUALITY
# ---------------------------------------------------------------
print("=" * 60)
print("EKSPLORACJA DANYCH - WINE QUALITY")
print("=" * 60)

red_wine = safe_read_csv(
    ["data/wine-quality/winequality-red.csv", "data/winequality-red.csv"],
    sep=";"
)
white_wine = safe_read_csv(
    ["data/wine-quality/winequality-white.csv", "data/winequality-white.csv"],
    sep=";"
)

if red_wine is not None and white_wine is not None:
    red_wine["type"] = 0
    white_wine["type"] = 1
    wine_data = pd.concat([red_wine, white_wine], axis=0, ignore_index=True)

    # Binaryzacja jakości
    wine_data["quality_binary"] = (wine_data["quality"] > 5).astype(int)

    print(f"Liczba próbek: {len(wine_data)}")
    print(f"Liczba cech: {wine_data.shape[1] - 2}")
    print(f"Rozkład klas: {wine_data['quality_binary'].value_counts().to_dict()}")

    # --- (1) Distribution of Quality ---
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].hist(wine_data["quality"], bins=10, edgecolor="black", color="steelblue")
    ax[0].set_title("Distribution of Wine Quality Scores")
    ax[0].set_xlabel("Quality Score")
    ax[0].set_ylabel("Count")

    ax[1].bar(
        [0, 1],
        wine_data["quality_binary"].value_counts().sort_index().values,
        color=["salmon", "lightgreen"],
        edgecolor="black"
    )
    ax[1].set_xticks([0, 1])
    ax[1].set_xticklabels(["Low (≤5)", "High (>5)"])
    ax[1].set_title("Binary Classification Distribution")
    for a in ax:
        a.grid(True, alpha=0.3)
    save_plot(fig, "wine_class_distribution.png")

    # --- (2) Correlation Matrix ---
    fig, ax = plt.subplots(figsize=(12, 10))
    corr = wine_data.drop(columns=["quality", "quality_binary"], errors="ignore").corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Wine Quality - Feature Correlation Matrix")
    save_plot(fig, "wine_correlation.png")

    # --- (3) Feature Distributions ---
    features = [
        "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
        "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
        "pH", "sulphates", "alcohol"
    ]
    fig, axes = plt.subplots(3, 4, figsize=(18, 12))
    axes = axes.flatten()
    for i, feature in enumerate(features):
        axes[i].hist(wine_data[feature], bins=30, edgecolor="black", alpha=0.7, color="steelblue")
        axes[i].set_title(feature)
        axes[i].grid(True, alpha=0.3)
    for ax in axes[len(features):]:
        ax.axis("off")
    fig.suptitle("Wine Quality - Feature Distributions", fontsize=16)
    save_plot(fig, "wine_feature_distributions.png")

    # --- (4) Pairplot (Key Features) ---
    key_features = ["alcohol", "volatile acidity", "sulphates", "citric acid", "quality_binary"]
    pairplot_data = wine_data[key_features].sample(min(1000, len(wine_data)), random_state=42)
    sns.pairplot(
        pairplot_data,
        hue="quality_binary",
        palette={0: "salmon", 1: "lightgreen"},
        diag_kind="hist",
        plot_kws={"alpha": 0.6}
    )
    plt.suptitle("Wine Quality - Key Features Pairplot", y=1.01, fontsize=16)
    plt.tight_layout()
    plt.savefig("results/wine_pairplot.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("✓ Zapisano: results/wine_pairplot.png")
else:
    print("⚠️ Nie udało się wczytać danych Wine Quality.")

# ---------------------------------------------------------------
# 2️⃣  EKSPLORACJA DANYCH — CONCRETE STRENGTH
# ---------------------------------------------------------------
print("\n" + "=" * 60)
print("EKSPLORACJA DANYCH - CONCRETE STRENGTH")
print("=" * 60)

concrete = safe_read_csv(["data/concrete-strength/Concrete_Data.csv"])

if concrete is not None:
    target_name = concrete.columns[-1]
    print(f"Liczba próbek: {len(concrete)}")
    print(f"Liczba cech: {concrete.shape[1] - 1}")
    print(f"Średnia wytrzymałość: {concrete[target_name].mean():.2f} MPa")

    # --- (5) Distribution of Target ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].hist(concrete[target_name], bins=30, edgecolor="black", color="coral")
    axes[0].set_title("Distribution of Concrete Compressive Strength")
    axes[0].set_xlabel("Compressive Strength (MPa)")
    axes[0].set_ylabel("Frequency")

    axes[1].boxplot(concrete[target_name], vert=True)
    axes[1].set_title("Boxplot of Target Variable")
    axes[1].set_ylabel("Compressive Strength (MPa)")
    for ax in axes:
        ax.grid(True, alpha=0.3)
    save_plot(fig, "concrete_distribution.png")

    # --- (6) Correlation Matrix ---
    fig, ax = plt.subplots(figsize=(10, 8))
    corr = concrete.corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="YlOrRd", center=0, ax=ax)
    ax.set_title("Concrete Strength - Feature Correlation Matrix")
    save_plot(fig, "concrete_correlation.png")
else:
    print("⚠️ Brak pliku Concrete_Data.csv — pomijam analizę Concrete.")

print("\n✅ Eksploracja danych zakończona pomyślnie!")
