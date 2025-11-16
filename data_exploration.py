"""
Eksploracja danych dla projektu ANFIS vs ML Models
Autorzy: Zespół IV_ROK 2025
--------------------------------------------------
Generuje wykresy:
 - rozkłady jakości wina / wytrzymałości betonu (Target)
 - macierze korelacji (Wine / Concrete)
 - rozkłady cech (Wine / Concrete)
 - pairploty (Wine / Concrete)
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
    """
    Próbuje wczytać plik CSV z listy możliwych ścieżek.
    
    Iteruje przez listę ścieżek i zwraca DataFrame z pierwszego istniejącego pliku.
    Jeśli żaden plik nie istnieje, zwraca None.
    
    Args:
        paths: lista ścieżek do sprawdzenia
        **kwargs: dodatkowe argumenty przekazywane do pd.read_csv()
        
    Returns:
        DataFrame lub None
    """
    for path in paths:
        if os.path.exists(path):
            try:
                return pd.read_csv(path, **kwargs)
            except Exception as e:
                print(f"Błąd przy wczytywaniu {path}: {e}")
    print(f"Nie znaleziono pliku CSV w ścieżkach: {paths}")
    return None


def save_plot(fig, filename):
    """
    Bezpiecznie zapisuje wykres do katalogu results/.
    
    Automatycznie stosuje tight_layout i zamyka figurę po zapisie.
    
    Args:
        fig: obiekt Figure z matplotlib
        filename: nazwa pliku wyjściowego (np. "wykres.png")
    """
    out_path = os.path.join("results", filename)
    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Zapisano: {out_path}")


# ---------------------------------------------------------------
# EKSPLORACJA DANYCH — WINE QUALITY
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

    # --- (1) Distribution of Target (Quality) ---
    """
    Figure 1: Wine Quality Class Distribution
    
    Left plot: HISTOGRAM of original quality scores (0-10)
      - Bin count: 10 bins (one per quality score)
      - X-axis: Quality score (integer 0-10)
      - Y-axis: Frequency count (number of samples)
      - Method: Matplotlib hist() with bins=10, each bin represents one quality level
    
    Right plot: BAR CHART of binarized classes
      - Two bars: Class 0 (Low quality, score ≤5) and Class 1 (High quality, score >5)
      - X-axis: Binary class labels
      - Y-axis: Count of samples in each class
      - Purpose: Show class imbalance for binary classification task
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Histogram of raw quality scores
    axes[0].hist(wine_data["quality"], bins=10, edgecolor="black", color="steelblue", alpha=0.8)
    axes[0].set_title("Distribution of Wine Quality Scores\n(Histogram: 10 bins, 1 per score)", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("Quality Score (0-10)", fontsize=11, fontweight='bold')
    axes[0].set_ylabel("Frequency (Count)", fontsize=11, fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Add count labels on bars
    counts, bins, patches = axes[0].hist(wine_data["quality"], bins=10, edgecolor="black", color="steelblue", alpha=0.8)
    for i, (count, patch) in enumerate(zip(counts, patches)):
        if count > 0:
            axes[0].text(patch.get_x() + patch.get_width()/2, patch.get_height() + 50, 
                        int(count), ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Right: Bar chart of binary classes
    class_counts = wine_data["quality_binary"].value_counts().sort_index()
    bars = axes[1].bar(
        [0, 1],
        class_counts.values,
        color=["salmon", "lightgreen"],
        edgecolor="black",
        width=0.6,
        alpha=0.8
    )
    axes[1].set_xticks([0, 1])
    axes[1].set_xticklabels(["Low (≤5)", "High (>5)"], fontsize=10)
    axes[1].set_xlabel("Binary Class", fontsize=11, fontweight='bold')
    axes[1].set_ylabel("Count", fontsize=11, fontweight='bold')
    axes[1].set_title("Binary Classification Distribution\n(Bar Chart: 2 classes)", fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # Add count labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2, height + 50,
                    f'{int(height)}\n({100*height/len(wine_data):.1f}%)',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.suptitle("Wine Quality Dataset - Target Variable Analysis", fontsize=14, fontweight='bold', y=0.98)
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
    print("Nie udało się wczytać danych Wine Quality.")

# ---------------------------------------------------------------
# EKSPLORACJA DANYCH — CONCRETE STRENGTH
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

    # --- (1) Distribution of Target (Compressive Strength) ---
    # Tworzymy binarną klasyfikację: np. high strength > median
    median_strength = concrete[target_name].median()
    concrete["strength_binary"] = (concrete[target_name] > median_strength).astype(int)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Histogram oryginalnych wartości
    axes[0].hist(concrete[target_name], bins=30, edgecolor="black", color="coral")
    axes[0].set_title("Distribution of Concrete Compressive Strength")
    axes[0].set_xlabel("Compressive Strength (MPa)")
    axes[0].set_ylabel("Frequency")

    # Słupkowy rozkład binarny (jak w Wine)
    counts = concrete["strength_binary"].value_counts().sort_index()
    axes[1].bar(
        [0, 1],
        counts.values,
        color=["salmon", "lightgreen"],
        edgecolor="black"
    )
    axes[1].set_xticks([0, 1])
    axes[1].set_xticklabels([f"Low (≤{median_strength:.1f})", f"High (>{median_strength:.1f})"])
    axes[1].set_title("Binary Classification Distribution (by Median)")
    for ax in axes:
        ax.grid(True, alpha=0.3)
    save_plot(fig, "concrete_target_distribution.png")

    # --- (2) Correlation Matrix ---
    fig, ax = plt.subplots(figsize=(10, 8))
    corr = concrete.corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="YlOrRd", center=0, ax=ax)
    ax.set_title("Concrete Strength - Feature Correlation Matrix")
    save_plot(fig, "concrete_correlation.png")

    # --- (3) Feature Distributions ---
    features = concrete.columns[:-1].tolist()  # wszystkie cechy oprócz targetu
    n_features = len(features)
    n_rows = (n_features + 3) // 4  # 4 kolumny
    fig, axes = plt.subplots(n_rows, 4, figsize=(18, 4 * n_rows))
    axes = axes.flatten()
    for i, feature in enumerate(features):
        axes[i].hist(concrete[feature], bins=30, edgecolor="black", alpha=0.7, color="coral")
        axes[i].set_title(feature)
        axes[i].grid(True, alpha=0.3)
    for ax in axes[n_features:]:
        ax.axis("off")
    fig.suptitle("Concrete Strength - Feature Distributions", fontsize=16)
    save_plot(fig, "concrete_feature_distributions.png")

        # --- (4) Pairplot (Key Features) - z pełną kontrolą nad etykietami i kolorami ---
    # Wybieramy kluczowe cechy — pierwsze 3 + target
    key_features = [features[0], features[1], features[2], target_name]

    # Skrócenie nazw cech dla czytelności
    short_names = {
        "cement (component 1)(kg in a m^3 mixture)": "Cement",
        "blast furnace slag (component 2)(kg in a m^3 mixture)": "Blast Slag",
        "fly ash (component 3)(kg in a m^3 mixture)": "Fly Ash",
        "water (component 4)(kg in a m^3 mixture)": "Water",
        "superplasticizer (component 5)(kg in a m^3 mixture)": "Superplast.",
        "coarse aggregate (component 6)(kg in a m^3 mixture)": "Coarse Agg.",
        "fine aggregate (component 7)(kg in a m^3 mixture)": "Fine Agg.",
        "age (day)": "Age",
        "Concrete compressive strength(MPa, megapascals)": "Strength"
    }

    pairplot_data = concrete[key_features].copy()
    pairplot_data.rename(columns=short_names, inplace=True)

    g = sns.PairGrid(pairplot_data, diag_sharey=False)
    g.map_upper(sns.scatterplot, alpha=0.6, color="steelblue")
    g.map_lower(sns.scatterplot, alpha=0.6, color="steelblue")
    g.map_diag(sns.histplot, color="coral", alpha=0.7)

    for i, col in enumerate(pairplot_data.columns):
        for j in range(len(pairplot_data.columns)):
            ax = g.axes[i, j]
            if j == 0:
                ax.set_ylabel(col, fontsize=10, rotation=0, ha='right', va='center')
            else:
                ax.set_ylabel("")
            if i == len(pairplot_data.columns) - 1:
                ax.set_xlabel(col, fontsize=10, rotation=90, ha='center', va='top')
            else:
                ax.set_xlabel("")

    # Dodajemy tytuł
    g.fig.suptitle("Concrete Strength - Key Features Pairplot", y=1.01, fontsize=16)

    # Optymalizujemy marginesy i rozmiar
    plt.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.1, hspace=0.3, wspace=0.3)

    # Zapisujemy
    plt.savefig("results/concrete_pairplot.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("✓ Zapisano: results/concrete_pairplot.png")
else:
    print("Brak pliku Concrete_Data.csv — pomijam analizę Concrete.")

print("\nEksploracja danych zakończona pomyślnie!")