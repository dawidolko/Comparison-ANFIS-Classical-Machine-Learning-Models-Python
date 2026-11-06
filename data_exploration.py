"""
Data exploration for the ANFIS vs ML Models project
Authors: Team IV_ROK 2025
--------------------------------------------------
The following plots are generated:
 - wine quality / concrete strength distributions (target variable),
 - correlation matrices (Wine / Concrete),
 - feature distributions (Wine / Concrete),
 - pairplots (Wine / Concrete).
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------------
# Global settings
# ---------------------------------------------------------------
sns.set(style="whitegrid", font_scale=1.0)
os.makedirs("results", exist_ok=True)

# ---------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------
def safe_read_csv(paths, **kwargs):
    """
    Try to read a CSV file from a list of possible paths.

    Iterates over the paths and returns a DataFrame from the first existing file.
    If no file exists, returns None.

    Args:
        paths: list of paths to check
        **kwargs: extra arguments passed to pd.read_csv()

    Returns:
        DataFrame or None
    """
    for path in paths:
        if os.path.exists(path):
            try:
                return pd.read_csv(path, **kwargs)
            except Exception as e:
                print(f"Error loading {path}: {e}")
    print(f"CSV file not found in paths: {paths}")
    return None


def save_plot(fig, filename):
    """
    Save the plot safely to the results/ directory.

    Applies tight_layout and closes the figure after saving.

    Args:
        fig: matplotlib Figure object
        filename: output file name (e.g. "plot.png")
    """
    out_path = os.path.join("results", filename)
    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Zapisano: {out_path}")


# ---------------------------------------------------------------
# DATA EXPLORATION - WINE QUALITY
# ---------------------------------------------------------------
print("=" * 60)
print("DATA EXPLORATION - WINE QUALITY")
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

    # Convert quality to a binary variable
    wine_data["quality_binary"] = (wine_data["quality"] > 5).astype(int)

    print(f"Samples: {len(wine_data)}")
    print(f"Features: {wine_data.shape[1] - 2}")
    print(f"Class distribution: {wine_data['quality_binary'].value_counts().to_dict()}")

    # --- (1) Target variable distribution (wine quality) ---
    """
    Generate figure 1: wine quality class distribution.

    Left plot: HISTOGRAM of the original quality scores (0-10)
      - 10 bins are used (one per possible score),
      - X axis: quality score (integer 0-10),
      - Y axis: number of samples,
      - Uses Matplotlib hist() with bins=10, each bin represents one score.

    Right plot: BAR CHART of the binary classes
      - Two classes are used: 0 (low quality, score <=5) and 1 (high quality,
        score >5),
      - X axis: binary class labels,
      - Y axis: number of samples per class,
      - The goal is to show the class imbalance in the binary classification
        task.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: histogram of the original quality scores
    axes[0].hist(wine_data["quality"], bins=10, edgecolor="black", color="steelblue", alpha=0.8)
    axes[0].set_title("Rozkład ocen jakości wina\n(Histogram: 10 przedziałów, po jednym na ocenę)", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("Ocena jakości (0–10)", fontsize=11, fontweight='bold')
    axes[0].set_ylabel("Liczba próbek", fontsize=11, fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Add count labels on the bars
    counts, bins, patches = axes[0].hist(wine_data["quality"], bins=10, edgecolor="black", color="steelblue", alpha=0.8)
    for i, (count, patch) in enumerate(zip(counts, patches)):
        if count > 0:
            axes[0].text(patch.get_x() + patch.get_width()/2, patch.get_height() + 50, 
                        int(count), ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Right: bar chart of the binary classes
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
    axes[1].set_xticklabels(["Niska (≤5)", "Wysoka (>5)"], fontsize=10)
    axes[1].set_xlabel("Klasa binarna", fontsize=11, fontweight='bold')
    axes[1].set_ylabel("Liczba próbek", fontsize=11, fontweight='bold')
    axes[1].set_title("Rozkład klasyfikacji binarnej\n(Wykres słupkowy: 2 klasy)", fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # Add count labels on the bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2, height + 50,
                    f'{int(height)}\n({100*height/len(wine_data):.1f}%)',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.suptitle("Zbiór danych Wine Quality – Analiza zmiennej docelowej", fontsize=14, fontweight='bold', y=0.98)
    save_plot(fig, "wine_class_distribution.png")

    # --- (2) Correlation matrix ---
    fig, ax = plt.subplots(figsize=(12, 10))
    corr = wine_data.drop(columns=["quality", "quality_binary"], errors="ignore").corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Wine Quality – Macierz korelacji cech")
    save_plot(fig, "wine_correlation.png")

    # --- (3) Feature distributions ---
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
    fig.suptitle("Wine Quality – Rozkłady cech", fontsize=16)
    save_plot(fig, "wine_feature_distributions.png")

    # --- (4) Pairplot (key features) ---
    key_features = ["alcohol", "volatile acidity", "sulphates", "citric acid", "quality_binary"]
    pairplot_data = wine_data[key_features].sample(min(1000, len(wine_data)), random_state=42)
    sns.pairplot(
        pairplot_data,
        hue="quality_binary",
        palette={0: "salmon", 1: "lightgreen"},
        diag_kind="hist",
        plot_kws={"alpha": 0.6}
    )
    plt.suptitle("Wine Quality – Pairplot kluczowych cech", y=1.01, fontsize=16)
    plt.tight_layout()
    plt.savefig("results/wine_pairplot.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("✓ Zapisano: results/wine_pairplot.png")
else:
    print("Failed to load Wine Quality data.")

# ---------------------------------------------------------------
# DATA EXPLORATION - CONCRETE STRENGTH
# ---------------------------------------------------------------
print("\n" + "=" * 60)
print("DATA EXPLORATION - CONCRETE STRENGTH")
print("=" * 60)

concrete = safe_read_csv(["data/concrete-strength/Concrete_Data.csv"])

if concrete is not None:
    target_name = concrete.columns[-1]
    print(f"Samples: {len(concrete)}")
    print(f"Features: {concrete.shape[1] - 1}")
    print(f"Mean strength: {concrete[target_name].mean():.2f} MPa")

    # --- (1) Target variable distribution (concrete strength) ---
    # Binary classification is used: high strength > median
    median_strength = concrete[target_name].median()
    concrete["strength_binary"] = (concrete[target_name] > median_strength).astype(int)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Histogram of the original strength values
    axes[0].hist(concrete[target_name], bins=30, edgecolor="black", color="coral")
    axes[0].set_title("Rozkład wytrzymałości betonu na ścieiskanie")
    axes[0].set_xlabel("Wytrzymałość na ścieiskanie (MPa)")
    axes[0].set_ylabel("Liczba próbek")

    # Bar chart of the binary distribution
    counts = concrete["strength_binary"].value_counts().sort_index()
    axes[1].bar(
        [0, 1],
        counts.values,
        color=["salmon", "lightgreen"],
        edgecolor="black"
    )
    axes[1].set_xticks([0, 1])
    axes[1].set_xticklabels([f"Niska (≤{median_strength:.1f})", f"Wysoka (>{median_strength:.1f})"])
    axes[1].set_title("Rozkład klasyfikacji binarnej (wg mediany)")
    for ax in axes:
        ax.grid(True, alpha=0.3)
    save_plot(fig, "concrete_target_distribution.png")

    # --- (2) Correlation matrix ---
    fig, ax = plt.subplots(figsize=(10, 8))
    corr = concrete.corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="YlOrRd", center=0, ax=ax)
    ax.set_title("Concrete Strength – Macierz korelacji cech")
    save_plot(fig, "concrete_correlation.png")

    # --- (3) Feature distributions ---
    features = concrete.columns[:-1].tolist()  # all features except the target variable
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
    fig.suptitle("Concrete Strength – Rozkłady cech", fontsize=16)
    save_plot(fig, "concrete_feature_distributions.png")

    # --- (4) Pairplot (key features) - with full control over labels and colors ---
    # Key features selected: the first three components + the target variable
    key_features = [features[0], features[1], features[2], target_name]

    # Shorten feature names for readability
    short_names = {
        "cement (component 1)(kg in a m^3 mixture)": "Cement",
        "blast furnace slag (component 2)(kg in a m^3 mixture)": "Blast Slag",
        "fly ash (component 3)(kg in a m^3 mixture)": "Fly Ash",
        "water (component 4)(kg in a m^3 mixture)": "Woda",
        "superplasticizer (component 5)(kg in a m^3 mixture)": "Superplastyfikator",
        "coarse aggregate (component 6)(kg in a m^3 mixture)": "Kruszywo grube",
        "fine aggregate (component 7)(kg in a m^3 mixture)": "Kruszywo drobne",
        "age (day)": "Wiek",
        "Concrete compressive strength(MPa, megapascals)": "Wytrzymałość"
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

    g.fig.suptitle("Concrete Strength – Pairplot kluczowych cech", y=1.01, fontsize=16)

    plt.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.1, hspace=0.3, wspace=0.3)

    plt.savefig("results/concrete_pairplot.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("✓ Zapisano: results/concrete_pairplot.png")
else:
    print("Missing Concrete_Data.csv, Concrete analysis skipped.")

print("\nData exploration finished successfully!")