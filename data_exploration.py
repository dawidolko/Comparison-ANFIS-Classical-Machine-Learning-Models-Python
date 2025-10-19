import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams["figure.dpi"] = 100
os.makedirs("results", exist_ok=True)

red_wine = pd.read_csv('data/winequality-red.csv', sep=';')
white_wine = pd.read_csv('data/winequality-white.csv', sep=';')

red_wine['type'] = 0
white_wine['type'] = 1

wine_data = pd.concat([red_wine, white_wine], axis=0, ignore_index=True)

num_cols = wine_data.select_dtypes(include=[np.number]).columns.tolist()
feature_cols = [c for c in num_cols if c not in ('quality', 'type')]

print("=== PODSTAWOWE INFORMACJE ===")
print(f"Liczba próbek: {len(wine_data)}")
print(f"Liczba cech:   {len(feature_cols)}")
print("\nKolumny:")
print(wine_data.columns.tolist())
print("\nPierwsze wiersze:")
print(wine_data.head())

print("\n=== STATYSTYKI (tylko numeryczne) ===")
print(wine_data[num_cols].describe())

print("\n=== BRAKUJĄCE WARTOŚCI (liczba) ===")
print(wine_data.isnull().sum())

dup_count = wine_data.duplicated().sum()
print(f"\n=== DUPLIKATY WIERSZY ===\nLiczba duplikatów: {dup_count}")

print("\n=== ROZKŁAD JAKOŚCI WINA (globalnie) ===")
print(wine_data['quality'].value_counts().sort_index())

print("\n=== ROZKŁAD JAKOŚCI wg TYPU (0=red, 1=white) ===")
print(wine_data.groupby('type')['quality'].value_counts().unstack(fill_value=0).sort_index())

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
wine_data['quality'].plot(kind='hist', bins=10, edgecolor='black')
plt.xlabel('Jakość wina')
plt.ylabel('Liczba próbek')
plt.title('Rozkład jakości wina (hist)')

plt.subplot(1, 2, 2)
quality_counts = wine_data['quality'].value_counts().sort_index()
plt.bar(quality_counts.index, quality_counts.values, edgecolor='black')
plt.xlabel('Jakość wina')
plt.ylabel('Liczba próbek')
plt.title('Rozkład jakości wina (bar)')

plt.tight_layout()
plt.savefig('results/quality_distribution.png', dpi=300, bbox_inches='tight')
plt.show()
print("✓ Zapisano: results/quality_distribution.png")

ct = wine_data.groupby(['type', 'quality']).size().unstack(fill_value=0)
ct.index = ct.index.map({0: 'Red', 1: 'White'})
ct = ct[sorted(ct.columns)]
ct.plot(kind='bar', stacked=True, figsize=(10, 5), edgecolor='black')
plt.xlabel('Typ wina')
plt.ylabel('Liczba próbek')
plt.title('Rozkład jakości wg typu (stacked)')
plt.tight_layout()
plt.savefig('results/quality_by_type.png', dpi=300, bbox_inches='tight')
plt.show()
print("✓ Zapisano: results/quality_by_type.png")

plt.figure(figsize=(12, 10))
corr = wine_data[num_cols].corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0, square=False)
plt.title('Macierz korelacji (numeryczne)')
plt.tight_layout()
plt.savefig('results/correlation_matrix.png', dpi=300, bbox_inches='tight')
plt.show()
print("✓ Zapisano: results/correlation_matrix.png")

fig, axes = plt.subplots(nrows=int(np.ceil(len(feature_cols)/3)), ncols=3, figsize=(14, 4*np.ceil(len(feature_cols)/3)))
axes = axes.ravel()
for ax, col in zip(axes, feature_cols):
    sns.kdeplot(data=wine_data, x=col, hue='type', common_norm=False, ax=ax, fill=True, alpha=0.35,
                hue_order=[0,1], palette={0:'tab:red', 1:'tab:blue'})
    ax.set_title(col)
for ax in axes[len(feature_cols):]:
    ax.axis('off')
plt.tight_layout()
plt.savefig('results/features_kde_by_type.png', dpi=300, bbox_inches='tight')
plt.show()
print("✓ Zapisano: results/features_kde_by_type.png")
