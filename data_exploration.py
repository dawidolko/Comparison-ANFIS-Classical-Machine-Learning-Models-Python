import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Wyłącza wyświetlanie okien - tylko zapis do plików
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Wczytaj dane (nowa ścieżka: data/wine-quality/)
red_wine = pd.read_csv(os.path.join('data', 'wine-quality', 'winequality-red.csv'), sep=';')
white_wine = pd.read_csv(os.path.join('data', 'wine-quality', 'winequality-white.csv'), sep=';')

# Dodaj kolumnę typu wina
red_wine['type'] = 0
white_wine['type'] = 1

# Połącz oba datasety
wine_data = pd.concat([red_wine, white_wine], axis=0)

print("=== PODSTAWOWE INFORMACJE ===")
print(f"Liczba próbek: {len(wine_data)}")
print(f"Liczba cech: {wine_data.shape[1] - 1}")  # -1 bo quality to target
print("\nKolumny:")
print(wine_data.columns.tolist())
print("\nPierwsze wiersze:")
print(wine_data.head())

print("\n=== STATYSTYKI ===")
print(wine_data.describe())

print("\n=== BRAKUJĄCE WARTOŚCI ===")
print(wine_data.isnull().sum())

print("\n=== ROZKŁAD JAKOŚCI WINA ===")
print(wine_data['quality'].value_counts().sort_index())

# Wizualizacja rozkładu jakości
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
wine_data['quality'].hist(bins=10, edgecolor='black')
plt.xlabel('Jakość wina')
plt.ylabel('Liczba próbek')
plt.title('Rozkład jakości wina')

plt.subplot(1, 2, 2)
quality_counts = wine_data['quality'].value_counts().sort_index()
plt.bar(quality_counts.index, quality_counts.values, edgecolor='black')
plt.xlabel('Jakość wina')
plt.ylabel('Liczba próbek')
plt.title('Rozkład jakości wina (bar)')
plt.tight_layout()
plt.savefig('results/quality_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Wykres zapisany: results/quality_distribution.png")

# Korelacja między cechami
plt.figure(figsize=(12, 10))
correlation = wine_data.corr()
sns.heatmap(correlation, annot=True, fmt='.2f', cmap='coolwarm', center=0)
plt.title('Macierz korelacji cech')
plt.tight_layout()
plt.savefig('results/correlation_matrix.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Wykres zapisany: results/correlation_matrix.png")