import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs('results', exist_ok=True)

print("="*60)
print("EKSPLORACJA DANYCH - WINE QUALITY")
print("="*60)

# Wczytaj dane Wine Quality
try:
    red_wine = pd.read_csv('data/wine-quality/winequality-red.csv', sep=';')
    white_wine = pd.read_csv('data/wine-quality/winequality-white.csv', sep=';')
except:
    red_wine = pd.read_csv('data/winequality-red.csv', sep=';')
    white_wine = pd.read_csv('data/winequality-white.csv', sep=';')

red_wine['type'] = 0
white_wine['type'] = 1
wine_data = pd.concat([red_wine, white_wine], axis=0)

# Binaryzacja jakości: 1 = dobra (quality > 5), 0 = słaba
wine_data['quality_binary'] = (wine_data['quality'] > 5).astype(int)

print(f"Liczba próbek: {len(wine_data)}")
print(f"Liczba cech: {wine_data.shape[1] - 2}")
print(f"Rozkład klas: {wine_data['quality_binary'].value_counts().to_dict()}")

# 1. Wine: Class Distribution
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
ax[0].hist(wine_data['quality'], bins=10, edgecolor='black', color='steelblue')
ax[0].set_xlabel('Quality Score')
ax[0].set_ylabel('Count')
ax[0].set_title('Distribution of Wine Quality Scores')
ax[0].grid(True, alpha=0.3)

ax[1].bar([0, 1], wine_data['quality_binary'].value_counts().sort_index().values, 
          color=['salmon', 'lightgreen'], edgecolor='black')
ax[1].set_xticks([0, 1])
ax[1].set_xticklabels(['Low (≤5)', 'High (>5)'])
ax[1].set_ylabel('Count')
ax[1].set_title('Binary Classification Distribution')
ax[1].grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('results/wine_class_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Zapisano: results/wine_class_distribution.png")

# 2. Wine: Correlation Matrix
fig, ax = plt.subplots(figsize=(12, 10))
corr = wine_data.drop(['quality', 'quality_binary'], axis=1).corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax)
ax.set_title('Wine Quality - Feature Correlation Matrix')
plt.tight_layout()
plt.savefig('results/wine_correlation.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Zapisano: results/wine_correlation.png")

# 3. Wine: Feature Distributions
features_to_plot = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
                    'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
                    'pH', 'sulphates', 'alcohol']
fig, axes = plt.subplots(3, 4, figsize=(18, 12))
axes = axes.flatten()
for i, feature in enumerate(features_to_plot):
    axes[i].hist(wine_data[feature], bins=30, edgecolor='black', alpha=0.7, color='steelblue')
    axes[i].set_xlabel(feature)
    axes[i].set_ylabel('Frequency')
    axes[i].set_title(f'Distribution: {feature}')
    axes[i].grid(True, alpha=0.3)
axes[-1].axis('off')
plt.suptitle('Wine Quality - Feature Distributions', fontsize=16)
plt.tight_layout()
plt.savefig('results/wine_feature_distributions.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Zapisano: results/wine_feature_distributions.png")

# 4. Wine: Pairplot (kluczowe cechy)
key_features = ['alcohol', 'volatile acidity', 'sulphates', 'citric acid', 'quality_binary']
pairplot_data = wine_data[key_features].sample(min(1000, len(wine_data)), random_state=42)
sns.pairplot(pairplot_data, hue='quality_binary', palette={0: 'salmon', 1: 'lightgreen'}, 
             diag_kind='hist', plot_kws={'alpha': 0.6})
plt.suptitle('Wine Quality - Key Features Pairplot', y=1.01, fontsize=16)
plt.tight_layout()
plt.savefig('results/wine_pairplot.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Zapisano: results/wine_pairplot.png")

print("\n" + "="*60)
print("EKSPLORACJA DANYCH - CONCRETE STRENGTH")
print("="*60)

# Wczytaj dane Concrete Strength
try:
    concrete = pd.read_csv('data/concrete-strength/Concrete_Data.csv')
except:
    print("⚠️ Brak pliku Concrete_Data.csv - pomijam concrete")
    concrete = None

if concrete is not None:
    print(f"Liczba próbek: {len(concrete)}")
    print(f"Liczba cech: {concrete.shape[1] - 1}")
    print(f"Target mean: {concrete.iloc[:, -1].mean():.2f}")
    
    # 5. Concrete: Distribution of Target
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].hist(concrete.iloc[:, -1], bins=30, edgecolor='black', color='coral')
    axes[0].set_xlabel('Compressive Strength (MPa)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Distribution of Concrete Compressive Strength')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].boxplot(concrete.iloc[:, -1], vert=True)
    axes[1].set_ylabel('Compressive Strength (MPa)')
    axes[1].set_title('Boxplot of Target Variable')
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/concrete_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Zapisano: results/concrete_distribution.png")
    
    # 6. Concrete: Correlation Matrix
    fig, ax = plt.subplots(figsize=(10, 8))
    corr = concrete.corr()
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='YlOrRd', center=0, ax=ax)
    ax.set_title('Concrete Strength - Feature Correlation Matrix')
    plt.tight_layout()
    plt.savefig('results/concrete_correlation.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Zapisano: results/concrete_correlation.png")

print("\n✅ Eksploracja danych zakończona!")