import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle


def load_and_preprocess_data():
    """
    Wczytuje i przetwarza dane o winie
    """
    # Wczytaj dane
    red_wine = pd.read_csv('data/winequality-red.csv', sep=';')
    white_wine = pd.read_csv('data/winequality-white.csv', sep=';')

    # Połącz
    wine_data = pd.concat([red_wine, white_wine], axis=0)

    print(f"Całkowita liczba próbek: {len(wine_data)}")

    # Przekształć na problem binarny
    # 0 = zła jakość (≤5), 1 = dobra jakość (>5)
    wine_data['quality_binary'] = (wine_data['quality'] > 5).astype(int)

    print("\n=== ROZKŁAD KLAS ===")
    print(f"Zła jakość (0): {(wine_data['quality_binary'] == 0).sum()}")
    print(f"Dobra jakość (1): {(wine_data['quality_binary'] == 1).sum()}")

    # Wybierz cechy
    feature_columns = [
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

    X = wine_data[feature_columns].values
    y = wine_data['quality_binary'].values

    # Podziel na train/test (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Standaryzacja (ważne dla ANFIS!)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Zapisz scaler
    with open('models/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    print(f"\nRozmiar zbioru treningowego: {X_train_scaled.shape}")
    print(f"Rozmiar zbioru testowego: {X_test_scaled.shape}")

    return X_train_scaled, X_test_scaled, y_train, y_test, feature_columns


if __name__ == "__main__":
    # Stwórz folder models jeśli nie istnieje
    import os

    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)

    X_train, X_test, y_train, y_test, features = load_and_preprocess_data()

    # Zapisz przetworzone dane
    np.save('data/X_train.npy', X_train)
    np.save('data/X_test.npy', X_test)
    np.save('data/y_train.npy', y_train)
    np.save('data/y_test.npy', y_test)

    print("\n✓ Dane przetworzone i zapisane!")