import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle


def load_and_preprocess_data():
    """
    Wczytuje i przetwarza dane o winie
    """
    # Wczytaj dane (nowa lokalizacja: data/wine-quality)
    red_wine = pd.read_csv(os.path.join('data', 'wine-quality', 'winequality-red.csv'), sep=';')
    white_wine = pd.read_csv(os.path.join('data', 'wine-quality', 'winequality-white.csv'), sep=';')

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

    # Zapisz przetworzone dane (nowa lokalizacja: data/wine-quality)
    os.makedirs(os.path.join('data', 'wine-quality'), exist_ok=True)
    np.save(os.path.join('data', 'wine-quality', 'X_train.npy'), X_train)
    np.save(os.path.join('data', 'wine-quality', 'X_test.npy'), X_test)
    np.save(os.path.join('data', 'wine-quality', 'y_train.npy'), y_train)
    np.save(os.path.join('data', 'wine-quality', 'y_test.npy'), y_test)

    print("\n✓ Dane przetworzone i zapisane!")

    # --- DODATKOWO: przetwarzanie dla concrete-strength (jeśli plik istnieje)
    concrete_csv = os.path.join('data', 'concrete-strength', 'Concrete_Data.csv')
    if os.path.exists(concrete_csv):
        try:
            df_c = pd.read_csv(concrete_csv)
            # Zakładamy, że ostatnia kolumna to target (strength)
            Xc = df_c.iloc[:, :-1].values
            yc = df_c.iloc[:, -1].values

            Xc_train, Xc_test, yc_train, yc_test = train_test_split(
                Xc, yc, test_size=0.2, random_state=42
            )

            scaler_c = StandardScaler()
            Xc_train_scaled = scaler_c.fit_transform(Xc_train)
            Xc_test_scaled = scaler_c.transform(Xc_test)

            # Zapisz scaler i dane w katalogu models/concrete-strength/
            os.makedirs(os.path.join('models', 'concrete-strength'), exist_ok=True)
            with open(os.path.join('models', 'concrete-strength', 'scaler.pkl'), 'wb') as f:
                pickle.dump(scaler_c, f)

            # Zapisz przetworzone dane dla concrete
            os.makedirs(os.path.join('data', 'concrete-strength'), exist_ok=True)
            np.save(os.path.join('data', 'concrete-strength', 'X_train.npy'), Xc_train_scaled)
            np.save(os.path.join('data', 'concrete-strength', 'X_test.npy'), Xc_test_scaled)
            np.save(os.path.join('data', 'concrete-strength', 'y_train.npy'), yc_train)
            np.save(os.path.join('data', 'concrete-strength', 'y_test.npy'), yc_test)

            # Wygeneruj podstawowe wykresy: rozkład siły i macierz korelacji
            import matplotlib.pyplot as plt
            import seaborn as sns

            os.makedirs('results', exist_ok=True)

            plt.figure(figsize=(8, 5))
            plt.hist(yc, bins=30, edgecolor='black')
            plt.xlabel('Concrete compressive strength (MPa)')
            plt.ylabel('Count')
            plt.title('Distribution of Concrete Strength')
            out1 = os.path.join('results', 'concrete_distribution.png')
            plt.tight_layout()
            plt.savefig(out1, dpi=300)
            plt.close()

            # Correlation matrix
            plt.figure(figsize=(10, 8))
            corr = pd.DataFrame(Xc, columns=df_c.columns[:-1]).corr()
            sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0)
            plt.title('Concrete Features Correlation')
            out2 = os.path.join('results', 'concrete_correlation.png')
            plt.tight_layout()
            plt.savefig(out2, dpi=300)
            plt.close()

            print(f"✓ Scaler i dane dla concrete-strength zapisane: models/concrete-strength/scaler.pkl")
            print(f"✓ Wykresy zapisane: {out1}, {out2}")

        except Exception as e:
            print(f"⚠ Błąd podczas przetwarzania concrete-strength: {e}")
    else:
        print("⚠ Brak pliku concrete-strength/Concrete_Data.csv — pomijam przetwarzanie dla betonu")