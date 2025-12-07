import numpy as np  # Biblioteka do operacji na tablicach numerycznych
import matplotlib  # Biblioteka do tworzenia wykresów
matplotlib.use('Agg')  # Ustawia backend matplotlib bez GUI (do zapisu plików bez wyświetlania)
import matplotlib.pyplot as plt  # Moduł do tworzenia wykresów
from anfis import ANFISModel  # Importuje klasę modelu ANFIS
import os  # Biblioteka do operacji na systemie plików
import argparse  # Biblioteka do parsowania argumentów wiersza poleceń


def visualize_membership_functions(n_memb=2, dataset='all'):
    """
    Wizualizuje gaussowskie funkcje przynależności dla kluczowych cech.
    
    Dla każdej ważnej cechy rysuje n_memb funkcji gaussowskich z wytrenowanego modelu.
    Zakres osi X jest dostosowany do rzeczywistych wartości w danych treningowych.
    
    Args:
        n_memb: liczba funkcji przynależności (2 lub 3)
        dataset: nazwa zestawu ('concrete', 'all', 'red', 'white')
    """
    print(f"\n📈 Wizualizacja MF: dataset={dataset}, n_memb={n_memb}")  # Wypisuje informacje o konfiguracji wizualizacji

    model_path = f"models/anfis_{dataset}_best_{n_memb}memb.weights.h5"  # Konstruuje ścieżkę do pliku wag modelu
    if not os.path.exists(model_path):  # Sprawdza czy plik modelu istnieje
        print(f"✗ Model {model_path} nie istnieje!")  # Informuje o braku pliku
        return  # Kończy wykonanie funkcji

    # Ładowanie danych (dla zakresów)
    if dataset == "concrete":  # Sprawdza czy to zbiór betonu
        X_train = np.load("data/concrete-strength/X_train.npy")  # Ładuje dane treningowe betonu
        feature_names = [  # Lista nazw cech dla betonu
            "Cement", "Blast furnace slag", "Fly ash", "Water",
            "Superplasticizer", "Coarse aggregate", "Fine aggregate", "Age"
        ]
        important_features = list(range(min(6, X_train.shape[1])))  # Wybiera pierwsze 6 najważniejszych cech
    else:  # Dla zbiorów wina (all, red, white)
        try:  # Próbuje załadować odpowiedni plik
            if dataset == "all":  # Jeśli to połączone dane wina
                X_train = np.load("data/X_train.npy")  # Ładuje dane treningowe dla wszystkich win
            else:  # Dla konkretnego typu wina
                X_train = np.load(f"data/X_train_{dataset}.npy")  # Ładuje dane dla czerwonego lub białego wina
        except Exception:  # Łapie błędy ładowania
            print(f"⚠️ Nie znaleziono danych dla {dataset}, pomijam.")  # Informuje o braku danych
            return  # Kończy wykonanie funkcji

        feature_names = [  # Lista nazw cech dla wina
            "Fixed acidity", "Volatile acidity", "Citric acid", "Residual sugar",
            "Chlorides", "Free SO₂", "Total SO₂", "Density",
            "pH", "Sulphates", "Alcohol"
        ]
        important_features = [10, 1, 8, 9, 0, 7]  # Indeksy najważniejszych cech wina (Alcohol, Volatile acidity, pH, Sulphates, Fixed acidity, Density)

    n_features = X_train.shape[1]  # Pobiera liczbę cech wejściowych z danych
    important_features = [f for f in important_features if f < n_features]  # Filtruje indeksy cech aby nie przekroczyć liczby dostępnych cech

    # Inicjalizacja i pobranie parametrów MF
    anfis_model = ANFISModel(n_input=n_features, n_memb=n_memb, batch_size=32)  # Tworzy model ANFIS z odpowiednimi parametrami
    anfis_model.model.load_weights(model_path)  # Ładuje wytrenowane wagi z pliku
    anfis_model.update_weights()  # Aktualizuje lokalne kopie wag w modelu
    centers, sigmas = anfis_model.get_membership_functions()  # Pobiera centra i sigmy funkcji gaussowskich

    # Zakres danych dynamicznie (±15% margines)
    mins, maxs = X_train.min(axis=0), X_train.max(axis=0)  # Oblicza minima i maksima każdej cechy
    margins = (maxs - mins) * 0.15  # Oblicza 15% margines dla lepszej wizualizacji

    # Liczba subplotów dopasowana automatycznie
    n_cols = 3  # Liczba kolumn w siatce wykresów
    n_rows = int(np.ceil(len(important_features) / n_cols))  # Oblicza liczbę wierszy potrzebną dla wszystkich cech
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 5 * n_rows))  # Tworzy siatkę subplotów
    axes = axes.flatten()  # Spłaszcza tablicę osi do 1D dla łatwiejszej iteracji

    colors = ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e', '#9467bd', '#8c564b']  # Paleta kolorów dla różnych funkcji przynależności

    for idx, feat_idx in enumerate(important_features):  # Iteruje przez wszystkie ważne cechy
        ax = axes[idx]  # Pobiera bieżącą oś wykresu
        x_range = np.linspace(mins[feat_idx] - margins[feat_idx],
                              maxs[feat_idx] + margins[feat_idx], 400)  # Generuje 400 punktów w zakresie cechy z marginesami
        c = centers[:, feat_idx]  # Pobiera centra funkcji gaussowskich dla bieżącej cechy
        s = sigmas[:, feat_idx]  # Pobiera sigmy funkcji gaussowskich dla bieżącej cechy

        for i in range(n_memb):  # Iteruje przez wszystkie funkcje przynależności
            mu = np.exp(-((x_range - c[i]) ** 2) / (2 * s[i] ** 2))  # Oblicza wartości funkcji gaussowskiej: exp(-(x-c)^2/(2*sigma^2))
            ax.plot(x_range, mu, color=colors[i % len(colors)],
                    linewidth=2, label=f'MF {i+1}')  # Rysuje funkcję przynależności z etykietą

        fname = feature_names[feat_idx] if feat_idx < len(feature_names) else f"Feature {feat_idx}"  # Pobiera nazwę cechy lub generuje domyślną
        ax.set_title(fname, fontsize=12)  # Ustawia tytuł wykresu (nazwa cechy)
        ax.set_xlabel('Feature value')  # Ustawia etykietę osi X
        ax.set_ylabel('Membership μ(x)')  # Ustawia etykietę osi Y (stopień przynależności)
        ax.legend()  # Dodaje legendę z funkcjami MF
        ax.grid(True, alpha=0.3)  # Dodaje siatkę z przezroczystością 0.3
        ax.set_ylim([-0.05, 1.05])  # Ustawia zakres osi Y od -0.05 do 1.05

    # Układ i zapis
    for j in range(len(important_features), len(axes)):  # Iteruje przez nieużywane subplot
        axes[j].axis('off')  # Wyłącza wyświetlanie pustych subplotów

    plt.suptitle(f"ANFIS Membership Functions ({dataset}, {n_memb} MF)", fontsize=16, fontweight='bold')  # Dodaje główny tytuł wykresu
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Dopasowuje układ subplotów zostawiając miejsce na tytuł

    os.makedirs("results", exist_ok=True)  # Tworzy katalog results jeśli nie istnieje
    out_path = f"results/membership_functions_{dataset}_{n_memb}memb.png"  # Konstruuje ścieżkę do zapisu pliku
    plt.savefig(out_path, dpi=300, bbox_inches="tight")  # Zapisuje wykres do pliku PNG z wysoką rozdzielczością
    plt.close()  # Zamyka wykres aby zwolnić pamięć
    print(f"✓ Zapisano wykres: {out_path}")  # Informuje o pomyślnym zapisie


# ===========================================================
# GŁÓWNY BLOK
# ===========================================================
if __name__ == "__main__":  # Sprawdza czy skrypt jest uruchamiany bezpośrednio (nie importowany)
    os.makedirs("results", exist_ok=True)  # Tworzy katalog results jeśli nie istnieje
    parser = argparse.ArgumentParser()  # Tworzy parser argumentów wiersza poleceń
    parser.add_argument("--datasets", nargs="+", default=["all"], choices=["concrete", "all", "red", "white"])  # Argument do wyboru zbiorów danych
    parser.add_argument("--memb", nargs="+", type=int, default=[2, 3])  # Argument do wyboru liczby funkcji przynależności
    args = parser.parse_args()  # Parsuje argumenty wiersza poleceń

    for dataset in args.datasets:  # Iteruje przez wszystkie wybrane zbiory danych
        for n_memb in args.memb:  # Iteruje przez wszystkie wybrane liczby funkcji przynależności
            try:  # Próbuje wygenerować wizualizację
                visualize_membership_functions(n_memb, dataset)  # Wywołuje funkcję wizualizacji
            except Exception as e:  # Łapie wyjątki podczas wizualizacji
                print(f"✗ Błąd dla dataset={dataset}, n_memb={n_memb}: {e}")  # Wypisuje komunikat o błędzie

    print("\n✓ Wizualizacja MF zakończona!")  # Informuje o ukończeniu wszystkich wizualizacji
