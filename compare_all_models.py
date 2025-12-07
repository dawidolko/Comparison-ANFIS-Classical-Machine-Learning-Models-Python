"""
Generuje wykresy porównawcze modeli dla OBU problemów:
- Wine Quality (dataset 'all')
- Concrete Strength

Zapisuje je jako:
  - results/model_comparison_bar_wine.png
  - results/overfitting_analysis_wine.png
  - results/model_comparison_bar_concrete.png
  - results/overfitting_analysis_concrete.png

Uruchamianie:
    python3 compare_all_models.py
"""

import json  # Biblioteka do obsługi plików JSON
import os  # Biblioteka do operacji na systemie plików
import matplotlib  # Biblioteka do tworzenia wykresów
matplotlib.use("Agg")  # Ustawia backend matplotlib bez GUI (do zapisu plików)
import matplotlib.pyplot as plt  # Moduł do tworzenia wykresów
import numpy as np  # Biblioteka do operacji na tablicach numerycznych
import seaborn as sns  # Biblioteka do zaawansowanych wizualizacji

# ---------------------------------------------------------------------
# FUNKCJE POMOCNICZE
# ---------------------------------------------------------------------
def load_wine_results():
    """
    Wczytuje wyniki wszystkich modeli dla Wine Quality (dataset 'all').
    
    Obsługuje:
    - 2 warianty ANFIS (2 i 3 funkcje przynależności)
    - 3 modele klasyczne (NN, SVM, Random Forest)
    
    Returns:
        Dict[nazwa_modelu, wyniki_json]
    """
    paths = {  # Słownik mapujący nazwy modeli na ścieżki do plików JSON
        "ANFIS (2 MF)": "results/anfis_all_2memb_results.json",  # Ścieżka do wyników ANFIS z 2 funkcjami przynależności
        "ANFIS (3 MF)": "results/anfis_all_3memb_results.json",  # Ścieżka do wyników ANFIS z 3 funkcjami przynależności
        "Neural Network": "results/nn_wine_results.json",  # Ścieżka do wyników sieci neuronowej
        "SVM": "results/svm_wine_results.json",  # Ścieżka do wyników Support Vector Machine
        "Random Forest": "results/rf_wine_results.json",  # Ścieżka do wyników Random Forest
    }
    results = {}  # Inicjalizuje pusty słownik na wyniki
    for name, path in paths.items():  # Iteruje przez wszystkie ścieżki
        if os.path.exists(path):  # Sprawdza czy plik istnieje
            try:  # Próbuje wczytać plik
                with open(path, "r", encoding="utf-8") as f:  # Otwiera plik JSON
                    data = json.load(f)  # Wczytuje dane JSON
                    if name in ["Neural Network", "SVM", "Random Forest"]:  # Sprawdza czy to model klasyczny
                        if "test_accuracy" in data:  # Sprawdza czy plik zawiera accuracy (dla wine)
                            results[name] = data  # Dodaje wyniki do słownika
                        else:  # Jeśli brak test_accuracy
                            print(f"Plik {path} nie zawiera 'test_accuracy' — pomijam dla Wine.")  # Informuje o braku metryki
                    else:  # Dla modeli ANFIS
                        results[name] = data  # Dodaje wyniki do słownika bez dodatkowej walidacji
            except Exception as e:  # Łapie błędy wczytywania
                print(f"Błąd wczytywania {path}: {e}")  # Wypisuje komunikat o błędzie
        else:  # Jeśli plik nie istnieje
            print(f"Brak pliku: {path}")  # Informuje o braku pliku
    return results  # Zwraca słownik z wynikami


def load_concrete_results():
    """
    Wczytuje wyniki wszystkich modeli dla Concrete Strength.
    
    Obsługuje:
    - 2 warianty ANFIS (2 i 3 funkcje przynależności)
    - 3 modele klasyczne (NN, SVM, Random Forest)
    
    Returns:
        Dict[nazwa_modelu, wyniki_json]
    """
    paths = {  # Słownik mapujący nazwy modeli na ścieżki do plików JSON dla betonu
        "ANFIS (2 MF)": "results/anfis_concrete_2memb_results.json",  # ANFIS z 2 funkcjami dla betonu
        "ANFIS (3 MF)": "results/anfis_concrete_3memb_results.json",  # ANFIS z 3 funkcjami dla betonu
        "Neural Network": "results/nn_concrete_results.json",  # Sieć neuronowa dla betonu
        "SVM": "results/svm_concrete_results.json",  # SVM dla betonu
        "Random Forest": "results/rf_concrete_results.json",  # Random Forest dla betonu
    }
    results = {}  # Inicjalizuje pusty słownik na wyniki
    for name, path in paths.items():  # Iteruje przez wszystkie ścieżki
        if os.path.exists(path):  # Sprawdza czy plik istnieje
            try:  # Próbuje wczytać plik
                with open(path, "r", encoding="utf-8") as f:  # Otwiera plik JSON
                    data = json.load(f)  # Wczytuje dane JSON
                    if name in ["Neural Network", "SVM", "Random Forest"]:  # Sprawdza czy to model klasyczny
                        if "test_mae" in data:  # Sprawdza czy plik zawiera MAE (dla regresji betonu)
                            results[name] = data  # Dodaje wyniki do słownika
                        else:  # Jeśli brak test_mae
                            print(f"Plik {path} nie zawiera 'test_mae' — pomijam dla Concrete.")  # Informuje o braku metryki
                    else:  # Dla modeli ANFIS
                        results[name] = data  # Dodaje wyniki bez dodatkowej walidacji
            except Exception as e:  # Łapie błędy wczytywania
                print(f"Błąd wczytywania {path}: {e}")  # Wypisuje komunikat o błędzie
        else:  # Jeśli plik nie istnieje
            print(f"Brak pliku: {path}")  # Informuje o braku pliku
    return results  # Zwraca słownik z wynikami


def plot_comparison_bar_chart(results, is_regression, output_path, title_suffix):  # Funkcja generująca wykres słupkowy porównujący modele
    if not results:  # Sprawdza czy są wyniki do wyświetlenia
        print(f"Pomijam generowanie {output_path} — brak wyników.")  # Informuje o braku danych
        return  # Kończy wykonanie funkcji

    models = list(results.keys())  # Pobiera listę nazw modeli
    train_vals, test_vals = [], []  # Inicjalizuje listy na wartości treningowe i testowe
    for m in models:  # Iteruje przez wszystkie modele
        res = results[m]  # Pobiera wyniki dla bieżącego modelu
        if is_regression:  # Sprawdza czy to zadanie regresji
            train_vals.append(res.get("train_mae", np.nan))  # Dodaje train MAE (lub NaN jeśli brak)
            test_vals.append(res.get("test_mae", np.nan))  # Dodaje test MAE (lub NaN jeśli brak)
        else:  # Dla zadania klasyfikacji
            train_vals.append(res.get("train_accuracy", np.nan) * 100)  # Konwertuje accuracy na procenty (0.95 -> 95%)
            test_vals.append(res.get("test_accuracy", np.nan) * 100)  # Konwertuje test accuracy na procenty

    x = np.arange(len(models))  # Tworzy tablicę pozycji dla modeli (0, 1, 2, ...)
    width = 0.35  # Szerokość słupków (35% odstępu między pozycjami)
    fig, ax = plt.subplots(figsize=(12, 7))  # Tworzy figurę wykresu o wymiarach 12x7 cali

    label_train = "Train MAE" if is_regression else "Train Accuracy (%)"  # Ustawia etykietę dla danych treningowych
    label_test = "Test MAE" if is_regression else "Test Accuracy (%)"  # Ustawia etykietę dla danych testowych

    bars1 = ax.bar(x - width / 2, train_vals, width, label=label_train, color="steelblue", alpha=0.8, edgecolor="black")  # Rysuje słupki treningowe (przesunięte w lewo)
    bars2 = ax.bar(x + width / 2, test_vals, width, label=label_test, color="coral", alpha=0.8, edgecolor="black")  # Rysuje słupki testowe (przesunięte w prawo)

    ax.set_xlabel("Model", fontsize=14, fontweight="bold")  # Ustawia etykietę osi X
    ylabel = "MAE (niżej = lepiej)" if is_regression else "Dokładność (%)"  # Określa etykietę osi Y w zależności od typu zadania
    ax.set_ylabel(ylabel, fontsize=14, fontweight="bold")  # Ustawia etykietę osi Y
    ax.set_title(f"Porównanie modeli — {title_suffix}", fontsize=16, fontweight="bold", pad=20)  # Ustawia tytuł wykresu z paddingiem 20
    ax.set_xticks(x)  # Ustawia pozycje znaczników na osi X
    ax.set_xticklabels(models, rotation=15, ha="right")  # Ustawia nazwy modeli obrócone o 15° i wyrównane do prawej
    ax.legend(fontsize=12)  # Dodaje legendę z rozmiarem czcionki 12
    ax.grid(axis="y", alpha=0.3, linestyle="--")  # Dodaje poziomą siatkę z przerywaną linią

    for bars in [bars1, bars2]:  # Iteruje przez obydwa zestawy słupków
        for bar in bars:  # Iteruje przez każdy słupek
            height = bar.get_height()  # Pobiera wysokość słupka (wartość metryki)
            if np.isnan(height):  # Sprawdza czy wartość jest NaN
                continue  # Pomija NaN wartości
            text = f"{height:.2f}" if is_regression else f"{height:.1f}%"  # Formatuje tekst (2 miejsca dla MAE, 1 dla %)
            offset = 0.02 if is_regression else 0.5  # Ustawia przesunięcie tekstu nad słupkiem
            ax.text(bar.get_x() + bar.get_width() / 2., height + offset,  # Umieszcza tekst nad środkiem słupka
                    text, ha="center", va="bottom", fontsize=9)  # Wyrównanie centralne, od dołu, rozmiar 9

    plt.tight_layout()  # Automatycznie dopasowuje układ
    plt.savefig(output_path, dpi=300, bbox_inches="tight")  # Zapisuje wykres w wysokiej rozdzielczości
    plt.close()  # Zamyka wykres aby zwolnić pamięć
    print(f"✓ Zapisano: {output_path}")  # Informuje o zapisie pliku


def plot_overfitting_analysis(results, is_regression, output_path):  # Funkcja analizująca overfitting (różnice train vs test)
    if not results:  # Sprawdza czy są wyniki
        print(f"⚠️ Pomijam generowanie {output_path} — brak wyników.")  # Informuje o braku danych
        return  # Kończy wykonanie

    models = list(results.keys())  # Pobiera listę nazw modeli
    train_vals, test_vals = [], []  # Inicjalizuje listy na wartości
    for m in models:  # Iteruje przez modele
        res = results[m]  # Pobiera wyniki modelu
        if is_regression:  # Dla regresji
            train_vals.append(res.get("train_mae", np.nan))  # Pobiera train MAE
            test_vals.append(res.get("test_mae", np.nan))  # Pobiera test MAE
        else:  # Dla klasyfikacji
            train_vals.append(res.get("train_accuracy", np.nan) * 100)  # Konwertuje accuracy na procenty
            test_vals.append(res.get("test_accuracy", np.nan) * 100)  # Konwertuje test accuracy na procenty

    overfit_gap = []  # Lista na różnice (gap) wskazujące overfitting
    for t, v in zip(train_vals, test_vals):  # Iteruje przez pary wartości
        if np.isnan(t) or np.isnan(v):  # Sprawdza czy któraś wartość to NaN
            overfit_gap.append(np.nan)  # Dodaje NaN do listy
        else:  # Jeśli obydwie wartości są poprawne
            gap = (t - v) if not is_regression else (v - t)  # Oblicza różnicę (train-test dla klasyfikacji, test-train dla regresji)
            overfit_gap.append(gap)  # Dodaje różnicę do listy

    fig, ax = plt.subplots(figsize=(10, 6))  # Tworzy figurę wykresu poziomego
    colors = []  # Lista na kolory słupków (zależne od wielkości gap)
    for gap in overfit_gap:  # Iteruje przez wszystkie różnice
        if np.isnan(gap):  # Jeśli brak danych
            colors.append("gray")  # Szary kolor dla NaN
        elif abs(gap) < (1 if not is_regression else 2):  # Jeśli różnica bardzo mała (<1% lub <2 MAE)
            colors.append("green")  # Zielony = dobry model (mały overfitting)
        elif abs(gap) < (5 if not is_regression else 5):  # Jeśli różnica średnia (<5% lub <5 MAE)
            colors.append("orange")  # Pomarańczowy = umiarkowany overfitting
        else:  # Jeśli różnica duża (≥5% lub ≥5 MAE)
            colors.append("red")  # Czerwony = duży overfitting

    bars = ax.barh(models, overfit_gap, color=colors, alpha=0.8, edgecolor="black")  # Rysuje poziome słupki z kolorami
    label_x = "Różnica (Train - Test) [%]" if not is_regression else "Różnica (Test - Train) [MAE]"  # Etykieta osi X
    ax.set_xlabel(label_x, fontsize=13, fontweight="bold")  # Ustawia etykietę osi X
    ax.set_title("Analiza Overfittingu (mniejsza różnica = lepiej)", fontsize=15, fontweight="bold", pad=15)  # Tytuł wykresu
    ax.grid(axis="x", alpha=0.3, linestyle="--")  # Dodaje pionową siatkę

    # Ustaw granice osi X, aby obejmowały wszystkie wartości
    min_val = min([x for x in overfit_gap if not np.isnan(x)] + [0])  # Znajduje minimalną wartość (włącznie z 0)
    max_val = max([x for x in overfit_gap if not np.isnan(x)] + [0])  # Znajduje maksymalną wartość (włącznie z 0)
    ax.set_xlim(left=min_val - 0.5, right=max_val + 0.5)  # Ustawia granice osi X z marginesem 0.5

    for i, (bar, val) in enumerate(zip(bars, overfit_gap)):  # Iteruje przez słupki i wartości
        if np.isnan(val):  # Pomija NaN wartości
            continue  # Przechodzi do następnej iteracji
        text_x = bar.get_width() + 0.05  # Oblicza pozycję X tekstu (za końcem słupka)
        ax.text(text_x, i, f"{val:.2f}", va="center", ha='left', fontsize=10, fontweight="bold", color="black")  # Dodaje wartość jako tekst

    plt.tight_layout()  # Dopasowuje układ
    plt.savefig(output_path, dpi=300, bbox_inches="tight")  # Zapisuje wykres
    plt.close()  # Zamyka wykres
    print(f"✓ Zapisano: {output_path}")  # Informuje o zapisie


# ---------------------------------------------------------------------
# GŁÓWNY BLOK — generuje wszystko automatycznie
# ---------------------------------------------------------------------
if __name__ == "__main__":  # Sprawdza czy skrypt uruchamiany bezpośrednio
    print("======================================")  # Separator wizualny
    print("STEP 5: Model Comparison")  # Wypisuje nazwę kroku
    print("======================================")  # Separator wizualny

    # --- Wine Quality (all) ---
    print("\n🍷 Ładuję wyniki dla Wine Quality (dataset 'all')...")  # Informuje o ładowaniu wyników wina
    wine_results = load_wine_results()  # Ładuje wyniki wszystkich modeli dla wine
    if wine_results:  # Sprawdza czy są wyniki
        plot_comparison_bar_chart(wine_results, is_regression=False,  # Generuje wykres słupkowy (klasyfikacja)
                                  output_path="results/model_comparison_bar_wine.png",
                                  title_suffix="Wine Quality (all)")
        plot_overfitting_analysis(wine_results, is_regression=False,  # Generuje analizę overfitting
                                  output_path="results/overfitting_analysis_wine.png")
    else:  # Jeśli brak wyników
        print("Pomijam Wine — brak wyników.")  # Informuje o pominięciu

    # --- Concrete Strength ---
    print("\n🏗️ Ładuję wyniki dla Concrete Strength...")  # Informuje o ładowaniu wyników betonu
    concrete_results = load_concrete_results()  # Ładuje wyniki wszystkich modeli dla concrete
    if concrete_results:  # Sprawdza czy są wyniki
        plot_comparison_bar_chart(concrete_results, is_regression=True,  # Generuje wykres słupkowy (regresja)
                                  output_path="results/model_comparison_bar_concrete.png",
                                  title_suffix="Concrete Strength")
        plot_overfitting_analysis(concrete_results, is_regression=True,  # Generuje analizę overfitting
                                  output_path="results/overfitting_analysis_concrete.png")
    else:  # Jeśli brak wyników
        print("Pomijam Concrete — brak wyników.")  # Informuje o pominięciu

    print("\nPorównanie modeli zakończone!")  # Informuje o zakończeniu