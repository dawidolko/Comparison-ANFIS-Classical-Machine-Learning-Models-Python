"""
Eksploracja danych dla projektu ANFIS vs ML Models
Autorzy: Zespół IV_ROK 2025
--------------------------------------------------
Wygenerowano następujące wykresy:
 - rozkłady jakości wina / wytrzymałości betonu (zmienna docelowa),
 - macierze korelacji (Wine / Concrete),
 - rozkłady cech (Wine / Concrete),
 - pairploty (Wine / Concrete).
"""

import os  # Biblioteka do operacji na systemie plików
import pandas as pd  # Biblioteka do operacji na ramkach danych
import numpy as np  # Biblioteka do operacji na tablicach numerycznych
import matplotlib  # Biblioteka do tworzenia wykresów
matplotlib.use("Agg")  # Ustawia backend bez GUI (do zapisu plików)
import matplotlib.pyplot as plt  # Moduł do tworzenia wykresów
import seaborn as sns  # Biblioteka do zaawansowanych wizualizacji statystycznych

# ---------------------------------------------------------------
# Ustawienia globalne
# ---------------------------------------------------------------
sns.set(style="whitegrid", font_scale=1.0)  # Ustawia styl seaborn: białą siatkę i rozmiar czcionki 1.0
os.makedirs("results", exist_ok=True)  # Tworzy katalog results jeśli nie istnieje

# ---------------------------------------------------------------
# Pomocnicze funkcje
# ---------------------------------------------------------------
def safe_read_csv(paths, **kwargs):
    """
    Próbowano wczytać plik CSV z listy możliwych ścieżek.
    
    Iterowano po liście ścieżek i zwrócono DataFrame z pierwszego istniejącego pliku.
    Jeżeli żaden plik nie istniał, zwrócono None.
    
    Args:
        paths: lista ścieżek do sprawdzenia
        **kwargs: dodatkowe argumenty przekazywane do pd.read_csv()
        
    Returns:
        DataFrame lub None
    """
    for path in paths:  # Iteruje przez wszystkie możliwe ścieżki
        if os.path.exists(path):  # Sprawdza czy plik istnieje
            try:  # Próbuje wczytać plik
                return pd.read_csv(path, **kwargs)  # Wczytuje CSV i zwraca DataFrame
            except Exception as e:  # Łapie błędy wczytywania
                print(f"Błąd przy wczytywaniu {path}: {e}")  # Informuje o błędzie
    print(f"Nie znaleziono pliku CSV w ścieżkach: {paths}")  # Informuje że nie znaleziono żadnego pliku
    return None  # Zwraca None jeśli wszystkie ścieżki zawiodły


def save_plot(fig, filename):
    """
    Wykres został zapisany bezpiecznie do katalogu results/.
    
    Zastosowano tight_layout, a figura została zamknięta po zapisie.
    
    Args:
        fig: obiekt Figure z matplotlib
        filename: nazwa pliku wyjściowego (np. "wykres.png")
    """
    out_path = os.path.join("results", filename)  # Tworzy pełną ścieżkę do pliku wyjściowego
    plt.tight_layout()  # Automatycznie dopasowuje układ elementów wykresu
    fig.savefig(out_path, dpi=300, bbox_inches="tight")  # Zapisuje wykres w wysokiej rozdzielczości
    plt.close(fig)  # Zamyka figurę aby zwolnić pamięć
    print(f"✓ Zapisano: {out_path}")  # Informuje o pomyślnym zapisie


# ---------------------------------------------------------------
# EKSPLORACJA DANYCH — WINE QUALITY
# ---------------------------------------------------------------
print("=" * 60)  # Wypisuje separator wizualny
print("EKSPLORACJA DANYCH - WINE QUALITY")  # Wypisuje nagłówek sekcji
print("=" * 60)  # Wypisuje separator wizualny

red_wine = safe_read_csv(  # Ładuje dane czerwonego wina z możliwych ścieżek
    ["data/wine-quality/winequality-red.csv", "data/winequality-red.csv"],
    sep=";"  # Separator kolumn w pliku CSV
)
white_wine = safe_read_csv(  # Ładuje dane białego wina z możliwych ścieżek
    ["data/wine-quality/winequality-white.csv", "data/winequality-white.csv"],
    sep=";"  # Separator kolumn w pliku CSV
)

if red_wine is not None and white_wine is not None:  # Sprawdza czy obydwa pliki zostały wczytane
    red_wine["type"] = 0  # Dodaje kolumnę typu: 0 dla czerwonego wina
    white_wine["type"] = 1  # Dodaje kolumnę typu: 1 dla białego wina
    wine_data = pd.concat([red_wine, white_wine], axis=0, ignore_index=True)  # Łączy obydwa zbiory danych w jeden DataFrame

    # Jakość została przekształcona na zmienną binarną
    wine_data["quality_binary"] = (wine_data["quality"] > 5).astype(int)  # Tworzy binarną etykietę: 1 jeśli quality > 5, 0 w przeciwnym razie

    print(f"Liczba próbek: {len(wine_data)}")  # Wypisuje całkowitą liczbę próbek
    print(f"Liczba cech: {wine_data.shape[1] - 2}")  # Wypisuje liczbę cech (kolumny - type - quality_binary)
    print(f"Rozkład klas: {wine_data['quality_binary'].value_counts().to_dict()}")  # Wypisuje liczność każdej klasy

    # --- (1) Rozkład zmiennej docelowej (jakość wina) ---
    """
    Wygenerowano rysunek 1: rozkład klasy jakości wina.

    Lewy wykres: HISTOGRAM oryginalnych ocen jakości (0–10)
      - Przyjęto 10 przedziałów (po jednym na każdy możliwy wynik),
      - Oś X: ocena jakości (liczba całkowita 0–10),
      - Oś Y: liczba próbek,
      - Zastosowano metodę Matplotlib hist() z bins=10, każdy przedział reprezentuje jedną ocenę.

    Prawy wykres: WYKRES SŁUPKOWY binarnych klas
      - Przyjęto dwie klasy: 0 (niska jakość, ocena ≤5) i 1 (wysoka jakość, ocena >5),
      - Oś X: etykiety klas binarnych,
      - Oś Y: liczba próbek w klasie,
      - Celem było zobrazowanie niezrównoważenia klas w zadaniu klasyfikacji binarnej.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Lewy: Histogram oryginalnych ocen jakości
    axes[0].hist(wine_data["quality"], bins=10, edgecolor="black", color="steelblue", alpha=0.8)  # Rysuje histogram ocen jakości z 10 binami
    axes[0].set_title("Rozkład ocen jakości wina\n(Histogram: 10 przedziałów, po jednym na ocenę)", fontsize=12, fontweight='bold')  # Ustawia tytuł wykresu z opisem metody
    axes[0].set_xlabel("Ocena jakości (0–10)", fontsize=11, fontweight='bold')  # Ustawia etykietę osi X
    axes[0].set_ylabel("Liczba próbek", fontsize=11, fontweight='bold')  # Ustawia etykietę osi Y
    axes[0].grid(True, alpha=0.3, axis='y')  # Włącza siatkę na osi Y z przezroczystością
    
    # Dodano etykiety liczbowe na słupkach
    counts, bins, patches = axes[0].hist(wine_data["quality"], bins=10, edgecolor="black", color="steelblue", alpha=0.8)  # Ponownie rysuje histogram aby otrzymać obiekty słupków
    for i, (count, patch) in enumerate(zip(counts, patches)):  # Iteruje przez każdy słupek
        if count > 0:  # Sprawdza czy słupek ma wartość większą od zera
            axes[0].text(patch.get_x() + patch.get_width()/2, patch.get_height() + 50,   # Dodaje etykietę tekstową nad słupkiem
                        int(count), ha='center', va='bottom', fontsize=9, fontweight='bold')  # Wyświetla liczbę próbek jako tekst

    # Prawy: Wykres słupkowy klas binarnych
    class_counts = wine_data["quality_binary"].value_counts().sort_index()  # Liczy wystąpienia każdej klasy binarnej i sortuje według indeksu
    bars = axes[1].bar(  # Tworzy wykres słupkowy i zapisuje obiekty słupków
        [0, 1],  # Pozycje słupków na osi X (klasa 0 i 1)
        class_counts.values,  # Wysokości słupków (liczba próbek)
        color=["salmon", "lightgreen"],  # Kolory słupków: czerwony dla niskiej, zielony dla wysokiej jakości
        edgecolor="black",  # Kolor krawędzi słupków
        width=0.6,  # Szerokość słupków
        alpha=0.8  # Przezroczystość słupków
    )
    axes[1].set_xticks([0, 1])  # Ustawia pozycje etykiet na osi X
    axes[1].set_xticklabels(["Niska (≤5)", "Wysoka (>5)"], fontsize=10)  # Ustawia etykiety tekstowe dla klas
    axes[1].set_xlabel("Klasa binarna", fontsize=11, fontweight='bold')  # Ustawia etykietę osi X
    axes[1].set_ylabel("Liczba próbek", fontsize=11, fontweight='bold')  # Ustawia etykietę osi Y
    axes[1].set_title("Rozkład klasyfikacji binarnej\n(Wykres słupkowy: 2 klasy)", fontsize=12, fontweight='bold')  # Ustawia tytuł wykresu
    axes[1].grid(True, alpha=0.3, axis='y')  # Włącza siatkę na osi Y z przezroczystością
    
    # Dodano etykiety liczbowe na słupkach
    for i, bar in enumerate(bars):  # Iteruje przez każdy słupek
        height = bar.get_height()  # Pobiera wysokość słupka
        axes[1].text(bar.get_x() + bar.get_width()/2, height + 50,  # Dodaje etykietę tekstową nad słupkiem
                    f'{int(height)}\n({100*height/len(wine_data):.1f}%)',  # Wyświetla liczbę i procent próbek
                    ha='center', va='bottom', fontsize=10, fontweight='bold')  # Formatuje tekst (wycentrowany, pogrubiony)
    
    plt.suptitle("Zbiór danych Wine Quality – Analiza zmiennej docelowej", fontsize=14, fontweight='bold', y=0.98)  # Ustawia główny tytuł całej figury
    save_plot(fig, "wine_class_distribution.png")  # Zapisuje wykres do pliku

    # --- (2) Macierz korelacji ---
    fig, ax = plt.subplots(figsize=(12, 10))  # Tworzy nową figurę o rozmiarze 12x10 cali
    corr = wine_data.drop(columns=["quality", "quality_binary"], errors="ignore").corr()  # Oblicza macierz korelacji Pearsona dla wszystkich cech (bez zmiennych docelowych)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)  # Rysuje heatmapę z adnotacjami liczbowymi, kolorami od niebieskiego (ujemne) przez biały (0) do czerwonego (dodatnie)
    ax.set_title("Wine Quality – Macierz korelacji cech")  # Ustawia tytuł wykresu
    save_plot(fig, "wine_correlation.png")  # Zapisuje wykres do pliku

    # --- (3) Rozkłady cech ---
    features = [  # Lista wszystkich 11 cech wina do wizualizacji
        "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
        "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
        "pH", "sulphates", "alcohol"
    ]
    fig, axes = plt.subplots(3, 4, figsize=(18, 12))  # Tworzy siatkę 3x4 subplotów (12 miejsc dla 11 cech)
    axes = axes.flatten()  # Spłaszcza macierz 2D subplotów do listy 1D dla łatwiejszej iteracji
    for i, feature in enumerate(features):  # Iteruje przez każdą cechę z jej indeksem
        axes[i].hist(wine_data[feature], bins=30, edgecolor="black", alpha=0.7, color="steelblue")  # Rysuje histogram z 30 binami dla danej cechy
        axes[i].set_title(feature)  # Ustawia tytuł jako nazwę cechy
        axes[i].grid(True, alpha=0.3)  # Włącza siatkę z przezroczystością
    for ax in axes[len(features):]:  # Iteruje przez niewykorzystane subploty (12 miejsc - 11 cech = 1 pusty)
        ax.axis("off")  # Wyłącza wyświetlanie pustego subplotu
    fig.suptitle("Wine Quality – Rozkłady cech", fontsize=16)  # Ustawia główny tytuł dla całej figury
    save_plot(fig, "wine_feature_distributions.png")  # Zapisuje wykres do pliku

    # --- (4) Pairplot (kluczowe cechy) ---
    key_features = ["alcohol", "volatile acidity", "sulphates", "citric acid", "quality_binary"]  # Lista 4 najważniejszych cech + zmienna docelowa binarna
    pairplot_data = wine_data[key_features].sample(min(1000, len(wine_data)), random_state=42)  # Pobiera próbkę maksymalnie 1000 wierszy (dla szybkości) z losowym ziarnem 42
    sns.pairplot(  # Tworzy pairplot - macierz wykresów rozproszenia dla każdej pary cech
        pairplot_data,  # Dane do wizualizacji
        hue="quality_binary",  # Koloruje punkty według klasy binarnej
        palette={0: "salmon", 1: "lightgreen"},  # Ustawia kolory: czerwony dla klasy 0, zielony dla klasy 1
        diag_kind="hist",  # Na przekątnej rysuje histogramy (zamiast wykresów rozproszenia)
        plot_kws={"alpha": 0.6}  # Ustawia przezroczystość punktów na 0.6
    )
    plt.suptitle("Wine Quality – Pairplot kluczowych cech", y=1.01, fontsize=16)  # Ustawia główny tytuł powyżej całego pairplotu
    plt.tight_layout()  # Dopasowuje układ elementów wykresu
    plt.savefig("results/wine_pairplot.png", dpi=300, bbox_inches="tight")  # Zapisuje pairplot w wysokiej rozdzielczości
    plt.close()  # Zamyka figurę aby zwolnić pamięć
    print("✓ Zapisano: results/wine_pairplot.png")  # Informuje o pomyślnym zapisie
else:  # Wykonuje się gdy nie udało się wczytać plików wine
    print("Nie udało się wczytać danych Wine Quality.")  # Informuje o błędzie wczytywania

# ---------------------------------------------------------------
# EKSPLORACJA DANYCH — CONCRETE STRENGTH
# ---------------------------------------------------------------
print("\n" + "=" * 60)  # Wypisuje pusty wiersz i separator wizualny
print("EKSPLORACJA DANYCH - CONCRETE STRENGTH")  # Wypisuje nagłówek sekcji
print("=" * 60)  # Wypisuje separator wizualny

concrete = safe_read_csv(["data/concrete-strength/Concrete_Data.csv"])  # Ładuje dane betonu z możliwej ścieżki

if concrete is not None:  # Sprawdza czy plik został pomyślnie wczytany
    target_name = concrete.columns[-1]  # Pobiera nazwę ostatniej kolumny (zmienna docelowa - wytrzymałość)
    print(f"Liczba próbek: {len(concrete)}")  # Wypisuje całkowitą liczbę próbek
    print(f"Liczba cech: {concrete.shape[1] - 1}")  # Wypisuje liczbę cech (kolumny - zmienna docelowa)
    print(f"Średnia wytrzymałość: {concrete[target_name].mean():.2f} MPa")  # Wypisuje średnią wytrzymałość betonu w megapascalach

    # --- (1) Rozkład zmiennej docelowej (wytrzymałość betonu) ---
    # Przyjęto binarną klasyfikację: wysoka wytrzymałość > mediana
    median_strength = concrete[target_name].median()  # Oblicza medianę wytrzymałości jako próg podziału
    concrete["strength_binary"] = (concrete[target_name] > median_strength).astype(int)  # Tworzy binarną etykietę: 1 jeśli wytrzymałość > mediana, 0 w przeciwnym razie

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))  # Tworzy figurę z 2 subplotami w jednym wierszu

    # Histogram oryginalnych wartości wytrzymałości
    axes[0].hist(concrete[target_name], bins=30, edgecolor="black", color="coral")  # Rysuje histogram wytrzymałości z 30 binami
    axes[0].set_title("Rozkład wytrzymałości betonu na ścieiskanie")  # Ustawia tytuł wykresu
    axes[0].set_xlabel("Wytrzymałość na ścieiskanie (MPa)")  # Ustawia etykietę osi X (megapascale)
    axes[0].set_ylabel("Liczba próbek")  # Ustawia etykietę osi Y

    # Wykres słupkowy rozkładu binarnego
    counts = concrete["strength_binary"].value_counts().sort_index()  # Liczy wystąpienia każdej klasy binarnej i sortuje według indeksu
    axes[1].bar(  # Tworzy wykres słupkowy
        [0, 1],  # Pozycje słupków na osi X (klasa 0 i 1)
        counts.values,  # Wysokości słupków (liczba próbek)
        color=["salmon", "lightgreen"],  # Kolory słupków: czerwony dla niskiej, zielony dla wysokiej wytrzymałości
        edgecolor="black"  # Kolor krawędzi słupków
    )
    axes[1].set_xticks([0, 1])  # Ustawia pozycje etykiet na osi X
    axes[1].set_xticklabels([f"Niska (≤{median_strength:.1f})", f"Wysoka (>{median_strength:.1f})"])  # Ustawia etykiety z wartościami mediany
    axes[1].set_title("Rozkład klasyfikacji binarnej (wg mediany)")  # Ustawia tytuł wykresu
    for ax in axes:  # Iteruje przez obydwa subploty
        ax.grid(True, alpha=0.3)  # Włącza siatkę z przezroczystością
    save_plot(fig, "concrete_target_distribution.png")  # Zapisuje wykres do pliku

    # --- (2) Macierz korelacji ---
    fig, ax = plt.subplots(figsize=(10, 8))  # Tworzy nową figurę o rozmiarze 10x8 cali
    corr = concrete.corr()  # Oblicza macierz korelacji Pearsona dla wszystkich kolumn
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="YlOrRd", center=0, ax=ax)  # Rysuje heatmapę z adnotacjami, kolorami od żółtego przez pomarańczowy do czerwonego
    ax.set_title("Concrete Strength – Macierz korelacji cech")  # Ustawia tytuł wykresu
    save_plot(fig, "concrete_correlation.png")  # Zapisuje wykres do pliku

    # --- (3) Rozkłady cech ---
    features = concrete.columns[:-1].tolist()  # Pobiera listę wszystkich cech oprócz ostatniej kolumny (zmienna docelowa)
    n_features = len(features)  # Liczy całkowitą liczbę cech
    n_rows = (n_features + 3) // 4  # Oblicza potrzebną liczbę wierszy dla siatki 4-kolumnowej (zaokrąglenie w górę)
    fig, axes = plt.subplots(n_rows, 4, figsize=(18, 4 * n_rows))  # Tworzy siatkę subplotów o dynamicznej liczbie wierszy
    axes = axes.flatten()  # Spłaszcza macierz 2D subplotów do listy 1D dla łatwiejszej iteracji
    for i, feature in enumerate(features):  # Iteruje przez każdą cechę z jej indeksem
        axes[i].hist(concrete[feature], bins=30, edgecolor="black", alpha=0.7, color="coral")  # Rysuje histogram z 30 binami dla danej cechy
        axes[i].set_title(feature)  # Ustawia tytuł jako nazwę cechy
        axes[i].grid(True, alpha=0.3)  # Włącza siatkę z przezroczystością
    for ax in axes[n_features:]:  # Iteruje przez niewykorzystane subploty
        ax.axis("off")  # Wyłącza wyświetlanie pustych subplotów
    fig.suptitle("Concrete Strength – Rozkłady cech", fontsize=16)  # Ustawia główny tytuł dla całej figury
    save_plot(fig, "concrete_feature_distributions.png")  # Zapisuje wykres do pliku

    # --- (4) Pairplot (kluczowe cechy) – z pełną kontrolą nad etykietami i kolorami ---
    # Wybrano kluczowe cechy: pierwsze trzy składniki + zmienna docelowa
    key_features = [features[0], features[1], features[2], target_name]  # Lista 3 najważniejszych składników betonu + wytrzymałość

    # Skrócono nazwy cech dla poprawy czytelności
    short_names = {  # Słownik mapujący długie nazwy na krótkie
        "cement (component 1)(kg in a m^3 mixture)": "Cement",  # Skrót dla cementu
        "blast furnace slag (component 2)(kg in a m^3 mixture)": "Blast Slag",  # Skrót dla żużla wielkopiecowego
        "fly ash (component 3)(kg in a m^3 mixture)": "Fly Ash",  # Skrót dla popiołów lotnych
        "water (component 4)(kg in a m^3 mixture)": "Woda",  # Skrót dla wody
        "superplasticizer (component 5)(kg in a m^3 mixture)": "Superplastyfikator",  # Skrót dla superplastyfikatora
        "coarse aggregate (component 6)(kg in a m^3 mixture)": "Kruszywo grube",  # Skrót dla kruszywa grubego
        "fine aggregate (component 7)(kg in a m^3 mixture)": "Kruszywo drobne",  # Skrót dla kruszywa drobnego
        "age (day)": "Wiek",  # Skrót dla wieku betonu
        "Concrete compressive strength(MPa, megapascals)": "Wytrzymałość"  # Skrót dla wytrzymałości
    }

    pairplot_data = concrete[key_features].copy()  # Tworzy kopię DataFrame z wybranymi cechami
    pairplot_data.rename(columns=short_names, inplace=True)  # Zmienia nazwy kolumn na krótkie w miejscu

    g = sns.PairGrid(pairplot_data, diag_sharey=False)  # Tworzy siatkę PairGrid z niezależnymi skalami Y dla przekątnej
    g.map_upper(sns.scatterplot, alpha=0.6, color="steelblue")  # Rysuje wykresy rozproszenia w górnym trójkącie matrycy
    g.map_lower(sns.scatterplot, alpha=0.6, color="steelblue")  # Rysuje wykresy rozproszenia w dolnym trójkącie matrycy
    g.map_diag(sns.histplot, color="coral", alpha=0.7)  # Rysuje histogramy na przekątnej

    for i, col in enumerate(pairplot_data.columns):  # Iteruje przez każdą kolumnę z indeksem
        for j in range(len(pairplot_data.columns)):  # Iteruje przez każdy subplot w wierszu
            ax = g.axes[i, j]  # Pobiera subplot na pozycji [i,j]
            if j == 0:  # Jeśli to pierwszy subplot w wierszu
                ax.set_ylabel(col, fontsize=10, rotation=0, ha='right', va='center')  # Ustawia etykietę Y (nazwa cechy poziomo po prawej)
            else:  # W przeciwnym razie
                ax.set_ylabel("")  # Usuwa etykietę Y
            if i == len(pairplot_data.columns) - 1:  # Jeśli to ostatni wiersz
                ax.set_xlabel(col, fontsize=10, rotation=90, ha='center', va='top')  # Ustawia etykietę X (nazwa cechy pionowo u dołu)
            else:  # W przeciwnym razie
                ax.set_xlabel("")  # Usuwa etykietę X

    # Dodano tytuł
    g.fig.suptitle("Concrete Strength – Pairplot kluczowych cech", y=1.01, fontsize=16)  # Ustawia główny tytuł powyżej całego pairplotu

    # Zoptymalizowano marginesy i odstępy
    plt.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.1, hspace=0.3, wspace=0.3)  # Dopasowuje marginesy: lewy, prawy, górny, dolny i odstępy między subplotami

    # Wykres został zapisany
    plt.savefig("results/concrete_pairplot.png", dpi=300, bbox_inches="tight")  # Zapisuje pairplot w wysokiej rozdzielczości
    plt.close()  # Zamyka figurę aby zwolnić pamięć
    print("✓ Zapisano: results/concrete_pairplot.png")  # Informuje o pomyślnym zapisie
else:  # Wykonuje się gdy nie udało się wczytać pliku concrete
    print("Brak pliku Concrete_Data.csv — pominięto analizę Concrete.")  # Informuje o braku pliku

print("\nEksploracja danych zakończona pomyślnie!")  # Wypisuje komunikat o pomyślnym zakończeniu skryptu