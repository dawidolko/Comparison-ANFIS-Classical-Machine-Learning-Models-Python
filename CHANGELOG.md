# CHANGELOG - Zmiany w projekcie

## Data: 2025-10-20

### Główne zmiany wprowadzone do projektu:

## 1. ✅ Aktualizacja .gitignore

**Plik:** `.gitignore`

**Zmiany:**

- Dodano ignorowanie wygenerowanych plików danych: `*.npy`, `*.csv` (z wyjątkiem oryginalnych plików dataset)
- Dodano ignorowanie modeli: `*.h5`, `*.keras`, `*.pkl`, `*.weights.h5`
- Dodano ignorowanie wyników: `*.png`, `*.json`, `*.txt` w folderze `results/`
- Dodano ignorowanie środowisk wirtualnych: `.venv/`, `venv/`, `env/`
- Dodano ignorowanie IDE: `.idea/`, `.vscode/`
- Dodano ignorowanie plików systemowych macOS i Windows

**Powód:** Zapobiega commitowaniu dużych plików binarnych i tymczasowych do repozytorium Git.

---

## 2. ✅ Wyłączenie wyświetlania wykresów (plt.show())

**Pliki zmienione:**

- `data_exploration.py`
- `train_anfis.py`
- `train_comparison_models.py`
- `compare_all_models.py`
- `visualize_membership_functions.py`

**Zmiany:**

- Na początku każdego pliku dodano:
  ```python
  import matplotlib
  matplotlib.use('Agg')  # Wyłącza wyświetlanie okien - tylko zapis do plików
  ```
- Usunięto wszystkie wywołania `plt.show()`
- Zastąpiono je przez `plt.close()` po zapisaniu wykresu
- Dodano komunikaty `print("✓ Wykres zapisany: ścieżka/do/pliku.png")`

**Powód:**

- Aplikacja była blokowana dopóki użytkownik nie zamknął okienek wykresów
- Backend 'Agg' wymusza zapis wykresów tylko do plików bez wyświetlania okien
- Wykresy są nadal dostępne w folderze `results/` i w aplikacji Streamlit

---

## 3. ✅ Utworzenie modułu scaller.py

**Nowy plik:** `scaller.py`

**Zawartość:**

- Funkcja `load_scalers()` - ładuje oba scalery (11D i 12D)
- Funkcja `get_scaler_11d()` - ładuje tylko scaler 11D (dla ANFIS)
- Funkcja `get_scaler_12d()` - ładuje tylko scaler 12D (dla NN/SVM/RF)

**Obsługiwane ścieżki:**

- `models/scaler.pkl` - scaler 11D (standardowy, używany przez ANFIS)
- `models/scaler_nn.pkl` - scaler 12D (używany przez Neural Network)

**Powód:** W `app.py` był import `from scaller import load_scalers`, ale tego pliku brakowało w projekcie.

---

## 4. ✅ Utworzenie modułu utils.py

**Nowy plik:** `utils.py`

**Zawartość:**

- Funkcja `load_anfis_model()` - ładuje model ANFIS i wykonuje predykcję
  - Przeniesiona z `app.py` funkcja `_load_anfis()`
  - Obsługuje automatyczne wykrywanie kształtu wag z pliku H5
  - Wspiera warianty 11D i 12D
- Funkcja `load_results()` - ładuje wyniki wszystkich modeli z plików JSON
  - Przeniesiona z `app.py`
  - Wczytuje wyniki: ANFIS (2 i 3 funkcje), NN, SVM, RF

**Powód:** Separacja logiki biznesowej od kodu Streamlit. W `app.py` powinien być tylko kod interfejsu użytkownika.

---

## 5. ✅ Uproszczenie app.py

**Plik:** `app.py`

**Zmiany:**

- Usunięto funkcję `_load_anfis()` (przeniesiona do `utils.py`)
- Usunięto funkcję `load_results()` (przeniesiona do `utils.py`)
- Dodano importy z nowych modułów:
  ```python
  from utils import load_anfis_model, load_results
  from scaller import load_scalers
  ```
- Uproszczono kod - teraz zawiera tylko logikę interfejsu Streamlit

**Powód:** Czystszy kod, łatwiejszy w utrzymaniu, separacja odpowiedzialności (UI vs logika).

---

## Struktura projektu po zmianach:

```
Comparison-ANFIS-Classical-Machine-Learning-Models-Python/
├── .gitignore                          # ✨ Zaktualizowany
├── anfis.py                           # Model ANFIS
├── app.py                             # ✨ Uproszczony - tylko Streamlit UI
├── compare_all_models.py              # ✨ Bez plt.show()
├── data_exploration.py                # ✨ Bez plt.show()
├── data_preprocessing.py              # Preprocessing danych
├── main.py                            # Orkiestrator pipeline'u
├── requirements.txt                   # Zależności projektu
├── scaller.py                         # ✨ NOWY - ładowanie scalerów
├── train_anfis.py                     # ✨ Bez plt.show()
├── train_comparison_models.py         # ✨ Bez plt.show()
├── utils.py                           # ✨ NOWY - funkcje pomocnicze
├── visualize_membership_functions.py  # ✨ Bez plt.show()
├── data/                              # Dane (ignorowane w git)
│   ├── winequality-red.csv
│   ├── winequality-white.csv
│   ├── X_train.npy                    # Ignorowane
│   └── ...
├── models/                            # Modele (ignorowane w git)
│   ├── anfis_best_2memb.weights.h5    # Ignorowane
│   ├── anfis_best_3memb.weights.h5    # Ignorowane
│   ├── nn_best.keras                  # Ignorowane
│   ├── scaler.pkl                     # Ignorowane
│   └── ...
└── results/                           # Wyniki (ignorowane w git)
    ├── *.png                          # Ignorowane
    ├── *.json                         # Ignorowane
    └── ...
```

---

## Instrukcje uruchomienia:

### 1. Instalacja zależności:

```bash
pip install -r requirements.txt
```

### 2. Uruchomienie pipeline'u (trenowanie modeli):

```bash
python main.py
```

**Efekt:** Skrypt wykona wszystkie kroki automatycznie:

1. Eksploracja danych → wykresy w `results/`
2. Preprocessing → pliki `.npy` w `data/`
3. Trening ANFIS → modele i wykresy w `models/` i `results/`
4. Trening modeli porównawczych (NN, SVM, RF)
5. Wizualizacja funkcji przynależności
6. Porównanie wszystkich modeli

**WAŻNE:** Aplikacja **NIE BĘDZIE** blokowana oknami wykresów! Wszystkie wykresy zapisują się automatycznie do plików.

### 3. Uruchomienie aplikacji Streamlit:

```bash
streamlit run app.py
```

**Funkcje aplikacji:**

- 🏠 Strona główna - opis projektu i statystyki
- 📊 Wyniki modeli - ranking i porównanie
- 🧠 ANFIS - szczegóły modelu rozmytego
- 🔍 Eksploracja danych - wizualizacje datasetu
- 🔮 Predykcja - interaktywne przewidywanie jakości wina

---

## Problemy rozwiązane:

1. ✅ **Blokowanie aplikacji przez wykresy**

   - Rozwiązanie: `matplotlib.use('Agg')` + usunięcie `plt.show()`

2. ✅ **Brak modułu scaller.py**

   - Rozwiązanie: Utworzenie modułu z funkcjami ładowania scalerów

3. ✅ **Mieszanie logiki biznesowej i UI w app.py**

   - Rozwiązanie: Przeniesienie funkcji do utils.py

4. ✅ **Nieoptymalne .gitignore**

   - Rozwiązanie: Dodanie ignorowania plików binarnych i wygenerowanych

5. ✅ **Konieczność ręcznego zamykania wykresów w pipeline**
   - Rozwiązanie: Automatyczny zapis bez wyświetlania

---

## Zgodność z wytycznymi projektu:

✅ **Wykorzystanie AI zgodnie z zasadami:**

- AI użyte do refaktoryzacji i poprawy struktury kodu
- Wszystkie zmiany są transparentne i opisane
- Kod pozostaje czytelny i zgodny z wymaganiami projektu

✅ **Struktura zgodna z "Systemy rozmyte - projekt zaliczeniowy":**

- Implementacja ANFIS z Gaussowskimi funkcjami przynależności
- Porównanie z klasycznymi modelami ML
- Wizualizacja wyuczonych funkcji przynależności
- Interaktywna aplikacja do demonstracji

---

## Następne kroki (opcjonalne):

- [ ] Dodanie testów jednostkowych
- [ ] Rozszerzenie dokumentacji o szczegóły implementacji ANFIS
- [ ] Dodanie możliwości wyboru liczby funkcji przynależności w GUI
- [ ] Export wyników do PDF/Markdown

---

**Autor zmian:** AI Assistant (zgodnie z wytycznymi projektu)
**Data:** 2025-10-20
**Wersja:** 1.1.0
