# 📚 README - Projekt: Klasyfikacja Jakości Wina za pomocą ANFIS

## 📋 Spis treści
1. [Opis projektu](#opis-projektu)
2. [Struktura projektu](#struktura-projektu)
3. [Wymagania](#wymagania)
4. [Instrukcja uruchomienia](#instrukcja-uruchomienia)
5. [Opis plików](#opis-plików)
6. [Wyniki](#wyniki)

---

## 🎯 Opis projektu

Projekt porównuje algorytm **ANFIS (Adaptive Neuro-Fuzzy Inference System)** z klasycznymi metodami uczenia maszynowego w zadaniu klasyfikacji jakości wina. ANFIS to hybrydowy model łączący:
- **Logikę rozmytą** - interpretowalne reguły IF-THEN
- **Sieci neuronowe** - uczenie parametrów za pomocą propagacji wstecznej

### Główne cele:
✅ Implementacja algorytmu ANFIS w TensorFlow/Keras  
✅ Porównanie ANFIS z klasycznymi modelami (NN, SVM, Random Forest)  
✅ Analiza interpretowalności modelu rozmytego  
✅ Wizualizacja wyuczonych funkcji przynależności  

---

## 📁 Struktura projektu

```
wine_quality_anfis/
├── data/                          # Dane (generowane automatycznie)
│   ├── winequality-red.csv
│   ├── winequality-white.csv
│   ├── X_train.npy
│   ├── X_test.npy
│   ├── y_train.npy
│   └── y_test.npy
├── models/                        # Wytrenowane modele
│   ├── anfis_best_2memb.keras
│   ├── anfis_best_3memb.keras
│   ├── nn_best.keras
│   ├── svm_model.pkl
│   ├── rf_model.pkl
│   └── scaler.pkl
├── results/                       # Wykresy i wyniki
│   ├── all_models_comparison.png
│   ├── overfitting_analysis.png
│   ├── anfis_2memb_training.png
│   ├── anfis_3memb_training.png
│   └── *.json (wyniki)
├── anfis.py                       # Implementacja ANFIS
├── data_exploration.py            # Eksploracja danych
├── data_preprocessing.py          # Przygotowanie danych
├── train_anfis.py                 # Trening modeli ANFIS
├── train_comparison_models.py     # Trening modeli porównawczych
├── compare_all_models.py          # Porównanie wyników
└── requirements.txt               # Zależności
```

---

## 🔧 Wymagania

### Wymagane biblioteki:
```
tensorflow==2.17.0
numpy==1.26.4
pandas==2.2.3
scikit-learn==1.5.2
matplotlib==3.9.2
seaborn==0.12.2
```

### Instalacja:
```bash
pip install -r requirements.txt
```

---

## 🚀 Instrukcja uruchomienia

### **KROK 1: Eksploracja danych** 📊
```bash
python data_exploration.py
```

**Co robi ten skrypt:**
- Pobiera dataset Wine Quality (czerwone i białe wino)
- Łączy oba datasety (6497 próbek)
- Analizuje rozkład jakości wina (skala 3-9)
- Sprawdza braki danych i korelacje między cechami
- Generuje wykresy:
  - `quality_distribution.png` - rozkład jakości wina
  - `correlation_matrix.png` - macierz korelacji cech

**Rezultat:**
- ✅ Pobrane dane o winie
- ✅ Wygenerowane wykresy analityczne

---

### **KROK 2: Przygotowanie danych** 🔄
```bash
python data_preprocessing.py
```

**Co robi ten skrypt:**
- Przekształca problem na klasyfikację binarną:
  - **Klasa 0** (zła jakość): jakość ≤ 5
  - **Klasa 1** (dobra jakość): jakość > 5
- Wybiera 11 najważniejszych cech (fixed acidity, alcohol, pH, itd.)
- Dzieli dane na zbiór treningowy (80%) i testowy (20%)
- **Standaryzuje dane** (StandardScaler) - kluczowe dla ANFIS!
- Zapisuje przetworzone dane do plików `.npy`

**Rezultat:**
- ✅ 5197 próbek treningowych
- ✅ 1300 próbek testowych
- ✅ Rozkład klas: 2384 złej jakości / 4113 dobrej jakości

---

### **KROK 3: Trening modeli ANFIS** 🧠
```bash
python train_anfis.py
```

**Co robi ten skrypt:**
- Trenuje 2 modele ANFIS:
  - **ANFIS z 2 funkcjami przynależności** (2048 reguł)
  - **ANFIS z 3 funkcjami przynależności** (177,147 reguł)
- Każdy model trenuje się przez 20 epok
- Używa optymalizatora NADAM + binary crossentropy
- Zapisuje najlepsze wagi modelu (ModelCheckpoint)
- Early stopping po 15 epokach bez poprawy
- Generuje wykresy treningu dla każdego modelu

**Warstwy modelu ANFIS:**
1. **FuzzyLayer** - fuzzyfikacja (gaussowska funkcja przynależności)
2. **RuleLayer** - generowanie reguł rozmytych (AND)
3. **NormLayer** - normalizacja wag reguł
4. **DefuzzLayer** - defuzzyfikacja (kombinacja liniowa Takagi-Sugeno)
5. **SummationLayer** - agregacja wyników

**Rezultat:**
- ✅ ANFIS (2 funkcje): Test Accuracy = **69.06%**
- ✅ ANFIS (3 funkcje): Test Accuracy = **76.48%**
- ✅ Zapisane modele w `models/`
- ✅ Wykresy treningu w `results/`

**Czas wykonania:** ~2 minuty

---

### **KROK 4: Trening modeli porównawczych** 🤖
```bash
python train_comparison_models.py
```

**Co robi ten skrypt:**
Trenuje 3 klasyczne modele uczenia maszynowego:

#### **4.1. Neural Network (NN)**
- Architektura: 16 → Dropout(0.3) → 8 → Dropout(0.2) → 1
- Funkcje aktywacji: ReLU + Sigmoid
- Optymalizator: Adam
- 50 epok z early stopping

#### **4.2. Support Vector Machine (SVM)**
- Kernel: RBF (Radial Basis Function)
- C=1.0, gamma='scale'
- Trenowany na całym zbiorze

#### **4.3. Random Forest**
- 200 drzew decyzyjnych
- max_depth=15
- Trening równoległy (n_jobs=-1)

**Rezultat:**
- ✅ Neural Network: Test Accuracy = **75.69%**
- ✅ SVM: Test Accuracy = **77.85%**
- ✅ Random Forest: Test Accuracy = **83.23%** 🏆
- ✅ Wszystkie modele zapisane w `models/`

**Czas wykonania:** ~5-10 minut

---

### **KROK 5: Porównanie wszystkich modeli** 📈
```bash
python compare_all_models.py
```

**Co robi ten skrypt:**
- Wczytuje wyniki wszystkich 5 modeli
- Generuje 2 wykresy porównawcze:
  - **all_models_comparison.png** - wykres słupkowy Train vs Test
  - **overfitting_analysis.png** - analiza różnicy Train-Test
- Wyświetla szczegółową tabelę rankingową

**Rezultat:**
```
🥇 #1: Random Forest    - 83.23% (ale overfitting: 14.46%)
🥈 #2: SVM              - 77.85% (minimal overfitting: 1.47%)
🥉 #3: ANFIS (3 funkcje)- 76.48% (lekki overfitting: 4.59%)
   #4: Neural Network   - 75.69% (minimal overfitting: 1.76%)
   #5: ANFIS (2 funkcje)- 69.06% (brak overfittingu: 0.75%)
```

---

## 📄 Opis plików

### **anfis.py**
Główna implementacja algorytmu ANFIS w TensorFlow/Keras.

**Klasy:**
- `ANFISModel` - główny model ANFIS
- `FuzzyLayer` - warstwa fuzzyfikacji z gaussowskimi funkcjami przynależności
- `RuleLayer` - warstwa reguł rozmytych (T-norma = mnożenie)
- `NormLayer` - normalizacja wag reguł
- `DefuzzLayer` - defuzzyfikacja metodą Takagi-Sugeno-Kanga
- `SummationLayer` - agregacja końcowa

**Kluczowe funkcje:**
- `fit()` - trenowanie modelu
- `get_membership_functions()` - zwraca wyuczone parametry funkcji przynależności

---

### **data_exploration.py**
Skrypt do analizy eksploracyjnej danych Wine Quality.

**Funkcje:**
- Pobieranie danych z UCI ML Repository
- Statystyki opisowe (mean, std, min, max)
- Sprawdzanie braków danych
- Wizualizacja rozkładu jakości wina
- Macierz korelacji Pearsona

---

### **data_preprocessing.py**
Przygotowanie danych do treningu.

**Funkcje:**
- `load_and_preprocess_data()` - główna funkcja przetwarzania
  - Łączenie red + white wine
  - Binaryzacja etykiet (quality > 5)
  - Selekcja 11 cech
  - Podział train/test (80/20, stratified)
  - Standaryzacja (StandardScaler)
  - Zapis do `.npy`

---

### **train_anfis.py**
Trening modeli ANFIS z różną liczbą funkcji przynależności.

**Funkcje:**
- `train_anfis_model(n_memb, epochs, batch_size)` - trenuje jeden model ANFIS
- `plot_training_history(history, n_memb)` - wizualizacja treningu

**Parametry:**
- `n_memb` - liczba funkcji przynależności (2 lub 3)
- `epochs` - liczba epok (domyślnie 20)
- `batch_size` - rozmiar batcha (32)

---

### **train_comparison_models.py**
Trening modeli porównawczych (NN, SVM, RF).

**Funkcje:**
- `train_neural_network()` - trenuje klasyczną sieć neuronową
- `train_svm()` - trenuje SVM z RBF kernel
- `train_random_forest()` - trenuje Random Forest
- `plot_training_history()` - wykresy dla NN

---

### **compare_all_models.py**
Porównanie i wizualizacja wyników wszystkich modeli.

**Funkcje:**
- `load_all_results()` - wczytuje wyniki z plików JSON
- `plot_comparison_bar_chart()` - wykres słupkowy
- `plot_overfitting_analysis()` - analiza overfittingu
- `create_summary_table()` - tabela rankingowa

---

## 📊 Wyniki

### Finalne porównanie modeli:

| Ranking | Model | Test Accuracy | Train Accuracy | Overfitting | Interpretacja |
|---------|-------|---------------|----------------|-------------|---------------|
| 🥇 | Random Forest | **83.23%** | 97.69% | 14.46% ⚠️ | ❌ Czarna skrzynka |
| 🥈 | SVM | **77.85%** | 79.31% | 1.47% ✅ | ❌ Czarna skrzynka |
| 🥉 | ANFIS (3 funkcje) | **76.48%** | 81.08% | 4.59% ✅ | ✅ **Reguły rozmyte!** |
| 4 | Neural Network | **75.69%** | 77.45% | 1.76% ✅ | ❌ Czarna skrzynka |
| 5 | ANFIS (2 funkcje) | **69.06%** | 69.81% | 0.75% ✅ | ✅ **Reguły rozmyte!** |

### Kluczowe obserwacje:

✅ **ANFIS jest konkurencyjny!**
- ANFIS (3 funkcje) osiąga 76.48% - tylko 6.75% gorszy od najlepszego modelu
- Lepszy niż klasyczna sieć neuronowa (75.69%)
- Minimalny overfitting (4.59%)

✅ **ANFIS ma INTERPRETACJĘ!**
- Możemy zobaczyć wyuczone funkcje przynależności
- Możemy zidentyfikować najważniejsze reguły rozmyte
- Inne modele to "czarne skrzynki"

⚠️ **Random Forest overfituje**
- Najwyższa dokładność testowa (83.23%)
- Ale ogromny overfitting (14.46%)
- Train accuracy = 97.69% (prawie idealne dopasowanie do danych treningowych)

---

## 🔬 Elementy logiki rozmytej w ANFIS

### Gaussowska funkcja przynależności:
```
μ(x) = exp(-(x - c)² / 2σ²)
```
gdzie:
- `c` - centrum funkcji (uczone)
- `σ` - szerokość funkcji (uczone)

### Reguły rozmyte (przykład):
```
JEŚLI alkohol jest WYSOKI AND kwasowość jest NISKA 
TO jakość wina jest DOBRA
```

### Defuzzyfikacja (Takagi-Sugeno):
```
Wyjście = Σ(wᵢ × (aᵢx₁ + bᵢx₂ + ... + cᵢ))
```
gdzie `wᵢ` to znormalizowane wagi reguł

---

## 🎓 Wnioski

1. **ANFIS łączy zalety dwóch światów:**
   - Uczenie się jak sieć neuronowa
   - Interpretacja jak system ekspercki

2. **3 funkcje przynależności >> 2 funkcje:**
   - +7.42% dokładności (76.48% vs 69.06%)
   - Więcej reguł = lepsza reprezentacja danych

3. **ANFIS vs Klasyczne modele:**
   - Random Forest najlepszy, ale overfituje
   - SVM solidny wybór (77.85%, minimalny overfitting)
   - **ANFIS świetny kompromis:** dobra dokładność + interpretowalność

4. **Problem jakości wina:**
   - 11 cech numerycznych, 6497 próbek
   - Niezbalansowanie klas (37% złej / 63% dobrej jakości)
   - Wszystkie modele osiągają >75% dokładności

---

## 📞 Autor

**Dawid Olko, Piotr Smoła, Jakub Opar, Michał Pilecki**  
Kierunek: Informatyka, grupa Lab 01  
Przedmiot: Systemy rozmyte  
Prowadzący: mgr inż. Marcin Mrukowicz  
Rzeszów, r.a. 2025/2026

---

## 📚 Bibliografia

1. Jang, J-S. R. (1993). ANFIS: adaptive-network-based fuzzy inference system. IEEE Transactions on Systems, Man, and Cybernetics.
2. Implementacja bazowa: [Gregor Lenhard - ANFIS TensorFlow 2.0](https://github.com/gregorLen/AnfisTensorflow2.0)
3. Dataset: [UCI ML Repository - Wine Quality Dataset](https://archive.ics.uci.edu/dataset/186/wine+quality)

---

**✅ Projekt gotowy do uruchomienia!**  
Postępuj zgodnie z instrukcją krok po kroku (KROK 1-5) aby odtworzyć wszystkie wyniki.