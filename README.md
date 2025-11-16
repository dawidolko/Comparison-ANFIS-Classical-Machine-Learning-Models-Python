# 🤖 ANFIS vs Classical Machine Learning Models

Comprehensive comparison of **ANFIS (Adaptive Neuro-Fuzzy Inference System)** with classical machine learning algorithms on two real-world datasets.

---

## 📊 Datasets

### 1. **Wine Quality Classification** 🍷

- **Source**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/wine+quality)
- **Samples**: 6,497 (1,599 red + 4,898 white)
- **Features**: 11 physicochemical properties
  - Fixed acidity, volatile acidity, citric acid
  - Residual sugar, chlorides
  - Free/total sulfur dioxide
  - Density, pH, sulphates, alcohol
- **Task**: Binary classification (quality > 5 vs ≤ 5)
- **Variants**:
  - `all`: Combined red + white wines
  - `red`: Red wines only
  - `white`: White wines only

### 2. **Concrete Compressive Strength Prediction** 🏗️

- **Source**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Concrete+Compressive+Strength)
- **Samples**: 1,030
- **Features**: 8 components
  - Cement, blast furnace slag, fly ash
  - Water, superplasticizer
  - Coarse/fine aggregate, age (days)
- **Task**: Regression (predict compressive strength in MPa)

---

## 🧠 Models Compared

| Model              | Type           | Configuration                       |
| ------------------ | -------------- | ----------------------------------- |
| **ANFIS**          | Neuro-Fuzzy    | 2 & 3 Gaussian membership functions |
| **Neural Network** | Deep Learning  | Multi-layer perceptron              |
| **SVM**            | Kernel Methods | RBF kernel                          |
| **Random Forest**  | Ensemble       | 300 trees                           |

---

## 🏗️ ANFIS Architecture

**5-Layer Takagi-Sugeno-Kang System:**

```
Input → Fuzzy Layer → Rule Layer → Norm Layer → Defuzz Layer → Output
```

1. **Fuzzy Layer**: Gaussian membership functions

   - μ(x) = exp(-(x-c)²/(2σ²))
   - Each feature: 2 or 3 MFs

2. **Rule Layer**: Fuzzy rule generation

   - Rules = n_memb ^ n_features
   - Example: 11 features × 2 MF = 2,048 rules

3. **Norm Layer**: Rule weight normalization

4. **Defuzz Layer**: TSK-type defuzzification

   - f_i = w₀ + w₁x₁ + ... + wₙxₙ

5. **Summation Layer**: Weighted output aggregation

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8+ (tested on 3.12)
- pip package manager
- 4GB RAM minimum
- ~1GB disk space

### One-Command Setup

**Linux/macOS:**

```bash
chmod +x setup.sh
./setup.sh
```

**Windows:**

```bash
setup.bat
```

This single command will:

1. Create virtual environment
2. Install all dependencies
3. Preprocess both datasets
4. Train ANFIS models (all variants)
5. Perform 5-fold cross-validation
6. Visualize membership functions
7. Generate data exploration plots
8. Train comparison models (NN, SVM, RF)
9. Create comparison charts
10. Launch Streamlit GUI at `http://localhost:8501`

**⏱️ Estimated time**: 15-30 minutes (depending on your CPU)

---

## 📁 Project Structure

```
├── setup.sh / setup.bat          # Automated setup script
├── requirements.txt               # Python dependencies
│
├── data/                          # Raw datasets
│   ├── wine-quality/
│   │   ├── winequality-red.csv
│   │   └── winequality-white.csv
│   └── concrete-strength/
│       └── Concrete_Data.csv
│
├── anfis.py                       # ANFIS core implementation
├── data_preprocessing.py          # Data loading & normalization
├── train_anfis.py                 # ANFIS training pipeline
├── train_comparison_models.py     # Train NN, SVM, RF
├── compare_all_models.py          # Generate comparison plots
├── visualize_membership_functions.py
├── data_exploration.py            # EDA visualizations
│
├── app.py                         # Streamlit web interface
│
├── models/                        # Trained model weights
└── results/                       # Generated plots & metrics
```

---

## 📊 Results & Visualizations

The automated pipeline generates:

### ANFIS Results (per dataset × MF configuration):

- Training curves (accuracy/MAE + loss)
- Prediction scatter plots
- Membership function plots
- Cross-validation metrics (5-fold)
- Fuzzy rule extraction (top-K rules)

### Data Exploration:

- Class/target distribution plots
- Feature correlation heatmaps
- Feature distribution histograms
- Pairplots for key features

### Model Comparison:

- Accuracy/MAE bar charts
- Overfitting analysis (train-test gap)
- Performance ranking table

---

## 🎯 Key Features

✅ **Fully Automated**: Single command setup  
✅ **Two Problem Types**: Classification + Regression  
✅ **Multiple Datasets**: 4 configurations (concrete, all, red, white)  
✅ **Cross-Validation**: 5-fold stratified/standard  
✅ **Interactive GUI**: Streamlit web dashboard  
✅ **Rule Extraction**: Interpretable fuzzy rules  
✅ **Comprehensive Comparison**: 4 ML algorithms  
✅ **Publication-Ready Plots**: 300 DPI PNG exports

---

## 🔬 Technical Details

### Preprocessing

- **Wine**: StandardScaler per dataset variant, 80/20 split
- **Concrete**: StandardScaler, 80/20 split
- **ANFIS Input Range**: Normalized to [-3, 3]

### Training Configuration

- **Optimizer**: Nadam (lr=0.001)
- **Epochs**: 20 (early stopping patience=10)
- **Batch Size**: 32
- **Loss Functions**:
  - Wine: Binary crossentropy
  - Concrete: Mean Squared Error

### Cross-Validation

- **Wine**: 5-fold Stratified (preserves class balance)
- **Concrete**: 5-fold Standard (regression)

---

## 📖 Documentation

- **[MANUAL_INSTRUCTION.md](MANUAL_INSTRUCTION.md)**: Detailed step-by-step installation guide
- **Code Documentation**: All functions have Polish docstrings

---

## 👥 Authors

- **Dawid Olko** - Project Lead
- **Piotr Smoła** - ML Implementation
- **Jakub Opar** - Data Analysis
- **Michał Pilecki** - Visualization

---

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

---

## 📚 References

1. **ANFIS**: J.-S. R. Jang, "ANFIS: adaptive-network-based fuzzy inference system," IEEE Transactions on Systems, Man, and Cybernetics, vol. 23, no. 3, pp. 665-685, 1993.
2. **Wine Quality Dataset**: P. Cortez et al., "Modeling wine preferences by data mining from physicochemical properties," Decision Support Systems, 2009.
3. **Concrete Dataset**: I-C. Yeh, "Modeling of strength of high-performance concrete using artificial neural networks," Cement and Concrete Research, 1998.

---

## 🐛 Troubleshooting

**Issue**: Streamlit doesn't launch automatically  
**Solution**: Manually run `streamlit run app.py` after setup completes

**Issue**: TensorFlow installation fails  
**Solution**: Ensure Python 3.8-3.12. TensorFlow 2.17 not compatible with 3.13+

**Issue**: Out of memory during training  
**Solution**: Reduce batch size in `train_anfis.py` (line 95: `batch_size=16`)

---

## ⭐ Star This Repo!

If this project helped your research or learning, please consider giving it a star ⭐

**Questions?** Open an issue on GitHub!

1. **Fuzzy Layer**: Gaussian membership functions├── results/ # Wykresy i wyniki (generowane)

   - μ(x) = exp(-(x-c)²/(2σ²))│ ├── all_models_comparison.png

   - Each feature: 2 or 3 MFs│ ├── overfitting_analysis.png

│ ├── anfis_2memb_training.png

2. **Rule Layer**: Fuzzy rule generation│ ├── anfis_3memb_training.png

   - Rules = n_memb ^ n_features│ ├── membership_functions_visualization.png

   - Example: 11 features × 2 MF = 2,048 rules│ └── \*.json (wyniki liczbowe)

├── anfis.py # ⚙️ Implementacja ANFIS

3. **Norm Layer**: Rule weight normalization├── data_exploration.py # 📊 Eksploracja danych

├── data_preprocessing.py # 🔄 Przygotowanie danych

4. **Defuzz Layer**: TSK-type defuzzification├── train_anfis.py # 🧠 Trening modeli ANFIS

   - f_i = w₀ + w₁x₁ + ... + wₙxₙ├── train_comparison_models.py # 🤖 Trening modeli porównawczych

├── compare_all_models.py # 📈 Porównanie wyników

5. **Summation Layer**: Weighted output aggregation├── visualize_membership_functions.py # 📉 Wizualizacja funkcji przynależności

├── utils.py # 🛠️ Funkcje pomocnicze (NOWE v1.1.0)

---├── scaller.py # 📐 Ładowanie scalerów (NOWE v1.1.0)

├── app.py # 🍷 Interfejs Streamlit

## 🚀 Quick Start├── main.py # 🚀 Główny pipeline

├── requirements.txt # 📦 Zależności

### Prerequisites└── .gitignore # 🚫 Pliki ignorowane przez Git

- Python 3.8+ (tested on 3.12)```

- pip package manager

- 4GB RAM minimum**Legenda:**

- ~1GB disk space

- 📁 Foldery generowane automatycznie podczas uruchomienia

### One-Command Setup- 🆕 **NOWE w v1.1.0:** Moduły `utils.py` i `scaller.py` do separacji logiki biznesowej

**Linux/macOS:**---

````bash

chmod +x setup.sh## 🔧 Wymagania

./setup.sh

```### Wymagane biblioteki:



**Windows:**```

```bashtensorflow==2.17.0

setup.batnumpy==1.26.4

```pandas==2.2.3

scikit-learn==1.5.2

This single command will:matplotlib==3.9.2

1. Create virtual environmentseaborn==0.12.2

2. Install all dependenciesstreamlit==1.39.0

3. Preprocess both datasetsh5py==3.12.1

4. Train ANFIS models (all variants)pillow==11.0.0

5. Perform 5-fold cross-validation```

6. Visualize membership functions

7. Generate data exploration plots### Instalacja:

8. Train comparison models (NN, SVM, RF)

9. Create comparison charts```bash

10. Launch Streamlit GUI at `http://localhost:8501`pip install -r requirements.txt

````

**⏱️ Estimated time**: 15-30 minutes (depending on your CPU)

### Automatyczne skrypty setup:

---

```bash

## 📁 Project Structure# Windows

setup.bat

```

├── setup.sh / setup.bat # Automated setup script# Linux/macOS

├── requirements.txt # Python dependencieschmod +x setup.sh

│./setup.sh

├── data/ # Raw datasets```

│ ├── wine-quality/

│ │ ├── winequality-red.csv---

│ │ └── winequality-white.csv

│ └── concrete-strength/## 🆕 Zmiany w wersji 1.1.0

│ └── Concrete_Data.csv

│### ✅ Optymalizacje wykonane:

├── anfis.py # ANFIS core implementation

├── data_preprocessing.py # Data loading & normalization1. **🖼️ Naprawiono blokowanie przez matplotlib**

├── train_anfis.py # ANFIS training pipeline

├── train_comparison_models.py # Train NN, SVM, RF - Dodano `matplotlib.use('Agg')` do wszystkich skryptów

├── compare_all_models.py # Generate comparison plots - Usunięto wszystkie `plt.show()` - wykresy zapisują się automatycznie

├── visualize_membership_functions.py - **Efekt:** Pipeline wykonuje się bez zatrzymywania na oknach!

├── data_exploration.py # EDA visualizations

│2. **📦 Separacja logiki biznesowej**

├── app.py # Streamlit web interface

│ - Utworzono `utils.py` - funkcje ładowania modeli ANFIS i wyników

├── models/ # Trained model weights - Utworzono `scaller.py` - centralne zarządzanie scalerami

└── results/ # Generated plots & metrics - **Efekt:** `app.py` zawiera tylko kod UI Streamlit

````

3. **🚫 Rozszerzony .gitignore**

---

   - Dodano ignorowanie wygenerowanych plików (_.npy, _.h5, _.pkl, _.png)

## 📊 Results & Visualizations   - **Efekt:** Repozytorium nie zawiera binarnych artefaktów



The automated pipeline generates:4. **📚 Pełna dokumentacja**

   - `CHANGELOG.md` - szczegółowy opis zmian technicznych

### ANFIS Results (per dataset × MF configuration):   - `PODSUMOWANIE.md` - instrukcje testowania i ocena jakości

- Training curves (accuracy/MAE + loss)

- Prediction scatter plots**Kompatybilność:** Wszystkie zmiany są wstecznie kompatybilne ✅

- Membership function plots

- Cross-validation metrics (5-fold)---

- Fuzzy rule extraction (top-K rules)

## 🚀 Instrukcja uruchomienia

### Data Exploration:

- Class/target distribution plots### **SZYBKI START** ⚡

- Feature correlation heatmaps

- Feature distribution histogramsProjekt został zoptymalizowany do bezproblemowego uruchomienia:

- Pairplots for key features

```bash

### Model Comparison:# 1. Instalacja zależności

- Accuracy/MAE bar chartspip install -r requirements.txt

- Overfitting analysis (train-test gap)

- Performance ranking table# 2. Uruchomienie pełnego pipeline'u (wszystkie kroki automatycznie)

python main.py

---

# 3. Uruchomienie interfejsu Streamlit

## 🎯 Key Featuresstreamlit run app.py

````

✅ **Fully Automated**: Single command setup

✅ **Two Problem Types**: Classification + Regression **Uwaga:** Od wersji 1.1.0 wszystkie wykresy generują się automatycznie do plików bez wyświetlania okien! 🎉

✅ **Multiple Datasets**: 4 configurations (concrete, all, red, white)

✅ **Cross-Validation**: 5-fold stratified/standard ---

✅ **Interactive GUI**: Streamlit web dashboard

✅ **Rule Extraction**: Interpretable fuzzy rules ### **KROK 1: Eksploracja danych** 📊

✅ **Comprehensive Comparison**: 4 ML algorithms

✅ **Publication-Ready Plots**: 300 DPI PNG exports```bash

python data_exploration.py

---```

## 🔬 Technical Details**Co robi ten skrypt:**

### Preprocessing- Pobiera dataset Wine Quality (czerwone i białe wino)

- **Wine**: StandardScaler per dataset variant, 80/20 split- Łączy oba datasety (6497 próbek)

- **Concrete**: StandardScaler, 80/20 split- Analizuje rozkład jakości wina (skala 3-9)

- **ANFIS Input Range**: Normalized to [-3, 3]- Sprawdza braki danych i korelacje między cechami

- Generuje wykresy:

### Training Configuration - `quality_distribution.png` - rozkład jakości wina

- **Optimizer**: Nadam (lr=0.001) - `correlation_matrix.png` - macierz korelacji cech

- **Epochs**: 20 (early stopping patience=10)

- **Batch Size**: 32**Rezultat:**

- **Loss Functions**:

  - Wine: Binary crossentropy- ✅ Pobrane dane o winie

  - Concrete: Mean Squared Error- ✅ Wygenerowane wykresy analityczne

### Cross-Validation---

- **Wine**: 5-fold Stratified (preserves class balance)

- **Concrete**: 5-fold Standard (regression)### **KROK 2: Przygotowanie danych** 🔄

---```bash

python data_preprocessing.py

## 📖 Documentation```

- **[MANUAL_INSTRUCTION.md](MANUAL_INSTRUCTION.md)**: Detailed step-by-step installation guide**Co robi ten skrypt:**

- **[CHANGES.md](CHANGES.md)**: Project evolution history

- Przekształca problem na klasyfikację binarną:

--- - **Klasa 0** (zła jakość): jakość ≤ 5

- **Klasa 1** (dobra jakość): jakość > 5

## 👥 Authors- Wybiera 11 najważniejszych cech (fixed acidity, alcohol, pH, itd.)

- Dzieli dane na zbiór treningowy (80%) i testowy (20%)

- **Dawid Olko** - Project Lead- **Standaryzuje dane** (StandardScaler) - kluczowe dla ANFIS!

- **Piotr Smoła** - ML Implementation- Zapisuje przetworzone dane do plików `.npy`

- **Jakub Opar** - Data Analysis

- **Michał Pilecki** - Visualization**Rezultat:**

---- ✅ 5197 próbek treningowych

- ✅ 1300 próbek testowych

## 📄 License- ✅ Rozkład klas: 2384 złej jakości / 4113 dobrej jakości

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.---

---### **KROK 3: Trening modeli ANFIS** 🧠

## 📚 References```bash

python train_anfis.py

1. **ANFIS**: J.-S. R. Jang, "ANFIS: adaptive-network-based fuzzy inference system," IEEE Transactions on Systems, Man, and Cybernetics, vol. 23, no. 3, pp. 665-685, 1993.```

2. **Wine Quality Dataset**: P. Cortez et al., "Modeling wine preferences by data mining from physicochemical properties," Decision Support Systems, 2009.**Co robi ten skrypt:**

3. **Concrete Dataset**: I-C. Yeh, "Modeling of strength of high-performance concrete using artificial neural networks," Cement and Concrete Research, 1998.- Trenuje 2 modele ANFIS:

- **ANFIS z 2 funkcjami przynależności** (2048 reguł)

--- - **ANFIS z 3 funkcjami przynależności** (177,147 reguł)

- Każdy model trenuje się przez 20 epok

## 🐛 Troubleshooting- Używa optymalizatora NADAM + binary crossentropy

- Zapisuje najlepsze wagi modelu (ModelCheckpoint)

**Issue**: Streamlit doesn't launch automatically - Early stopping po 15 epokach bez poprawy

**Solution**: Manually run `streamlit run app.py` after setup completes- Generuje wykresy treningu dla każdego modelu

**Issue**: TensorFlow installation fails **Warstwy modelu ANFIS:**

**Solution**: Ensure Python 3.8-3.12. TensorFlow 2.17 not compatible with 3.13+

1. **FuzzyLayer** - fuzzyfikacja (gaussowska funkcja przynależności)

**Issue**: Out of memory during training 2. **RuleLayer** - generowanie reguł rozmytych (AND)

**Solution**: Reduce batch size in `train_anfis.py` (line 95: `batch_size=16`)3. **NormLayer** - normalizacja wag reguł

4. **DefuzzLayer** - defuzzyfikacja (kombinacja liniowa Takagi-Sugeno)

---5. **SummationLayer** - agregacja wyników

## ⭐ Star This Repo!**Rezultat:**

If this project helped your research or learning, please consider giving it a star ⭐- ✅ ANFIS (2 funkcje): Test Accuracy = **69.06%**

- ✅ ANFIS (3 funkcje): Test Accuracy = **76.48%**

**Questions?** Open an issue on GitHub!- ✅ Zapisane modele w `models/`

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

### **utils.py** 🆕

Moduł pomocniczy z logiką biznesową (v1.1.0).

**Funkcje:**

- `load_anfis_model(weights_path)` - ładuje model ANFIS z pliku H5
  - Automatycznie wykrywa liczbę wejść i funkcji przynależności
  - Obsługuje pliki `.weights.h5`
- `load_results()` - agreguje wyniki wszystkich 5 modeli z JSON
  - Zwraca słownik z metrykami Train/Test Accuracy

---

### **scaller.py** 🆕

Moduł zarządzający scalerami danych (v1.1.0).

**Funkcje:**

- `load_scalers()` - ładuje oba scalery (11D i 12D)
- `get_scaler_11d()` - zwraca scaler dla ANFIS (11 cech)
- `get_scaler_12d()` - zwraca scaler dla NN/SVM/RF (12 cech z wine_type)

---

### **data_exploration.py**

Skrypt do analizy eksploracyjnej danych Wine Quality.

**Funkcje:**

- Pobieranie danych z UCI ML Repository
- Statystyki opisowe (mean, std, min, max)
- Sprawdzanie braków danych
- Wizualizacja rozkładu jakości wina
- Macierz korelacji Pearsona

**Zmiany v1.1.0:**

- ✅ Dodano `matplotlib.use('Agg')` - bez wyświetlania okien
- ✅ Usunięto `plt.show()` - automatyczny zapis do plików

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

**Zmiany v1.1.0:**

- ✅ Dodano `matplotlib.use('Agg')`
- ✅ `plot_training_history()` zapisuje zamiast wyświetlać

---

### **train_comparison_models.py**

Trening modeli porównawczych (NN, SVM, RF).

**Funkcje:**

- `train_neural_network()` - trenuje klasyczną sieć neuronową
- `train_svm()` - trenuje SVM z RBF kernel
- `train_random_forest()` - trenuje Random Forest
- `plot_training_history()` - wykresy dla NN

**Zmiany v1.1.0:**

- ✅ Dodano `matplotlib.use('Agg')`

---

### **compare_all_models.py**

Porównanie i wizualizacja wyników wszystkich modeli.

**Funkcje:**

- `load_all_results()` - wczytuje wyniki z plików JSON
- `plot_comparison_bar_chart()` - wykres słupkowy
- `plot_overfitting_analysis()` - analiza overfittingu
- `create_summary_table()` - tabela rankingowa

**Zmiany v1.1.0:**

- ✅ Usunięto `plt.show()` z dwóch funkcji wykresów

---

### **visualize_membership_functions.py**

Wizualizacja wyuczonych funkcji przynależności ANFIS.

**Funkcje:**

- Ładowanie wag modelu ANFIS
- Wykresy gaussowskich funkcji dla 6 najważniejszych cech
- Zapis do `membership_functions_visualization.png`

**Zmiany v1.1.0:**

- ✅ Dodano `matplotlib.use('Agg')`

---

### **app.py**

Interfejs użytkownika Streamlit (w trakcie rekonstrukcji).

**Strony:**

- 🏠 Strona główna - statystyki projektu
- 📊 Wyniki modeli - porównanie i ranking
- 🧠 ANFIS - teoria i wizualizacje
- 📈 Eksploracja danych - podgląd datasetu
- 🍷 Predykcja - interaktywne przewidywanie jakości wina

**Zmiany v1.1.0:**

- ✅ Przeniesiono logikę do `utils.py` i `scaller.py`
- ⚠️ Wymaga pełnej rekonstrukcji funkcjonalności

---

### **main.py**

Główny pipeline wykonujący wszystkie kroki automatycznie.

**Kolejność wykonania:**

1. data_preprocessing.py ← Przygotowanie danych
2. train_anfis.py ← Trening modeli ANFIS
3. visualize_membership_functions.py ← Wizualizacja funkcji przynależności
4. train_comparison_models.py ← Trening modeli porównawczych (NN, SVM, RF)
5. data_exploration.py ← Analiza eksploracyjna danych
6. compare_all_models.py ← Porównanie wszystkich modeli
7. app.py ← Uruchomienie GUI Streamlit

---

## 📊 Wyniki

### Finalne porównanie modeli:

| Ranking | Model             | Test Accuracy | Train Accuracy | Overfitting | Interpretacja          |
| ------- | ----------------- | ------------- | -------------- | ----------- | ---------------------- |
| 🥇      | Random Forest     | **83.23%**    | 97.69%         | 14.46% ⚠️   | ❌ Czarna skrzynka     |
| 🥈      | SVM               | **77.85%**    | 79.31%         | 1.47% ✅    | ❌ Czarna skrzynka     |
| 🥉      | ANFIS (3 funkcje) | **76.48%**    | 81.08%         | 4.59% ✅    | ✅ **Reguły rozmyte!** |
| 4       | Neural Network    | **75.69%**    | 77.45%         | 1.76% ✅    | ❌ Czarna skrzynka     |
| 5       | ANFIS (2 funkcje) | **69.06%**    | 69.81%         | 0.75% ✅    | ✅ **Reguły rozmyte!** |

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
