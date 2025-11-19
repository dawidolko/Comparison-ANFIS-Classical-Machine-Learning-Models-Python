# Jak działa projekt ANFIS - Kompletny przewodnik

## Część 1: Co robi setup.sh od początku do końca

Kiedy uruchamiasz `./setup.sh`, wykonuje się 7 kroków w określonej kolejności. Oto dokładnie co się dzieje:

### KROK 1: Przygotowanie danych (data_preprocessing.py)

Skrypt bierze surowe pliki CSV i przygotowuje je do treningu modeli.

**Dla wina:**

- Wczytuje `winequality-red.csv` i `winequality-white.csv`
- Łączy je w jeden zbiór ALBO używa osobno (3 warianty: all, red, white)
- Zamienia kolumnę `quality` (liczby 3-9) na wartości binarne: quality > 5 → 1 (dobre wino), reszta → 0 (słabe wino)
- Normalizuje wszystkie 11 cech (fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free sulfur dioxide, total sulfur dioxide, density, pH, sulphates, alcohol) używając StandardScaler - każda cecha ma średnią=0 i odchylenie=1
- Dzieli dane na train (80%) i test (20%)
- Zapisuje jako pliki `.npy`: `X_train_all.npy`, `y_train_all.npy`, `X_test_all.npy`, `y_test_all.npy` (podobnie dla red i white)
- Zapisuje scaler do `scaler_all.pkl` żeby móc później odwrócić normalizację

**Dla betonu:**

- Wczytuje `Concrete_Data.csv`
- 8 cech wejściowych (cement, slag, fly ash, water, superplasticizer, coarse aggregate, fine aggregate, age)
- 1 cel: concrete compressive strength (wytrzymałość w MPa) - WAŻNE: to jest regresja, nie klasyfikacja!
- Normalizacja StandardScaler
- Podział 80/20
- Zapisuje do `concrete-strength/X_train.npy`, `y_train.npy`, `X_test.npy`, `y_test.npy`

Po tym kroku masz gotowe dane w formacie numpy arrays, znormalizowane i podzielone.

---

### KROK 2: Trening ANFIS (train_anfis.py)

Teraz zaczyna się właściwy trening modeli ANFIS. Skrypt trenuje 8 różnych modeli:

**6 modeli dla wina:**

1. `all_2memb` - wszystkie wina, 2 funkcje przynależności na cechę
2. `all_3memb` - wszystkie wina, 3 funkcje przynależności na cechę
3. `red_2memb` - tylko czerwone wina, 2 funkcje
4. `red_3memb` - tylko czerwone wina, 3 funkcje
5. `white_2memb` - tylko białe wina, 2 funkcje
6. `white_3memb` - tylko białe wina, 3 funkcje

**2 modele dla betonu:** 7. `concrete_2memb` - beton, 2 funkcje przynależności 8. `concrete_3memb` - beton, 3 funkcje przynależności

Dla każdego modelu skrypt:

**A) Tworzy model ANFIS:**

```python
model = ANFISModel(n_input=11, n_memb=2, regression=False)  # Wine
# LUB
model = ANFISModel(n_input=8, n_memb=3, regression=True)  # Concrete
```

**B) Kompiluje go:**

- Wine (klasyfikacja): loss='binary_crossentropy', optimizer='nadam', metrics=['accuracy']
- Concrete (regresja): loss='mse', optimizer='nadam', metrics=['mae']

**C) Trenuje przez max 20 epok:**

- Każda epoka: model przetwarza wszystkie batche (32 próbki naraz)
- Walidacja na zbiorze testowym co epokę
- Early stopping: jeśli val_loss nie poprawia się przez 10 epok → stop
- Model checkpoint: zapisuje najlepsze wagi do `models/anfis_all_best_2memb.weights.h5`

**D) Po treningu zapisuje:**

- `results/anfis_all_2memb_results.json` - metryki (train/test accuracy, loss, MAE, R²)
- `results/anfis_all_2memb_rules.json` - wyekstrahowane reguły rozmyte (więcej o tym później)
- `results/anfis_all_2memb_training.png` - wykresy krzywych uczenia (accuracy vs epoki, loss vs epoki)
- `results/anfis_all_2memb_fit.png` - scatter plot: prawdziwe wartości vs predykcje

**E) Cross-validation (5-fold):**

- Dzieli dane treningowe na 5 części
- 5 razy: trenuje na 4 częściach, testuje na 1
- Zapisuje `results/anfis_all_2memb_cv.json` - średnie i odchylenie standardowe metryk

Cały ten proces zajmuje 15-30 minut w zależności od procesora.

---

### KROK 3: Wizualizacja funkcji przynależności (visualize_membership_functions.py)

Dla każdego wytrenowanego modelu ANFIS skrypt tworzy wykresy pokazujące jak wyglądają funkcje Gaussa.

Model ma wyuczone parametry `c` (centrum) i `sigma` (szerokość) dla każdej funkcji przynależności. Skrypt:

- Bierze wartości minimalne i maksymalne każdej cechy ze zbioru treningowego
- Generuje 200 punktów w tym zakresie
- Dla każdego punktu oblicza wartość funkcji Gaussa: `exp(-(x - c)² / (2*sigma²))`
- Rysuje wykresy dla wybranych cech (np. dla wina: alcohol, pH, sulphates; dla betonu: cement, water, age)

Zapisuje jako `results/membership_functions_all_2memb.png` (subplot z kilkoma cechami).

To pozwala zobaczyć jak model "widzi" dane - jakie zakresy wartości uważa za "niskie", "średnie", "wysokie".

---

### KROK 4: Trening modeli porównawczych (train_comparison_models.py)

Teraz trenujemy 3 klasyczne modele ML do porównania z ANFIS.

**Dla wina (klasyfikacja):**

1. **Neural Network:**

   - Architektura: Input(11) → Dense(16, relu) → Dropout(0.3) → Dense(8, relu) → Dropout(0.2) → Dense(1, sigmoid)
   - Adam optimizer, learning rate=0.001
   - Early stopping po 15 epokach bez poprawy
   - Class weights bo dane są niezbalansowane (więcej dobrych win)
   - Zapisuje: `models/nn_wine.keras`, `results/nn_wine_results.json`

2. **SVM (Support Vector Machine):**

   - RBF kernel (Radial Basis Function)
   - C=1.0 (parametr regularyzacji)
   - gamma='scale' (szerokość kernela)
   - Class weights
   - Zapisuje: `results/svm_wine_results.json`

3. **Random Forest:**
   - 200 drzew decyzyjnych
   - max_depth=15 (maksymalna głębokość drzewa)
   - min_samples_split=5
   - Class weights
   - Zapisuje: `results/rf_wine_results.json`

**Dla betonu (regresja):**

1. **Neural Network:**

   - Input(8) → Dense(32, relu) → Dropout(0.3) → Dense(16, relu) → Dropout(0.2) → Dense(1, linear)
   - Loss: MSE, metryka: MAE
   - Zapisuje: `models/nn_concrete.keras`, `results/nn_concrete_results.json`

2. **SVM Regressor:**

   - RBF kernel
   - C=1.0, epsilon=0.1
   - Zapisuje: `results/svm_concrete_results.json`

3. **Random Forest Regressor:**
   - 200 drzew, max_depth=15
   - Zapisuje: `results/rf_concrete_results.json`

Każdy model zapisuje metryki: train accuracy/MAE, test accuracy/MAE, czas treningu.

---

### KROK 5: Eksploracja danych (data_exploration.py)

Tworzy wykresy analizy danych (EDA - Exploratory Data Analysis).

**Dla wina:**

- `wine_class_distribution.png` - wykres słupkowy: ile próbek ma quality=0 vs quality=1
- `wine_correlation.png` - macierz korelacji (heatmap) pokazująca zależności między cechami
- `wine_feature_distributions.png` - histogramy wszystkich 11 cech
- `wine_pairplot.png` - scatter plots par cech (np. alcohol vs pH)

**Dla betonu:**

- `concrete_strength_distribution.png` - histogram wytrzymałości betonu
- `concrete_correlation.png` - macierz korelacji 8 cech
- `concrete_feature_distributions.png` - histogramy cech
- `concrete_strength_vs_features.png` - scatter plots: cement vs strength, water vs strength, age vs strength

Wszystko zapisane w `results/`.

---

### KROK 6: Porównanie wszystkich modeli (compare_all_models.py)

Wczytuje wszystkie pliki `*_results.json` i tworzy porównawcze wykresy.

**Dla wina:**

- `model_comparison_bar_wine.png` - wykres słupkowy porównujący accuracy wszystkich modeli (ANFIS 2MF, ANFIS 3MF, NN, SVM, RF)
- `overfitting_analysis_wine.png` - wykres train accuracy vs test accuracy dla każdego modelu (pokazuje czy model się przeuczył)

**Dla betonu:**

- `model_comparison_bar_concrete.png` - wykres słupkowy porównujący MAE wszystkich modeli
- `overfitting_analysis_concrete.png` - train MAE vs test MAE

To pozwala łatwo zobaczyć który model działa najlepiej.

---

### KROK 7: Uruchomienie GUI (streamlit run app.py)

Na końcu setup.sh uruchamia aplikację webową Streamlit w tle.

Otwiera się przeglądarka na `http://localhost:8501` z 5 zakładkami:

- Home - opis projektu
- ANFIS Results - interaktywne przeglądanie wyników ANFIS
- Rules & History - reguły rozmyte i historia treningu
- Data Analysis - wykresy EDA
- Model Comparison - porównanie ANFIS vs NN/SVM/RF

Aplikacja działa dopóki nie zamkniesz terminala lub nie wciśniesz Ctrl+C.

---

## Część 2: Co to jest ANFIS i jak działa

ANFIS to skrót od Adaptive Neuro-Fuzzy Inference System. To model łączący sieci neuronowe z logiką rozmytą.

### Dlaczego ANFIS?

Zwykła sieć neuronowa to "czarna skrzynka" - nie wiesz dlaczego podjęła decyzję.
ANFIS generuje REGUŁY które możesz przeczytać:

```
JEŚLI alcohol jest WYSOKI (0.85) AND pH jest ŚREDNI (0.60)
TO quality = 0.5 + 0.03*alcohol - 0.02*pH = 0.76 (prawdopodobieństwo dobrego wina)
```

Taka reguła ma interpretację: "Wino z wysokim alkoholem i średnim pH będzie raczej dobre".

### Architektura ANFIS - 5 warstw

ANFIS to przepływ danych przez 5 warstw. Każda robi coś innego.

```
Wejście (x) → [Warstwa 1] → [Warstwa 2] → [Warstwa 3] → [Warstwa 4] → [Warstwa 5] → Wyjście (y)
             Fuzzyfikacja   Reguły       Normalizacja  Defuzzyfikacja  Agregacja
```

### **WARSTWA 1: Fuzzyfikacja (FuzzyLayer)** 🌫️

**Co robi:** Zamienia liczby ostre na "stopnie przynależności" do zbiorów rozmytych.

**Wzór - Gaussowska funkcja przynależności:**

```
μᵢⱼ(x) = exp(-(xⱼ - cᵢⱼ)² / (2σᵢⱼ²))
```

Gdzie:

- `xⱼ` = wartość j-tej cechy wejściowej (np. alkohol = 12.5%)
- `cᵢⱼ` = **centrum** i-tej funkcji przynależności dla cechy j (UCZONY parametr!)
- `σᵢⱼ` = **szerokość** funkcji (UCZONY parametr!)
- `μᵢⱼ` = stopień przynależności (0.0 do 1.0)

**Przykład:**

```
Dla cechy "alkohol":
- Funkcja 1 (LOW):  c=10, σ=1.5 → dla x=9  → μ=0.85 (wysoki stopień "LOW")
- Funkcja 2 (HIGH): c=14, σ=1.5 → dla x=9  → μ=0.03 (niski stopień "HIGH")
```

**Wyjście:** Tensor kształtu **(batch_size, n_memb, n_features)**  
Dla Wine: `(32, 2, 11)` - 2 funkcje przynależności dla każdej z 11 cech

**Kod w `anfis.py`:**

```python
class FuzzyLayer(tf.keras.layers.Layer):
    def call(self, x):
        x = tf.expand_dims(x, axis=1)  # (B, n) → (B, 1, n)
        z = (x - self.c) / (self.sigma + 1e-8)
        mu = tf.exp(-0.5 * tf.square(z))  # Gaussa
        return tf.clip_by_value(mu, 1e-8, 1.0)
```

---

### **WARSTWA 2: Tworzenie Reguł (RuleLayer)** 📜

**Co robi:** Tworzy WSZYSTKIE możliwe kombinacje funkcji przynależności = reguły rozmyte.

**Wzór - T-norma (AND) przez iloczyn:**

```
wₖ = μ₁ₖ₁ × μ₂ₖ₂ × ... × μₙₖₙ
```

Gdzie:

- `wₖ` = siła k-tej reguły (0.0 do 1.0)
- `k = (k₁, k₂, ..., kₙ)` = kombinacja indeksów funkcji przynależności

**Przykład dla 2 cech × 2 MF:**

```
Reguła 1: μ₁(LOW) × μ₂(LOW)   = 0.85 × 0.90 = 0.765
Reguła 2: μ₁(LOW) × μ₂(HIGH)  = 0.85 × 0.10 = 0.085
Reguła 3: μ₁(HIGH) × μ₂(LOW)  = 0.15 × 0.90 = 0.135
Reguła 4: μ₁(HIGH) × μ₂(HIGH) = 0.15 × 0.10 = 0.015
```

**Liczba reguł:** `n_memb^n_features`

- Wine (11 cech, 2 MF): 2^11 = **2,048 reguł**
- Wine (11 cech, 3 MF): 3^11 = **177,147 reguł** 😱
- Concrete (8 cech, 3 MF): 3^8 = **6,561 reguł**

**Wyjście:** Tensor **(batch_size, n_rules)**  
Dla Wine 3MF: `(32, 177147)` - siła każdej reguły

**Kod w `anfis.py`:**

```python
class RuleLayer(tf.keras.layers.Layer):
    def call(self, mu):
        out = mu[:, :, 0]  # Pierwsza cecha
        for i in range(1, self.n):
            out = tf.einsum("bm,bn->bmn", out, mu[:, :, i])  # Iloczyn
            out = tf.reshape(out, (tf.shape(mu)[0], -1))
        return out
```

---

### **WARSTWA 3: Normalizacja (NormLayer)** ⚖️

**Co robi:** Normalizuje siły reguł tak, aby sumowały się do 1.

**Wzór:**

```
w̄ₖ = wₖ / (w₁ + w₂ + ... + wₙ)
```

**Przykład:**

```
w = [0.765, 0.085, 0.135, 0.015]  → suma = 1.0
w̄ = [0.765, 0.085, 0.135, 0.015] (już znormalizowane)
```

**Wyjście:** Tensor **(batch_size, n_rules)** - znormalizowane wagi

**Kod w `anfis.py`:**

```python
class NormLayer(tf.keras.layers.Layer):
    def call(self, w):
        s = tf.reduce_sum(w, axis=1, keepdims=True)
        return w / (s + 1e-8)
```

---

### **WARSTWA 4: Defuzzyfikacja (DefuzzLayer)** 🎯

**Co robi:** Oblicza **konsekwent** każdej reguły (część THEN) według modelu TSK-1.

**Wzór konsekwentu k-tej reguły:**

```
fₖ = w₀ₖ + w₁ₖx₁ + w₂ₖx₂ + ... + wₙₖxₙ
```

Gdzie:

- `w₀ₖ` = **bias** k-tej reguły (UCZONY parametr!)
- `w₁ₖ, w₂ₖ, ...` = **wagi** konsekwentu (UCZONE parametry!)

**Potem mnoży przez znormalizowaną wagę:**

```
yₖ = w̄ₖ × fₖ
```

**Przykład:**

```
Reguła 1: f₁ = 0.5 + 0.3×alkohol - 0.1×kwasowość = 0.5 + 0.3×12 - 0.1×5 = 4.1
         y₁ = 0.765 × 4.1 = 3.14

Reguła 2: f₂ = -0.2 + 0.5×alkohol + 0.2×kwasowość = -0.2 + 0.5×12 + 0.2×5 = 6.8
         y₂ = 0.085 × 6.8 = 0.58
```

**Wyjście:** Tensor **(batch_size, n_rules)** - wkład każdej reguły

**Kod w `anfis.py`:**

```python
class DefuzzLayer(tf.keras.layers.Layer):
    def call(self, w_norm, x):
        y = tf.matmul(x, self.CP_weight) + self.CP_bias  # Konsekwent TSK
        return w_norm * y  # Mnożenie przez wagę reguły
```

---

### **WARSTWA 5: Agregacja (SummationLayer)** ➕

**Co robi:** Sumuje wkłady wszystkich reguł = końcowe wyjście ANFIS.

**Wzór:**

```
y = Σₖ yₖ = Σₖ (w̄ₖ × fₖ)
```

**Przykład:**

```
y = 3.14 + 0.58 + 0.40 + 0.05 = 4.17
```

Dla **klasyfikacji** (Wine): `y` przechodzi przez **sigmoid** → prawdopodobieństwo (0-1)  
Dla **regresji** (Concrete): `y` pozostaje **linear** → wartość MPa (0-100)

**Wyjście:** Tensor **(batch_size, 1)** - predykcja końcowa

**Kod w `anfis.py`:**

```python
class SummationLayer(tf.keras.layers.Layer):
    def call(self, per_rule):
        return tf.reduce_sum(per_rule, axis=1, keepdims=True)
```

---

## 🎓 Jak działa trening?

### 1. **Forward pass** (przepływ w przód)

```
x → FuzzyLayer → RuleLayer → NormLayer → DefuzzLayer → SummationLayer → Activation → ŷ
```

### 2. **Obliczenie błędu (Loss)**

**Dla Wine (klasyfikacja):**

```
Loss = Binary Cross-Entropy = -[y×log(ŷ) + (1-y)×log(1-ŷ)]
```

**Dla Concrete (regresja):**

```
Loss = MSE = (y - ŷ)²
MAE = |y - ŷ|  (metry pomocnicza)
```

### 3. **Backward pass** (propagacja wsteczna)

TensorFlow automatycznie oblicza gradienty dla WSZYSTKICH parametrów:

- Centra `c` i szerokości `σ` funkcji Gaussa (Warstwa 1)
- Wagi `w` i bias `b` konsekwentów TSK (Warstwa 4)

### 4. **Aktualizacja parametrów**

Optymalizator **Nadam** (Adam z Nesterov momentum):

```
θ_new = θ_old - learning_rate × gradient
```

Learning rate = **0.001** (stała)

### 5. **Early Stopping**

Trening kończy się gdy **val_loss** nie poprawia się przez **10 epok**.

---

## 📊 Przepływ danych w setup.sh

### **KROK 1: data_preprocessing.py**

```
CSV → pandas → binary labels → StandardScaler → train/test split → .npy + .pkl
```

**Dla Wine:**

```python
quality → (quality > 5).astype(int) → y_binary
11 cech → StandardScaler.fit_transform() → X_normalized
```

**Dla Concrete:**

```python
8 cech → StandardScaler.fit_transform() → X_normalized
strength (MPa) → y (bez zmian, regresja)
```

---

### **KROK 2: train_anfis.py**

#### **2.1 Tworzenie modelu**

```python
# Dla Wine (klasyfikacja)
model = ANFISModel(n_input=11, n_memb=2, regression=False)
model.compile(loss="binary_crossentropy", optimizer="nadam", metrics=["accuracy"])

# Dla Concrete (regresja)
model = ANFISModel(n_input=8, n_memb=3, regression=True)
model.compile(loss="mse", optimizer="nadam", metrics=["mae"])
```

#### **2.2 Trening**

```python
model.fit(X_train, y_train, validation_data=(X_test, y_test),
          epochs=20, callbacks=[ModelCheckpoint, EarlyStopping])
```

**Callbacks:**

- `ModelCheckpoint` - zapisuje najlepsze wagi do `models/anfis_*.weights.h5`
- `EarlyStopping` - zatrzymuje trening po 10 epokach bez poprawy

#### **2.3 Wizualizacja**

```python
plot_training_history()  # Krzywe accuracy/MAE + loss
plot_fit_on_train()      # Scatter plot y_true vs y_pred + R²
```

#### **2.4 Ekstrakcja reguł**

```python
centers, sigmas = model.get_membership_functions()
weights, bias = model.weights, model.bias

# Dla każdej reguły k:
rule_k = {
    "membership_indices": [k1, k2, ..., kn],  # Które MF są aktywne
    "consequent": {
        "weights": [w1k, w2k, ..., wnk],
        "bias": w0k
    }
}
```

**Zapis do `results/anfis_*_rules.json`**

---

### **KROK 3: visualize_membership_functions.py**

```python
# Dla każdej cechy j:
x_range = np.linspace(X_min[j], X_max[j], 200)

for i in range(n_memb):
    mu = exp(-(x_range - c[i,j])² / (2*sigma[i,j]²))
    plt.plot(x_range, mu, label=f"MF {i+1}")
```

**Wyjście:** `results/membership_functions_*.png` - wykresy Gaussa

---

### **KROK 4: train_comparison_models.py**

Trenuje 3 klasyczne modele:

**Neural Network:**

```
Input(11) → Dense(16, relu) → Dropout(0.3) → Dense(8, relu) → Dropout(0.2) → Dense(1, sigmoid)
```

**SVM:**

```
RBF kernel, C=1.0, gamma='scale'
```

**Random Forest:**

```
200 drzew, max_depth=15
```

---

### **KROK 5: data_exploration.py**

Generuje wykresy EDA:

- Rozkład klas (`wine_class_distribution.png`)
- Macierz korelacji (`wine_correlation.png`)
- Histogramy cech (`wine_feature_distributions.png`)
- Pairplot (`wine_pairplot.png`)

---

### **KROK 6: compare_all_models.py**

Wczytuje wszystkie `*_results.json` i tworzy:

- `model_comparison_bar.png` - wykres słupkowy accuracy/MAE
- `overfitting_analysis.png` - train vs test gap

---

### **KROK 7: app.py (Streamlit)**

```python
streamlit run app.py
```

Uruchamia GUI na `http://localhost:8501` z 5 zakładkami.

---

## 🔢 Przykład numeryczny ANFIS

### **Problem:** Przewidzieć jakość wina na podstawie alkoholu i kwasowości

**Dane:**

```
x₁ = alkohol = 12.0%
x₂ = kwasowość = 5.0
y = jakość = 1 (dobra)
```

**Model:** ANFIS z 2 funkcjami przynależności (LOW, HIGH)

---

### **WARSTWA 1: Fuzzyfikacja**

**Parametry (wyuczone):**

```
c₁₁=10, σ₁₁=2  (alkohol LOW)
c₂₁=14, σ₂₁=2  (alkohol HIGH)
c₁₂=4, σ₁₂=1   (kwasowość LOW)
c₂₂=7, σ₂₂=1   (kwasowość HIGH)
```

**Obliczenia:**

```
μ₁₁ = exp(-(12-10)²/(2×2²)) = exp(-0.5) = 0.606  (alkohol LOW)
μ₂₁ = exp(-(12-14)²/(2×2²)) = exp(-0.5) = 0.606  (alkohol HIGH)
μ₁₂ = exp(-(5-4)²/(2×1²))   = exp(-0.5) = 0.606  (kwasowość LOW)
μ₂₂ = exp(-(5-7)²/(2×1²))   = exp(-2.0) = 0.135  (kwasowość HIGH)
```

---

### **WARSTWA 2: Reguły**

**4 reguły (2×2):**

```
w₁ = μ₁₁ × μ₁₂ = 0.606 × 0.606 = 0.367  (LOW alkohol AND LOW kwasowość)
w₂ = μ₁₁ × μ₂₂ = 0.606 × 0.135 = 0.082  (LOW alkohol AND HIGH kwasowość)
w₃ = μ₂₁ × μ₁₂ = 0.606 × 0.606 = 0.367  (HIGH alkohol AND LOW kwasowość)
w₄ = μ₂₁ × μ₂₂ = 0.606 × 0.135 = 0.082  (HIGH alkohol AND HIGH kwasowość)
```

---

### **WARSTWA 3: Normalizacja**

```
suma = 0.367 + 0.082 + 0.367 + 0.082 = 0.898

w̄₁ = 0.367/0.898 = 0.409
w̄₂ = 0.082/0.898 = 0.091
w̄₃ = 0.367/0.898 = 0.409
w̄₄ = 0.082/0.898 = 0.091
```

---

### **WARSTWA 4: Defuzzyfikacja**

**Parametry konsekwentów (wyuczone):**

```
Reguła 1: w₀=0.5, w₁=0.03, w₂=-0.05
Reguła 2: w₀=-0.2, w₁=0.02, w₂=0.08
Reguła 3: w₀=0.8, w₁=0.05, w₂=-0.03
Reguła 4: w₀=0.1, w₁=0.04, w₂=0.02
```

**Obliczenia konsekwentów:**

```
f₁ = 0.5 + 0.03×12 - 0.05×5 = 0.61
f₂ = -0.2 + 0.02×12 + 0.08×5 = 0.44
f₃ = 0.8 + 0.05×12 - 0.03×5 = 1.25
f₄ = 0.1 + 0.04×12 + 0.02×5 = 0.68
```

**Wkłady reguł:**

```
y₁ = 0.409 × 0.61 = 0.249
y₂ = 0.091 × 0.44 = 0.040
y₃ = 0.409 × 1.25 = 0.511
y₄ = 0.091 × 0.68 = 0.062
```

---

### **WARSTWA 5: Agregacja + Activation**

```
y_raw = 0.249 + 0.040 + 0.511 + 0.062 = 0.862

y_final = sigmoid(0.862) = 1/(1+e^(-0.862)) = 0.703
```

**Interpretacja:** Model przewiduje prawdopodobieństwo **70.3%**, że wino jest dobrej jakości.

**Rzeczywista etykieta:** `y=1` → wino jest dobre ✓

**Loss (Binary Cross-Entropy):**

```
Loss = -[1×log(0.703) + 0×log(0.297)] = -log(0.703) = 0.352
```

---

## 🎯 Kluczowe wnioski

### 1. **ANFIS to "biała skrzynka"**

- Możesz zobaczyć, które reguły są aktywne
- Możesz zinterpretować wagi konsekwentów
- Przykład: "Jeśli alkohol wysoki i kwasowość niska → jakość dobra (waga 0.511)"

### 2. **Parametry uczą się automatycznie**

- Centra i szerokości funkcji Gaussa
- Wagi i bias konsekwentów TSK
- **Gradient descent** przez TensorFlow

### 3. **Liczba reguł rośnie wykładniczo**

- 2 MF, 11 cech → 2,048 reguł
- 3 MF, 11 cech → 177,147 reguł (!!)
- Dlatego używamy **top-K** reguł w ekstrakcji

### 4. **ANFIS działa dobrze na małych zbiorach**

- Wine: 5,197 próbek treningowych
- Concrete: 824 próbki
- NN/SVM/RF często wymagają więcej danych

---

## 📚 Bibliografia

1. **Jang, J.-S. R. (1993)**. "ANFIS: Adaptive-Network-Based Fuzzy Inference System"  
   _IEEE Transactions on Systems, Man, and Cybernetics_, vol. 23, no. 3, pp. 665-685.

2. **Takagi, T., & Sugeno, M. (1985)**. "Fuzzy identification of systems and its applications to modeling and control"  
   _IEEE Transactions on Systems, Man, and Cybernetics_, vol. 15, no. 1, pp. 116-132.

---

## 🔗 Dodatkowe materiały

- **Kod źródłowy ANFIS:** [`anfis.py`](anfis.py)
- **Skrypt treningu:** [`train_anfis.py`](train_anfis.py)
- **Dokumentacja widoków GUI:** [`WIDOKI_APLIKACJI.md`](WIDOKI_APLIKACJI.md)
- **README projektu:** [`README.md`](README.md)
