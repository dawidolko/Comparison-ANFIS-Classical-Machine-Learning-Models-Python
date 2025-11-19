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

Parametr `regression=True/False` decyduje czy ostatnia warstwa ma aktywację liniową (regresja) czy sigmoid (klasyfikacja).

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

---

#### WARSTWA 1: Fuzzyfikacja

Ta warstwa zamienia ostre liczby na "stopnie przynależności".

Przykład: masz alkohol = 12%. Zamiast powiedzieć "to 12%", ANFIS mówi:

- "To jest w 85% niski alkohol"
- "To jest w 15% wysoki alkohol"

Używa do tego funkcji Gaussa (krzywa dzwonowa):

```
μ(x) = exp(-(x - c)² / (2*sigma²))
```

Gdzie:

- `x` to wartość cechy (np. 12% alkoholu)
- `c` to centrum funkcji (np. c=10 dla "niskiego alkoholu")
- `sigma` to szerokość (im większa, tym funkcja jest szersza)

Model uczy się parametrów `c` i `sigma` automatycznie podczas treningu.

Dla wina z 11 cechami i 2 funkcjami przynależności masz 2×11 = 22 funkcje Gaussa.
Dla betonu z 8 cechami i 3 funkcjami masz 3×8 = 24 funkcje.

**Co wyjdzie z tej warstwy:**
Tensor o kształcie (batch_size, n_memb, n_features)
Dla wina: (32, 2, 11) - dla każdej z 32 próbek w batchu masz 2 stopnie przynależności dla każdej z 11 cech

---

#### WARSTWA 2: Tworzenie reguł

Ta warstwa tworzy wszystkie możliwe kombinacje funkcji przynależności = reguły.

Dla każdej cechy bierzesz jedną funkcję przynależności i mnożysz je ze sobą (operator AND w logice rozmytej).

Przykład dla 2 cech (alkohol, pH) i 2 funkcji na cechę:

```
Reguła 1: alkohol_LOW × pH_LOW
Reguła 2: alkohol_LOW × pH_HIGH
Reguła 3: alkohol_HIGH × pH_LOW
Reguła 4: alkohol_HIGH × pH_HIGH
```

Liczba reguł = n_memb^n_features

Dla wina (11 cech, 2 MF): 2^11 = 2048 reguł
Dla wina (11 cech, 3 MF): 3^11 = 177147 reguł (!)
Dla betonu (8 cech, 3 MF): 3^8 = 6561 reguł

Każda reguła ma swoją "siłę" (wartość od 0 do 1) - to są te pomnożone stopnie przynależności.

**Co wyjdzie z tej warstwy:**
Tensor (batch_size, n_rules)
Dla wina 2MF: (32, 2048) - siła każdej z 2048 reguł dla każdej próbki

---

#### WARSTWA 3: Normalizacja

Bierze siły reguł i normalizuje je tak żeby sumowały się do 1.

```
w_znormalizowane = w / (suma wszystkich w)
```

To sprawia że wyjście będzie stabilne niezależnie od tego jak mocne są funkcje przynależności.

**Co wyjdzie:**
Tensor (batch_size, n_rules) - znormalizowane siły reguł

---

#### WARSTWA 4: Defuzzyfikacja (konsekwenty TSK)

To jest najbardziej kluczowa warstwa. Tutaj każda reguła oblicza swój wkład do końcowego wyniku.

Model Takagi-Sugeno-Kang (TSK) mówi że konsekwent każdej reguły to równanie liniowe:

```
f_k = w0 + w1*x1 + w2*x2 + ... + wn*xn
```

Gdzie `w0, w1, w2, ...` to wagi które model się uczy.

Przykład dla reguły "alkohol_HIGH AND pH_LOW":

```
f = 0.5 + 0.03*alkohol - 0.02*pH + 0.01*sulphates + ...
```

Jeśli alkohol=12, pH=3.2, sulphates=0.5:

```
f = 0.5 + 0.03*12 - 0.02*3.2 + 0.01*0.5 = 0.5 + 0.36 - 0.064 + 0.005 = 0.801
```

Potem mnożysz to przez znormalizowaną siłę reguły:

```
wkład_reguły = w_znormalizowane * f
```

Jeśli ta reguła ma siłę 0.35:

```
wkład = 0.35 * 0.801 = 0.280
```

**Co wyjdzie:**
Tensor (batch_size, n_rules) - wkład każdej reguły do końcowego wyniku

---

#### WARSTWA 5: Agregacja

Po prostu sumuje wkłady wszystkich reguł:

```
y = suma wszystkich wkładów
```

Dla klasyfikacji (wino) ten wynik przechodzi przez sigmoid → wartość 0-1 (prawdopodobieństwo)
Dla regresji (beton) pozostaje jak jest → wartość w MPa

**Co wyjdzie:**
Tensor (batch_size, 1) - końcowa predykcja dla każdej próbki

---

## Konkretny przykład numeryczny

Załóżmy że chcemy przewidzieć jakość wina na podstawie 2 cech: alkohol i pH.
Model ma 2 funkcje przynależności (LOW, HIGH).

**Dane wejściowe:**

```
alkohol = 12.0
pH = 3.2
```

### Krok 1: Fuzzyfikacja

Wyuczone parametry:

```
alkohol_LOW:  c=10, sigma=2
alkohol_HIGH: c=14, sigma=2
pH_LOW:       c=3, sigma=0.5
pH_HIGH:      c=4, sigma=0.5
```

Obliczenia:

```
alkohol_LOW = exp(-(12-10)²/(2*2²)) = exp(-0.5) = 0.606
alkohol_HIGH = exp(-(12-14)²/(2*2²)) = exp(-0.5) = 0.606
pH_LOW = exp(-(3.2-3)²/(2*0.5²)) = exp(-0.08) = 0.923
pH_HIGH = exp(-(3.2-4)²/(2*0.5²)) = exp(-1.28) = 0.278
```

### Krok 2: Reguły

```
Reguła 1 (alkohol_LOW AND pH_LOW):   0.606 * 0.923 = 0.559
Reguła 2 (alkohol_LOW AND pH_HIGH):  0.606 * 0.278 = 0.168
Reguła 3 (alkohol_HIGH AND pH_LOW):  0.606 * 0.923 = 0.559
Reguła 4 (alkohol_HIGH AND pH_HIGH): 0.606 * 0.278 = 0.168
```

### Krok 3: Normalizacja

```
suma = 0.559 + 0.168 + 0.559 + 0.168 = 1.454

Reguła 1: 0.559 / 1.454 = 0.384
Reguła 2: 0.168 / 1.454 = 0.116
Reguła 3: 0.559 / 1.454 = 0.384
Reguła 4: 0.168 / 1.454 = 0.116
```

### Krok 4: Konsekwenty

Wyuczone wagi dla każdej reguły (to są parametry które model się nauczył):

```
Reguła 1: f1 = 0.5 + 0.03*alkohol - 0.05*pH = 0.5 + 0.03*12 - 0.05*3.2 = 0.70
Reguła 2: f2 = -0.2 + 0.02*alkohol + 0.08*pH = -0.2 + 0.02*12 + 0.08*3.2 = 0.296
Reguła 3: f3 = 0.8 + 0.05*alkohol - 0.03*pH = 0.8 + 0.05*12 - 0.03*3.2 = 1.304
Reguła 4: f4 = 0.1 + 0.04*alkohol + 0.02*pH = 0.1 + 0.04*12 + 0.02*3.2 = 0.644
```

Wkłady:

```
Reguła 1: 0.384 * 0.70 = 0.269
Reguła 2: 0.116 * 0.296 = 0.034
Reguła 3: 0.384 * 1.304 = 0.501
Reguła 4: 0.116 * 0.644 = 0.075
```

### Krok 5: Agregacja

```
y_raw = 0.269 + 0.034 + 0.501 + 0.075 = 0.879
y_final = sigmoid(0.879) = 1/(1+e^(-0.879)) = 0.707
```

**Wynik:** Model przewiduje że to wino ma 70.7% szans na to że jest dobrej jakości (quality > 5).

---

## Jak działa trening?

Trening ANFIS to standardowy gradient descent jak w sieciach neuronowych.

### Co się uczy?

1. **Centra i szerokości funkcji Gaussa** (Warstwa 1)
   - Dla wina 2MF: 2\*11 = 22 centra + 22 sigmy = 44 parametry
2. **Wagi konsekwentów TSK** (Warstwa 4)
   - Dla wina 2MF: 2048 reguł × (11 wag + 1 bias) = 24576 parametrów

W sumie dla ANFIS wina 2MF: ~24620 parametrów do wyuczenia.

### Proces treningu:

1. **Forward pass** - przepuść batch przez 5 warstw, dostań predykcję
2. **Oblicz loss:**
   - Wine: Binary Cross-Entropy = -[y*log(ŷ) + (1-y)*log(1-ŷ)]
   - Concrete: MSE = (y - ŷ)²
3. **Backward pass** - TensorFlow automatycznie oblicza gradienty dla wszystkich parametrów
4. **Update parametrów** - optymalizator Nadam (Adam z momentum) aktualizuje wagi:

   ```
   parametr_nowy = parametr_stary - learning_rate * gradient
   ```

   learning_rate = 0.001

5. **Powtarzaj** przez kolejne batche i epoki dopóki val_loss się poprawia

### Early stopping:

Jeśli val_loss nie poprawia się przez 10 epok → zatrzymaj trening.
To zapobiega przeuczeniu (overfitting).

---

## Ekstrakcja reguł

Po treningu model ma wyuczone parametry. Możesz je wyciągnąć i zapisać jako reguły w JSON.

Przykładowa reguła w `anfis_all_2memb_rules.json`:

```json
{
  "rule_0": {
    "membership_indices": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "consequent": {
      "weights": [
        0.03, -0.05, 0.01, 0.02, -0.01, 0.0, 0.01, -0.02, 0.04, 0.01, 0.05
      ],
      "bias": 0.5
    }
  }
}
```

Interpretacja:

- `membership_indices` = [0,0,0,0,0,0,0,0,0,0,0] oznacza że wszystkie cechy używają funkcji przynależności nr 0 (LOW)
- `weights` = wagi w konsekwencie TSK: f = 0.50 + 0.03*x1 - 0.05*x2 + ...
- `bias` = wyraz wolny w konsekwencie

Możesz to przetłumaczyć na:

```
JEŚLI fixed_acidity jest LOW AND volatile_acidity jest LOW AND ... AND alcohol jest LOW
TO quality = 0.50 + 0.03*fixed_acidity - 0.05*volatile_acidity + ... + 0.05*alcohol
```

---

## Podsumowanie działania projektu

1. **setup.sh uruchamia się** → wywołuje 7 skryptów po kolei
2. **data_preprocessing.py** → przygotowuje dane (normalizacja, podział train/test)
3. **train_anfis.py** → trenuje 8 modeli ANFIS (2 i 3 MF dla wina i betonu), zapisuje wyniki, reguły, wykresy
4. **visualize_membership_functions.py** → rysuje funkcje Gaussa dla każdego modelu
5. **train_comparison_models.py** → trenuje NN, SVM, RF dla porównania
6. **data_exploration.py** → tworzy wykresy EDA (rozkłady, korelacje)
7. **compare_all_models.py** → porównuje wszystkie modele na wykresach
8. **streamlit run app.py** → uruchamia GUI gdzie możesz wszystko zobaczyć

Cały proces zajmuje 15-30 minut. Na końcu masz:

- Wytrenowane modele w `models/`
- Wyniki w `results/*.json`
- Wykresy w `results/*.png`
- Działające GUI na localhost:8501

ANFIS to model który łączy zalety sieci neuronowych (automatyczne uczenie) z logiką rozmytą (interpretowalne reguły). Działa przez 5 warstw: fuzzyfikacja → reguły → normalizacja → konsekwenty TSK → agregacja. Każda reguła to kombinacja funkcji przynależności + równanie liniowe. Model uczy się parametrów funkcji Gaussa i wag konsekwentów przez gradient descent.
