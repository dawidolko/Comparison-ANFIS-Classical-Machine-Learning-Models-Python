# 🎓 OBRONA PROJEKTU ANFIS - Kompletny Przewodnik

## Spis treści

1. [Podstawowe pytania o projekt](#1-podstawowe-pytania-o-projekt)
2. [Pytania o ANFIS](#2-pytania-o-anfis)
3. [Zadania do wykonania](#3-zadania-do-wykonania)
4. [Mapa projektu - gdzie co jest](#4-mapa-projektu---gdzie-co-jest)
5. [Porównanie ANFIS vs modele klasyczne](#5-porównanie-anfis-vs-modele-klasyczne)

---

## 1. Podstawowe pytania o projekt

### 🔹1.1 Co to jest model TSK (Takagi-Sugeno-Kang)?

**Odpowiedź:** Model TSK to rodzaj systemu rozmytego, w którym **konsekwent reguły jest funkcją liniową** zamiast tradycyjnego zbioru rozmytego. W klasycznych systemach rozmytych (Mamdani) wynik reguły to zbiór rozmyty (np. "temperatura jest WYSOKA"), natomiast w TSK wynik to konkretna wartość liczbowa obliczona ze wzoru liniowego. Dzięki temu model TSK jest łatwiejszy do optymalizacji metodami gradientowymi i daje precyzyjne wyniki numeryczne, co czyni go idealnym do zastosowań w sieciach neuronowych jak ANFIS.

**Reguła TSK pierwszego rzędu:**

```
JEŚLI x₁ jest A₁ ORAZ x₂ jest A₂ ORAZ ... xₙ jest Aₙ
TO y = w₀ + w₁·x₁ + w₂·x₂ + ... + wₙ·xₙ
```

**Gdzie w kodzie:** [anfis.py](anfis.py#L1-L10) - komentarz na początku pliku oraz `DefuzzLayer` (linie 232-264)

```python
# W DefuzzLayer.call():
y = tf.matmul(x, self.CP_weight) + self.CP_bias  # f_i(x) = x·W_i + b_i
return w_norm * y  # Ważona kombinacja
```

**Zalety TSK:**

- Wyjście jest gładką funkcją (łatwiejsze w uczeniu)
- Może aproksymować dowolną funkcję ciągłą
- Nadaje się zarówno do klasyfikacji jak i regresji

---

### 🔹 1.2 Co to jest funkcja przynależności (Membership Function)?

**Odpowiedź:** Funkcja przynależności μ(x) określa **stopień przynależności** elementu x do zbioru rozmytego - czyli "jak bardzo" dany element pasuje do danej kategorii. W przeciwieństwie do logiki klasycznej (gdzie element albo należy, albo nie należy do zbioru), logika rozmyta pozwala na częściową przynależność wyrażoną liczbą z przedziału [0, 1]. Na przykład, wino o zawartości alkoholu 11% może mieć przynależność 0.7 do zbioru "mocne" i 0.3 do zbioru "średnie" - to pozwala modelować niepewność i płynne przejścia między kategoriami.

**Wartości μ(x) ∈ [0, 1]:**

- 0 = element w ogóle nie należy do zbioru
- 1 = element w pełni należy do zbioru
- wartości pośrednie = częściowa przynależność (np. 0.7 = "raczej należy")

**W projekcie używamy funkcji Gaussa:**

$$\mu(x) = e^{-\frac{1}{2}\left(\frac{x - c}{\sigma}\right)^2}$$

Gdzie:

- **c** = centrum (mean) - punkt maksymalnej przynależności
- **σ** = szerokość (sigma) - jak szybko spada przynależność

**Gdzie w kodzie:** [anfis.py](anfis.py#L158-L200) - klasa `FuzzyLayer`

```python
def call(self, x):
    x = tf.expand_dims(x, axis=1)                       # (B, 1, n)
    sigma_eff = tf.maximum(self.sigma, self.eps)        # zabezpieczenie przed dzieleniem przez 0
    z = (x - self.c[None, :, :]) / (sigma_eff[None, :, :] + self.eps)
    mu = tf.exp(-0.5 * tf.square(z))                    # Gaussowska MF
    return tf.clip_by_value(mu, 1e-8, 1.0)
```

**Wizualizacja:** Wykresy funkcji przynależności są zapisywane do `results/membership_functions_*.png`

---

### 🔹 1.3 Co to jest preprocessing (przetwarzanie wstępne)?

**Odpowiedź:** Preprocessing to **przygotowanie surowych danych** przed uczeniem modelu - jest to kluczowy etap, który bezpośrednio wpływa na jakość wyników. Surowe dane często mają różne skale (np. pH 0-14, alkohol 8-15%), zawierają braki lub są w nieodpowiednim formacie, co utrudnia uczenie modeli. Preprocessing ujednolica dane, usuwa szum i przekształca je do postaci optymalnej dla algorytmów uczenia maszynowego - bez tego modele mogłyby faworyzować cechy o większych wartościach lub w ogóle nie zbiegać.

**Główne kroki preprocessingu:**

1. **Normalizację/Standaryzację** - sprowadzenie cech do porównywalnej skali (średnia=0, odchylenie=1)
2. **Podział danych** na zbiór treningowy (80%) i testowy (20%) - żeby uczciwie ocenić model
3. **Transformację etykiet** (np. binaryzacja jakości wina: >5 = dobre, ≤5 = słabe)

**Gdzie w kodzie:** [data_preprocessing.py](data_preprocessing.py) - cały plik

**Co robi preprocessing w projekcie:**

| Krok                   | Opis                              | Kod         |
| ---------------------- | --------------------------------- | ----------- |
| 1. Wczytanie CSV       | `pd.read_csv()`                   | linia 31-32 |
| 2. Binaryzacja jakości | `quality > 5 → 1, else 0`         | linia 44    |
| 3. Podział train/test  | `train_test_split(test_size=0.2)` | linia 57-59 |
| 4. StandardScaler      | `μ=0, σ=1` dla każdej cechy       | linia 61-63 |
| 5. Zapis do .npy       | `np.save()`                       | linia 65-72 |

```python
# Binaryzacja jakości wina
wine_data['quality_binary'] = (wine_data['quality'] > 5).astype(int)

# Standaryzacja (średnia=0, odchylenie=1)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)  # dopasuj i transformuj
X_test = scaler.transform(X_test)        # tylko transformuj (te same parametry!)
```

**WAŻNE:** Ten sam preprocessing musi być użyty dla ANFIS i modeli porównawczych!

---

### 🔹 1.4 Co to jest ANFIS?

**Odpowiedź:** ANFIS (Adaptive Neuro-Fuzzy Inference System) to **hybrydowa architektura** łącząca zalety dwóch podejść: logiki rozmytej i sieci neuronowych. System rozmyty zapewnia interpretowalność - możemy odczytać reguły typu "JEŚLI alkohol jest WYSOKI i kwasowość jest NISKA TO wino jest DOBRE", podczas gdy sieć neuronowa automatycznie uczy się optymalnych parametrów tych reguł z danych. ANFIS łączy więc "białą skrzynkę" (zrozumiałe reguły) z mocą uczenia "czarnej skrzynki" (sieci neuronowe), dając model który jest zarówno skuteczny jak i wyjaśnialny.

**Połączenie dwóch światów:**

- **System wnioskowania rozmytego** (Fuzzy Inference System) - tworzy zrozumiałe reguły IF-THEN
- **Sieć neuronowa** - automatycznie optymalizuje parametry funkcji przynależności i wagi reguł

**5 warstw ANFIS:**

```
Wejście → [1.Fuzzy] → [2.Rule] → [3.Norm] → [4.Defuzz] → [5.Sum] → Wyjście
```

| Warstwa | Nazwa          | Funkcja                        | W kodzie         |
| ------- | -------------- | ------------------------------ | ---------------- |
| 1       | FuzzyLayer     | Oblicza μ(x) dla każdej MF     | anfis.py:158-200 |
| 2       | RuleLayer      | Iloczyn (T-norma AND)          | anfis.py:203-221 |
| 3       | NormLayer      | Normalizacja wag: w̄ᵢ = wᵢ/Σwⱼ  | anfis.py:224-231 |
| 4       | DefuzzLayer    | Konsekwent TSK: fᵢ = x·Wᵢ + bᵢ | anfis.py:234-264 |
| 5       | SummationLayer | Suma: y = Σw̄ᵢ·fᵢ               | anfis.py:267-273 |

**Gdzie w kodzie:** [anfis.py](anfis.py) - cały plik definiuje architekturę

---

### 🔹 1.5 Jak działa "losowa linia" (inicjalizacja wag)?

**Odpowiedź:** Przy tworzeniu modelu wszystkie parametry (wagi) są inicjalizowane **losowo** z określonych rozkładów - to kluczowe dla prawidłowego uczenia. Gdybyśmy zainicjowali wszystkie wagi tak samo (np. zerami), to wszystkie neurony/reguły uczyłyby się tego samego - nie byłoby różnorodności. Losowa inicjalizacja "łamie symetrię" i pozwala różnym częściom sieci specjalizować się w różnych wzorcach.

**W ANFIS inicjalizujemy losowo:**

1. **Centra funkcji przynależności (c)** - gdzie na osi X jest "środek" każdej funkcji Gaussa
2. **Szerokości funkcji przynależności (σ)** - jak "szerokie" są funkcje Gaussa
3. **Wagi konkluzji (CP_weight, CP_bias)** - parametry reguł TSK

**Dlaczego używamy seed=42?** Ustawienie ziarna generatora losowego (seed) zapewnia **powtarzalność** - za każdym razem gdy uruchomimy trening, dostaniemy te same "losowe" wartości początkowe. Dzięki temu eksperymenty są odtwarzalne i możemy porównywać wyniki.

**Gdzie w kodzie:** [anfis.py](anfis.py#L178-L190)

```python
# FuzzyLayer - parametry funkcji przynależności
self.c = self.add_weight(
    name="c",
    shape=(self.m, self.n),
    initializer=tf.keras.initializers.RandomUniform(minval=-1.5, maxval=1.5, seed=42),
    trainable=True,
)
self.sigma = self.add_weight(
    name="sigma",
    shape=(self.m, self.n),
    initializer=tf.keras.initializers.RandomUniform(minval=0.5, maxval=1.5, seed=42),
    trainable=True,
)
```

**Dlaczego losowo?**

- Przerywa symetrię (różne neurony uczą się różnych cech)
- `seed=42` zapewnia powtarzalność eksperymentów
- Zakres `-1.5 do 1.5` dla centrów (dane są znormalizowane do ~±3)
- Zakres `0.5 do 1.5` dla sigma (rozsądna szerokość MF)

---

## 2. Pytania o ANFIS

### 🔹 1.6 Co to jest walidacja krzyżowa (cross-validation)?

**Odpowiedź:** K-krotna walidacja krzyżowa to technika oceny modelu, która pozwala wiarygodnie oszacować jak model będzie działał na nowych danych. Problem z pojedynczym podziałem train/test polega na tym, że wynik zależy od "szczęścia" - który zestaw danych trafił do testu. Cross-validation rozwiązuje to przez wielokrotne testowanie: każda próbka jest dokładnie raz w zbiorze testowym, więc dostajemy stabilną ocenę uśrednioną z K eksperymentów.

**Jak działa (dla K=5):**

1. Dane dzielimy na 5 równych części (foldów)
2. 5 razy trenujemy model: za każdym razem 4 części to trening, 1 część to test
3. Uśredniamy wyniki z 5 testów → dostajemy wiarygodną ocenę ± odchylenie standardowe

**Zalety:**

- Wykorzystuje 100% danych zarówno do treningu jak i testu (każda próbka jest raz testowana)
- Zmniejsza wariancję oszacowania błędu - wynik nie zależy od losowego podziału
- Wykrywa overfitting - jeśli model dobrze działa na treningu ale słabo na CV, to się przeuczył

**Gdzie w kodzie:** [train_anfis.py](train_anfis.py#L507-L572) - funkcja `cross_validate_anfis()`

```python
def cross_validate_anfis(n_memb=2, batch_size=32, dataset="all", n_splits=5, epochs=10):
    # Używa StratifiedKFold dla klasyfikacji (zachowuje proporcje klas)
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=42) \
        if dataset == "concrete" else StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    for fold, (tr_idx, va_idx) in enumerate(splitter.split(X, y), 1):
        # Trenuj na tr_idx, testuj na va_idx
        ...
```

**Wyniki CV:** zapisywane do `results/anfis_*_cv.json`

---

### 🔹 1.7 Co mierzy Accuracy?

**Odpowiedź:** Accuracy (dokładność) mierzy **procent poprawnych klasyfikacji** - czyli ile razy model trafnie przewidział klasę spośród wszystkich próbek. Jest to najprostsza i najbardziej intuicyjna metryka: jeśli accuracy = 75%, oznacza to że model poprawnie sklasyfikował 75 na 100 próbek. Accuracy odpowiada na pytanie "jak często model ma rację?", ale nie rozróżnia między typami błędów (fałszywe alarmy vs przeoczenia).

**Wzór:**
$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN} = \frac{\text{poprawne predykcje}}{\text{wszystkie próbki}}$$

**Składniki (macierz pomyłek):**

- TP = True Positive - model powiedział "dobre wino" i miał rację
- TN = True Negative - model powiedział "słabe wino" i miał rację
- FP = False Positive - model powiedział "dobre" ale wino było słabe (fałszywy alarm)
- FN = False Negative - model powiedział "słabe" ale wino było dobre (przeoczenie)

**Gdzie w kodzie:** [train_anfis.py](train_anfis.py#L117-L121)

```python
anfis_model.model.compile(
    optimizer=tf.keras.optimizers.Nadam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy"]  # <- tutaj definiujemy metrykę
)
```

**Wady Accuracy:**

- Myląca przy niezbalansowanych klasach (np. 90% klasy A → model może zawsze zgadywać A i mieć 90% accuracy)

---

### 🔹1.8 Co mierzy ROC AUC?

**Odpowiedź:** ROC AUC (Area Under Receiver Operating Characteristic Curve) mierzy **zdolność modelu do rozróżniania klas** - czyli jak dobrze model potrafi oddzielić pozytywne przykłady od negatywnych. Wyobraź sobie, że model daje każdemu winu "score" od 0 do 1 - ROC AUC mówi, jak często wino dobre dostaje wyższy score niż wino słabe. AUC = 0.85 oznacza, że w 85% przypadków losowo wybrane dobre wino będzie miało wyższy score niż losowo wybrane słabe wino.

**Dlaczego AUC a nie Accuracy?** Bo AUC nie zależy od progu (czy klasyfikujemy jako "dobre" przy score > 0.5, > 0.3 czy > 0.7) - mierzy ogólną jakość rankingu, nie konkretnych decyzji.

**Interpretacja wartości:**

- AUC = 1.0 → idealny klasyfikator (wszystkie dobre mają wyższy score niż wszystkie słabe)
- AUC = 0.5 → losowy klasyfikator (jak rzut monetą - model nic nie wie)
- AUC < 0.5 → gorszy niż losowy (model myli klasy - odwróć predykcje!)
- **AUC > 0.8** → dobry model, **AUC > 0.9** → bardzo dobry model

**Zalety AUC:**

- Nie zależy od progu klasyfikacji - ocenia model całościowo
- Działa dobrze dla niezbalansowanych danych (np. 90% jednej klasy)
- Mierzy jakość rankingu, nie tylko binarnych decyzji

**Gdzie w kodzie:** [train_comparison_models.py](train_comparison_models.py#L175)

```python
from sklearn.metrics import roc_auc_score
results['nn'] = {
    ...
    "roc_auc": float(roc_auc_score(y_test_r, y_proba)),
}
```

---

### 🔹 1.9 Co mierzy MSE / MAE?

**Odpowiedź:** MSE i MAE to metryki dla zadań regresji, które mierzą **średni błąd predykcji** - czyli jak bardzo wartości przewidziane przez model różnią się od prawdziwych wartości.

**MSE (Mean Squared Error) - Średni Błąd Kwadratowy:**
MSE oblicza średnią z kwadratów różnic między predykcją a rzeczywistością. Ponieważ błędy są podnoszone do kwadratu, duże błędy są karane znacznie mocniej niż małe - błąd 10 MPa daje karę 100, ale błąd 2 MPa daje karę tylko 4. To sprawia, że MSE jest wrażliwe na wartości odstające (outliers) i "zmusza" model do unikania dużych pomyłek.

$$\text{MSE} = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$$

**MAE (Mean Absolute Error) - Średni Błąd Bezwzględny:**
MAE oblicza średnią z wartości bezwzględnych różnic - każdy błąd jest traktowany proporcjonalnie do jego wielkości. Błąd 10 MPa jest karany 5x mocniej niż błąd 2 MPa (a nie 25x jak w MSE). MAE jest bardziej "sprawiedliwe" i łatwiejsze do interpretacji - jeśli MAE = 5 MPa, to średnio model myli się o 5 MPa.

$$\text{MAE} = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|$$

**Porównanie - kiedy co używać:**

| Metryka | Co mierzy               | Wrażliwość na outliers      | Interpretacja             |
| ------- | ----------------------- | --------------------------- | ------------------------- |
| **MSE** | Średni kwadrat błędu    | Wysoka (karze duże błędy ²) | Trudniejsza (jednostki²)  |
| **MAE** | Średni błąd bezwzględny | Niska (równo traktuje)      | Łatwa (te same jednostki) |

**Przykład:** Jeśli przewidujemy wytrzymałość betonu i MAE = 4.5 MPa, to średnio mylimy się o 4.5 megapaskala.

**Gdzie w kodzie:** [train_anfis.py](train_anfis.py#L109-L115)

```python
if dataset == "concrete":
    anfis_model.model.compile(
        optimizer=tf.keras.optimizers.Nadam(learning_rate=0.001),
        loss="mean_squared_error",  # MSE jako funkcja straty
        metrics=["mae"]              # MAE jako metryka
    )
```

---

### 🔹1.10 Czym różni się hybrydowe uczenie ANFIS od uczenia standardowego?

**Odpowiedź:** Główna różnica polega na tym, **jak trenowane są dwa rodzaje parametrów** w ANFIS: parametry przeslanki (premise) - czyli centra i szerokości funkcji przynależności, oraz parametry konkluzji (consequent) - czyli wagi w regule TSK.

**Uczenie hybrydowe (klasyczny ANFIS Janga):** Rozdziela trening na dwie fazy w każdej epoce. Najpierw zamraża parametry przeslanki i optymalizuje konkluzje metodą najmniejszych kwadratów (LSE) - to jest szybkie bo ma rozwiązanie analityczne. Potem zamraża konkluzje i uczy przesłanki gradientem. To przyspiesza zbieżność, bo LSE znajduje optymalne konkluzje w jednym kroku.

**Uczenie standardowe (używane w tym projekcie):** Trenuje wszystkie parametry jednocześnie zwykłym gradientem (backpropagation). Jest prostsze w implementacji (używamy Keras/TensorFlow), ale może być wolniejsze. Współczesne optymalizatory (Adam, Nadam) częściowo niwelują tę różnicę.

| Aspekt                | Standardowe (ten projekt)        | Hybrydowe (klasyczne)        |
| --------------------- | -------------------------------- | ---------------------------- |
| **Co trenuje**        | Wszystkie parametry jednocześnie | Rozdziela premise/consequent |
| **Metoda premise**    | Gradient descent                 | Gradient descent             |
| **Metoda consequent** | Gradient descent                 | **Least Squares Estimation** |
| **Implementacja**     | Prosta (gotowe Keras)            | Bardziej złożona             |
| **Zbieżność**         | Wolniejsza (więcej epok)         | Szybsza                      |

**W tym projekcie:** Używamy **standardowego uczenia (end-to-end backpropagation)** przez TensorFlow/Keras, które trenuje wszystkie parametry jednocześnie metodą gradientową.

**Gdzie w kodzie:** [anfis.py](anfis.py#L22-L55) - model jest zwykłą siecią Keras

```python
# Standardowe uczenie - wszystko przez backpropagation
self.model = tf.keras.Model(inputs=[x_in], outputs=[out], name=model_name)
# Wagi premise (c, sigma) i consequent (CP_weight, CP_bias) są wszystkie trainable=True
```

---

### 🔹1.11 Czym ANFIS różni się od sieci MLP?

**Odpowiedź:** Najważniejsza różnica to **interpretowalność**: ANFIS można "przeczytać" jako zbiór reguł IF-THEN zrozumiałych dla człowieka, podczas gdy MLP to "czarna skrzynka" gdzie wagi nie mają intuicyjnego znaczenia. W ANFIS wiesz, że "JEŚLI alkohol jest WYSOKI i kwasowość jest NISKA TO wino jest dobre" - w MLP masz tylko macierz liczb.

**Druga różnica to sposób \u0142ączenia informacji:** ANFIS używa iloczynu (T-norma AND) do kombinowania wejść, co odpowiada logicznemu "ORAZ" - wszystkie warunki muszą być spełnione. MLP używa sumy ważonej, gdzie różne cechy mogą się kompensować.

**Trzecia różnica to eksplozja złożoności:** W ANFIS liczba reguł rośnie wykładniczo z liczbą cech - dla 11 cech i 2 funkcji przynależności mamy 2^11 = 2048 reguł. W MLP możemy mieć dowolną liczbę neuronów niezależnie od wejść.

| Cecha                 | ANFIS                              | MLP (Multi-Layer Perceptron)     |
| --------------------- | ---------------------------------- | -------------------------------- |
| **Interpretowalność** | ✅ Wysoka (reguły IF-THEN)         | ❌ Niska (czarna skrzynka)       |
| **Struktura**         | Stała (5 warstw, wynika z logiki)  | Dowolna liczba warstw/neuronów   |
| **Funkcje aktywacji** | Gaussowskie funkcje przynależności | ReLU, sigmoid, tanh              |
| **Łączenie wejść**    | Iloczyn (T-norma "ORAZ")           | Suma ważona                      |
| **Konsekwent**        | Funkcja liniowa TSK                | Dowolna nieliniowa transformacja |
| **Złożoność**         | n_memb^n_features reguł            | Konfigurowalana                  |

**Przykład eksplozji reguł:**

- 11 cech × 2 MF = 2^11 = **2,048 reguł** (zarządzalne)
- 11 cech × 3 MF = 3^11 = **177,147 reguł** (dużo!)
- 11 cech × 4 MF = 4^11 = **4,194,304 reguł** (niemozliwe do interpretacji)

---

### 🔹1.12 Co to jest funkcja celu (loss function)?

**Odpowiedź:** Funkcja celu (straty) mierzy **jak bardzo predykcje modelu różnią się od prawdziwych wartości** - to "ocena" którą model dostaje za swoje predykcje. Im mniejsza wartość loss, tym lepiej model przewiduje. Podczas uczenia model stara się zminimalizować tę funkcję, modyfikując swoje wagi - to jak uczeń poprawiający swoje odpowiedzi, żeby dostać lepszą ocenę. Funkcja loss musi być różniczkowalna, żeby można było obliczyć gradient i wiedzieć "w którą stronę" zmieniać wagi.

**Dlaczego różne funkcje dla różnych zadań?**

- **Klasyfikacja** (0 lub 1): Binary Cross-Entropy karze "pewność siebie" modelu gdy się myli - jeśli model jest 99% pewny że wino jest dobre, a ono jest złe, dostaje bardzo dużą karę
- **Regresja** (wartość ciągła): MSE karze proporcjonalnie do kwadratu błędu - im bardziej się mylisz, tym większa kara

**W projekcie używamy:**

| Zadanie             | Loss Function        | Co robi                              | Wzór                                                      |
| ------------------- | -------------------- | ------------------------------------ | --------------------------------------------------------- |
| Klasyfikacja (wine) | Binary Cross-Entropy | Karze pewność w błędnych predykcjach | $-\frac{1}{n}\sum[y\log(\hat{y}) + (1-y)\log(1-\hat{y})]$ |
| Regresja (concrete) | MSE                  | Karze kwadrat odchylenia od celu     | $\frac{1}{n}\sum(y - \hat{y})^2$                          |

**Gdzie w kodzie:** [train_anfis.py](train_anfis.py#L109-L121)

```python
# Klasyfikacja (wine)
loss="binary_crossentropy"

# Regresja (concrete)
loss="mean_squared_error"
```

---

### 🔹 1.13 Dlaczego optymalizator jest niezbędny?

**Odpowiedź:** Optymalizator jest niezbędny, bo to on umożliwia uczenie się modelu – bez niego wagi sieci nie zmieniałyby się i model nie poprawiałby swoich predykcji. Optymalizator decyduje, jak i o ile zmienić parametry na podstawie gradientu, by minimalizować błąd (funkcję celu). Bez optymalizatora model byłby statyczny i nie nauczyłby się niczego z danych – to on „napędza” cały proces uczenia. Różne optymalizatory (SGD, Adam, Nadam) różnią się strategią aktualizacji wag, ale każdy z nich jest absolutnie konieczny, by model mógł się uczyć.

**Podstawowa idea - Gradient Descent:**
$$w_{t+1} = w_t - \eta \cdot \nabla L(w_t)$$

- $w_t$ = aktualne wagi
- $\eta$ = learning rate (jak duży krok)
- $\nabla L$ = gradient funkcji straty (kierunek "w dół")

**Optymalizatory w projekcie:**

| Optymalizator       | Co robi                                     | Kiedy używać                                |
| ------------------- | ------------------------------------------- | ------------------------------------------- |
| **Nadam** (w ANFIS) | Adam + Nesterov momentum - "patrzy w przód" | Dobry domyślny wybór                        |
| **Adam** (w NN)     | Adaptive learning rate + momentum           | Najpopularniejszy, działa dobrze            |
| **SGD**             | Podstawowy gradient descent                 | Wymaga tuningu, ale może dać lepsze minimum |

**Gdzie w kodzie:** [train_anfis.py](train_anfis.py#L110)

```python
optimizer=tf.keras.optimizers.Nadam(learning_rate=0.001)
```

---

### 🔹 1.14 Dlaczego stosuje się mini-batch?

**Odpowiedź:** Mini-batch to sposób przetwarzania danych podczas uczenia, gdzie zamiast używać wszystkich próbek naraz (zbyt wolne) lub pojedynczych próbek (zbyt chaotyczne), używamy małych porcji np. 32 próbek. To jak jedzenie - nie jesz całego obiadu na raz (zakrztusisz się), ani kęs po kęsie przez 3 godziny (za wolno), tylko normalne porcje. Mini-batch daje stabilniejszy gradient niż pojedyncze próbki, ale jest szybszy niż pełny batch i mieści się w pamięci GPU.

**Porównanie podejść:**

| Tryb           | Batch size    | Zalety                      | Wady                                 |
| -------------- | ------------- | --------------------------- | ------------------------------------ |
| **SGD online** | 1 próbka      | Bardzo szybka aktualizacja  | Chaotyczny gradient, wolna zbieżność |
| **Batch GD**   | wszystkie (n) | Stabilny, dokładny gradient | Bardzo wolne, wymaga dużo RAM        |
| **Mini-batch** | 32-256        | Złoty środek                | -                                    |

**Konkretne zalety mini-batch:**

1. **Regularyzacja** - szum w gradiencie (bo liczymy z próbki, nie całości) pomaga uciec z lokalnych minimów i zapobiega przeuczeniu
2. **Efektywność GPU** - karty graficzne są zoptymalizowane pod operacje macierzowe na wielu danych naraz (32 próbki × 11 cech = macierz 32×11)
3. **Pamięć** - nie trzeba trzymać całego zbioru (6497 próbek) w pamięci GPU, wystarczy 32

**Gdzie w kodzie:** [train_anfis.py](train_anfis.py#L141-L145)

```python
history = anfis_model.model.fit(
    X_train, y_train,
    ...
    batch_size=batch_size,  # domyślnie 32
    ...
)
```

---

## 3. Zadania do wykonania

### ✅ 2.1 Zmiana liczby funkcji przynależności

**Lokalizacja:** [train_anfis.py](train_anfis.py#L579-L585) lub wywołanie z CLI

```bash
# Z linii komend:
python train_anfis.py --datasets all --memb 2 3 4 --epochs 20

# W kodzie - bezpośrednie wywołanie:
train_anfis_model(n_memb=4, epochs=20, dataset="all")
```

**Jak to wpływa na model:**

- Więcej MF = więcej reguł = większa ekspresywność
- n_memb=2 → 2^11 = 2048 reguł
- n_memb=3 → 3^11 = 177,147 reguł

---

### ✅ 2.2 Zmiana liczby iteracji (epok)

**Lokalizacja:** [train_anfis.py](train_anfis.py#L583)

```bash
python train_anfis.py --epochs 50  # zamiast 20
```

**Lub w kodzie:**

```python
train_anfis_model(n_memb=2, epochs=50, dataset="all")
```

---

### ✅ 2.3 Uczenie na wybranych atrybutach

**Jak zmodyfikować:** Zmień listę `feature_columns` w [data_preprocessing.py](data_preprocessing.py#L40-42)

```python
# Oryginalne - wszystkie 11 cech:
feature_columns = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
                  'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
                  'pH', 'sulphates', 'alcohol']

# Zmodyfikowane - tylko 5 najważniejszych (przykład):
feature_columns = ['alcohol', 'volatile acidity', 'sulphates', 'citric acid', 'density']
```

**Po zmianie:**

1. Uruchom `python data_preprocessing.py`
2. Uruchom `python train_anfis.py`

---

### ✅ 2.4 Wypisanie reguł ANFIS

**Lokalizacja:** [train_anfis.py](train_anfis.py#L429-L500) - funkcja `extract_and_save_rules()`

**Wyniki:** `results/anfis_*_rules.json`

**Format reguły:**

```json
{
  "rule_index": 0,
  "membership_indices": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  "consequent": {
    "weights": [0.123, -0.456, ...],  // wagi dla każdej cechy
    "bias": 0.789
  }
}
```

**Interpretacja:** Reguła 0 mówi:

> JEŚLI cecha_0 jest LOW (MF 0) ORAZ cecha_1 jest LOW ORAZ ... ORAZ cecha_10 jest LOW
> TO y = 0.789 + 0.123·x₀ - 0.456·x₁ + ...

---

### ✅ 2.5Wyświetlenie funkcji przynależności PRZED i PO uczeniu

**Lokalizacja:** [train_anfis.py](train_anfis.py#L140-L160) - automatycznie zapisuje MF przed i po treningu

**Co się dzieje automatycznie:**
Podczas treningu ANFIS automatycznie zapisywane są parametry funkcji przynależności:

- **PRZED treningiem:** `results/mf_centers_before_{dataset}_{n_memb}memb.npy` i `results/mf_sigmas_before_{dataset}_{n_memb}memb.npy`
- **PO treningu:** `results/mf_centers_after_{dataset}_{n_memb}memb.npy` i `results/mf_sigmas_after_{dataset}_{n_memb}memb.npy`

**Kod w train_anfis.py (automatycznie wykonywany):**

```python
# PRZED TRENINGIEM - zapisz początkowe MF
anfis_model.update_weights()
centers_before, sigmas_before = anfis_model.get_membership_functions()
np.save(f"results/mf_centers_before_{dataset}_{n_memb}memb.npy", centers_before)
np.save(f"results/mf_sigmas_before_{dataset}_{n_memb}memb.npy", sigmas_before)

# ... trening model.fit() ...

# PO TRENINGU - zapisz końcowe MF
anfis_model.update_weights()
centers_after, sigmas_after = anfis_model.get_membership_functions()
np.save(f"results/mf_centers_after_{dataset}_{n_memb}memb.npy", centers_after)
np.save(f"results/mf_sigmas_after_{dataset}_{n_memb}memb.npy", sigmas_after)
```

**Jak wyświetlić porównanie PRZED vs PO:**

```python
import numpy as np
import matplotlib.pyplot as plt

# Wczytaj parametry
dataset, n_memb = "all", 2
centers_before = np.load(f"results/mf_centers_before_{dataset}_{n_memb}memb.npy")
sigmas_before = np.load(f"results/mf_sigmas_before_{dataset}_{n_memb}memb.npy")
centers_after = np.load(f"results/mf_centers_after_{dataset}_{n_memb}memb.npy")
sigmas_after = np.load(f"results/mf_sigmas_after_{dataset}_{n_memb}memb.npy")

# Porównaj np. dla cechy 0 (fixed acidity)
feature_idx = 0
x = np.linspace(-3, 3, 200)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
for mf in range(n_memb):
    c, s = centers_before[mf, feature_idx], sigmas_before[mf, feature_idx]
    ax1.plot(x, np.exp(-0.5 * ((x - c) / s)**2), label=f'MF {mf}')
ax1.set_title("PRZED treningiem")
ax1.legend()

for mf in range(n_memb):
    c, s = centers_after[mf, feature_idx], sigmas_after[mf, feature_idx]
    ax2.plot(x, np.exp(-0.5 * ((x - c) / s)**2), label=f'MF {mf}')
ax2.set_title("PO treningu")
ax2.legend()

plt.savefig("results/mf_comparison_before_after.png")
plt.show()
```

**Co obserwować:**

- **Centra (c)** - czy przesunęły się do bardziej znaczących wartości cech
- **Szerokości (σ)** - czy zwęziły się (większa precyzja) lub rozszerzyły (większa generalizacja)
- Duże zmiany = model dużo się nauczył na tej cesze
- Małe zmiany = cecha mniej istotna lub już była dobrze zainicjalizowana

**Wykresy PO treningu:** `results/membership_functions_*.png` (generowane przez `visualize_membership_functions.py`)

---

### ✅ 2.6 Zmiana optymalizatora / learning rate

**Lokalizacja:** [train_anfis.py](train_anfis.py#L109-L121)

```python
# Oryginał:
optimizer=tf.keras.optimizers.Nadam(learning_rate=0.001)

# Zmiana na Adam:
optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005)

# Zmiana na SGD z momentum:
optimizer=tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)

# Zmiana na RMSprop:
optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001)
```

---

### ✅ 2.7 Omówienie wykresu historii funkcji celu

**Lokalizacja wykresów:** `results/anfis_*_training.png`

**Co pokazują wykresy:**

1. **Lewy wykres (Accuracy/MAE):**

   - Niebieska linia = Train
   - Pomarańczowa linia = Validation
   - Czerwony punkt = najlepsza epoka

2. **Prawy wykres (Loss):**
   - Zielona linia = Train Loss
   - Czerwona linia = Validation Loss

**Jak interpretować:**

| Obserwacja                    | Diagnoza                                    |
| ----------------------------- | ------------------------------------------- |
| Train i Val maleją razem      | ✅ Dobry trening                            |
| Train maleje, Val stoi/rośnie | ⚠️ Overfitting                              |
| Obie krzywe stoją (stagnacja) | ⚠️ Learning rate za mały lub złe minimum    |
| Oscylacje                     | ⚠️ Learning rate za duży                    |
| Szybki spadek → plateau       | ✅ Normalne (szybkie uczenie → fine-tuning) |

**Gdzie w kodzie:** [train_anfis.py](train_anfis.py#L200-L282) - `plot_training_history()`

---

## 🎯 Szybka ściągawka na obronę

### Najważniejsze definicje

| Pojęcie              | Definicja jednozdaniowa                                        |
| -------------------- | -------------------------------------------------------------- |
| **ANFIS**            | Hybrydowy system łączący logikę rozmytą z sieciami neuronowymi |
| **TSK**              | Model rozmyty z liniowymi konsekwentami (y = ax + b)           |
| **MF**               | Funkcja określająca stopień przynależności do zbioru rozmytego |
| **Cross-validation** | K-krotny podział danych do stabilnej oceny modelu              |
| **Loss function**    | Funkcja mierząca błąd predykcji (minimalizujemy ją)            |
| **Optimizer**        | Algorytm aktualizujący wagi w kierunku mniejszego błędu        |
| **Mini-batch**       | Podział danych na porcje dla efektywniejszego treningu         |

## Co to jest gradient? (po ludzku)

Gradient to po prostu "kierunek najszybszego spadku" – pokazuje, w którą stronę trzeba zmienić parametry (np. wagi w sieci), żeby najszybciej zmniejszyć błąd. Wyobraź sobie, że stoisz na górce i chcesz zejść na sam dół: gradient to strzałka pokazująca, gdzie jest najbardziej stromo w dół. W uczeniu maszynowym algorytm korzysta z gradientu, by krok po kroku poprawiać model i zbliżać się do najlepszego rozwiązania.
