"""
ANFIS (Takagi–Sugeno–Kang, typ-1)
Uogólniony model dla klasyfikacji (sigmoid output) i regresji (linear output).

Warstwy:
1. FuzzyLayer     — Gaussowskie funkcje przynależności μ(x)
2. RuleLayer      — T-norma (AND) przez iloczyn → kombinacje reguł
3. NormLayer      — Normalizacja wag reguł
4. DefuzzLayer    — TSK-1: y_i = x·W_i + b_i
5. SummationLayer — Agregacja po regułach
"""

import tensorflow as tf  # Framework do budowy i treningu sieci neuronowych
import numpy as np  # Biblioteka do operacji na macierzach i tablicach numerycznych


class ANFISModel:  # Główna klasa modelu ANFIS (Adaptive Neuro-Fuzzy Inference System)
    """
    Klasa modelu ANFIS (TSK-1) z możliwością użycia w klasyfikacji lub regresji.
    """

    def __init__(self, n_input: int, n_memb: int, batch_size: int = 32, regression: bool = False):
        """
        Inicjalizuje model ANFIS z określoną liczbą wejść i funkcji przynależności.
        
        Args:
            n_input: liczba cech wejściowych
            n_memb: liczba funkcji przynależności na cechę
            batch_size: rozmiar batcha dla predykcji
            regression: True dla regresji (aktywacja liniowa), False dla klasyfikacji (sigmoid)
        """
        self.n = int(n_input)  # Zapisuje liczbę cech wejściowych jako atrybut instancji
        self.m = int(n_memb)  # Zapisuje liczbę funkcji przynależności na cechę
        self.batch_size = int(batch_size)  # Zapisuje rozmiar batcha używany podczas predykcji
        self.regression = bool(regression)  # Flaga określająca czy to regresja (True) czy klasyfikacja (False)

        # --- Warstwy modelu ---
        x_in = tf.keras.layers.Input(shape=(self.n,), name="inputLayer")  # Tworzy warstwę wejściową o kształcie (batch_size, n_input)

        mu = FuzzyLayer(self.n, self.m, name="fuzzy_layer")(x_in)  # Warstwa 1: Fuzzyfikacja - oblicza stopnie przynależności dla każdej cechy (B, m, n)
        w = RuleLayer(self.n, self.m, name="ruleLayer")(mu)  # Warstwa 2: Oblicza siłę każdej reguły rozmytej przez iloczyn stopni przynależności (B, R)
        w_norm = NormLayer(name="normLayer")(w)  # Warstwa 3: Normalizuje siły reguł tak aby sumowały się do 1 (B, R)
        per_rule = DefuzzLayer(self.n, self.m, name="defuzzy_layer")(w_norm, x_in)  # Warstwa 4: Oblicza wkład każdej reguły (konsekwent TSK-1) (B, R)
        y_lin = SummationLayer(name="sumLayer")(per_rule)  # Warstwa 5: Sumuje wkłady wszystkich reguł do końcowego wyniku (B, 1)

        # --- Wyjście ---
        if regression:  # Sprawdza czy model służy do regresji
            out = tf.keras.layers.Activation("linear", name="output")(y_lin)  # Dla regresji: aktywacja liniowa (bez przekształcenia wartości)
            model_name = "ANFIS_Concrete"  # Nazwa modelu dla problemu regresji (beton)
        else:  # W przeciwnym razie to klasyfikacja
            out = tf.keras.layers.Activation("sigmoid", name="prob")(y_lin)  # Dla klasyfikacji: sigmoid zamienia wyjście na prawdopodobieństwo (0-1)
            model_name = "ANFIS_WineQuality"  # Nazwa modelu dla problemu klasyfikacji (wino)

        self.model = tf.keras.Model(inputs=[x_in], outputs=[out], name=model_name)  # Tworzy model Keras łączący wejście z wyjściem przez wszystkie warstwy
        self.update_weights()  # Inicjalizuje lokalne kopie wag z warstw Keras

    # --- Interfejs wysokiego poziomu ---

    def __call__(self, X):  # Metoda magiczna umożliwiająca wywołanie modelu jak funkcji: model(X)
        """
        Wykonuje predykcję dla danych wejściowych X.
        
        Args:
            X: macierz cech (n_samples, n_features)
            
        Returns:
            Tablica predykcji (n_samples, 1)
        """
        return self.model.predict(X, batch_size=self.batch_size, verbose=0)  # Wywołuje predykcję Keras z wybranym batch_size bez logowania

    def fit(self, X, y, **kwargs):  # Metoda trenująca model na danych treningowych z dowolnymi dodatkowymi argumentami
        """
        Trenuje model ANFIS na danych treningowych.
        
        Args:
            X: macierz cech treningowych
            y: wektor etykiet
            **kwargs: dodatkowe argumenty przekazywane do model.fit()
            
        Returns:
            Historia treningu (History object)
        """
        hist = self.model.fit(X, y, **kwargs)  # Wywołuje standardową metodę fit() z Keras przekazując wszystkie argumenty
        self.update_weights()  # Po treningu aktualizuje lokalne kopie wag (centra, sigmy, wagi konsekwentów)
        return hist  # Zwraca obiekt History zawierający metryki z każdej epoki

    def update_weights(self):  # Metoda aktualizująca lokalne kopie parametrów z warstw Keras
        """
        Aktualizuje lokalne kopie wag z warstw Keras.
        
        Pobiera parametry z warstw fuzzy_layer (centra i sigmy) 
        oraz defuzzy_layer (wagi konsekwentne i bias).
        """
        fz = self.model.get_layer("fuzzy_layer")  # Pobiera referencję do warstwy fuzzyfikacji po nazwie
        self.cs, self.sigmas = fz.get_weights()  # Wyciąga tablice numpy z centrami i sigmami funkcji Gaussa

        df = self.model.get_layer("defuzzy_layer")  # Pobiera referencję do warstwy defuzzyfikacji po nazwie
        self.bias, self.weights = df.get_weights()  # Wyciąga bias i wagi konsekwentów TSK-1

    def get_membership_functions(self):  # Metoda zwracająca parametry funkcji przynależności
        """
        Zwraca parametry gaussowskich funkcji przynależności.
        
        Returns:
            Tuple (centra, sigmy) - oba jako tablice numpy shape (n_memb, n_features)
        """
        return self.cs, self.sigmas  # Zwraca kopie lokalnych tablic z centrami i sigmami

    # --- Eksport reguł ---

    def to_rules_json(self):  # Metoda eksportująca wyuczone reguły do formatu JSON
        """
        Generuje listę reguł w formacie kompatybilnym z extract_and_save_rules().
        """
        n_features = self.n  # Pobiera liczbę cech wejściowych z atrybutu modelu
        n_memb = self.m  # Pobiera liczbę funkcji przynależności z atrybutu modelu
        weights = self.weights  # Pobiera macierz wag konsekwentów (n_features, n_rules)
        bias = self.bias  # Pobiera wektor biasów reguł (1, n_rules)

        n_rules = n_memb ** n_features  # Oblicza całkowitą liczbę reguł: n_memb^n_features
        rules = []  # Inicjalizuje pustą listę do przechowywania reguł
        for ridx in range(n_rules):  # Iteruje przez wszystkie indeksy reguł (0 do n_rules-1)
            combo = self._rule_index_to_tuple(ridx, n_features, n_memb)  # Konwertuje indeks reguły na krotkę indeksów funkcji przynależności
            cons_w = weights[:, ridx].tolist()  # Wyciąga wagi konsekwentu dla tej reguły i konwertuje na listę Python
            cons_b = float(bias[0, ridx])  # Wyciąga bias konsekwentu dla tej reguły i konwertuje na float
            rules.append({  # Dodaje słownik z parametrami reguły do listy
                "rule_index": int(ridx),  # Indeks reguły jako liczba całkowita
                "membership_indices": combo,  # Lista indeksów funkcji przynależności dla każdej cechy
                "consequent": {"weights": cons_w, "bias": cons_b}  # Słownik z wagami i biasem konsekwentu
            })

        return {  # Zwraca słownik z pełnymi informacjami o wszystkich regułach
            "n_features": n_features,  # Liczba cech wejściowych
            "n_memb": n_memb,  # Liczba funkcji przynależności na cechę
            "n_rules_total": n_rules,  # Całkowita liczba reguł
            "rules_listed": len(rules),  # Liczba reguł w liście (powinna być równa n_rules)
            "rules": rules  # Lista wszystkich reguł z ich parametrami
        }

    @staticmethod
    def _rule_index_to_tuple(idx: int, n_features: int, n_memb: int):
        """
        Konwertuje płaski indeks reguły na kombinację indeksów funkcji przynależności.
        
        Args:
            idx: indeks reguły (0 do n_memb^n_features - 1)
            n_features: liczba cech
            n_memb: liczba funkcji przynależności na cechę
            
        Returns:
            Lista indeksów MF dla każdej cechy
        """
        combo = []  # Inicjalizuje pustą listę na indeksy funkcji przynależności
        for _ in range(n_features):  # Iteruje przez wszystkie cechy wejściowe
            combo.append(idx % n_memb)  # Ekstrahuje indeks MF dla bieżącej cechy (reszta z dzielenia)
            idx //= n_memb  # Przesuwa się do kolejnej pozycji w systemie liczbowym o podstawie n_memb
        return list(reversed(combo))  # Odwraca listę aby otrzymać poprawną kolejność cech


# =========================================================================== #
#                                 WARSTWY                                     #
# =========================================================================== #

class FuzzyLayer(tf.keras.layers.Layer):
    """
    Warstwa fuzzyfikacji z gaussowskimi funkcjami przynależności.
    
    Dla każdej cechy wejściowej tworzy n_memb funkcji gaussowskich.
    Zwraca tensor kształtu (batch_size, n_memb, n_input) z wartościami przynależności.
    """
    def __init__(self, n_input, n_memb, **kwargs):  # Konstruktor warstwy fuzzyfikacji z parametrami liczby cech i funkcji przynależności
        super().__init__(**kwargs)  # Wywołuje konstruktor klasy bazowej tf.keras.layers.Layer
        self.n = int(n_input)  # Zapisuje liczbę cech wejściowych jako atrybut
        self.m = int(n_memb)  # Zapisuje liczbę funkcji przynależności na cechę jako atrybut
        self.eps = 1e-6  # Mała wartość epsilon zapobiegająca dzieleniu przez zero

    def build(self, input_shape):  # Metoda wywoływana automatycznie przez Keras podczas pierwszego użycia warstwy
        self.c = self.add_weight(  # Tworzy wagi reprezentujące centra funkcji gaussowskich (uczone przez gradient descent)
            name="c",  # Nazwa wag do identyfikacji w modelu
            shape=(self.m, self.n),  # Kształt: (liczba_MF, liczba_cech) - każda cecha ma m centrów
            initializer=tf.keras.initializers.RandomUniform(minval=-1.5, maxval=1.5, seed=42),  # Losowa inicjalizacja z zakresu [-1.5, 1.5]
            trainable=True,  # Wagi są aktualizowane podczas treningu
        )
        self.sigma = self.add_weight(  # Tworzy wagi reprezentujące szerokości (odchylenia standardowe) funkcji gaussowskich
            name="sigma",  # Nazwa wag do identyfikacji w modelu
            shape=(self.m, self.n),  # Kształt: (liczba_MF, liczba_cech) - każda cecha ma m sigm
            initializer=tf.keras.initializers.RandomUniform(minval=0.5, maxval=1.5, seed=42),  # Losowa inicjalizacja z zakresu [0.5, 1.5]
            trainable=True,  # Wagi są aktualizowane podczas treningu
        )
        super().build(input_shape)  # Wywołuje metodę build klasy bazowej aby zakończyć budowę warstwy

    def call(self, x):  # Główna metoda obliczeniowa warstwy - wywoływana podczas forward pass
        x = tf.expand_dims(x, axis=1)  # Dodaje wymiar: (B, n) -> (B, 1, n) aby umożliwić broadcast z (m, n)
        sigma_eff = tf.maximum(self.sigma, self.eps)  # O granicza sigma od dołu aby uniknąć dzielenia przez zero
        z = (x - self.c[None, :, :]) / (sigma_eff[None, :, :] + self.eps)  # Oblicza znormalizowaną odległość od centrów: (x-c)/sigma
        mu = tf.exp(-0.5 * tf.square(z))  # Oblicza wartości funkcji gaussowskiej: exp(-0.5*z^2) -> stopnie przynależności
        return tf.clip_by_value(mu, 1e-8, 1.0)  # Ogranicza wartości do zakresu [1e-8, 1.0] dla stabilności numerycznej


class RuleLayer(tf.keras.layers.Layer):
    """
    Warstwa obliczająca siłę aktywacji wszystkich reguł poprzez iloczyn (T-norma).
    
    Tworzy wszystkie możliwe kombinacje funkcji przynależności.
    Zwraca tensor kształtu (batch_size, n_memb^n_input) reprezentujący siłę każdej reguły.
    """
    def __init__(self, n_input, n_memb, **kwargs):
        super().__init__(**kwargs)
        self.n = int(n_input)
        self.m = int(n_memb)

    def call(self, mu):  # Główna metoda obliczeniowa warstwy - tworzy wszystkie kombinacje reguł fuzzy
        if self.n < 1:  # Sprawdza czy liczba cech jest większa lub równa 1
            raise ValueError("n_input must be >= 1")  # Rzuca wyjątek jeśli liczba cech jest nieprawidłowa
        out = mu[:, :, 0]  # Inicjalizuje wyjście pierwszą cechą: (B, m)
        for i in range(1, self.n):  # Iteruje przez pozostałe cechy (od 1 do n-1)
            out = tf.einsum("bm,bn->bmn", out, mu[:, :, i])  # Mnoży obecne reguły przez stopnie przynależności kolejnej cechy (iloczyn kartezjański)
            out = tf.reshape(out, (tf.shape(mu)[0], -1))  # Spłaszcza tensor do kształtu (B, m^(i+1)) aby kontynuować iterację
        return out  # Zwraca siły aktywacji wszystkich reguł: (B, m^n)


class NormLayer(tf.keras.layers.Layer):
    """
    Warstwa normalizująca siły aktywacji reguł.
    
    Każda siła reguły jest dzielona przez sumę wszystkich sił,
    co zapewnia że suma znormalizowanych wag wynosi 1.
    """
    def call(self, w):  # Główna metoda obliczeniowa warstwy - normalizuje siły aktywacji reguł
        s = tf.reduce_sum(w, axis=1, keepdims=True)  # Oblicza sumę wszystkich sił reguł dla każdego przykładu w batchu
        return w / (s + 1e-8)  # Dzieli każdą siłę przez sumę (normalizacja) dodając epsilon dla stabilności


class DefuzzLayer(tf.keras.layers.Layer):
    """
    Warstwa defuzzyfikacji zgodnie z modelem Takagi-Sugeno-Kang pierwszego rzędu.
    
    Dla każdej reguły oblicza konsekwent liniowy: f_i(x) = x·W_i + b_i,
    a następnie mnoży przez znormalizowaną siłę aktywacji reguły.
    """
    def __init__(self, n_input, n_memb, **kwargs):  # Konstruktor warstwy defuzzyfikacji z parametrami liczby cech i funkcji przynależności
        super().__init__(**kwargs)  # Wywołuje konstruktor klasy bazowej tf.keras.layers.Layer
        self.n = int(n_input)  # Zapisuje liczbę cech wejściowych jako atrybut
        self.m = int(n_memb)  # Zapisuje liczbę funkcji przynależności na cechę jako atrybut
        R = self.m ** self.n  # Oblicza całkowitą liczbę reguł: m^n (wszystkie kombinacje)

        self.CP_bias = self.add_weight(  # Tworzy wagi biasów konsekwentów (wyraz wolny w funkcji liniowej każdej reguły)
            name="Consequence_bias",  # Nazwa wag do identyfikacji w modelu
            shape=(1, R),  # Kształt: (1, liczba_reguł) - jeden bias na każdą regułę
            initializer=tf.keras.initializers.RandomUniform(minval=-2, maxval=2, seed=42),  # Losowa inicjalizacja z zakresu [-2, 2]
            trainable=True,  # Wagi są aktualizowane podczas treningu
        )
        self.CP_weight = self.add_weight(  # Tworzy wagi konsekwentów (współczynniki funkcji liniowej każdej reguły)
            name="Consequence_weight",  # Nazwa wag do identyfikacji w modelu
            shape=(self.n, R),  # Kształt: (liczba_cech, liczba_reguł) - każda reguła ma n współczynników
            initializer=tf.keras.initializers.RandomUniform(minval=-2, maxval=2, seed=42),  # Losowa inicjalizacja z zakresu [-2, 2]
            trainable=True,  # Wagi są aktualizowane podczas treningu
        )

    def call(self, w_norm, x):  # Główna metoda obliczeniowa warstwy - oblicza ważone konsekwenty reguł
        y = tf.matmul(x, self.CP_weight) + self.CP_bias  # Oblicza konsekwenty liniowe: f_i(x) = x·W_i + b_i dla wszystkich reguł
        return w_norm * y  # Mnoży konsekwenty przez znormalizowane siły aktywacji reguł: w_norm_i * f_i(x)


class SummationLayer(tf.keras.layers.Layer):
    """
    Warstwa agregacji końcowej - sumuje wyniki wszystkich reguł.
    
    Wyjście modelu jest sumą ważonych konsekwentów ze wszystkich aktywnych reguł.
    """
    def call(self, per_rule):  # Główna metoda obliczeniowa warstwy - agreguje wyniki ze wszystkich reguł
        return tf.reduce_sum(per_rule, axis=1, keepdims=True)  # Sumuje ważone konsekwenty wszystkich reguł otrzymując końcowe wyjście modelu
