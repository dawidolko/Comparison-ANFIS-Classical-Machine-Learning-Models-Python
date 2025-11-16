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

import tensorflow as tf
import numpy as np


class ANFISModel:
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
        self.n = int(n_input)
        self.m = int(n_memb)
        self.batch_size = int(batch_size)
        self.regression = bool(regression)

        # --- Warstwy modelu ---
        x_in = tf.keras.layers.Input(shape=(self.n,), name="inputLayer")

        mu = FuzzyLayer(self.n, self.m, name="fuzzy_layer")(x_in)          # (B, m, n)
        w = RuleLayer(self.n, self.m, name="ruleLayer")(mu)                # (B, R)
        w_norm = NormLayer(name="normLayer")(w)                            # (B, R)
        per_rule = DefuzzLayer(self.n, self.m, name="defuzzy_layer")(w_norm, x_in)  # (B, R)
        y_lin = SummationLayer(name="sumLayer")(per_rule)                  # (B, 1)

        # --- Wyjście ---
        if regression:
            out = tf.keras.layers.Activation("linear", name="output")(y_lin)
            model_name = "ANFIS_Concrete"
        else:
            out = tf.keras.layers.Activation("sigmoid", name="prob")(y_lin)
            model_name = "ANFIS_WineQuality"

        self.model = tf.keras.Model(inputs=[x_in], outputs=[out], name=model_name)
        self.update_weights()

    # --- Interfejs wysokiego poziomu ---

    def __call__(self, X):
        """
        Wykonuje predykcję dla danych wejściowych X.
        
        Args:
            X: macierz cech (n_samples, n_features)
            
        Returns:
            Tablica predykcji (n_samples, 1)
        """
        return self.model.predict(X, batch_size=self.batch_size, verbose=0)

    def fit(self, X, y, **kwargs):
        """
        Trenuje model ANFIS na danych treningowych.
        
        Args:
            X: macierz cech treningowych
            y: wektor etykiet
            **kwargs: dodatkowe argumenty przekazywane do model.fit()
            
        Returns:
            Historia treningu (History object)
        """
        hist = self.model.fit(X, y, **kwargs)
        self.update_weights()
        return hist

    def update_weights(self):
        """
        Aktualizuje lokalne kopie wag z warstw Keras.
        
        Pobiera parametry z warstw fuzzy_layer (centra i sigmy) 
        oraz defuzzy_layer (wagi konsekwentne i bias).
        """
        fz = self.model.get_layer("fuzzy_layer")
        self.cs, self.sigmas = fz.get_weights()

        df = self.model.get_layer("defuzzy_layer")
        self.bias, self.weights = df.get_weights()

    def get_membership_functions(self):
        """
        Zwraca parametry gaussowskich funkcji przynależności.
        
        Returns:
            Tuple (centra, sigmy) - oba jako tablice numpy shape (n_memb, n_features)
        """
        return self.cs, self.sigmas

    # --- Eksport reguł ---

    def to_rules_json(self):
        """
        Generuje listę reguł w formacie kompatybilnym z extract_and_save_rules().
        """
        n_features = self.n
        n_memb = self.m
        weights = self.weights
        bias = self.bias

        n_rules = n_memb ** n_features
        rules = []
        for ridx in range(n_rules):
            combo = self._rule_index_to_tuple(ridx, n_features, n_memb)
            cons_w = weights[:, ridx].tolist()
            cons_b = float(bias[0, ridx])
            rules.append({
                "rule_index": int(ridx),
                "membership_indices": combo,
                "consequent": {"weights": cons_w, "bias": cons_b}
            })

        return {
            "n_features": n_features,
            "n_memb": n_memb,
            "n_rules_total": n_rules,
            "rules_listed": len(rules),
            "rules": rules
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
        combo = []
        for _ in range(n_features):
            combo.append(idx % n_memb)
            idx //= n_memb
        return list(reversed(combo))


# =========================================================================== #
#                                 WARSTWY                                     #
# =========================================================================== #

class FuzzyLayer(tf.keras.layers.Layer):
    """
    Warstwa fuzzyfikacji z gaussowskimi funkcjami przynależności.
    
    Dla każdej cechy wejściowej tworzy n_memb funkcji gaussowskich.
    Zwraca tensor kształtu (batch_size, n_memb, n_input) z wartościami przynależności.
    """
    def __init__(self, n_input, n_memb, **kwargs):
        super().__init__(**kwargs)
        self.n = int(n_input)
        self.m = int(n_memb)
        self.eps = 1e-6

    def build(self, input_shape):
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
        super().build(input_shape)

    def call(self, x):
        x = tf.expand_dims(x, axis=1)                       # (B, 1, n)
        sigma_eff = tf.maximum(self.sigma, self.eps)        # (m, n)
        z = (x - self.c[None, :, :]) / (sigma_eff[None, :, :] + self.eps)
        mu = tf.exp(-0.5 * tf.square(z))                    # (B, m, n)
        return tf.clip_by_value(mu, 1e-8, 1.0)              # clamp 0–1 (stabilność)


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

    def call(self, mu):
        if self.n < 1:
            raise ValueError("n_input must be >= 1")
        out = mu[:, :, 0]
        for i in range(1, self.n):
            out = tf.einsum("bm,bn->bmn", out, mu[:, :, i])
            out = tf.reshape(out, (tf.shape(mu)[0], -1))
        return out


class NormLayer(tf.keras.layers.Layer):
    """
    Warstwa normalizująca siły aktywacji reguł.
    
    Każda siła reguły jest dzielona przez sumę wszystkich sił,
    co zapewnia że suma znormalizowanych wag wynosi 1.
    """
    def call(self, w):
        s = tf.reduce_sum(w, axis=1, keepdims=True)
        return w / (s + 1e-8)


class DefuzzLayer(tf.keras.layers.Layer):
    """
    Warstwa defuzzyfikacji zgodnie z modelem Takagi-Sugeno-Kang pierwszego rzędu.
    
    Dla każdej reguły oblicza konsekwent liniowy: f_i(x) = x·W_i + b_i,
    a następnie mnoży przez znormalizowaną siłę aktywacji reguły.
    """
    def __init__(self, n_input, n_memb, **kwargs):
        super().__init__(**kwargs)
        self.n = int(n_input)
        self.m = int(n_memb)
        R = self.m ** self.n

        self.CP_bias = self.add_weight(
            name="Consequence_bias",
            shape=(1, R),
            initializer=tf.keras.initializers.RandomUniform(minval=-2, maxval=2, seed=42),
            trainable=True,
        )
        self.CP_weight = self.add_weight(
            name="Consequence_weight",
            shape=(self.n, R),
            initializer=tf.keras.initializers.RandomUniform(minval=-2, maxval=2, seed=42),
            trainable=True,
        )

    def call(self, w_norm, x):
        y = tf.matmul(x, self.CP_weight) + self.CP_bias  # (B, R)
        return w_norm * y                                # (B, R)


class SummationLayer(tf.keras.layers.Layer):
    """
    Warstwa agregacji końcowej - sumuje wyniki wszystkich reguł.
    
    Wyjście modelu jest sumą ważonych konsekwentów ze wszystkich aktywnych reguł.
    """
    def call(self, per_rule):
        return tf.reduce_sum(per_rule, axis=1, keepdims=True)
