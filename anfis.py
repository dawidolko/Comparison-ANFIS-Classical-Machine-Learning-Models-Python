"""
ANFIS (TSK-1) for Wine Quality
- dynamiczny batch (bez wymuszania batch_size w warstwach)
- stabilna fuzzyfikacja (broadcast, clamp sigma)
- wyjście z sigmoid (prawdopodobieństwo dla BCE)
"""

import tensorflow as tf


class ANFISModel:
    def __init__(self, n_input, n_memb, batch_size=32):
        self.n = int(n_input)
        self.m = int(n_memb)
        self.batch_size = int(batch_size)

        x_in = tf.keras.layers.Input(shape=(self.n,), name='inputLayer')  # brak sztywnego batcha
        mu   = FuzzyLayer(self.n, self.m, name='fuzzy_layer')(x_in)       # (B,m,n)
        w    = RuleLayer(self.n, self.m, name='ruleLayer')(mu)            # (B,R)
        wN   = NormLayer(name='normLayer')(w)                              # (B,R)
        per  = DefuzzLayer(self.n, self.m, name='defuzzy_layer')(wN, x_in) # (B,R)
        ylin = SummationLayer(name='sumLayer')(per)                        # (B,1)
        prob = tf.keras.layers.Activation('sigmoid', name='prob')(ylin)    # (B,1)
        self.model = tf.keras.Model(inputs=[x_in], outputs=[prob], name='ANFIS_WineQuality')

        # pomocnicze uchwyty
        self.update_weights()

    def __call__(self, X):
        return self.model.predict(X, batch_size=self.batch_size, verbose=0)

    def update_weights(self):
        fz = self.model.get_layer('fuzzy_layer')
        self.cs, self.sigmas = fz.get_weights()  # 'c', 'sigma'
        df = self.model.get_layer('defuzzy_layer')
        self.bias, self.weights = df.get_weights()  # 'Consequence_bias', 'Consequence_weight'

    def fit(self, X, y, **kwargs):
        hist = self.model.fit(X, y, **kwargs)
        self.update_weights()
        return hist

    def get_membership_functions(self):
        return self.cs, self.sigmas


class FuzzyLayer(tf.keras.layers.Layer):
    """Gaussowskie MF: zwraca μ(x) o kształcie (B, m, n)."""
    def __init__(self, n_input, n_memb, **kwargs):
        super().__init__(**kwargs)
        self.n = int(n_input)
        self.m = int(n_memb)
        self.eps = 1e-6

    def build(self, input_shape):
        self.c = self.add_weight(
            name='c',
            shape=(self.m, self.n),
            initializer=tf.keras.initializers.RandomUniform(minval=-1.5, maxval=1.5, seed=42),
            trainable=True,
        )
        # zachowujemy nazwę 'sigma' (kompatybilność), w call pilnujemy dodatniości
        self.sigma = self.add_weight(
            name='sigma',
            shape=(self.m, self.n),
            initializer=tf.keras.initializers.RandomUniform(minval=0.5, maxval=1.5, seed=42),
            trainable=True,
        )
        super().build(input_shape)

    def call(self, x):
        # x: (B,n) -> (B,1,n); c,sigma: (m,n) -> (1,m,n)
        x = tf.expand_dims(x, axis=1)
        sigma_eff = tf.maximum(self.sigma, self.eps)
        z = (x - self.c[None, :, :]) / (sigma_eff[None, :, :] + self.eps)
        return tf.exp(-0.5 * tf.square(z))  # (B,m,n)


class RuleLayer(tf.keras.layers.Layer):
    """AND reguł przez iloczyn: wynik (B, m^n)."""
    def __init__(self, n_input, n_memb, **kwargs):
        super().__init__(**kwargs)
        self.n = int(n_input)
        self.m = int(n_memb)

    def call(self, mu):
        if self.n < 1:
            raise ValueError("n_input must be >= 1")
        out = mu[:, :, 0]  # (B,m)
        for i in range(1, self.n):
            out = tf.einsum('bm,bn->bmn', out, mu[:, :, i])  # (B,m,m,...)
            out = tf.reshape(out, (tf.shape(mu)[0], -1))     # (B, m^(i+1))
        return out  # (B, m^n)


class NormLayer(tf.keras.layers.Layer):
    """Normalizacja po regułach."""
    def call(self, w):
        s = tf.reduce_sum(w, axis=1, keepdims=True)
        return w / (s + 1e-8)


class DefuzzLayer(tf.keras.layers.Layer):
    """
    TSK-1: f_i(x) = x @ W[:, i] + b[i], a następnie w_norm * f(x).
    Zostawiamy nazwy/kształty wag jak poprzednio (kompatybilność).
    """
    def __init__(self, n_input, n_memb, **kwargs):
        super().__init__(**kwargs)
        self.n = int(n_input)
        self.m = int(n_memb)
        R = self.m ** self.n
        self.CP_bias = self.add_weight(
            name='Consequence_bias',
            shape=(1, R),
            initializer=tf.keras.initializers.RandomUniform(minval=-2, maxval=2, seed=42),
            trainable=True,
        )
        self.CP_weight = self.add_weight(
            name='Consequence_weight',
            shape=(self.n, R),
            initializer=tf.keras.initializers.RandomUniform(minval=-2, maxval=2, seed=42),
            trainable=True,
        )

    def call(self, w_norm, x):
        y = tf.matmul(x, self.CP_weight) + self.CP_bias  # (B,R)
        return w_norm * y                                 # (B,R)


class SummationLayer(tf.keras.layers.Layer):
    """Suma po regułach -> (B,1)."""
    def call(self, per_rule):
        return tf.reduce_sum(per_rule, axis=1, keepdims=True)
