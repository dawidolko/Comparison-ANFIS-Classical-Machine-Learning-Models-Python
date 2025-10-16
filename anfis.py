"""
ANFIS algorithm based on Gregor Lenhard from University of Basel implementation
Modified for Wine Quality Classification Project

https://github.com/gregorLen/AnfisTensorflow2.0/blob/master/Models/myanfis.py
"""

import tensorflow as tf

class ANFISModel:
    def __init__(self, n_input, n_memb, batch_size=32):
        """
        n_input: liczba cech wejściowych
        n_memb: liczba funkcji przynależności na cechę
        batch_size: rozmiar batcha
        """
        self.n = n_input
        self.m = n_memb
        self.batch_size = batch_size

        input_ = tf.keras.layers.Input(shape=(n_input,), name='inputLayer', batch_size=self.batch_size)
        L1 = FuzzyLayer(n_input, n_memb, name='fuzzy_layer')(input_)
        L2 = RuleLayer(n_input, n_memb, name='ruleLayer')(L1)
        L3 = NormLayer(name='normLayer')(L2)
        L4 = DefuzzLayer(n_input, n_memb, name='defuzzy_layer')(L3, input_)
        L5 = SummationLayer(name='sumLayer')(L4)

        self.model = tf.keras.Model(inputs=[input_], outputs=[L5], name='ANFIS_WineQuality')
        self.update_weights()

    def __call__(self, X):
        preds = self.model.predict(X, batch_size=self.batch_size, verbose=0)
        return preds

    def update_weights(self):
        """Aktualizuje zapisane wagi po treningu"""
        self.cs, self.sigmas = self.model.get_layer('fuzzy_layer').get_weights()
        self.bias, self.weights = self.model.get_layer('defuzzy_layer').get_weights()

    def fit(self, X, y, **kwargs):
        """Trenuje model"""
        self.init_weights = self.model.get_layer('fuzzy_layer').get_weights()
        history = self.model.fit(X, y, **kwargs)
        self.update_weights()
        tf.keras.backend.clear_session()
        return history

    def get_membership_functions(self):
        """Zwraca parametry funkcji przynależności"""
        return self.cs, self.sigmas


class FuzzyLayer(tf.keras.layers.Layer):
    """Warstwa fuzzyfikacji - przekształca wartości na stopnie przynależności"""
    def __init__(self, n_input, n_memb, **kwargs):
        super(FuzzyLayer, self).__init__(**kwargs)
        self.n = n_input
        self.m = n_memb

    def build(self, batch_input_shape):
        self.batch_size = batch_input_shape[0]

        # Centra funkcji gaussowskich
        self.c = self.add_weight(name='c',
                                  shape=(self.m, self.n),
                                  initializer=tf.keras.initializers.RandomUniform(minval=-1.5, maxval=1.5, seed=42),
                                  trainable=True)

        # Szerokości funkcji gaussowskich
        self.sigma = self.add_weight(name='sigma',
                                     shape=(self.m, self.n),
                                     initializer=tf.keras.initializers.RandomUniform(minval=0.5, maxval=1.5, seed=42),
                                     trainable=True)

        super(FuzzyLayer, self).build(batch_input_shape)

    def call(self, x_inputs):
        """Gaussowska funkcja przynależności"""
        L1_output = tf.exp(-1 * tf.square(tf.subtract(
                                tf.reshape(
                                    tf.tile(x_inputs, (1, self.m)), (-1, self.m, self.n)), self.c
                            )) / (tf.square(self.sigma) + 1e-8))
        return L1_output


class RuleLayer(tf.keras.layers.Layer):
    """Warstwa reguł - oblicza stopień odpalenia każdej reguły"""
    def __init__(self, n_input, n_memb, **kwargs):
        super(RuleLayer, self).__init__(**kwargs)
        self.n = n_input
        self.m = n_memb
        self.batch_size = None

    def build(self, batch_input_shape):
        self.batch_size = batch_input_shape[0]
        super(RuleLayer, self).build(batch_input_shape)

    def call(self, input_):
        if self.n < 2:
            raise ValueError("Liczba cech musi być >= 2")

        reshaped_tensors = []

        for i in range(self.n):
            shape = [self.batch_size] + [1] * self.n
            shape[i+1] = -1
            reshaped_tensors.append(tf.reshape(input_[:, :, i], shape))

        L2_output = reshaped_tensors[0]

        for tensor in reshaped_tensors[1:]:
            L2_output *= tensor

        return tf.reshape(L2_output, [self.batch_size, -1])


class NormLayer(tf.keras.layers.Layer):
    """Warstwa normalizacji - normalizuje wagi reguł"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, w):
        w_sum = tf.reshape(tf.reduce_sum(w, axis=1), (-1, 1))
        w_norm = w / (w_sum + 1e-8)
        return w_norm


class DefuzzLayer(tf.keras.layers.Layer):
    """Warstwa defuzzyfikacji - przekształca wagi na wartości wyjściowe"""
    def __init__(self, n_input, n_memb, **kwargs):
        super().__init__(**kwargs)
        self.n = n_input
        self.m = n_memb

        # Bias dla każdej reguły
        self.CP_bias = self.add_weight(name='Consequence_bias',
                                       shape=(1, self.m ** self.n),
                                       initializer=tf.keras.initializers.RandomUniform(minval=-2, maxval=2, seed=42),
                                       trainable=True)

        # Wagi dla każdej cechy i reguły
        self.CP_weight = self.add_weight(name='Consequence_weight',
                                         shape=(self.n, self.m ** self.n),
                                         initializer=tf.keras.initializers.RandomUniform(minval=-2, maxval=2, seed=42),
                                         trainable=True)

    def call(self, w_norm, input_):
        L4_L2_output = tf.multiply(w_norm, tf.matmul(input_, self.CP_weight) + self.CP_bias)
        return L4_L2_output


class SummationLayer(tf.keras.layers.Layer):
    """Warstwa sumowania - agreguje wyniki"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, batch_input_shape):
        self.batch_size = batch_input_shape[0]
        super(SummationLayer, self).build(batch_input_shape)

    def call(self, input_):
        L5_L2_output = tf.reduce_sum(input_, axis=1)
        L5_L2_output = tf.reshape(L5_L2_output, (-1, 1))
        return L5_L2_output