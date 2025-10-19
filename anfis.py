from __future__ import annotations
from typing import Optional, Tuple
import tensorflow as tf
import numpy as np

class FuzzyLayer(tf.keras.layers.Layer):
    def __init__(self, n_input: int, n_memb: int, **kwargs):
        super().__init__(**kwargs)
        self.n = int(n_input)
        self.m = int(n_memb)

    def build(self, input_shape):
        self.c = self.add_weight(
            name="centres",
            shape=(self.m, self.n),
            initializer=tf.keras.initializers.RandomUniform(minval=-1.0, maxval=1.0),
            trainable=True,
        )
        self.log_sigma = self.add_weight(
            name="log_sigma",
            shape=(self.m, self.n),
            initializer=tf.keras.initializers.RandomUniform(minval=-0.2, maxval=0.2),
            trainable=True,
        )
        self.eps = tf.constant(1e-6, dtype=self.dtype or tf.float32)
        super().build(input_shape)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        x_exp = tf.expand_dims(x, axis=1)
        sigma = tf.nn.softplus(self.log_sigma) + self.eps
        z = (x_exp - self.c) / (sigma + self.eps)
        mu = tf.exp(-0.5 * tf.square(z))
        return mu

    def get_membership_params(self) -> Tuple[np.ndarray, np.ndarray]:
        c = self.c.numpy()
        sigma = (np.log1p(np.exp(self.log_sigma.numpy())) + 1e-6)  # softplus
        return c, sigma

    def init_from_data(self, X: np.ndarray):
        X = np.asarray(X)
        assert X.ndim == 2 and X.shape[1] == self.n, "X must be (N, n_input)"
        qs = np.linspace(0.1, 0.9, num=self.m)
        centres = np.zeros((self.m, self.n), dtype=np.float32)
        widths = np.zeros_like(centres)
        for i in range(self.n):
            col = X[:, i]
            c_i = np.quantile(col, qs)
            centres[:, i] = c_i
            diffs = np.diff(c_i)
            if len(diffs) == 0 or np.all(diffs == 0):
                iqr = np.subtract(*np.percentile(col, [75, 25]))
                s = iqr / (2.0 * max(self.m, 2)) if iqr > 0 else (col.std() + 1e-3)
                widths[:, i] = s
            else:
                left = np.r_[diffs[0], diffs]
                right = np.r_[diffs, diffs[-1]]
                widths[:, i] = 0.5 * (left + right)
        widths = np.clip(widths, 1e-3, None)
        self.c.assign(centres)
        self.log_sigma.assign(np.log(np.expm1(widths)))


class RuleLayer(tf.keras.layers.Layer):
    def __init__(self, n_input: int, n_memb: int, **kwargs):
        super().__init__(**kwargs)
        self.n = int(n_input)
        self.m = int(n_memb)

    def call(self, mu: tf.Tensor) -> tf.Tensor:
        if self.n < 1:
            raise ValueError("n_input must be >= 1")
        parts = [mu[:, :, i] for i in range(self.n)]
        out = parts[0]
        for i in range(1, self.n):
            out = tf.einsum("bm,bn->bmn", out, parts[i])
            out = tf.reshape(out, (tf.shape(mu)[0], -1))
        return out


class NormLayer(tf.keras.layers.Layer):
    def call(self, w: tf.Tensor) -> tf.Tensor:
        s = tf.reduce_sum(w, axis=1, keepdims=True)
        return w / (s + 1e-8)


class DefuzzLayer(tf.keras.layers.Layer):
    def __init__(self, n_input: int, n_rules: int, **kwargs):
        super().__init__(**kwargs)
        self.n = int(n_input)
        self.r = int(n_rules)

    def build(self, input_shape):
        self.A = self.add_weight(
            name="A",
            shape=(self.n, self.r),
            initializer=tf.keras.initializers.RandomUniform(minval=-0.5, maxval=0.5),
            trainable=True,
        )
        self.b = self.add_weight(
            name="b",
            shape=(self.r,),
            initializer=tf.keras.initializers.Zeros(),
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs: Tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
        w_norm, x = inputs
        y = tf.linalg.matmul(x, self.A) + self.b
        return w_norm * y


class SummationLayer(tf.keras.layers.Layer):
    def call(self, per_rule: tf.Tensor) -> tf.Tensor:
        y = tf.reduce_sum(per_rule, axis=1, keepdims=True)
        return y


class ANFISModel:
    def __init__(self, n_input: int, n_memb: int, task: str = "classification"):
        self.n = int(n_input)
        self.m = int(n_memb)
        self.task = task

        x_in = tf.keras.Input(shape=(self.n,), name="inputs")
        mu = FuzzyLayer(self.n, self.m, name="fuzzy")(x_in)
        w = RuleLayer(self.n, self.m, name="rules")(mu)
        w_norm = NormLayer(name="norm")(w)
        R = (self.m ** self.n)
        per_rule = DefuzzLayer(self.n, R, name="defuzz")([w_norm, x_in])
        y_raw = SummationLayer(name="sum")(per_rule)

        if task == "classification":
            y_out = tf.keras.layers.Activation("sigmoid", name="out")(y_raw)
            loss = "binary_crossentropy"
            metrics = ["accuracy"]
        elif task == "regression":
            y_out = tf.identity(y_raw, name="out")
            loss = "mse"
            metrics = ["mse"]
        else:
            raise ValueError("task must be 'classification' or 'regression'")

        self.model = tf.keras.Model(inputs=x_in, outputs=y_out, name="ANFIS")
        self.model.compile(optimizer=tf.keras.optimizers.Nadam(1e-3), loss=loss, metrics=metrics)

    def summary(self):
        return self.model.summary()

    def predict(self, X: np.ndarray, batch_size: Optional[int] = None) -> np.ndarray:
        return self.model.predict(X, batch_size=batch_size or 32, verbose=0)

    def fit(self, X: np.ndarray, y: np.ndarray, *, epochs: int = 20, batch_size: int = 32,
            validation_split: float = 0.2, callbacks: Optional[list] = None, verbose: int = 1):
        return self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=verbose,
        )

    def get_membership_params(self) -> Tuple[np.ndarray, np.ndarray]:
        fl: FuzzyLayer = self.model.get_layer("fuzzy")
        return fl.get_membership_params()

    def init_memberships_from_data(self, X: np.ndarray):
        fl: FuzzyLayer = self.model.get_layer("fuzzy")
        fl.init_from_data(X)