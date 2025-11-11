"""
TRAIN COMPARISON MODELS
-----------------------
Trenuje trzy klasyczne modele ML:
- Neural Network (MLP)
- SVM (RBF)
- Random Forest

Wyniki zapisywane w formacie zgodnym z ANFIS:
{
  "metric_type": "accuracy",
  "train_accuracy": 0.95,
  "test_accuracy": 0.91,
  "train_loss": 0.23,
  "test_loss": 0.27,
  ...
}
"""

import os, json, random, time, pickle
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    average_precision_score, confusion_matrix,
    classification_report, log_loss
)

# ------------------------------------------------------
# Konfiguracja
# ------------------------------------------------------
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)
os.environ["TF_DETERMINISTIC_OPS"] = "1"

RESULTS_DIR = "results"
MODELS_DIR  = "models"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR,  exist_ok=True)


# ------------------------------------------------------
# Pomocnicze
# ------------------------------------------------------
def now_suffix():
    return time.strftime("%Y%m%d-%H%M%S")

def ensure_column_vector(y):
    y = np.asarray(y)
    return y.reshape(-1, 1) if y.ndim == 1 else y

def class_weight_from_labels(y):
    y = np.asarray(y).ravel()
    pos = np.sum(y == 1)
    neg = np.sum(y == 0)
    if pos == 0 or neg == 0:
        return None
    w_pos = neg / max(pos, 1)
    return {0: 1.0, 1: float(w_pos)}

def load_data():
    base = os.path.join("data", "wine-quality") if os.path.exists("data/wine-quality/X_train.npy") else "data"
    X_train = np.load(os.path.join(base, "X_train.npy"))
    y_train = np.load(os.path.join(base, "y_train.npy"))
    X_test  = np.load(os.path.join(base, "X_test.npy"))
    y_test  = np.load(os.path.join(base, "y_test.npy"))
    y_train = ensure_column_vector(y_train).astype(np.float32)
    y_test  = ensure_column_vector(y_test).astype(np.float32)
    return X_train, y_train, X_test, y_test


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


# ------------------------------------------------------
# Neural Network
# ------------------------------------------------------
def train_neural_network():
    print("\n" + "="*70)
    print("TRENING: NEURAL NETWORK (MLP)")
    print("="*70)

    X_train, y_train, X_test, y_test = load_data()

    scaler = StandardScaler().fit(X_train)
    X_train_s, X_test_s = scaler.transform(X_train), scaler.transform(X_test)
    pickle.dump(scaler, open(os.path.join(MODELS_DIR, "scaler_nn.pkl"), "wb"))

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(X_train_s.shape[1],)),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    ckpt_path = os.path.join(MODELS_DIR, "nn_best.keras")
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(ckpt_path, monitor="val_loss", save_best_only=True, mode="min", verbose=0),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, verbose=0),
    ]

    cw = class_weight_from_labels(y_train)
    history = model.fit(
        X_train_s, y_train, validation_split=0.2,
        epochs=50, batch_size=32, verbose=1,
        class_weight=cw, callbacks=callbacks
    )

    # Ewaluacja
    train_loss, train_acc = model.evaluate(X_train_s, y_train, verbose=0)
    test_loss, test_acc = model.evaluate(X_test_s, y_test, verbose=0)
    y_proba = model.predict(X_test_s, verbose=0).ravel()
    y_pred  = (y_proba >= 0.5).astype(int)

    metrics = {
        "metric_type": "accuracy",
        "train_loss": float(train_loss),
        "test_loss": float(test_loss),
        "train_accuracy": float(train_acc),
        "test_accuracy": float(test_acc),
        "f1": float(f1_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "pr_auc": float(average_precision_score(y_test, y_proba)),
    }

    save_json(metrics, os.path.join(RESULTS_DIR, "nn_results.json"))
    model.save(ckpt_path, overwrite=True)
    print(f"✓ Zapisano: nn_results.json, model, scaler")
    return model, metrics


# ------------------------------------------------------
# SVM
# ------------------------------------------------------
def train_svm():
    print("\n" + "="*70)
    print("TRENING: SVM (RBF)")
    print("="*70)

    X_train, y_train, X_test, y_test = load_data()
    y_train_r, y_test_r = y_train.ravel(), y_test.ravel()

    clf = make_pipeline(
        StandardScaler(),
        SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=SEED)
    )
    clf.fit(X_train, y_train_r)

    y_proba = clf.predict_proba(X_test)[:, 1]
    y_pred  = (y_proba >= 0.5).astype(int)

    metrics = {
        "metric_type": "accuracy",
        "train_accuracy": float(clf.score(X_train, y_train_r)),
        "test_accuracy": float(accuracy_score(y_test_r, y_pred)),
        "train_loss": float(log_loss(y_train_r, clf.predict_proba(X_train)[:, 1])),
        "test_loss": float(log_loss(y_test_r, y_proba)),
        "f1": float(f1_score(y_test_r, y_pred)),
        "roc_auc": float(roc_auc_score(y_test_r, y_proba)),
        "pr_auc": float(average_precision_score(y_test_r, y_proba))
    }

    save_json(metrics, os.path.join(RESULTS_DIR, "svm_results.json"))
    pickle.dump(clf, open(os.path.join(MODELS_DIR, "svm_model.pkl"), "wb"))
    print("✓ Zapisano: svm_results.json, model.pkl")
    return clf, metrics


# ------------------------------------------------------
# RANDOM FOREST
# ------------------------------------------------------
def train_random_forest():
    print("\n" + "="*70)
    print("TRENING: RANDOM FOREST")
    print("="*70)

    X_train, y_train, X_test, y_test = load_data()
    y_train_r, y_test_r = y_train.ravel(), y_test.ravel()

    rf = RandomForestClassifier(
        n_estimators=300, max_depth=None,
        min_samples_split=5, min_samples_leaf=2,
        max_features="sqrt", class_weight="balanced",
        oob_score=True, random_state=SEED, n_jobs=-1
    )
    rf.fit(X_train, y_train_r)

    y_proba = rf.predict_proba(X_test)[:, 1]
    y_pred  = (y_proba >= 0.5).astype(int)

    metrics = {
        "metric_type": "accuracy",
        "train_accuracy": float(rf.score(X_train, y_train_r)),
        "test_accuracy": float(accuracy_score(y_test_r, y_pred)),
        "train_loss": float(log_loss(y_train_r, rf.predict_proba(X_train)[:, 1])),
        "test_loss": float(log_loss(y_test_r, y_proba)),
        "f1": float(f1_score(y_test_r, y_pred)),
        "roc_auc": float(roc_auc_score(y_test_r, y_proba)),
        "pr_auc": float(average_precision_score(y_test_r, y_proba)),
        "oob_score": float(rf.oob_score_)
    }

    save_json(metrics, os.path.join(RESULTS_DIR, "rf_results.json"))
    pickle.dump(rf, open(os.path.join(MODELS_DIR, "rf_model.pkl"), "wb"))
    np.save(os.path.join(RESULTS_DIR, "rf_feature_importances.npy"), rf.feature_importances_)
    print("✓ Zapisano: rf_results.json, model.pkl, feature_importances.npy")
    return rf, metrics


# ------------------------------------------------------
# GŁÓWNY BLOK
# ------------------------------------------------------
if __name__ == "__main__":
    all_results = {}

    for name, func in [
        ("Neural Network", train_neural_network),
        ("SVM", train_svm),
        ("Random Forest", train_random_forest),
    ]:
        try:
            _, metrics = func()
            all_results[name] = metrics
        except Exception as e:
            print(f"❌ Błąd podczas trenowania {name}: {e}")

    print("\n" + "="*80)
    print("PODSUMOWANIE TESTOWE:")
    print("="*80)
    for name, res in all_results.items():
        acc = res.get("test_accuracy", np.nan)
        loss = res.get("test_loss", np.nan)
        auc = res.get("roc_auc", np.nan)
        print(f"{name:<20} ACC={acc:.4f} | LOSS={loss:.4f} | AUC={auc:.4f}")
