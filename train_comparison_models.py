"""
TRAIN COMPARISON MODELS (Wine & Concrete)
-----------------------------------------
Trenuje trzy klasyczne modele ML dla OBU problemów:
- Wine Quality (klasyfikacja binarna)
- Concrete Strength (regresja)

Zapisuje wyniki w osobnych plikach:
  - nn_wine_results.json, nn_concrete_results.json
  - svm_wine_results.json, svm_concrete_results.json
  - rf_wine_results.json, rf_concrete_results.json
"""

import os, json, random, time, pickle
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    average_precision_score, log_loss, mean_absolute_error
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
MODELS_DIR = "models"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


# ------------------------------------------------------
# Pomocnicze
# ------------------------------------------------------
def now_suffix():
    """
    Generuje znacznik czasowy w formacie YYYYMMDD-HHMMSS.
    
    Returns:
        String z aktualną datą i czasem
    """
    return time.strftime("%Y%m%d-%H%M%S")

def ensure_column_vector(y, for_classification=False):
    """
    Konwertuje etykiety do odpowiedniego kształtu dla klasyfikacji lub regresji.
    
    Args:
        y: wektor etykiet
        for_classification: True = zwraca (n, 1), False = zwraca (n,)
        
    Returns:
        Przekształcona tablica numpy
    """
    y = np.asarray(y)
    if for_classification:
        return y.reshape(-1, 1) if y.ndim == 1 else y
    return y.ravel()  # Regresja: wektor 1D

def class_weight_from_labels(y):
    """
    Oblicza wagi klas dla niezbalansowanych zestawów danych.
    
    Używane do class_weight w modelach klasyfikacyjnych.
    Zwiększa wagę klasy mniejszościowej.
    
    Args:
        y: etykiety binarne (0/1)
        
    Returns:
        Dict {0: waga_0, 1: waga_1} lub None jeśli brak próbek
    """
    y = np.asarray(y).ravel()
    pos = np.sum(y == 1)
    neg = np.sum(y == 0)
    if pos == 0 or neg == 0:
        return None
    w_pos = neg / max(pos, 1)
    return {0: 1.0, 1: float(w_pos)}

def load_wine_data():
    """
    Ładuje znormalizowane dane Wine Quality z plików .npy.
    
    Returns:
        Tuple (X_train, y_train, X_test, y_test)
    """
    base = os.path.join("data", "wine-quality") if os.path.exists("data/wine-quality/X_train.npy") else "data"
    X_train = np.load(os.path.join(base, "X_train.npy"))
    y_train = np.load(os.path.join(base, "y_train.npy"))
    X_test = np.load(os.path.join(base, "X_test.npy"))
    y_test = np.load(os.path.join(base, "y_test.npy"))
    y_train = ensure_column_vector(y_train, for_classification=True).astype(np.float32)
    y_test = ensure_column_vector(y_test, for_classification=True).astype(np.float32)
    return X_train, y_train, X_test, y_test

def load_concrete_data():
    """
    Ładuje znormalizowane dane Concrete Strength z plików .npy.
    
    Returns:
        Tuple (X_train, y_train, X_test, y_test)
    """
    base_dir = "data/concrete-strength"
    X_train = np.load(os.path.join(base_dir, "X_train.npy"))
    y_train = np.load(os.path.join(base_dir, "y_train.npy"))
    X_test = np.load(os.path.join(base_dir, "X_test.npy"))
    y_test = np.load(os.path.join(base_dir, "y_test.npy"))
    y_train = ensure_column_vector(y_train, for_classification=False)
    y_test = ensure_column_vector(y_test, for_classification=False)
    return X_train, y_train, X_test, y_test

def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


# ------------------------------------------------------
# MODELE DLA WINE QUALITY (Klasyfikacja)
# ------------------------------------------------------
def train_models_for_wine():
    print("\n🍷 TRENING MODELI DLA WINE QUALITY")
    X_train, y_train, X_test, y_test = load_wine_data()
    y_train_r, y_test_r = y_train.ravel(), y_test.ravel()

    results = {}

    # --- Neural Network ---
    print("  - Neural Network")
    scaler = StandardScaler().fit(X_train)
    X_train_s, X_test_s = scaler.transform(X_train), scaler.transform(X_test)
    pickle.dump(scaler, open(os.path.join(MODELS_DIR, "scaler_nn_wine.pkl"), "wb"))

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(X_train_s.shape[1],)),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    cw = class_weight_from_labels(y_train_r)
    model.fit(X_train_s, y_train, validation_split=0.2, epochs=50, batch_size=32, verbose=0, class_weight=cw)
    train_loss, train_acc = model.evaluate(X_train_s, y_train, verbose=0)
    test_loss, test_acc = model.evaluate(X_test_s, y_test, verbose=0)
    y_proba = model.predict(X_test_s, verbose=0).ravel()
    y_pred = (y_proba >= 0.5).astype(int)
    results['nn'] = {
        "metric_type": "accuracy",
        "train_accuracy": float(train_acc),
        "test_accuracy": float(test_acc),
        "train_loss": float(train_loss),
        "test_loss": float(test_loss),
        "f1": float(f1_score(y_test_r, y_pred)),
        "roc_auc": float(roc_auc_score(y_test_r, y_proba)),
    }
    model.save(os.path.join(MODELS_DIR, "nn_wine.keras"))
    save_json(results['nn'], os.path.join(RESULTS_DIR, "nn_wine_results.json"))

    # --- SVM ---
    print("  - SVM")
    svm = make_pipeline(StandardScaler(), SVC(probability=True, random_state=SEED))
    svm.fit(X_train, y_train_r)
    y_proba = svm.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)
    results['svm'] = {
        "metric_type": "accuracy",
        "train_accuracy": float(svm.score(X_train, y_train_r)),
        "test_accuracy": float(accuracy_score(y_test_r, y_pred)),
        "train_loss": float(log_loss(y_train_r, svm.predict_proba(X_train)[:, 1])),
        "test_loss": float(log_loss(y_test_r, y_proba)),
        "f1": float(f1_score(y_test_r, y_pred)),
        "roc_auc": float(roc_auc_score(y_test_r, y_proba)),
    }
    pickle.dump(svm, open(os.path.join(MODELS_DIR, "svm_wine.pkl"), "wb"))
    save_json(results['svm'], os.path.join(RESULTS_DIR, "svm_wine_results.json"))

    # --- Random Forest ---
    print("  - Random Forest")
    rf = RandomForestClassifier(class_weight="balanced", random_state=SEED, n_jobs=-1)
    rf.fit(X_train, y_train_r)
    y_proba = rf.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)
    results['rf'] = {
        "metric_type": "accuracy",
        "train_accuracy": float(rf.score(X_train, y_train_r)),
        "test_accuracy": float(accuracy_score(y_test_r, y_pred)),
        "train_loss": float(log_loss(y_train_r, rf.predict_proba(X_train)[:, 1])),
        "test_loss": float(log_loss(y_test_r, y_proba)),
        "f1": float(f1_score(y_test_r, y_pred)),
        "roc_auc": float(roc_auc_score(y_test_r, y_proba)),
    }
    pickle.dump(rf, open(os.path.join(MODELS_DIR, "rf_wine.pkl"), "wb"))
    save_json(results['rf'], os.path.join(RESULTS_DIR, "rf_wine_results.json"))

    print("  ✅ Zakończono Wine")
    return results


# ------------------------------------------------------
# MODELE DLA CONCRETE STRENGTH (Regresja)
# ------------------------------------------------------
def train_models_for_concrete():
    print("\n🏗️ TRENING MODELI DLA CONCRETE STRENGTH")
    X_train, y_train, X_test, y_test = load_concrete_data()

    results = {}

    # --- Neural Network ---
    print("  - Neural Network")
    scaler = StandardScaler().fit(X_train)
    X_train_s, X_test_s = scaler.transform(X_train), scaler.transform(X_test)
    pickle.dump(scaler, open(os.path.join(MODELS_DIR, "scaler_nn_concrete.pkl"), "wb"))

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(X_train_s.shape[1],)),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer="adam", loss="mae")
    model.fit(X_train_s, y_train, validation_split=0.2, epochs=100, batch_size=32, verbose=0)
    train_mae = model.evaluate(X_train_s, y_train, verbose=0)
    test_mae = model.evaluate(X_test_s, y_test, verbose=0)
    results['nn'] = {
        "metric_type": "mae",
        "train_mae": float(train_mae),
        "test_mae": float(test_mae)
    }
    model.save(os.path.join(MODELS_DIR, "nn_concrete.keras"))
    save_json(results['nn'], os.path.join(RESULTS_DIR, "nn_concrete_results.json"))

    # --- SVM (SVR) ---
    print("  - SVM (SVR)")
    svr = make_pipeline(StandardScaler(), SVR(kernel="rbf", C=100, gamma="scale"))
    svr.fit(X_train, y_train)
    train_mae = mean_absolute_error(y_train, svr.predict(X_train))
    test_mae = mean_absolute_error(y_test, svr.predict(X_test))
    results['svm'] = {
        "metric_type": "mae",
        "train_mae": float(train_mae),
        "test_mae": float(test_mae)
    }
    pickle.dump(svr, open(os.path.join(MODELS_DIR, "svm_concrete.pkl"), "wb"))
    save_json(results['svm'], os.path.join(RESULTS_DIR, "svm_concrete_results.json"))

    # --- Random Forest (Regressor) ---
    print("  - Random Forest")
    rf = RandomForestRegressor(random_state=SEED, n_jobs=-1)
    rf.fit(X_train, y_train)
    train_mae = mean_absolute_error(y_train, rf.predict(X_train))
    test_mae = mean_absolute_error(y_test, rf.predict(X_test))
    results['rf'] = {
        "metric_type": "mae",
        "train_mae": float(train_mae),
        "test_mae": float(test_mae)
    }
    pickle.dump(rf, open(os.path.join(MODELS_DIR, "rf_concrete.pkl"), "wb"))
    save_json(results['rf'], os.path.join(RESULTS_DIR, "rf_concrete_results.json"))

    print("  ✅ Zakończono Concrete")
    return results


# ------------------------------------------------------
# GŁÓWNY BLOK
# ------------------------------------------------------
if __name__ == "__main__":
    print("TRENING KLASYCZNYCH MODELI ML")
    print("=================================")

    try:
        wine_results = train_models_for_wine()
    except Exception as e:
        print(f"Błąd podczas trenowania Wine: {e}")

    try:
        concrete_results = train_models_for_concrete()
    except Exception as e:
        print(f"Błąd podczas trenowania Concrete: {e}")

    print("\nWszystkie modele zostały wytrenowane i zapisane w 'results/' oraz 'models/'.")