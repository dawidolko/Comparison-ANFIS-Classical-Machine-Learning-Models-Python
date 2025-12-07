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

import os, json, random, time, pickle  # Biblioteki systemowe: pliki, JSON, losowość, czas, serializacja
import numpy as np  # Biblioteka do operacji na tablicach numerycznych
import tensorflow as tf  # Framework do budowy i treningu sieci neuronowych
import matplotlib  # Biblioteka do tworzenia wykresów
matplotlib.use('Agg')  # Ustawia backend bez GUI (do zapisu plików)
import matplotlib.pyplot as plt  # Moduł do tworzenia wykresów

from sklearn.preprocessing import StandardScaler  # Klasa do standaryzacji danych
from sklearn.pipeline import make_pipeline  # Funkcja do tworzenia pipeline (scaler + model)
from sklearn.svm import SVC, SVR  # Support Vector Machine: Classifier i Regressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor  # Random Forest dla klasyfikacji i regresji
from sklearn.metrics import (  # Metryki ewaluacji
    accuracy_score, f1_score, roc_auc_score,  # Metryki klasyfikacyjne: accuracy, F1, ROC AUC
    average_precision_score, log_loss, mean_absolute_error  # Average Precision, log loss, MAE
)

# ------------------------------------------------------
# Konfiguracja
# ------------------------------------------------------
SEED = 42  # Ziarno generatora losowego dla reprodukowalnych wyników
np.random.seed(SEED)  # Ustawia ziarno NumPy
tf.random.set_seed(SEED)  # Ustawia ziarno TensorFlow
random.seed(SEED)  # Ustawia ziarno modułu random
os.environ["TF_DETERMINISTIC_OPS"] = "1"  # Wymusza deterministyczne operacje TensorFlow

RESULTS_DIR = "results"  # Katalog do zapisu wyników (JSON, wykresy)
MODELS_DIR = "models"  # Katalog do zapisu wytrenowanych modeli
os.makedirs(RESULTS_DIR, exist_ok=True)  # Tworzy katalog results jeśli nie istnieje
os.makedirs(MODELS_DIR, exist_ok=True)  # Tworzy katalog models jeśli nie istnieje


# ------------------------------------------------------
# Pomocnicze
# ------------------------------------------------------
def now_suffix():
    """
    Generuje znacznik czasowy w formacie YYYYMMDD-HHMMSS.
    
    Returns:
        String z aktualną datą i czasem
    """
    return time.strftime("%Y%m%d-%H%M%S")  # Zwraca string z datą i czasem w formacie RRRRMMDD-GGMMSS

def ensure_column_vector(y, for_classification=False):  # Funkcja konwertująca etykiety do odpowiedniego kształtu
    """
    Konwertuje etykiety do odpowiedniego kształtu dla klasyfikacji lub regresji.
    
    Args:
        y: wektor etykiet
        for_classification: True = zwraca (n, 1), False = zwraca (n,)
        
    Returns:
        Przekształcona tablica numpy
    """
    y = np.asarray(y)  # Konwertuje do tablicy numpy
    if for_classification:  # Jeśli to klasyfikacja
        return y.reshape(-1, 1) if y.ndim == 1 else y  # Przekształca do kształtu (n, 1) jeśli 1D, inaczej zostawia
    return y.ravel()  # Regresja: wektor 1D (spłaszcza do jednego wymiaru)

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
    y = np.asarray(y).ravel()  # Konwertuje do tablicy 1D
    pos = np.sum(y == 1)  # Liczy liczbę przykładów klasy pozytywnej (1)
    neg = np.sum(y == 0)  # Liczy liczbę przykładów klasy negatywnej (0)
    if pos == 0 or neg == 0:  # Sprawdza czy któraś klasa nie ma próbek
        return None  # Zwraca None jeśli jedna z klas jest pusta
    w_pos = neg / max(pos, 1)  # Oblicza wagę klasy pozytywnej: (liczba_neg / liczba_pos)
    return {0: 1.0, 1: float(w_pos)}  # Zwraca słownik wag: klasa 0 ma wagę 1.0, klasa 1 ma obliczoną wagę

def load_wine_data():
    """
    Ładuje znormalizowane dane Wine Quality z plików .npy.
    
    Returns:
        Tuple (X_train, y_train, X_test, y_test)
    """
    base = os.path.join("data", "wine-quality") if os.path.exists("data/wine-quality/X_train.npy") else "data"  # Określa ścieżkę bazową w zależności od istnienia katalogu
    X_train = np.load(os.path.join(base, "X_train.npy"))  # Ładuje dane treningowe cech z pliku .npy
    y_train = np.load(os.path.join(base, "y_train.npy"))  # Ładuje etykiety treningowe z pliku .npy
    X_test = np.load(os.path.join(base, "X_test.npy"))  # Ładuje dane testowe cech z pliku .npy
    y_test = np.load(os.path.join(base, "y_test.npy"))  # Ładuje etykiety testowe z pliku .npy
    y_train = ensure_column_vector(y_train, for_classification=True).astype(np.float32)  # Konwertuje do kształtu (n, 1) i typu float32
    y_test = ensure_column_vector(y_test, for_classification=True).astype(np.float32)  # Konwertuje do kształtu (n, 1) i typu float32
    return X_train, y_train, X_test, y_test  # Zwraca krotkę z danymi

def load_concrete_data():
    """
    Ładuje znormalizowane dane Concrete Strength z plików .npy.
    
    Returns:
        Tuple (X_train, y_train, X_test, y_test)
    """
    base_dir = "data/concrete-strength"  # Określa katalog z danymi betonu
    X_train = np.load(os.path.join(base_dir, "X_train.npy"))  # Ładuje dane treningowe cech
    y_train = np.load(os.path.join(base_dir, "y_train.npy"))  # Ładuje wartości treningowe (wytrzymałość)
    X_test = np.load(os.path.join(base_dir, "X_test.npy"))  # Ładuje dane testowe cech
    y_test = np.load(os.path.join(base_dir, "y_test.npy"))  # Ładuje wartości testowe (wytrzymałość)
    y_train = ensure_column_vector(y_train, for_classification=False)  # Konwertuje do wektora 1D (regresja)
    y_test = ensure_column_vector(y_test, for_classification=False)  # Konwertuje do wektora 1D (regresja)
    return X_train, y_train, X_test, y_test  # Zwraca krotkę z danymi

def save_json(obj, path):  # Funkcja zapisująca obiekt do pliku JSON
    os.makedirs(os.path.dirname(path), exist_ok=True)  # Tworzy katalog jeśli nie istnieje
    with open(path, "w") as f:  # Otwiera plik do zapisu
        json.dump(obj, f, indent=2)  # Zapisuje obiekt jako JSON z wczęciami


# ------------------------------------------------------
# MODELE DLA WINE QUALITY (Klasyfikacja)
# ------------------------------------------------------
def train_models_for_wine():  # Główna funkcja trenująca wszystkie modele dla Wine Quality
    print("\n🍷 TRENING MODELI DLA WINE QUALITY")  # Informuje o rozpoczęciu treningu
    X_train, y_train, X_test, y_test = load_wine_data()  # Ładuje dane wina (już znormalizowane)
    y_train_r, y_test_r = y_train.ravel(), y_test.ravel()  # Konwertuje do wektorów 1D (dla sklearn)

    results = {}  # Słownik do przechowywania wyników wszystkich modeli

    # --- Neural Network ---
    print("  - Neural Network")  # Informuje o treningu NN
    scaler = StandardScaler().fit(X_train)  # Tworzy i dopasowuje scaler do danych treningowych
    X_train_s, X_test_s = scaler.transform(X_train), scaler.transform(X_test)  # Standaryzuje dane (mean=0, std=1)
    pickle.dump(scaler, open(os.path.join(MODELS_DIR, "scaler_nn_wine.pkl"), "wb"))  # Zapisuje scaler do pliku

    model = tf.keras.Sequential([  # Tworzy sekwencyjny model sieci neuronowej
        tf.keras.layers.Input(shape=(X_train_s.shape[1],)),  # Warstwa wejściowa o rozmiarze równym liczbie cech
        tf.keras.layers.Dense(32, activation="relu"),  # Pierwsza warstwa gęsta z 32 neuronami i aktywacją ReLU
        tf.keras.layers.Dropout(0.3),  # Dropout 30% aby zapobiec przeuczeniu
        tf.keras.layers.Dense(16, activation="relu"),  # Druga warstwa gęsta z 16 neuronami i aktywacją ReLU
        tf.keras.layers.Dropout(0.2),  # Dropout 20% aby zapobiec przeuczeniu
        tf.keras.layers.Dense(1, activation="sigmoid")  # Warstwa wyjściowa z 1 neuronem i aktywacją sigmoid (klasyfikacja binarna)
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])  # Kompiluje model z optymalizatorem Adam, funkcją straty entropii krzyżowej i metryką accuracy
    cw = class_weight_from_labels(y_train_r)  # Oblicza wagi klas aby zbalansować niezrównoważenie
    model.fit(X_train_s, y_train, validation_split=0.2, epochs=50, batch_size=32, verbose=0, class_weight=cw)  # Trenuje model przez 50 epok z walidacją 20% i wagami klas
    train_loss, train_acc = model.evaluate(X_train_s, y_train, verbose=0)  # Oblicza stratę i accuracy na zbiorze treningowym
    test_loss, test_acc = model.evaluate(X_test_s, y_test, verbose=0)  # Oblicza stratę i accuracy na zbiorze testowym
    y_proba = model.predict(X_test_s, verbose=0).ravel()  # Przewiduje prawdopodobieństwa dla zbioru testowego i spłaszcza do 1D
    y_pred = (y_proba >= 0.5).astype(int)  # Konwertuje prawdopodobieństwa na predykcje binarne (próg 0.5)
    results['nn'] = {  # Zapisuje wyniki do słownika
        "metric_type": "accuracy",  # Typ metryki - accuracy
        "train_accuracy": float(train_acc),  # Accuracy na zbiorze treningowym
        "test_accuracy": float(test_acc),  # Accuracy na zbiorze testowym
        "train_loss": float(train_loss),  # Strata na zbiorze treningowym
        "test_loss": float(test_loss),  # Strata na zbiorze testowym
        "f1": float(f1_score(y_test_r, y_pred)),  # Wskaźnik F1 (harmonic mean precyzji i recall)
        "roc_auc": float(roc_auc_score(y_test_r, y_proba)),  # Pole pod krzywą ROC
    }
    model.save(os.path.join(MODELS_DIR, "nn_wine.keras"))  # Zapisuje model do pliku Keras
    save_json(results['nn'], os.path.join(RESULTS_DIR, "nn_wine_results.json"))  # Zapisuje wyniki do pliku JSON

    # --- SVM ---
    print("  - SVM")  # Informuje o treningu SVM
    svm = make_pipeline(StandardScaler(), SVC(probability=True, random_state=SEED))  # Tworzy pipeline: standaryzacja + SVM z obliczaniem prawdopodobieństw
    svm.fit(X_train, y_train_r)  # Trenuje SVM na danych treningowych
    y_proba = svm.predict_proba(X_test)[:, 1]  # Przewiduje prawdopodobieństwa klasy pozytywnej dla zbioru testowego
    y_pred = (y_proba >= 0.5).astype(int)  # Konwertuje prawdopodobieństwa na predykcje binarne (próg 0.5)
    results['svm'] = {  # Zapisuje wyniki do słownika
        "metric_type": "accuracy",  # Typ metryki - accuracy
        "train_accuracy": float(svm.score(X_train, y_train_r)),  # Accuracy na zbiorze treningowym
        "test_accuracy": float(accuracy_score(y_test_r, y_pred)),  # Accuracy na zbiorze testowym
        "train_loss": float(log_loss(y_train_r, svm.predict_proba(X_train)[:, 1])),  # Log loss na zbiorze treningowym
        "test_loss": float(log_loss(y_test_r, y_proba)),  # Log loss na zbiorze testowym
        "f1": float(f1_score(y_test_r, y_pred)),  # Wskaźnik F1 (harmonic mean precyzji i recall)
        "roc_auc": float(roc_auc_score(y_test_r, y_proba)),  # Pole pod krzywą ROC
    }
    pickle.dump(svm, open(os.path.join(MODELS_DIR, "svm_wine.pkl"), "wb"))  # Zapisuje pipeline SVM do pliku pickle
    save_json(results['svm'], os.path.join(RESULTS_DIR, "svm_wine_results.json"))  # Zapisuje wyniki do pliku JSON

    # --- Random Forest ---
    print("  - Random Forest")
    rf = RandomForestClassifier(class_weight="balanced", random_state=SEED, n_jobs=-1)
    rf.fit(X_train, y_train_r)  # Trenuje Random Forest na danych treningowych
    y_proba = rf.predict_proba(X_test)[:, 1]  # Przewiduje prawdopodobieństwa klasy pozytywnej dla zbioru testowego
    y_pred = (y_proba >= 0.5).astype(int)  # Konwertuje prawdopodobieństwa na predykcje binarne (próg 0.5)
    results['rf'] = {  # Zapisuje wyniki do słownika
        "metric_type": "accuracy",  # Typ metryki - accuracy
        "train_accuracy": float(rf.score(X_train, y_train_r)),  # Accuracy na zbiorze treningowym
        "test_accuracy": float(accuracy_score(y_test_r, y_pred)),  # Accuracy na zbiorze testowym
        "train_loss": float(log_loss(y_train_r, rf.predict_proba(X_train)[:, 1])),  # Log loss na zbiorze treningowym
        "test_loss": float(log_loss(y_test_r, y_proba)),  # Log loss na zbiorze testowym
        "f1": float(f1_score(y_test_r, y_pred)),  # Wskaźnik F1 (harmonic mean precyzji i recall)
        "roc_auc": float(roc_auc_score(y_test_r, y_proba)),  # Pole pod krzywą ROC
    }
    pickle.dump(rf, open(os.path.join(MODELS_DIR, "rf_wine.pkl"), "wb"))  # Zapisuje model Random Forest do pliku pickle
    save_json(results['rf'], os.path.join(RESULTS_DIR, "rf_wine_results.json"))  # Zapisuje wyniki do pliku JSON

    print("  ✅ Zakończono Wine")  # Informuje o zakończeniu treningu dla Wine Quality
    return results  # Zwraca słownik z wynikami wszystkich modeli


# ------------------------------------------------------
# MODELE DLA CONCRETE STRENGTH (Regresja)
# ------------------------------------------------------
def train_models_for_concrete():  # Główna funkcja trenująca wszystkie modele dla Concrete Strength
    print("\n🏗️ TRENING MODELI DLA CONCRETE STRENGTH")  # Informuje o rozpoczęciu treningu
    X_train, y_train, X_test, y_test = load_concrete_data()  # Ładuje dane betonu (już znormalizowane)

    results = {}  # Słownik do przechowywania wyników wszystkich modeli

    # --- Neural Network ---
    print("  - Neural Network")  # Informuje o treningu NN
    scaler = StandardScaler().fit(X_train)  # Tworzy i dopasowuje scaler do danych treningowych
    X_train_s, X_test_s = scaler.transform(X_train), scaler.transform(X_test)  # Standaryzuje dane (mean=0, std=1)
    pickle.dump(scaler, open(os.path.join(MODELS_DIR, "scaler_nn_concrete.pkl"), "wb"))  # Zapisuje scaler do pliku

    model = tf.keras.Sequential([  # Tworzy sekwencyjny model sieci neuronowej
        tf.keras.layers.Input(shape=(X_train_s.shape[1],)),  # Warstwa wejściowa o rozmiarze równym liczbie cech
        tf.keras.layers.Dense(64, activation="relu"),  # Pierwsza warstwa gęsta z 64 neuronami i aktywacją ReLU
        tf.keras.layers.Dropout(0.2),  # Dropout 20% aby zapobiec przeuczeniu
        tf.keras.layers.Dense(32, activation="relu"),  # Druga warstwa gęsta z 32 neuronami i aktywacją ReLU
        tf.keras.layers.Dropout(0.2),  # Dropout 20% aby zapobiec przeuczeniu
        tf.keras.layers.Dense(1)  # Warstwa wyjściowa z 1 neuronem bez aktywacji (regresja)
    ])
    model.compile(optimizer="adam", loss="mae")  # Kompiluje model z optymalizatorem Adam i funkcją straty MAE (Mean Absolute Error)
    model.fit(X_train_s, y_train, validation_split=0.2, epochs=100, batch_size=32, verbose=0)  # Trenuje model przez 100 epok z walidacją 20%
    train_mae = model.evaluate(X_train_s, y_train, verbose=0)  # Oblicza MAE na zbiorze treningowym
    test_mae = model.evaluate(X_test_s, y_test, verbose=0)  # Oblicza MAE na zbiorze testowym
    results['nn'] = {  # Zapisuje wyniki do słownika
        "metric_type": "mae",  # Typ metryki - MAE (Mean Absolute Error)
        "train_mae": float(train_mae),  # MAE na zbiorze treningowym
        "test_mae": float(test_mae)  # MAE na zbiorze testowym
    }
    model.save(os.path.join(MODELS_DIR, "nn_concrete.keras"))  # Zapisuje model do pliku Keras
    save_json(results['nn'], os.path.join(RESULTS_DIR, "nn_concrete_results.json"))  # Zapisuje wyniki do pliku JSON

    # --- SVM (SVR) ---
    print("  - SVM (SVR)")  # Informuje o treningu SVR
    svr = make_pipeline(StandardScaler(), SVR(kernel="rbf", C=100, gamma="scale"))  # Tworzy pipeline: standaryzacja + SVR z jądrem RBF, parametrem C=100 i automatycznym gamma
    svr.fit(X_train, y_train)  # Trenuje SVR na danych treningowych
    train_mae = mean_absolute_error(y_train, svr.predict(X_train))  # Oblicza MAE na zbiorze treningowym
    test_mae = mean_absolute_error(y_test, svr.predict(X_test))  # Oblicza MAE na zbiorze testowym
    results['svm'] = {  # Zapisuje wyniki do słownika
        "metric_type": "mae",  # Typ metryki - MAE
        "train_mae": float(train_mae),  # MAE na zbiorze treningowym
        "test_mae": float(test_mae)  # MAE na zbiorze testowym
    }
    pickle.dump(svr, open(os.path.join(MODELS_DIR, "svm_concrete.pkl"), "wb"))  # Zapisuje pipeline SVR do pliku pickle
    save_json(results['svm'], os.path.join(RESULTS_DIR, "svm_concrete_results.json"))  # Zapisuje wyniki do pliku JSON

    # --- Random Forest (Regressor) ---
    print("  - Random Forest")  # Informuje o treningu Random Forest
    rf = RandomForestRegressor(random_state=SEED, n_jobs=-1)  # Tworzy Random Forest Regressor z ziarnem losowym i równoległym przetwarzaniem
    rf.fit(X_train, y_train)  # Trenuje Random Forest na danych treningowych
    train_mae = mean_absolute_error(y_train, rf.predict(X_train))  # Oblicza MAE na zbiorze treningowym
    test_mae = mean_absolute_error(y_test, rf.predict(X_test))  # Oblicza MAE na zbiorze testowym
    results['rf'] = {  # Zapisuje wyniki do słownika
        "metric_type": "mae",  # Typ metryki - MAE
        "train_mae": float(train_mae),  # MAE na zbiorze treningowym
        "test_mae": float(test_mae)  # MAE na zbiorze testowym
    }
    pickle.dump(rf, open(os.path.join(MODELS_DIR, "rf_concrete.pkl"), "wb"))  # Zapisuje model Random Forest do pliku pickle
    save_json(results['rf'], os.path.join(RESULTS_DIR, "rf_concrete_results.json"))  # Zapisuje wyniki do pliku JSON

    print("  ✅ Zakończono Concrete")  # Informuje o zakończeniu treningu dla Concrete Strength
    return results  # Zwraca słownik z wynikami wszystkich modeli


# ------------------------------------------------------
# GŁÓWNY BLOK
# ------------------------------------------------------
if __name__ == "__main__":  # Wykonuje się tylko gdy plik uruchomiony bezpośrednio (nie importowany)
    print("TRENING KLASYCZNYCH MODELI ML")  # Wypisuje nagłówek
    print("=================================")  # Wypisuje separator

    try:  # Próbuje wytrenować modele dla wine
        wine_results = train_models_for_wine()  # Trenuje wszystkie modele dla Wine Quality
    except Exception as e:  # Łapie wszelkie błędy
        print(f"Błąd podczas trenowania Wine: {e}")  # Informuje o błędzie

    try:  # Próbuje wytrenować modele dla concrete
        concrete_results = train_models_for_concrete()  # Trenuje wszystkie modele dla Concrete Strength
    except Exception as e:  # Łapie wszelkie błędy
        print(f"Błąd podczas trenowania Concrete: {e}")  # Informuje o błędzie

    print("\nWszystkie modele zostały wytrenowane i zapisane w 'results/' oraz 'models/'.")  # Informuje o pomyślnym zakończeniu