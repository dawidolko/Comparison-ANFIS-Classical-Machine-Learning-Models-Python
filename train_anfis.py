import numpy as np  # Biblioteka do operacji na tablicach numerycznych
import tensorflow as tf  # Framework do budowy i treningu sieci neuronowych
from sklearn.model_selection import StratifiedKFold, KFold  # Narzędzia do cross-validation (stratyfikowany i zwykły)
from sklearn.metrics import r2_score, confusion_matrix, classification_report  # Metryki do ewaluacji modeli
from anfis import ANFISModel  # Importuje klasę głównego modelu ANFIS z pliku anfis.py
import matplotlib.pyplot as plt  # Biblioteka do tworzenia wykresów
import json  # Biblioteka do zapisu i odczytu plików JSON
import os  # Biblioteka do operacji na systemie plików
import argparse  # Biblioteka do parsowania argumentów wiersza poleceń
import seaborn as sns  # Biblioteka do zaawansowanych wizualizacji statystycznych

np.random.seed(42)  # Ustawia ziarno generatora losowego NumPy dla reprodukowalnych wyników
tf.random.set_seed(42)  # Ustawia ziarno generatora losowego TensorFlow dla reprodukowalnych wyników


# -------------------------------------------------------------
# POMOCNICZE
# -------------------------------------------------------------
def _load_dataset(dataset: str):
    """
    Ładuje znormalizowane dane treningowe i testowe dla określonego zestawu.
    
    Args:
        dataset: nazwa zestawu ('concrete', 'all', 'red', 'white')
        
    Returns:
        Tuple (X_train, X_test, y_train, y_test) jako tablice numpy
        
    Raises:
        ValueError: jeśli dataset nie istnieje
    """
    base = "data"  # Definiuje bazową ścieżkę do katalogu z danymi
    paths = {  # Słownik mapujący nazwy zestawów danych na ścieżki do plików .npy
        "concrete": (f"{base}/concrete-strength/X_train.npy", f"{base}/concrete-strength/X_test.npy",
                     f"{base}/concrete-strength/y_train.npy", f"{base}/concrete-strength/y_test.npy"),  # Ścieżki dla danych betonu
        "all": (f"{base}/X_train.npy", f"{base}/X_test.npy",
                f"{base}/y_train.npy", f"{base}/y_test.npy"),  # Ścieżki dla połączonych danych wina (czerwone+białe)
        "red": (f"{base}/X_train_red.npy", f"{base}/X_test_red.npy",
                f"{base}/y_train_red.npy", f"{base}/y_test_red.npy"),  # Ścieżki dla danych czerwonego wina
        "white": (f"{base}/X_train_white.npy", f"{base}/X_test_white.npy",
                  f"{base}/y_train_white.npy", f"{base}/y_test_white.npy")  # Ścieżki dla danych białego wina
    }

    if dataset not in paths:  # Sprawdza czy podana nazwa zestawu istnieje w słowniku
        raise ValueError(f"Nieznany dataset: {dataset}")  # Rzuca wyjątek jeśli nazwa jest nieprawidłowa
    Xtr, Xte, ytr, yte = [np.load(p) for p in paths[dataset]]  # Ładuje wszystkie 4 pliki .npy dla wybranego zestawu
    return Xtr, Xte, ytr, yte  # Zwraca dane treningowe i testowe (cechy i etykiety)


def _rules_count(n_features: int, n_memb: int) -> int:
    """
    Oblicza całkowitą liczbę reguł rozmytych w modelu ANFIS.
    
    Liczba reguł = n_memb^n_features (wszystkie kombinacje funkcji przynależności)
    
    Args:
        n_features: liczba cech wejściowych
        n_memb: liczba funkcji przynależności na każdą cechę
        
    Returns:
        Całkowita liczba reguł
    """
    return int(n_memb ** n_features)  # Zwraca liczbę reguł: n_memb podniesione do potęgi n_features (wszystkie kombinacje)


# -------------------------------------------------------------
# GŁÓWNY TRENING
# -------------------------------------------------------------
def train_anfis_model(n_memb=2, epochs=20, batch_size=32, dataset="all"):
    """
    Trenuje pojedynczy model ANFIS dla określonego zestawu danych.
    
    Proces treningu:
    1. Ładuje i przygotowuje dane
    2. Tworzy model ANFIS z odpowiednią aktywacją (linear dla regresji, sigmoid dla klasyfikacji)
    3. Kompiluje z odpowiednią funkcją straty i metryką
    4. Trenuje z ModelCheckpoint i EarlyStopping
    5. Generuje wykresy treningu i dopasowania
    6. Ekstrahuje i zapisuje reguły
    7. Zapisuje wyniki do JSON
    
    Args:
        n_memb: liczba funkcji przynależności (2 lub 3)
        epochs: maksymalna liczba epok treningu
        batch_size: rozmiar batcha
        dataset: nazwa zestawu ('concrete', 'all', 'red', 'white')
        
    Returns:
        Tuple (test_metric, model) - metryka testowa i wytrenowany model
    """
    print(f"\n{'='*70}")  # Wypisuje separator na początku treningu
    print(f"TRENING ANFIS: dataset={dataset}, n_memb={n_memb}")  # Wypisuje informacje o konfiguracji treningu
    print(f"{'='*70}\n")  # Wypisuje separator kończący nagłówek

    X_train, X_test, y_train, y_test = _load_dataset(dataset)  # Ładuje dane treningowe i testowe dla wybranego zestawu
    n_features = X_train.shape[1]  # Pobiera liczbę cech wejściowych (kolumn) z danych treningowych
    n_rules = _rules_count(n_features, n_memb)  # Oblicza całkowitą liczbę reguł fuzzy: n_memb^n_features

    # dopasowanie batch_size (by uniknąć warningów TF)
    n_train = (len(X_train) // batch_size) * batch_size  # Oblicza liczbę przykładów treningowych podzielnych przez batch_size
    n_test = (len(X_test) // batch_size) * batch_size  # Oblicza liczbę przykładów testowych podzielnych przez batch_size
    X_train, y_train = X_train[:n_train], y_train[:n_train]  # Przycina zbiór treningowy do wielokrotności batch_size
    X_test, y_test = X_test[:n_test], y_test[:n_test]  # Przycina zbiór testowy do wielokrotności batch_size

    print(f"Features: {n_features}, MF: {n_memb}, Rules: {n_rules}")  # Wypisuje parametry modelu
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")  # Wypisuje rozmiary zbiorów po przycięciu

    # Dla regresji (concrete) ustawiono regression=True
    is_regression = (dataset == "concrete")  # Sprawdza czy to zadanie regresji (True dla betonu, False dla wina)
    anfis_model = ANFISModel(n_input=n_features, n_memb=n_memb, batch_size=batch_size, regression=is_regression)  # Tworzy model ANFIS z odpowiednimi parametrami

    # --------------------- konfiguracja metryk ---------------------
    if dataset == "concrete":  # Sprawdza czy to zadanie regresji (beton)
        anfis_model.model.compile(  # Kompiluje model z ustawieniami dla regresji
            optimizer=tf.keras.optimizers.Nadam(learning_rate=0.001),  # Optymalizator Nadam (Adam z Nesterov momentum) z learning rate 0.001
            loss="mean_squared_error",  # Funkcja straty MSE (błąd średniokwadratowy) dla regresji
            metrics=["mae"]  # Metryka MAE (mean absolute error) do monitorowania
        )
        monitor_metric = "val_loss"  # Monitoruje validation loss podczas treningu
        monitor_mode = "min"  # Tryb minimalizacji (nizsza wartość = lepiej)
        metric_name = "mae"  # Nazwa metryki do zapisania w historii
    else:  # Dla zadania klasyfikacji (wino)
        anfis_model.model.compile(  # Kompiluje model z ustawieniami dla klasyfikacji binarnej
            optimizer=tf.keras.optimizers.Nadam(learning_rate=0.001),  # Optymalizator Nadam z learning rate 0.001
            loss="binary_crossentropy",  # Funkcja straty entropia krzyżowa dla klasyfikacji binarnej
            metrics=["accuracy"]  # Metryka accuracy (dokładność) do monitorowania
        )
        monitor_metric = "val_accuracy"  # Monitoruje validation accuracy podczas treningu
        monitor_mode = "max"  # Tryb maksymalizacji (wyższa wartość = lepiej)
        metric_name = "accuracy"  # Nazwa metryki do zapisania w historii

    # ---------------------- callbacki ----------------------
    checkpoint_path = f"models/anfis_{dataset}_best_{n_memb}memb.weights.h5"  # Ścieżka do zapisu najlepszych wag modelu
    callbacks = [  # Lista callbacków do użycia podczas treningu
        tf.keras.callbacks.ModelCheckpoint(  # Callback zapisujący najlepszy model
            checkpoint_path, monitor=monitor_metric,  # Zapisuje wagi do pliku monitorując wybraną metrykę
            save_best_only=True, save_weights_only=True,  # Zapisuje tylko najlepsze wagi (nie całą architekturę)
            mode=monitor_mode, verbose=0  # Używa odpowiedniego trybu (min/max) bez verbose output
        ),
        tf.keras.callbacks.EarlyStopping(  # Callback zatrzymujący trening przy braku poprawy
            monitor=monitor_metric, patience=10,  # Monitoruje metrykę i czeka 10 epok bez poprawy
            restore_best_weights=True, verbose=0  # Przywraca najlepsze wagi po zatrzymaniu
        )
    ]

    # ---------------------- PRZED TRENINGIEM - zapisz początkowe MF ----------------------
    anfis_model.update_weights()
    centers_before, sigmas_before = anfis_model.get_membership_functions()
    np.save(f"results/mf_centers_before_{dataset}_{n_memb}memb.npy", centers_before)
    np.save(f"results/mf_sigmas_before_{dataset}_{n_memb}memb.npy", sigmas_before)
    print(f"✓ Zapisano MF PRZED treningiem: results/mf_*_before_{dataset}_{n_memb}memb.npy")

    print("Rozpoczynam trening...\n")  # Informuje użytkownika o rozpoczęciu treningu
    history = anfis_model.model.fit(  # Trenuje model i zapisuje historię treningu
        X_train, y_train,  # Dane treningowe (cechy i etykiety)
        validation_data=(X_test, y_test),  # Dane walidacyjne do ewaluacji po każdej epoce
        epochs=epochs, batch_size=batch_size,  # Liczba epok i rozmiar batcha
        callbacks=callbacks, verbose=1  # Używa zdefiniowanych callbacków i wypisuje postęp
    )

    # ---------------------- PO TRENINGU - zapisz końcowe MF ----------------------
    anfis_model.update_weights()
    centers_after, sigmas_after = anfis_model.get_membership_functions()
    np.save(f"results/mf_centers_after_{dataset}_{n_memb}memb.npy", centers_after)
    np.save(f"results/mf_sigmas_after_{dataset}_{n_memb}memb.npy", sigmas_after)
    print(f"✓ Zapisano MF PO treningu: results/mf_*_after_{dataset}_{n_memb}memb.npy")

    # ---------------------- ewaluacja ----------------------
    anfis_model.model.load_weights(checkpoint_path)  # Ładuje najlepsze wagi zapisane podczas treningu
    train_loss, train_metric = anfis_model.model.evaluate(X_train, y_train, verbose=0)  # Ewaluuje model na zbiorze treningowym
    test_loss, test_metric = anfis_model.model.evaluate(X_test, y_test, verbose=0)  # Ewaluuje model na zbiorze testowym

    # ---------------------- wyniki ----------------------
    results = {  # Słownik przechowujący wszystkie wyniki treningu
        "dataset": dataset,  # Nazwa zestawu danych
        "n_memb": n_memb,  # Liczba funkcji przynależności
        "n_features": int(n_features),  # Liczba cech wejściowych
        "n_rules": int(n_rules),  # Całkowita liczba reguł
        "train_loss": float(train_loss),  # Loss na zbiorze treningowym
        "test_loss": float(test_loss),  # Loss na zbiorze testowym
        "metric_type": "mae" if dataset == "concrete" else "accuracy",  # Typ metryki używany
        ("train_mae" if dataset == "concrete" else "train_accuracy"): float(train_metric),  # Metryka treningowa
        ("test_mae" if dataset == "concrete" else "test_accuracy"): float(test_metric),  # Metryka testowa
        "history": {  # Historia treningu (wartości metryk po każdej epoce)
            metric_name: [float(x) for x in history.history.get(metric_name, [])],  # Metryka treningowa przez epoki
            f"val_{metric_name}": [float(x) for x in history.history.get(f"val_{metric_name}", [])],  # Metryka walidacyjna przez epoki
            "loss": [float(x) for x in history.history["loss"]],  # Loss treningowy przez epoki
            "val_loss": [float(x) for x in history.history["val_loss"]],  # Loss walidacyjny przez epoki
        },
    }

    os.makedirs("results", exist_ok=True)  # Tworzy katalog results jeśli nie istnieje
    out_json = f"results/anfis_{dataset}_{n_memb}memb_results.json"  # Ścieżka do pliku JSON z wynikami
    with open(out_json, "w") as f:  # Otwiera plik JSON do zapisu
        json.dump(results, f, indent=4)  # Zapisuje słownik wyników do pliku z wczęciami (indent=4)
    print(f"✓ Zapisano {out_json}")  # Informuje o pomyślnym zapisie pliku

    # ---------------------- wykresy ----------------------
    plot_training_history(history, n_memb, dataset)  # Generuje wykres historii treningu (accuracy/MAE i loss)
    plot_fit_on_train(anfis_model, X_train, y_train, n_memb, dataset)  # Generuje wykres dopasowania predykcji do wartości rzeczywistych

    # ---------------------- reguły ----------------------
    try:  # Próbuje wyekstrahować i zapisać reguły fuzzy
        extract_and_save_rules(anfis_model, n_memb, dataset, X_train)  # Ekstrahuje reguły z modelu i zapisuje do JSON
    except Exception as e:  # Łapie wszelkie wyjątki podczas ekstrakcji
        print(f"⚠️ Błąd przy ekstrakcji reguł: {e}")  # Wypisuje komunikat o błędzie

    return anfis_model, history, results  # Zwraca wytrenowany model, historię i wyniki


# -------------------------------------------------------------
# WIZUALIZACJE
# -------------------------------------------------------------
def plot_training_history(history, n_memb, dataset):
    """
    Wygeneruj wizualizację historii treningu modelu ANFIS — bez tabelki.
    
    Dla KLASYFIKACJI (wine): krzywe Accuracy i Loss.
    Dla REGRESJI (concrete): krzywe MAE i Loss.
    
    Args:
        history: obiekt History z Keras (lub podobny słownik z metrykami)
        n_memb: liczba funkcji przynależności
        dataset: nazwa zbioru danych
    """
    # Ustal typ zadania i klucze metryk
    if "accuracy" in history.history:  # Sprawdza czy historia zawiera metrykę accuracy (klasyfikacja)
        mkey, vkey, label = "accuracy", "val_accuracy", "Accuracy"  # Ustawia klucze dla metryki accuracy
        is_classification = True  # Oznacza że to zadanie klasyfikacji
    elif "mae" in history.history:  # Sprawdza czy historia zawiera metrykę MAE (regresja)
        mkey, vkey, label = "mae", "val_mae", "MAE"  # Ustawia klucze dla metryki MAE
        is_classification = False  # Oznacza że to zadanie regresji
    else:  # Jeśli brak standardowych metryk
        # Jeśli nie ma ani accuracy, ani mae — użycie loss jako metryki
        mkey = None  # Brak metryki do wyświetlenia
        is_classification = False  # Domyślnie traktuj jako regresję

    # Tworzenie wykresów — 1 wiersz, 2 kolumny
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(14, 6))  # Tworzy figurę z dwoma wykresami obok siebie
    fig.suptitle(f"Historia treningu: {dataset.upper()} ({n_memb} funkcji przynależności)",  # Ustawia główny tytuł
                 fontsize=14, fontweight='bold', y=0.98)  # Formatowanie tytułu: rozmiar 14, pogrubiony, pozycja y=0.98

    # --- LEWY WYKRES: METRYKA (Accuracy / MAE) ---
    if mkey:  # Sprawdza czy jest metryka do wyświetlenia
        epochs = np.arange(1, len(history.history[mkey]) + 1)  # Tworzy tablicę numerów epok (1, 2, 3, ...)
        
        # Train i Validation
        ax0.plot(epochs, history.history[mkey], label="Train", lw=2.5, marker='o', markersize=4, color='steelblue')  # Rysuje krzywą treningową z okrągłymi markerami
        ax0.plot(epochs, history.history[vkey], label="Validation", lw=2.5, marker='s', markersize=4, color='coral')  # Rysuje krzywą walidacyjną z kwadratowymi markerami
        ax0.set_title(f"{label} Curve ({dataset}, {n_memb} MF)", fontsize=13, fontweight='bold')  # Ustawia tytuł wykresu
        ax0.set_xlabel("Epoch", fontsize=11, fontweight='bold')  # Etykieta osi X
        ax0.set_ylabel(label, fontsize=11, fontweight='bold')  # Etykieta osi Y (Accuracy lub MAE)
        ax0.legend(fontsize=10, loc='best')  # Dodaje legendę w najlepszej pozycji
        ax0.grid(True, alpha=0.3)  # Dodaje siatkę z przezroczystością 0.3
        
        # Zaznaczenie najlepszej epoki
        best_epoch_idx = np.argmax(history.history[vkey]) if is_classification else np.argmin(history.history[vkey])  # Znajduje indeks najlepszej epoki (max dla accuracy, min dla MAE)
        best_value = history.history[vkey][best_epoch_idx]  # Pobiera wartość metryki w najlepszej epoce
        best_epoch_num = best_epoch_idx + 1  # Konwertuje indeks na numer epoki (epoki numerowane od 1)
        
        ax0.scatter([best_epoch_num], [best_value], color='red', s=100, zorder=5,  # Rysuje czerwony punkt na najlepszej epoce
                    label=f'Best Epoch: {best_epoch_num}')  # Etykieta z numerem najlepszej epoki
        ax0.legend(fontsize=10, loc='best')  # Aktualizuje legendę z nowym punktem

    else:  # Jeśli brak metryki do wyświetlenia
        # Jeśli nie ma metryki (np. tylko loss), pusty wykres lub informacja
        ax0.text(0.5, 0.5, 'Brak metryki (accuracy/mae)', ha='center', va='center',  # Wyświetla tekst w środku wykresu
                 transform=ax0.transAxes, fontsize=12, color='gray')  # Używa współrzędnych transformowanych (0.5, 0.5 = środek)
        ax0.set_title(f"{dataset} - {n_memb}MF", fontsize=13, fontweight='bold')  # Ustawia tytuł wykresu
        ax0.set_xlabel("Epoch")  # Etykieta osi X
        ax0.set_ylabel("Metric")  # Etykieta osi Y
        ax0.grid(True, alpha=0.3)  # Dodaje siatkę

    # --- PRAWY WYKRES: LOSS ---
    epochs = np.arange(1, len(history.history["loss"]) + 1)  # Tworzy tablicę numerów epok dla funkcji straty
    ax1.plot(epochs, history.history["loss"], label="Train", lw=2.5, marker='o', markersize=4, color='darkgreen')  # Rysuje krzywą loss treningowego (ciemnozielona)
    ax1.plot(epochs, history.history["val_loss"], label="Validation", lw=2.5, marker='s', markersize=4, color='darkred')  # Rysuje krzywą loss walidacyjnego (ciemnoczerwona)
    ax1.set_title(f"Krzywa funkcji straty ({dataset}, {n_memb} MF)", fontsize=13, fontweight='bold')  # Tytuł wykresu
    ax1.set_xlabel("Epoch", fontsize=11, fontweight='bold')  # Etykieta osi X
    ax1.set_ylabel("Loss (MSE/BCE)", fontsize=11, fontweight='bold')  # Etykieta osi Y (MSE dla regresji, BCE dla klasyfikacji)
    ax1.legend(fontsize=10, loc='best')  # Dodaje legendę
    ax1.grid(True, alpha=0.3)  # Dodaje siatkę

    # Zaznaczenie najlepszej epoki dla loss
    best_loss_epoch_idx = np.argmin(history.history["val_loss"])  # Znajduje indeks epoki z najmniejszym validation loss
    best_loss_value = history.history["val_loss"][best_loss_epoch_idx]  # Pobiera wartość najniższego loss
    best_loss_epoch_num = best_loss_epoch_idx + 1  # Konwertuje indeks na numer epoki
    
    ax1.scatter([best_loss_epoch_num], [best_loss_value], color='red', s=100, zorder=5,  # Rysuje czerwony punkt na najlepszej epoce
                label=f'Best Loss Epoch: {best_loss_epoch_num}')  # Etykieta z numerem najlepszej epoki
    ax1.legend(fontsize=10, loc='best')  # Aktualizuje legendę

    plt.tight_layout()  # Automatycznie dopasowuje układ subplotów
    plt.savefig(f"results/anfis_{dataset}_{n_memb}memb_training.png", dpi=300, bbox_inches="tight")  # Zapisuje wykres do pliku PNG z wysoką rozdzielczością
    plt.close()  # Zamyka wykres aby zwolnić pamięć
    print(f"✓ Zapisano results/anfis_{dataset}_{n_memb}memb_training.png")  # Informuje o zapisie pliku


def plot_fit_on_train(model, X_train, y_train, n_memb, dataset):
    """
    Wizualizuje dopasowanie modelu na zbiorze treningowym.
    
    Dla regresji: scatter plot + histogram reszt + R²
    Dla klasyfikacji: macierz pomyłek + raport klasyfikacyjny
    
    Args:
        model: wytrenowany model ANFIS
        X_train: cechy treningowe
        y_train: etykiety treningowe
        n_memb: liczba funkcji przynależności
        dataset: nazwa zestawu danych
    """
    print(f"\n[DEBUG] plot_fit_on_train START - dataset={dataset}, n_memb={n_memb}")
    print(f"[DEBUG] y_train shape: {y_train.shape}, min: {y_train.min():.2f}, max: {y_train.max():.2f}")

    # Predykcje
    preds_raw = model(X_train)  # Wykonuje predykcje modelu na zbiorze treningowym
    if hasattr(preds_raw, 'numpy'):  # Sprawdza czy wynik jest tensorem TensorFlow
        preds = preds_raw.numpy().reshape(-1)  # Konwertuje tensor do numpy array i spłaszcza do 1D
    else:  # W przeciwnym razie
        preds = np.array(preds_raw).reshape(-1)  # Konwertuje do numpy array i spłaszcza do 1D

    print(f"[DEBUG] preds min: {preds.min():.4f}, max: {preds.max():.4f}, mean: {preds.mean():.4f}")  # Wypisuje statystyki predykcji do debugowania

    # --- KLASYFIKACJA ---
    if dataset in ["all", "red", "white"]:  # Sprawdza czy dataset to Wine Quality (klasyfikacja)
        print("[INFO] Klasyfikacja - generuję macierz pomyłek")  # Informuje o generowaniu macierzy pomyłek
        if len(np.unique(y_train)) <= 2:  # Sprawdza czy problem jest binarny
            preds_class = (preds > 0.5).astype(int)  # Konwertuje prawdopodobieństwa na klasy binarne (próg 0.5)
        else:  # W przeciwnym razie (klasyfikacja wieloklasowa)
            preds_class = np.round(preds).astype(int)  # Zaokrągla predykcje do najbliższej liczby całkowitej
            unique_y = np.unique(y_train)  # Pobiera unikalne klasy ze zbioru treningowego
            unique_p = np.unique(preds_class)  # Pobiera unikalne klasy z predykcji
            if not set(unique_p).issubset(set(unique_y)):  # Sprawdza czy predykcje zawierają klasy spoza zakresu
                print(f"[WARNING] Predykcje zawierają klasy poza zakresem: {unique_p} vs {unique_y}")  # Ostrzega o nieprawidłowych klasach
                preds_class = np.clip(preds_class, unique_y.min(), unique_y.max())  # Przyciąga predykcje do zakresu [min, max] klas

        # Macierz pomyłek
        cm = confusion_matrix(y_train, preds_class)  # Oblicza macierz pomyłek (confusion matrix)
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))  # Tworzy figurę z jednym subplotem
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,  # Rysuje heatmapę macierzy pomyłek z adnotacjami liczbowymi
                    xticklabels=np.unique(y_train), yticklabels=np.unique(y_train))  # Ustawia etykiety osi jako unikalne klasy
        ax.set_xlabel("Przewidziana klasa")  # Ustawia etykietę osi X
        ax.set_ylabel("Prawdziwa klasa")  # Ustawia etykietę osi Y
        ax.set_title(f"{dataset} - {n_memb}MF | Macierz pomyłek")  # Ustawia tytuł wykresu
        plt.tight_layout()  # Dopasowuje układ elementów
        plt.savefig(f"results/anfis_{dataset}_{n_memb}memb_confmat_train.png", dpi=300, bbox_inches="tight")  # Zapisuje wykres w wysokiej rozdzielczości
        plt.close()  # Zamyka figurę aby zwolnić pamięć

        # Raport klasyfikacyjny
        report = classification_report(y_train, preds_class, output_dict=True)  # Generuje raport klasyfikacyjny jako słownik
        print("\n[CLASSIFICATION REPORT]")  # Wypisuje nagłówek raportu
        print(classification_report(y_train, preds_class))  # Wypisuje raport klasyfikacyjny (precision, recall, f1-score)

        # Zapisanie raport do pliku
        with open(f"results/anfis_{dataset}_{n_memb}memb_class_report_train.txt", "w") as f:  # Otwiera plik do zapisu
            f.write(classification_report(y_train, preds_class))  # Zapisuje raport klasyfikacyjny do pliku

        print(f"✓ Zapisano macierz pomyłek i raport")  # Informuje o pomyślnym zapisie

    # --- REGRESJA ---
    else:  # Wykonuje się dla datasetów regresyjnych (concrete)
        print("[INFO] Regresja - generuję wykresy diagnostyczne")  # Informuje o generowaniu wykresów diagnostycznych
        
        # Obliczenie reszty
        residuals = y_train - preds  # Oblicza reszty (różnicę między rzeczywistymi a przewidywanymi wartościami)
        
        # R²
        r2 = r2_score(y_train, preds)  # Oblicza współczynnik determinacji R² (jakość dopasowania modelu)
        print(f"[INFO] R² = {r2:.4f}")  # Wypisuje współczynnik R²

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))  # Tworzy siatkę 2x2 subplotów
        fig.suptitle(f"{dataset} - {n_memb}MF | Diagnostyka modelu", fontsize=16)  # Ustawia główny tytuł całej figury

        # 1. Scatter: rzeczywiste vs predykcje
        ax = axes[0, 0]  # Wybiera pierwszy subplot
        ax.scatter(y_train, preds, s=20, alpha=0.6, color='blue')  # Rysuje wykres rozproszenia (scatter plot)
        min_val = min(y_train.min(), preds.min())  # Znajduje minimalną wartość spośród danych i predykcji
        max_val = max(y_train.max(), preds.max())  # Znajduje maksymalną wartość spośród danych i predykcji
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='y=x')  # Rysuje linię idealneg dopasowania (y=x)
        ax.set_xlabel("Rzeczywiste")  # Ustawia etykietę osi X
        ax.set_ylabel("Predykcja")  # Ustawia etykietę osi Y
        ax.set_title("Rzeczywiste vs Predykcja")  # Ustawia tytuł wykresu
        ax.legend()  # Wyświetla legendę
        ax.grid(True, alpha=0.3)  # Włącza siatkę z przezroczystością

        # 2. Histogram reszt
        ax = axes[0, 1]  # Wybiera drugi subplot
        ax.hist(residuals, bins=50, color='green', alpha=0.7, edgecolor='black')  # Rysuje histogram reszt z 50 binami
        ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Średnia = 0')  # Rysuje pionową linię na 0 (idealna reszta)
        ax.set_xlabel("Reszta (y - ŷ)")  # Ustawia etykietę osi X (różnica między rzeczywistym a przewidywanym)
        ax.set_ylabel("Liczba obserwacji")  # Ustawia etykietę osi Y
        ax.set_title(f"Histogram reszt\n(R²={r2:.3f})")  # Ustawia tytuł z współczynnikiem R²
        ax.legend()  # Wyświetla legendę
        ax.grid(True, alpha=0.3)  # Włącza siatkę z przezroczystością

        # 3. Reszty vs Predykcje (sprawdzenie heteroskedastyczności)
        ax = axes[1, 0]  # Wybiera trzeci subplot
        ax.scatter(preds, residuals, s=20, alpha=0.6, color='purple')  # Rysuje wykres rozproszenia reszt względem predykcji
        ax.axhline(0, color='red', linestyle='--', linewidth=2)  # Rysuje poziomą linię na 0 (idealna reszta)
        ax.set_xlabel("Predykcja")  # Ustawia etykietę osi X
        ax.set_ylabel("Reszta")  # Ustawia etykietę osi Y
        ax.set_title("Reszty vs Predykcja")  # Ustawia tytuł wykresu
        ax.grid(True, alpha=0.3)  # Włącza siatkę z przezroczystością

        # 4. QQ plot
        from scipy import stats  # Importuje moduł stats z scipy dla QQ plot
        ax = axes[1, 1]  # Wybiera czwarty subplot
        stats.probplot(residuals, dist="norm", plot=ax)  # Rysuje QQ plot porównujący rozkład reszt z rozkładem normalnym
        ax.set_title("QQ Plot reszt")  # Ustawia tytuł wykresu
        ax.grid(True, alpha=0.3)  # Włącza siatkę z przezroczystością

        plt.tight_layout()  # Dopasowuje układ elementów
        plt.savefig(f"results/anfis_{dataset}_{n_memb}memb_diag_train.png", dpi=300, bbox_inches="tight")  # Zapisuje wykres w wysokiej rozdzielczości
        plt.close()  # Zamyka figurę aby zwolnić pamięć

        print(f"✓ Zapisano wykresy diagnostyczne: results/anfis_{dataset}_{n_memb}memb_diag_train.png")  # Informuje o pomyślnym zapisie wykresów


# -------------------------------------------------------------
# EKSTRAKCJA REGUŁ
# -------------------------------------------------------------
def _rule_index_to_tuple(idx: int, n_features: int, n_memb: int):
    """
    Konwertuje płaski indeks reguły na wektor indeksów funkcji przynależności.
    
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


def extract_and_save_rules(model, n_memb: int, dataset: str, X_train):
    """
    Ekstrahuje reguły rozmyte z wytrenowanego modelu ANFIS i zapisuje do JSON.
    
    Jeśli liczba reguł > 4096, wybiera 100 najczęściej aktywowanych reguł
    na podstawie analizy danych treningowych.
    
    Zapisywane informacje:
    - Indeksy funkcji przynależności dla każdej reguły
    - Parametry konsekwentów (wagi i bias)
    - Centra i sigmy funkcji przynależności
    - Częstość aktywacji (dla dużych modeli)
    
    Args:
        model: wytrenowany model ANFIS
        n_memb: liczba funkcji przynależności
        dataset: nazwa zestawu danych
        X_train: dane treningowe (do analizy aktywacji)
    """
    model.update_weights()  # Aktualizuje wagi modelu aby mieć najnowsze parametry
    centers, sigmas = model.get_membership_functions()  # Pobiera centra i sigmy funkcji przynależności Gaussa
    bias, weights = model.bias, model.weights  # Pobiera bias i wagi konsekwentów TSK
    n_features = centers.shape[1]  # Liczy liczbę cech (kolumny w centers)
    n_rules = _rules_count(n_features, n_memb)  # Oblicza całkowitą liczbę reguł (n_memb^n_features)

    if n_rules <= 4096:  # Sprawdza czy liczba reguł jest mała (poniżej progu)
        top_rules_idx = list(range(n_rules))  # Wybiera wszystkie reguły
        approx_freq = None  # Brak częstości aktywacji (nie obliczane dla małych modeli)
    else:  # Wykonuje się gdy liczba reguł > 4096
        mvals = []  # Lista do przechowywania wartości funkcji przynależności dla każdej cechy
        for j in range(n_features):  # Iteruje przez każdą cechę
            xj = X_train[:, j][:, None]  # Pobiera j-tą kolumnę danych i przekształca do kształtu (n_samples, 1)
            cj = centers[:, j][None, :]  # Pobiera centra dla j-tej cechy i przekształca do kształtu (1, n_memb)
            sj = sigmas[:, j][None, :]  # Pobiera sigmy dla j-tej cechy i przekształca do kształtu (1, n_memb)
            mu = np.exp(-((xj - cj) ** 2) / (sj ** 2 + 1e-8))  # Oblicza wartości funkcji przynależności Gaussa dla wszystkich próbek i wszystkich MF
            mvals.append(mu)  # Dodaje obliczone wartości do listy
        best_idx = np.stack([np.argmax(mu, axis=1) for mu in mvals], axis=1)  # Dla każdej próbki i cechy znajduje indeks MF o najwyższej wartości
        pow_m = (n_memb ** np.arange(n_features)[::-1]).astype(int)  # Oblicza potęgi n_memb dla konwersji indeksów na indeks reguły
        rule_idx = (best_idx * pow_m).sum(axis=1)  # Konwertuje kombinację indeksów MF na pojedynczy indeks reguły
        vals, counts = np.unique(rule_idx, return_counts=True)  # Liczy częstość aktywacji każdej reguły
        order = np.argsort(counts)[::-1]  # Sortuje reguły według częstości (malejąco)
        vals, counts = vals[order], counts[order]  # Stosuje sortowanie do wartości i liczników
        top_rules_idx = vals[:100].tolist()  # Wybiera 100 najczęściej aktywowanych reguł
        approx_freq = {int(v): int(c) for v, c in zip(vals[:100], counts[:100])}  # Tworzy słownik z częstościami top 100 reguł

    rules = []  # Lista do przechowywania reguł
    for ridx in top_rules_idx:  # Iteruje przez wybrane indeksy reguł
        combo = _rule_index_to_tuple(ridx, n_features, n_memb)  # Konwertuje płaski indeks reguły na wektor indeksów MF
        rules.append({  # Dodaje słownik reguły do listy
            "rule_index": int(ridx),  # Indeks reguły
            "membership_indices": combo,  # Lista indeksów MF dla każdej cechy
            "consequent": {  # Konsekwent TSK (parametry liniowe)
                "weights": [float(w) for w in weights[:, ridx]],  # Wagi dla każdej cechy
                "bias": float(bias[0, ridx])  # Bias (wyraz wolny)
            }
        })

    payload = {  # Tworzy słownik z pełnymi informacjami o regule
        "dataset": dataset,  # Nazwa zestawu danych
        "n_features": n_features,  # Liczba cech
        "n_memb": n_memb,  # Liczba funkcji przynależności na cechę
        "n_rules_total": n_rules,  # Całkowita liczba reguł
        "rules_listed": len(rules),  # Liczba zapisanych reguł
        "approx_top_rule_frequency": approx_freq,  # Częstość aktywacji top reguł (jeśli obliczone)
        "membership_centers": centers.tolist(),  # Centra funkcji przynależności
        "membership_sigmas": sigmas.tolist(),  # Sigmy funkcji przynależności
        "rules": rules  # Lista reguł
    }

    out_json = f"results/anfis_{dataset}_{n_memb}memb_rules.json"  # Tworzy ścieżkę do pliku wyjściowego
    with open(out_json, "w") as f:  # Otwiera plik do zapisu
        json.dump(payload, f, indent=2)  # Zapisuje słownik jako JSON z wcięciami
    print(f"✓ Zapisano {out_json}")  # Informuje o pomyślnym zapisie


# -------------------------------------------------------------
# CROSS-WALIDACJA
# -------------------------------------------------------------
def cross_validate_anfis(n_memb=2, batch_size=32, dataset="all", n_splits=5, epochs=10):
    """
    Przeprowadza k-krotną walidację krzyżową modelu ANFIS.
    
    Używa:
    - KFold dla regresji (concrete)
    - StratifiedKFold dla klasyfikacji (wine)
    
    Każdy fold jest trenowany niezależnie, a wyniki są uśredniane.
    Zapisuje szczegółowe wyniki każdego folda oraz średnie do JSON.
    
    Args:
        n_memb: liczba funkcji przynależności
        batch_size: rozmiar batcha
        dataset: nazwa zestawu danych
        n_splits: liczba foldów (domyślnie 5)
        epochs: liczba epok treningu per fold
        
    Returns:
        Dict z wynikami cross-validation
    """
    print(f"\n{'='*70}")  # Wypisuje separator wizualny
    print(f"CROSS-VALIDATION: {dataset}, {n_memb} MF, {n_splits}-fold")  # Wypisuje nagłówek z parametrami CV
    print(f"{'='*70}\n")  # Wypisuje separator wizualny

    X_train, X_test, y_train, y_test = _load_dataset(dataset)  # Ładuje dane treningowe i testowe
    X = np.concatenate([X_train, X_test])  # Łączy dane treningowe i testowe w jeden zbiór
    y = np.concatenate([y_train, y_test])  # Łączy etykiety treningowe i testowe w jeden wektor

    # Tworzy KFold dla regresji (concrete) lub StratifiedKFold dla klasyfikacji (wine) aby zachować proporcje klas
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=42) if dataset == "concrete" else StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    fold_metrics = []  # Lista do przechowywania metryk każdego folda
    is_regression = (dataset == "concrete")  # Sprawdza czy problem jest regresyjny
    for fold, (tr_idx, va_idx) in enumerate(splitter.split(X, y if dataset != "concrete" else None), 1):  # Iteruje przez każdy fold CV (od 1)
        print(f"Fold {fold}/{n_splits}")  # Informuje o aktualnym foldzie
        Xt, Xv, yt, yv = X[tr_idx], X[va_idx], y[tr_idx], y[va_idx]  # Dzieli dane na treningowe i walidacyjne dla danego folda
        model = ANFISModel(n_input=X.shape[1], n_memb=n_memb, batch_size=batch_size, regression=is_regression)  # Tworzy nowy model ANFIS dla folda
        if dataset == "concrete":  # Jeśli regresja
            model.model.compile(optimizer=tf.keras.optimizers.Nadam(0.001),  # Kompiluje z optymalizatorem Nadam
                                loss="mean_squared_error", metrics=["mae"])  # Funkcja straty MSE i metryka MAE
            metric_key = "val_mae"  # Klucz metryki do zapisu
        else:  # Jeśli klasyfikacja
            model.model.compile(optimizer=tf.keras.optimizers.Nadam(0.001),  # Kompiluje z optymalizatorem Nadam
                                loss="binary_crossentropy", metrics=["accuracy"])  # Funkcja straty binary crossentropy i metryka accuracy
            metric_key = "val_accuracy"  # Klucz metryki do zapisu
        model.model.fit(Xt, yt, epochs=epochs, batch_size=batch_size, verbose=0)  # Trenuje model na danych treningowych folda
        loss, metric_val = model.model.evaluate(Xv, yv, verbose=0)  # Ewaluuje model na danych walidacyjnych folda
        fold_metrics.append({"fold": fold, metric_key: float(metric_val), "val_loss": float(loss)})  # Zapisuje wyniki folda do listy

    metric_vals = [list(f.values())[1] for f in fold_metrics]  # Ekstrahuje wartości metryki z każdego folda (drugi klucz w słowniku)
    mean_val, std_val = np.mean(metric_vals), np.std(metric_vals)  # Oblicza średnią i odchylenie standardowe metryki
    summary = {  # Tworzy słownik podsumowujący wyniki CV
        "dataset": dataset,  # Nazwa zestawu danych
        "n_memb": n_memb,  # Liczba funkcji przynależności
        "n_splits": n_splits,  # Liczba foldów
        "folds": fold_metrics,  # Lista wyników każdego folda
        ("mean_mae" if dataset == "concrete" else "mean_accuracy"): float(mean_val),  # Średnia metryka (MAE lub accuracy)
        ("std_mae" if dataset == "concrete" else "std_accuracy"): float(std_val),  # Odchylenie standardowe metryki
        "metric_type": "mae" if dataset == "concrete" else "accuracy",  # Typ metryki
    }

    out_json = f"results/anfis_{dataset}_{n_memb}memb_cv.json"  # Tworzy ścieżkę do pliku wyjściowego
    with open(out_json, "w") as f:  # Otwiera plik do zapisu
        json.dump(summary, f, indent=2)  # Zapisuje podsumowanie jako JSON z wcięciami
    print(f"✓ Zapisano {out_json}")  # Informuje o pomyślnym zapisie
    return summary  # Zwraca słownik z wynikami CV


# -------------------------------------------------------------
# MAIN
# -------------------------------------------------------------
def _parse_args():  # Funkcja parsująca argumenty wiersza poleceń
    p = argparse.ArgumentParser()  # Tworzy parser argumentów
    p.add_argument("--datasets", nargs="+", default=["all"], choices=["concrete", "all", "red", "white"])  # Argument: lista datasetów do treningu (domyślnie "all")
    p.add_argument("--memb", nargs="+", type=int, default=[2, 3])  # Argument: liczba funkcji przynależności (domyślnie 2 i 3)
    p.add_argument("--epochs", type=int, default=20)  # Argument: liczba epok treningu (domyślnie 20)
    p.add_argument("--batch-size", type=int, default=32)  # Argument: rozmiar batcha (domyślnie 32)
    p.add_argument("--cv", action="store_true")  # Argument: flaga włączająca cross-validation
    return p.parse_args()  # Zwraca sparsowane argumenty


if __name__ == "__main__":  # Wykonuje się tylko gdy plik uruchomiony bezpośrednio (nie importowany)
    os.makedirs("results", exist_ok=True)  # Tworzy katalog results jeśli nie istnieje
    os.makedirs("models", exist_ok=True)  # Tworzy katalog models jeśli nie istnieje
    args = _parse_args()  # Parsuje argumenty wiersza poleceń

    all_results = {}  # Słownik do przechowywania wyników wszystkich modeli
    for dataset in args.datasets:  # Iteruje przez wszystkie wybrane datasety
        for n_memb in args.memb:  # Iteruje przez wszystkie wybrane liczby funkcji przynależności
            try:  # Próbuje wytrenować model
                model, history, res = train_anfis_model(n_memb, args.epochs, args.batch_size, dataset)  # Trenuje model ANFIS
                all_results[f"{dataset}_{n_memb}MF"] = res  # Zapisuje wyniki do słownika
                if args.cv:  # Jeśli włączona cross-validation
                    cross_validate_anfis(n_memb, args.batch_size, dataset)  # Wykonuje cross-validation
            except Exception as e:  # Łapie wszelkie błędy
                print(f"⚠️ Błąd dla {dataset}, {n_memb}MF: {e}")  # Informuje o błędzie

    print("\nPODSUMOWANIE:")  # Wypisuje nagłówek podsumowania
    for name, res in all_results.items():  # Iteruje przez wszystkie wyniki
        metric = res.get("test_accuracy", res.get("test_mae"))  # Pobiera metrykę testową (accuracy lub mae)
        print(f"  {name}: test_metric={metric:.4f}")  # Wypisuje nazwę modelu i metrykę
