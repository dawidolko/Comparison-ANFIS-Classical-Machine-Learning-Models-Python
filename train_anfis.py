import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold, KFold
from anfis import ANFISModel
import matplotlib.pyplot as plt
import json
import os
import argparse

np.random.seed(42)
tf.random.set_seed(42)


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
    base = "data"
    paths = {
        "concrete": (f"{base}/concrete-strength/X_train.npy", f"{base}/concrete-strength/X_test.npy",
                     f"{base}/concrete-strength/y_train.npy", f"{base}/concrete-strength/y_test.npy"),
        "all": (f"{base}/X_train.npy", f"{base}/X_test.npy",
                f"{base}/y_train.npy", f"{base}/y_test.npy"),
        "red": (f"{base}/X_train_red.npy", f"{base}/X_test_red.npy",
                f"{base}/y_train_red.npy", f"{base}/y_test_red.npy"),
        "white": (f"{base}/X_train_white.npy", f"{base}/X_test_white.npy",
                  f"{base}/y_train_white.npy", f"{base}/y_test_white.npy")
    }

    if dataset not in paths:
        raise ValueError(f"Nieznany dataset: {dataset}")
    Xtr, Xte, ytr, yte = [np.load(p) for p in paths[dataset]]
    return Xtr, Xte, ytr, yte


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
    return int(n_memb ** n_features)


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
    print(f"\n{'='*70}")
    print(f"TRENING ANFIS: dataset={dataset}, n_memb={n_memb}")
    print(f"{'='*70}\n")

    X_train, X_test, y_train, y_test = _load_dataset(dataset)
    n_features = X_train.shape[1]
    n_rules = _rules_count(n_features, n_memb)

    # dopasowanie batch_size (by uniknąć warningów TF)
    n_train = (len(X_train) // batch_size) * batch_size
    n_test = (len(X_test) // batch_size) * batch_size
    X_train, y_train = X_train[:n_train], y_train[:n_train]
    X_test, y_test = X_test[:n_test], y_test[:n_test]

    print(f"Features: {n_features}, MF: {n_memb}, Rules: {n_rules}")
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")

    # Dla regresji (concrete) ustawiono regression=True
    is_regression = (dataset == "concrete")
    anfis_model = ANFISModel(n_input=n_features, n_memb=n_memb, batch_size=batch_size, regression=is_regression)

    # --------------------- konfiguracja metryk ---------------------
    if dataset == "concrete":
        anfis_model.model.compile(
            optimizer=tf.keras.optimizers.Nadam(learning_rate=0.001),
            loss="mean_squared_error",
            metrics=["mae"]
        )
        monitor_metric = "val_loss"
        monitor_mode = "min"
        metric_name = "mae"
    else:
        anfis_model.model.compile(
            optimizer=tf.keras.optimizers.Nadam(learning_rate=0.001),
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )
        monitor_metric = "val_accuracy"
        monitor_mode = "max"
        metric_name = "accuracy"

    # ---------------------- callbacki ----------------------
    checkpoint_path = f"models/anfis_{dataset}_best_{n_memb}memb.weights.h5"
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path, monitor=monitor_metric,
            save_best_only=True, save_weights_only=True,
            mode=monitor_mode, verbose=0
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor=monitor_metric, patience=10,
            restore_best_weights=True, verbose=0
        )
    ]

    print("Rozpoczynam trening...\n")
    history = anfis_model.model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs, batch_size=batch_size,
        callbacks=callbacks, verbose=1
    )

    # ---------------------- ewaluacja ----------------------
    anfis_model.model.load_weights(checkpoint_path)
    train_loss, train_metric = anfis_model.model.evaluate(X_train, y_train, verbose=0)
    test_loss, test_metric = anfis_model.model.evaluate(X_test, y_test, verbose=0)

    # ---------------------- wyniki ----------------------
    results = {
        "dataset": dataset,
        "n_memb": n_memb,
        "n_features": int(n_features),
        "n_rules": int(n_rules),
        "train_loss": float(train_loss),
        "test_loss": float(test_loss),
        "metric_type": "mae" if dataset == "concrete" else "accuracy",
        ("train_mae" if dataset == "concrete" else "train_accuracy"): float(train_metric),
        ("test_mae" if dataset == "concrete" else "test_accuracy"): float(test_metric),
        "history": {
            metric_name: [float(x) for x in history.history.get(metric_name, [])],
            f"val_{metric_name}": [float(x) for x in history.history.get(f"val_{metric_name}", [])],
            "loss": [float(x) for x in history.history["loss"]],
            "val_loss": [float(x) for x in history.history["val_loss"]],
        },
    }

    os.makedirs("results", exist_ok=True)
    out_json = f"results/anfis_{dataset}_{n_memb}memb_results.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=4)
    print(f"✓ Zapisano {out_json}")

    # ---------------------- wykresy ----------------------
    plot_training_history(history, n_memb, dataset)
    plot_fit_on_train(anfis_model, X_train, y_train, n_memb, dataset)

    # ---------------------- reguły ----------------------
    try:
        extract_and_save_rules(anfis_model, n_memb, dataset, X_train)
    except Exception as e:
        print(f"⚠️ Błąd przy ekstrakcji reguł: {e}")

    return anfis_model, history, results


# -------------------------------------------------------------
# WIZUALIZACJE
# -------------------------------------------------------------
def plot_training_history(history, n_memb, dataset):
    """
    Wygenerowano wizualizację historii treningu modelu ANFIS.
    
    Dla KLASYFIKACJI (wine):
    - Górny wiersz: krzywe Accuracy i Loss (zbiór treningowy vs walidacyjny),
    - Dolny wiersz: tabela podsumowująca metryki w wybranych epokach.
    
    Dla REGRESJI (concrete):
    - Krzywe MAE i Loss (zbiór treningowy vs walidacyjny).
    
    Args:
        history: obiekt History z Keras
        n_memb: liczba funkcji przynależności
        dataset: nazwa zbioru danych
    """
    fig = plt.figure(figsize=(16, 6))
    gs = fig.add_gridspec(2, 2, height_ratios=[3, 1])

    # Określono typ metryki na podstawie zawartości historii
    if "accuracy" in history.history:
        mkey, vkey, label = "accuracy", "val_accuracy", "Accuracy"
        is_classification = True
    elif "mae" in history.history:
        mkey, vkey, label = "mae", "val_mae", "MAE"
        is_classification = False
    else:
        mkey = None
        is_classification = False

    # Lewy górny wykres: podstawowa metryka (Accuracy/MAE)
    ax0 = fig.add_subplot(gs[0, 0])
    if mkey:
        epochs = np.arange(1, len(history.history[mkey]) + 1)
        ax0.plot(epochs, history.history[mkey], label="Train", lw=2.5, marker='o', markersize=4, color='steelblue')
        ax0.plot(epochs, history.history[vkey], label="Validation", lw=2.5, marker='s', markersize=4, color='coral')
        ax0.set_title(f"{label} Curve ({dataset}, {n_memb} MF)", fontsize=13, fontweight='bold')
        ax0.set_xlabel("Epoch", fontsize=11, fontweight='bold')
        ax0.set_ylabel(label, fontsize=11, fontweight='bold')
        ax0.legend(fontsize=10, loc='best')
        ax0.grid(True, alpha=0.3)
        
        # Zaznaczono najlepszą epokę (maksymalna wartość walidacyjna dla klasyfikacji, minimalna dla regresji)
        best_epoch = np.argmax(history.history[vkey]) if is_classification else np.argmin(history.history[vkey])
        best_value = history.history[vkey][best_epoch]
        ax0.scatter([best_epoch + 1], [best_value], color='red', s=100, zorder=5, label=f'Najlepsza epoka: {best_epoch + 1}')
        ax0.legend(fontsize=10, loc='best')

    # Prawy górny wykres: krzywa funkcji straty (Loss)
    ax1 = fig.add_subplot(gs[0, 1])
    epochs = np.arange(1, len(history.history["loss"]) + 1)
    ax1.plot(epochs, history.history["loss"], label="Train", lw=2.5, marker='o', markersize=4, color='darkgreen')
    ax1.plot(epochs, history.history["val_loss"], label="Validation", lw=2.5, marker='s', markersize=4, color='darkred')
    ax1.set_title(f"Krzywa funkcji straty ({dataset}, {n_memb} MF)", fontsize=13, fontweight='bold')
    ax1.set_xlabel("Epoch", fontsize=11, fontweight='bold')
    ax1.set_ylabel("Loss (MSE/BCE)", fontsize=11, fontweight='bold')
    ax1.legend(fontsize=10, loc='best')
    ax1.grid(True, alpha=0.3)
    
    # Dolny wiersz: tabela metryk (próbka epok)
    ax2 = fig.add_subplot(gs[1, :])
    ax2.axis('off')
    
    # Przyjęto próbę epok: co 5. epoka + ostatnia, maks. 8 wierszy
    total_epochs = len(history.history["loss"])
    sample_epochs = list(range(0, total_epochs, 5)) + [total_epochs - 1]
    sample_epochs = sorted(set(sample_epochs))[:8]

    if mkey:
        table_data = [
            ["Epoch"] + [str(e + 1) for e in sample_epochs],
            ["Train " + label] + [f"{history.history[mkey][e]:.4f}" for e in sample_epochs],
            ["Val " + label] + [f"{history.history[vkey][e]:.4f}" for e in sample_epochs],
            ["Train Loss"] + [f"{history.history['loss'][e]:.4f}" for e in sample_epochs],
            ["Val Loss"] + [f"{history.history['val_loss'][e]:.4f}" for e in sample_epochs]
        ]
    else:
        table_data = [
            ["Epoch"] + [str(e + 1) for e in sample_epochs],
            ["Train Loss"] + [f"{history.history['loss'][e]:.4f}" for e in sample_epochs],
            ["Val Loss"] + [f"{history.history['val_loss'][e]:.4f}" for e in sample_epochs]
        ]
    
    # Dodano tabelę do osi
    table = ax2.table(cellText=table_data, cellLoc='center', loc='center', 
                     colWidths=[0.12] * len(table_data[0]))
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Kolor nagłówka został ustawiony na zielony
    for i in range(len(table_data[0])):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')

    plt.suptitle(f"Historia treningu: {dataset.upper()} ({n_memb} funkcji przynależności)", 
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(f"results/anfis_{dataset}_{n_memb}memb_training.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_fit_on_train(model, X_train, y_train, n_memb, dataset):
    """
    Wizualizuje dopasowanie modelu na zbiorze treningowym.
    
    Generuje prosty wykres rozproszenia (scatter plot) pokazujący
    wartości rzeczywiste vs predykcje modelu.
    
    Dla regresji (concrete) dodatkowo oblicza i wyświetla współczynnik R².
    Linia y=x reprezentuje idealne dopasowanie.
    
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
    preds_raw = model(X_train)
    if hasattr(preds_raw, 'numpy'):
        preds = preds_raw.numpy().reshape(-1)
    else:
        preds = np.array(preds_raw).reshape(-1)
    
    print(f"[DEBUG] preds min: {preds.min():.4f}, max: {preds.max():.4f}, mean: {preds.mean():.4f}")
    
    # PROSTY WYKRES
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    
    # SCATTER
    ax.scatter(y_train, preds, s=30, alpha=0.6, color='blue')
    
    # Linia y=x
    min_val = min(y_train.min(), preds.min())
    max_val = max(y_train.max(), preds.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='y=x')
    
    ax.set_xlabel("Rzeczywiste", fontsize=14)
    ax.set_ylabel("Predykcja", fontsize=14)
    ax.set_title(f"{dataset} - {n_memb}MF", fontsize=16)
    ax.legend()
    ax.grid(True)
    
    # R2 dla regresji
    if dataset == "concrete":
        from sklearn.metrics import r2_score
        r2 = r2_score(y_train, preds)
        ax.text(0.05, 0.95, f'R²={r2:.3f}', transform=ax.transAxes, fontsize=12,
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(f"results/anfis_{dataset}_{n_memb}memb_fit_train.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Zapisano results/anfis_{dataset}_{n_memb}memb_fit_train.png")


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
    model.update_weights()
    centers, sigmas = model.get_membership_functions()
    bias, weights = model.bias, model.weights
    n_features = centers.shape[1]
    n_rules = _rules_count(n_features, n_memb)

    if n_rules <= 4096:
        top_rules_idx = list(range(n_rules))
        approx_freq = None
    else:
        mvals = []
        for j in range(n_features):
            xj = X_train[:, j][:, None]
            cj = centers[:, j][None, :]
            sj = sigmas[:, j][None, :]
            mu = np.exp(-((xj - cj) ** 2) / (sj ** 2 + 1e-8))
            mvals.append(mu)
        best_idx = np.stack([np.argmax(mu, axis=1) for mu in mvals], axis=1)
        pow_m = (n_memb ** np.arange(n_features)[::-1]).astype(int)
        rule_idx = (best_idx * pow_m).sum(axis=1)
        vals, counts = np.unique(rule_idx, return_counts=True)
        order = np.argsort(counts)[::-1]
        vals, counts = vals[order], counts[order]
        top_rules_idx = vals[:100].tolist()
        approx_freq = {int(v): int(c) for v, c in zip(vals[:100], counts[:100])}

    rules = []
    for ridx in top_rules_idx:
        combo = _rule_index_to_tuple(ridx, n_features, n_memb)
        rules.append({
            "rule_index": int(ridx),
            "membership_indices": combo,
            "consequent": {
                "weights": [float(w) for w in weights[:, ridx]],
                "bias": float(bias[0, ridx])
            }
        })

    payload = {
        "dataset": dataset,
        "n_features": n_features,
        "n_memb": n_memb,
        "n_rules_total": n_rules,
        "rules_listed": len(rules),
        "approx_top_rule_frequency": approx_freq,
        "membership_centers": centers.tolist(),
        "membership_sigmas": sigmas.tolist(),
        "rules": rules
    }

    out_json = f"results/anfis_{dataset}_{n_memb}memb_rules.json"
    with open(out_json, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"✓ Zapisano {out_json}")


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
    print(f"\n{'='*70}")
    print(f"CROSS-VALIDATION: {dataset}, {n_memb} MF, {n_splits}-fold")
    print(f"{'='*70}\n")

    X_train, X_test, y_train, y_test = _load_dataset(dataset)
    X = np.concatenate([X_train, X_test])
    y = np.concatenate([y_train, y_test])

    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=42) \
        if dataset == "concrete" else StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    fold_metrics = []
    is_regression = (dataset == "concrete")
    for fold, (tr_idx, va_idx) in enumerate(splitter.split(X, y if dataset != "concrete" else None), 1):
        print(f"Fold {fold}/{n_splits}")
        Xt, Xv, yt, yv = X[tr_idx], X[va_idx], y[tr_idx], y[va_idx]
        model = ANFISModel(n_input=X.shape[1], n_memb=n_memb, batch_size=batch_size, regression=is_regression)
        if dataset == "concrete":
            model.model.compile(optimizer=tf.keras.optimizers.Nadam(0.001),
                                loss="mean_squared_error", metrics=["mae"])
            metric_key = "val_mae"
        else:
            model.model.compile(optimizer=tf.keras.optimizers.Nadam(0.001),
                                loss="binary_crossentropy", metrics=["accuracy"])
            metric_key = "val_accuracy"
        model.model.fit(Xt, yt, epochs=epochs, batch_size=batch_size, verbose=0)
        loss, metric_val = model.model.evaluate(Xv, yv, verbose=0)
        fold_metrics.append({"fold": fold, metric_key: float(metric_val), "val_loss": float(loss)})

    metric_vals = [list(f.values())[1] for f in fold_metrics]
    mean_val, std_val = np.mean(metric_vals), np.std(metric_vals)
    summary = {
        "dataset": dataset,
        "n_memb": n_memb,
        "n_splits": n_splits,
        "folds": fold_metrics,
        ("mean_mae" if dataset == "concrete" else "mean_accuracy"): float(mean_val),
        ("std_mae" if dataset == "concrete" else "std_accuracy"): float(std_val),
        "metric_type": "mae" if dataset == "concrete" else "accuracy",
    }

    out_json = f"results/anfis_{dataset}_{n_memb}memb_cv.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Zapisano {out_json}")
    return summary


# -------------------------------------------------------------
# MAIN
# -------------------------------------------------------------
def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=["all"], choices=["concrete", "all", "red", "white"])
    p.add_argument("--memb", nargs="+", type=int, default=[2, 3])
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--cv", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    args = _parse_args()

    all_results = {}
    for dataset in args.datasets:
        for n_memb in args.memb:
            try:
                model, history, res = train_anfis_model(n_memb, args.epochs, args.batch_size, dataset)
                all_results[f"{dataset}_{n_memb}MF"] = res
                if args.cv:
                    cross_validate_anfis(n_memb, args.batch_size, dataset)
            except Exception as e:
                print(f"⚠️ Błąd dla {dataset}, {n_memb}MF: {e}")

    print("\nPODSUMOWANIE:")
    for name, res in all_results.items():
        metric = res.get("test_accuracy", res.get("test_mae"))
        print(f"  {name}: test_metric={metric:.4f}")
