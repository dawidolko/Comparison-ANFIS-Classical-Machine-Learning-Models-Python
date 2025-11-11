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
    """Ładuje dane z plików .npy dla danego datasetu."""
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
    return int(n_memb ** n_features)


# -------------------------------------------------------------
# GŁÓWNY TRENING
# -------------------------------------------------------------
def train_anfis_model(n_memb=2, epochs=20, batch_size=32, dataset="all"):
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

    anfis_model = ANFISModel(n_input=n_features, n_memb=n_memb, batch_size=batch_size)

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
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    if "accuracy" in history.history:
        mkey, vkey, label = "accuracy", "val_accuracy", "Dokładność"
    elif "mae" in history.history:
        mkey, vkey, label = "mae", "val_mae", "MAE"
    else:
        mkey = None

    if mkey:
        axes[0].plot(history.history[mkey], label="Train", lw=2)
        axes[0].plot(history.history[vkey], label="Validation", lw=2)
        axes[0].set_title(f"{label} ({dataset}, {n_memb} MF)")
        axes[0].set_xlabel("Epoka")
        axes[0].set_ylabel(label)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

    axes[1].plot(history.history["loss"], label="Train", lw=2)
    axes[1].plot(history.history["val_loss"], label="Validation", lw=2)
    axes[1].set_title(f"Strata ({dataset}, {n_memb} MF)")
    axes[1].set_xlabel("Epoka")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"results/anfis_{dataset}_{n_memb}memb_training.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_fit_on_train(model, X_train, y_train, n_memb, dataset):
    preds = model(X_train).reshape(-1)
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))

    if dataset == "concrete":
        ax[0].scatter(y_train, preds, s=8, alpha=0.6)
        ax[0].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], "r--")
        ax[0].set_xlabel("Rzeczywista wytrzymałość (MPa)")
        ax[0].set_ylabel("Predykcja ANFIS")
        ax[0].set_title("Concrete - Dopasowanie modelu")
        ax[1].hist(y_train - preds, bins=30, color="gray", alpha=0.7)
        ax[1].set_title("Rozkład błędów (y_true - y_pred)")
    else:
        ax[0].scatter(np.arange(len(y_train)), y_train, s=8, label="y_true", alpha=0.6)
        ax[0].scatter(np.arange(len(preds)), preds, s=8, label="y_pred", alpha=0.6)
        ax[0].set_title("Dopasowanie ANFIS na treningu")
        ax[0].legend()
        ax[1].hist(preds[y_train == 0], bins=30, alpha=0.7, label="y=0")
        ax[1].hist(preds[y_train == 1], bins=30, alpha=0.7, label="y=1")
        ax[1].set_title("Rozkład predykcji (train)")
        ax[1].legend()

    for a in ax:
        a.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"results/anfis_{dataset}_{n_memb}memb_fit_train.png", dpi=300, bbox_inches="tight")
    plt.close()


# -------------------------------------------------------------
# EKSTRAKCJA REGUŁ
# -------------------------------------------------------------
def _rule_index_to_tuple(idx: int, n_features: int, n_memb: int):
    combo = []
    for _ in range(n_features):
        combo.append(idx % n_memb)
        idx //= n_memb
    return list(reversed(combo))


def extract_and_save_rules(model, n_memb: int, dataset: str, X_train):
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
    print(f"\n{'='*70}")
    print(f"CROSS-VALIDATION: {dataset}, {n_memb} MF, {n_splits}-fold")
    print(f"{'='*70}\n")

    X_train, X_test, y_train, y_test = _load_dataset(dataset)
    X = np.concatenate([X_train, X_test])
    y = np.concatenate([y_train, y_test])

    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=42) \
        if dataset == "concrete" else StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    fold_metrics = []
    for fold, (tr_idx, va_idx) in enumerate(splitter.split(X, y if dataset != "concrete" else None), 1):
        print(f"Fold {fold}/{n_splits}")
        Xt, Xv, yt, yv = X[tr_idx], X[va_idx], y[tr_idx], y[va_idx]
        model = ANFISModel(n_input=X.shape[1], n_memb=n_memb, batch_size=batch_size)
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
