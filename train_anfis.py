import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from anfis import ANFISModel
import matplotlib.pyplot as plt
import json
import os
import argparse

np.random.seed(42)
tf.random.set_seed(42)


def _load_dataset(dataset: str):
    if dataset == 'concrete':
        X_train = np.load('data/concrete-strength/X_train.npy')
        y_train = np.load('data/concrete-strength/y_train.npy')
        X_test = np.load('data/concrete-strength/X_test.npy')
        y_test = np.load('data/concrete-strength/y_test.npy')
    elif dataset == 'all':
        X_train = np.load('data/X_train.npy')
        y_train = np.load('data/y_train.npy')
        X_test = np.load('data/X_test.npy')
        y_test = np.load('data/y_test.npy')
    elif dataset == 'red':
        X_train = np.load('data/X_train_red.npy')
        y_train = np.load('data/y_train_red.npy')
        X_test = np.load('data/X_test_red.npy')
        y_test = np.load('data/y_test_red.npy')
    elif dataset == 'white':
        X_train = np.load('data/X_train_white.npy')
        y_train = np.load('data/y_train_white.npy')
        X_test = np.load('data/X_test_white.npy')
        y_test = np.load('data/y_test_white.npy')
    else:
        raise ValueError('dataset must be one of: concrete, all, red, white')
    return X_train, X_test, y_train, y_test


def _rules_count(n_features: int, n_memb: int) -> int:
    return int(n_memb ** n_features)


def train_anfis_model(n_memb=2, epochs=20, batch_size=32, dataset='all'):
    print(f"\n{'='*70}")
    print(f"TRENING ANFIS: dataset={dataset}, n_memb={n_memb}")
    print(f"{'='*70}\n")
    
    X_train, X_test, y_train, y_test = _load_dataset(dataset)
    n_features = X_train.shape[1]
    n_rules = _rules_count(n_features, n_memb)

    n_train = (len(X_train) // batch_size) * batch_size
    n_test = (len(X_test) // batch_size) * batch_size
    X_train_batch = X_train[:n_train]
    y_train_batch = y_train[:n_train]
    X_test_batch = X_test[:n_test]
    y_test_batch = y_test[:n_test]

    print(f"Features: {n_features}, MF: {n_memb}, Rules: {n_rules}")
    print(f"Train samples: {len(X_train_batch)}, Test samples: {len(X_test_batch)}\n")

    anfis_model = ANFISModel(n_input=n_features, n_memb=n_memb, batch_size=batch_size)
    
    # Dla concrete (regresja) używamy MSE, dla wine (klasyfikacja) binary_crossentropy
    if dataset == 'concrete':
        anfis_model.model.compile(optimizer=tf.keras.optimizers.Nadam(learning_rate=0.001),
                                  loss='mean_squared_error', metrics=['mae'])
        monitor_metric = 'val_loss'
        monitor_mode = 'min'
        metric_name = 'mae'
    else:
        anfis_model.model.compile(optimizer=tf.keras.optimizers.Nadam(learning_rate=0.001),
                                  loss='binary_crossentropy', metrics=['accuracy'])
        monitor_metric = 'val_accuracy'
        monitor_mode = 'max'
        metric_name = 'accuracy'

    checkpoint_path = f'models/anfis_{dataset}_best_{n_memb}memb.weights.h5'
    checkpoint = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, monitor=monitor_metric,
                                                    save_best_only=True, save_weights_only=True,
                                                    mode=monitor_mode, verbose=1)
    early_stop = tf.keras.callbacks.EarlyStopping(monitor=monitor_metric, patience=10,
                                                  restore_best_weights=True, verbose=1)

    print("Rozpoczynam trening...\n")
    history = anfis_model.model.fit(
        X_train_batch, y_train_batch,
        validation_data=(X_test_batch, y_test_batch),
        epochs=epochs, batch_size=batch_size,
        callbacks=[checkpoint, early_stop], verbose=1
    )

    anfis_model.model.load_weights(checkpoint_path)
    train_loss, train_metric = anfis_model.model.evaluate(X_train_batch, y_train_batch, verbose=0)
    test_loss, test_metric = anfis_model.model.evaluate(X_test_batch, y_test_batch, verbose=0)

    print(f"\n{'='*70}")
    if dataset == 'concrete':
        print(f"WYNIKI: Train MAE={train_metric:.4f}, Test MAE={test_metric:.4f}")
        train_key, test_key = 'train_mae', 'test_mae'
    else:
        print(f"WYNIKI: Train Acc={train_metric:.4f}, Test Acc={test_metric:.4f}")
        train_key, test_key = 'train_accuracy', 'test_accuracy'
    print(f"{'='*70}\n")

    results = {
        'dataset': dataset,
        'n_memb': n_memb,
        'n_features': int(n_features),
        'n_rules': int(n_rules),
        train_key: float(train_metric),
        test_key: float(test_metric),
        'train_loss': float(train_loss),
        'test_loss': float(test_loss),
        'metric_type': 'mae' if dataset == 'concrete' else 'accuracy',
        'history': {
            metric_name: [float(x) for x in history.history[metric_name]],
            f'val_{metric_name}': [float(x) for x in history.history[f'val_{metric_name}']],
            'loss': [float(x) for x in history.history['loss']],
            'val_loss': [float(x) for x in history.history['val_loss']]
        }
    }

    os.makedirs('results', exist_ok=True)
    with open(f'results/anfis_{dataset}_{n_memb}memb_results.json', 'w') as f:
        json.dump(results, f, indent=4)

    plot_training_history(history, n_memb, dataset)
    plot_fit_on_train(anfis_model, X_train_batch, y_train_batch, n_memb, dataset)
    try:
        extract_and_save_rules(anfis_model, n_memb, dataset, X_train_batch)
    except Exception as e:
        print(f"Błąd przy ekstrakcji reguł: {e}")

    return anfis_model, history, results


def plot_training_history(history, n_memb, dataset):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(history.history['accuracy'], label='Train', linewidth=2)
    axes[0].plot(history.history['val_accuracy'], label='Validation', linewidth=2)
    axes[0].set_xlabel('Epoka')
    axes[0].set_ylabel('Dokładność')
    axes[0].set_title(f'ANFIS ({dataset}, {n_memb} MF) - Dokładność')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(history.history['loss'], label='Train', linewidth=2)
    axes[1].plot(history.history['val_loss'], label='Validation', linewidth=2)
    axes[1].set_xlabel('Epoka')
    axes[1].set_ylabel('Strata')
    axes[1].set_title(f'ANFIS ({dataset}, {n_memb} MF) - Strata')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    out = f'results/anfis_{dataset}_{n_memb}memb_training.png'
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Zapisano {out}")


def plot_fit_on_train(model: ANFISModel, X_train, y_train, n_memb, dataset):
    preds = model(X_train).reshape(-1)
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    ax[0].scatter(np.arange(len(y_train)), y_train, s=8, label='y_true', alpha=0.6)
    ax[0].scatter(np.arange(len(preds)), preds, s=8, label='y_pred', alpha=0.6)
    ax[0].set_title('Dopasowanie na zbiorze treningowym')
    ax[0].set_xlabel('Próbka')
    ax[0].set_ylabel('Prawdopodobieństwo klasy 1')
    ax[0].legend()
    ax[0].grid(True, alpha=0.3)
    ax[1].hist(preds[y_train==0], bins=30, alpha=0.7, label='y=0')
    ax[1].hist(preds[y_train==1], bins=30, alpha=0.7, label='y=1')
    ax[1].set_title('Rozkład predykcji na train')
    ax[1].set_xlabel('Prawdopodobieństwo')
    ax[1].legend()
    ax[1].grid(True, alpha=0.3)
    plt.tight_layout()
    out = f'results/anfis_{dataset}_{n_memb}memb_fit_train.png'
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Zapisano {out}")


def _rule_index_to_tuple(idx: int, n_features: int, n_memb: int):
    combo = []
    for _ in range(n_features):
        combo.append(idx % n_memb)
        idx //= n_memb
    return list(reversed(combo))


def extract_and_save_rules(model: ANFISModel, n_memb: int, dataset: str, X_train):
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
        best_idx_per_sample = np.stack([np.argmax(mu, axis=1) for mu in mvals], axis=1)
        pow_m = (n_memb ** np.arange(n_features)[::-1]).astype(int)
        rule_idx_per_sample = (best_idx_per_sample * pow_m).sum(axis=1)
        vals, counts = np.unique(rule_idx_per_sample, return_counts=True)
        order = np.argsort(counts)[::-1]
        vals = vals[order]
        counts = counts[order]
        top_k = int(min(100, len(vals)))
        top_rules_idx = vals[:top_k].tolist()
        approx_freq = {int(v): int(c) for v, c in zip(vals[:top_k], counts[:top_k])}

    rules = []
    for ridx in top_rules_idx:
        combo = _rule_index_to_tuple(ridx, n_features, n_memb)
        cons_w = weights[:, ridx].tolist()
        cons_b = float(bias[0, ridx])
        rules.append({
            'rule_index': int(ridx),
            'membership_indices': combo,
            'consequent': {'weights': cons_w, 'bias': cons_b}
        })
    payload = {
        'dataset': dataset,
        'n_features': int(n_features),
        'n_memb': int(n_memb),
        'n_rules_total': int(n_rules),
        'rules_listed': len(rules),
        'approx_top_rule_frequency': approx_freq,
        'rules': rules
    }
    out_json = f'results/anfis_{dataset}_{n_memb}memb_rules.json'
    with open(out_json, 'w') as f:
        json.dump(payload, f, indent=2)
    print(f"✓ Zapisano {out_json}")


def cross_validate_anfis(n_memb=2, batch_size=32, dataset='all', n_splits=5, epochs=15):
    print(f"\n{'='*70}")
    print(f"CROSS-WALIDACJA: dataset={dataset}, n_memb={n_memb}, {n_splits}-fold")
    print(f"{'='*70}\n")
    
    X_train, X_test, y_train, y_test = _load_dataset(dataset)
    X = np.concatenate([X_train, X_test], axis=0)
    y = np.concatenate([y_train, y_test], axis=0)
    
    # Dla concrete (regresja) używamy KFold, dla wine (klasyfikacja) StratifiedKFold
    if dataset == 'concrete':
        from sklearn.model_selection import KFold
        fold_splitter = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        split_iter = fold_splitter.split(X)
    else:
        fold_splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        split_iter = fold_splitter.split(X, y)
    fold_metrics = []
    n_features = X.shape[1]
    
    # Ustalamy loss i metryki w zależności od datasetu
    if dataset == 'concrete':
        loss_fn = 'mean_squared_error'
        metric_list = ['mae']
        metric_key = 'val_mae'
        print_metric = 'MAE'
    else:
        loss_fn = 'binary_crossentropy'
        metric_list = ['accuracy']
        metric_key = 'val_accuracy'
        print_metric = 'accuracy'
    
    for fold, (tr_idx, va_idx) in enumerate(split_iter, 1):
        print(f"Fold {fold}/{n_splits}...")
        Xt, Xv = X[tr_idx], X[va_idx]
        yt, yv = y[tr_idx], y[va_idx]
        n_tr = (len(Xt) // batch_size) * batch_size
        n_va = (len(Xv) // batch_size) * batch_size
        Xt, yt = Xt[:n_tr], yt[:n_tr]
        Xv, yv = Xv[:n_va], yv[:n_va]
        model = ANFISModel(n_input=n_features, n_memb=n_memb, batch_size=batch_size)
        model.model.compile(optimizer=tf.keras.optimizers.Nadam(0.001),
                            loss=loss_fn, metrics=metric_list)
        model.model.fit(Xt, yt, epochs=epochs, batch_size=batch_size, verbose=0)
        loss, metric_val = model.model.evaluate(Xv, yv, verbose=0)
        fold_metrics.append({'fold': fold, 'val_loss': float(loss), metric_key: float(metric_val)})
        print(f"  Fold {fold} {print_metric}: {metric_val:.4f}")
    
    metric_vals = [m[metric_key] for m in fold_metrics]
    
    # Dla concrete używamy mean_mae, dla wine mean_accuracy
    if dataset == 'concrete':
        summary_key = 'mean_mae'
        std_key = 'std_mae'
    else:
        summary_key = 'mean_accuracy'
        std_key = 'std_accuracy'
    
    out = {
        'dataset': dataset,
        'n_memb': n_memb,
        'n_splits': n_splits,
        'folds': fold_metrics,
        summary_key: float(np.mean(metric_vals)),
        std_key: float(np.std(metric_vals)),
        'metric_type': 'mae' if dataset == 'concrete' else 'accuracy'
    }
    with open(f'results/anfis_{dataset}_{n_memb}memb_cv.json', 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\n✓ Mean {print_metric}: {np.mean(metric_vals):.4f} ± {np.std(metric_vals):.4f}")
    print(f"✓ Zapisano results/anfis_{dataset}_{n_memb}memb_cv.json\n")
    return out


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--datasets', nargs='+', default=['all'], choices=['concrete', 'all', 'red', 'white'])
    p.add_argument('--memb', nargs='+', type=int, default=[2, 3])
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--cv', action='store_true')
    return p.parse_args()


if __name__ == "__main__":
    os.makedirs('results', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    args = _parse_args()
    all_results = {}
    for dataset in args.datasets:
        for n_memb in args.memb:
            try:
                model, history, results = train_anfis_model(
                    n_memb=n_memb, epochs=args.epochs, batch_size=args.batch_size, dataset=dataset
                )
                all_results[f'anfis_{dataset}_{n_memb}memb'] = results
                if args.cv:
                    cross_validate_anfis(n_memb=n_memb, batch_size=args.batch_size,
                                         dataset=dataset, n_splits=5, epochs=max(5, args.epochs//2))
            except Exception as e:
                print(f"BŁĄD dla dataset={dataset}, n_memb={n_memb}: {e}")
    print(f"\n{'='*70}")
    print("PODSUMOWANIE")
    print(f"{'='*70}")
    for name, res in all_results.items():
        print(f"{name}: Test={res['test_accuracy']:.4f} Train={res['train_accuracy']:.4f}")
