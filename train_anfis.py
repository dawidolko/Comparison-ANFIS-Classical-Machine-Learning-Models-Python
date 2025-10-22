import os
import json
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')  # Wyłącza wyświetlanie okien - tylko zapis do plików
import matplotlib.pyplot as plt
from anfis import ANFISModel

import os
import json
import argparse
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from anfis import ANFISModel

# Repro
np.random.seed(42)
tf.random.set_seed(42)


def load_dataset_npy(dataset: str):
    # prefer wine-quality folder for wine dataset
    if dataset == 'concrete-strength':
        base = os.path.join('data', 'concrete-strength')
    else:
        # prefer data/wine-quality if present, fallback to data/
        candidate = os.path.join('data', 'wine-quality', 'X_train.npy')
        if os.path.exists(candidate):
            base = os.path.join('data', 'wine-quality')
        else:
            base = 'data'

    X_train = np.load(os.path.join(base, 'X_train.npy')).astype(np.float32)
    y_train = np.load(os.path.join(base, 'y_train.npy')).astype(np.float32)
    X_test = np.load(os.path.join(base, 'X_test.npy')).astype(np.float32)
    y_test = np.load(os.path.join(base, 'y_test.npy')).astype(np.float32)
    if y_train.ndim == 1:
        y_train = y_train[:, None]
    if y_test.ndim == 1:
        y_test = y_test[:, None]
    return X_train, y_train, X_test, y_test


def plot_training_history(history, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(history.history.get('accuracy', []), label='Train')
    axes[0].plot(history.history.get('val_accuracy', []), label='Val')
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Accuracy'); axes[0].legend()

    axes[1].plot(history.history.get('loss', []), label='Train')
    axes[1].plot(history.history.get('val_loss', []), label='Val')
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Loss'); axes[1].legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def train_anfis_for_dataset(dataset: str, n_memb: int, epochs: int, batch_size: int):
    print(f"\n=== Trening ANFIS (dataset={dataset}, m={n_memb}) ===")
    X_train, y_train, X_test, y_test = load_dataset_npy(dataset)
    n_features = X_train.shape[1]
    print(f"Loaded dataset {dataset} - features: {n_features}")

    model = ANFISModel(n_input=n_features, n_memb=n_memb, batch_size=batch_size)
    model.model.compile(optimizer=tf.keras.optimizers.Nadam(1e-3), loss='binary_crossentropy', metrics=['accuracy'])
    model.model.summary()

    models_dir = 'models' if dataset == 'wine' else os.path.join('models', dataset)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs('results', exist_ok=True)

    ckpt_path = os.path.join(models_dir, f'anfis_best_{n_memb}memb.weights.h5')
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(ckpt_path, monitor='val_loss', mode='min', save_best_only=True, save_weights_only=True, verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5, verbose=1),
    ]

    history = model.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2, callbacks=callbacks, verbose=1)

    # load best weights and evaluate
    model.model.load_weights(ckpt_path)
    train_loss, train_acc = model.model.evaluate(X_train, y_train, verbose=0)
    test_loss, test_acc = model.model.evaluate(X_test, y_test, verbose=0)

    results = {
        'dataset': dataset,
        'n_memb': n_memb,
        'n_features': int(n_features),
        'train_accuracy': float(train_acc),
        'test_accuracy': float(test_acc),
        'train_loss': float(train_loss),
        'test_loss': float(test_loss),
        'history': {k: [float(x) for x in v] for k, v in history.history.items()}
    }

    results_name = f'anfis_{n_memb}memb_results.json' if dataset == 'wine' else f'anfis_{dataset}_{n_memb}memb_results.json'
    with open(os.path.join('results', results_name), 'w') as f:
        json.dump(results, f, indent=2)

    plot_training_history(history, os.path.join('results', f'anfis_{dataset}_{n_memb}memb_training.png'))
    print(f"Saved checkpoint to: {ckpt_path}")
    return ckpt_path, results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+', default=None, help='Datasets to train on: wine and/or concrete-strength')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch', type=int, default=32)
    args = parser.parse_args()

    # detect available datasets if not provided
    available = []
    if args.datasets:
        available = args.datasets
    else:
        # check both legacy location and new data/wine-quality
        if os.path.exists(os.path.join('data', 'X_train.npy')) or os.path.exists(os.path.join('data', 'wine-quality', 'X_train.npy')):
            available.append('wine')
        if os.path.exists(os.path.join('data', 'concrete-strength', 'X_train.npy')):
            available.append('concrete-strength')

    if not available:
        print('No preprocessed datasets found. Run data_preprocessing.py first.')
        raise SystemExit(1)

    for ds in available:
        for m in [2, 3]:
            train_anfis_for_dataset(ds, n_memb=m, epochs=args.epochs, batch_size=args.batch)

