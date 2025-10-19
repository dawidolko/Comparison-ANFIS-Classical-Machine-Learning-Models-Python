import os
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from anfis import ANFISModel

np.random.seed(42)
tf.random.set_seed(42)

def train_anfis_model(n_memb=2, epochs=20, batch_size=32):
    print("\n" + "="*60)
    print(f"Trening ANFIS (m={n_memb})")
    print("="*60)

    X_train = np.load('data/X_train.npy')
    y_train = np.load('data/y_train.npy').astype(np.float32)
    X_test  = np.load('data/X_test.npy')
    y_test  = np.load('data/y_test.npy').astype(np.float32)

    if y_train.ndim == 1:
        y_train = y_train[:, None]
    if y_test.ndim == 1:
        y_test = y_test[:, None]

    n_features = X_train.shape[1]
    n_rules = n_memb ** n_features
    print(f"Liczba cech: {n_features}")
    print(f"Liczba reguł: {n_rules}")

    if n_rules > 100_000:
        print("Bardzo duża liczba reguł — rozważ m=2 lub selekcję cech.")

    model = ANFISModel(n_input=n_features, n_memb=n_memb, task="classification")

    model.init_memberships_from_data(X_train)

    model.model.compile(
        optimizer=tf.keras.optimizers.Nadam(learning_rate=1e-3),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    print("\n" + "="*60)
    print("Architektura modelu ANFIS:")
    print("="*60)
    model.model.summary()

    os.makedirs('models', exist_ok=True)
    ckpt_path = f'models/anfis_best_{n_memb}memb.keras'

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            ckpt_path, monitor='val_loss', save_best_only=True, mode='min', verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=10, restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5, verbose=1
        ),
    ]

    class_weight = None

    print("\n" + "="*60)
    print("Rozpoczynam trening...")
    print("="*60 + "\n")

    history = model.model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=1
    )

    model.model.load_weights(ckpt_path)

    train_loss, train_acc = model.model.evaluate(X_train, y_train, verbose=0)
    test_loss,  test_acc  = model.model.evaluate(X_test,  y_test,  verbose=0)

    print("\n" + "="*60)
    print("WYNIKI FINALNE:")
    print("="*60)
    print(f"Train Accuracy: {train_acc:.4f}")
    print(f"Test  Accuracy: {test_acc:.4f}")
    print(f"Train Loss:     {train_loss:.4f}")
    print(f"Test  Loss:     {test_loss:.4f}")

    os.makedirs('results', exist_ok=True)

    results = {
        'n_memb': n_memb,
        'train_accuracy': float(train_acc),
        'test_accuracy':  float(test_acc),
        'train_loss':     float(train_loss),
        'test_loss':      float(test_loss),
        'history': {k: [float(v) for v in history.history[k]] for k in history.history}
    }

    with open(f'results/anfis_{n_memb}memb_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    plot_training_history(history, n_memb)

    return model, history, results


def plot_training_history(history, n_memb):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history.history['accuracy'], label='Train', linewidth=2)
    axes[0].plot(history.history.get('val_accuracy', []), label='Validation', linewidth=2)
    axes[0].set_xlabel('Epoka'); axes[0].set_ylabel('Dokładność')
    axes[0].set_title(f'ANFIS (m={n_memb}) — Accuracy'); axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(history.history['loss'], label='Train', linewidth=2)
    axes[1].plot(history.history.get('val_loss', []), label='Validation', linewidth=2)
    axes[1].set_xlabel('Epoka'); axes[1].set_ylabel('Strata')
    axes[1].set_title(f'ANFIS (m={n_memb}) — Loss'); axes[1].legend(); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    out = f'results/anfis_{n_memb}memb_training.png'
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Wykres zapisany: {out}")


if __name__ == "__main__":
    os.makedirs('results', exist_ok=True)
    os.makedirs('models', exist_ok=True)

    membership_functions = [2, 3]

    all_results = {}
    for m in membership_functions:
        try:
            model, history, res = train_anfis_model(n_memb=m, epochs=20, batch_size=32)
            all_results[f'anfis_{m}memb'] = res
        except Exception as e:
            print(f"\n❌ Błąd dla n_memb={m}: {e}")

    print("\n" + "="*60)
    print("PODSUMOWANIE WSZYSTKICH MODELI ANFIS")
    print("="*60)
    for name, r in all_results.items():
        print(f"\n{name}:")
        print(f"  Test Accuracy:  {r['test_accuracy']:.4f}")
        print(f"  Train Accuracy: {r['train_accuracy']:.4f}")
