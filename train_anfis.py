import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from anfis import ANFISModel
import matplotlib.pyplot as plt
import json
import os

# Ustaw seed dla powtarzalności
np.random.seed(42)
tf.random.set_seed(42)


def train_anfis_model(n_memb=2, epochs=20, batch_size=32):
    """
    Trenuje model ANFIS z zadanymi parametrami

    Args:
        n_memb: liczba funkcji przynależności (2, 3, lub 4)
        epochs: liczba epok treningu
        batch_size: rozmiar batcha
    """
    print(f"\n{'=' * 60}")
    print(f"Trening ANFIS z {n_memb} funkcjami przynależności")
    print(f"{'=' * 60}\n")

    # Wczytaj dane
    X_train = np.load('data/X_train.npy')
    y_train = np.load('data/y_train.npy')
    X_test = np.load('data/X_test.npy')
    y_test = np.load('data/y_test.npy')

    n_features = X_train.shape[1]
    print(f"Liczba cech: {n_features}")
    print(f"Liczba funkcji przynależności: {n_memb}")
    print(f"Liczba reguł: {n_memb ** n_features}")

    # Przygotuj dane w odpowiednim rozmiarze batcha
    n_train = (len(X_train) // batch_size) * batch_size
    n_test = (len(X_test) // batch_size) * batch_size

    X_train_batch = X_train[:n_train]
    y_train_batch = y_train[:n_train]
    X_test_batch = X_test[:n_test]
    y_test_batch = y_test[:n_test]

    print(f"\nRozmiar zbioru treningowego (po dostosowaniu): {X_train_batch.shape}")
    print(f"Rozmiar zbioru testowego (po dostosowaniu): {X_test_batch.shape}")

    # Stwórz model
    anfis_model = ANFISModel(
        n_input=n_features,
        n_memb=n_memb,
        batch_size=batch_size
    )

    # Kompiluj model
    anfis_model.model.compile(
        optimizer=tf.keras.optimizers.Nadam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    print("\n" + "=" * 60)
    print("Architektura modelu ANFIS:")
    print("=" * 60)
    anfis_model.model.summary()

    # Callback do zapisywania najlepszego modelu
    checkpoint_path = f'models/anfis_best_{n_memb}memb.weights.h5'
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_path,
        monitor='val_accuracy',
        save_best_only=True,
        save_weights_only=True,  # Tylko wagi, bez optymalizatora
        mode='max',
        verbose=1
    )

    # Early stopping
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=15,
        restore_best_weights=True,
        verbose=1
    )

    # Trening
    print(f"\n{'=' * 60}")
    print("Rozpoczynam trening...")
    print(f"{'=' * 60}\n")

    history = anfis_model.model.fit(
        X_train_batch, y_train_batch,
        validation_data=(X_test_batch, y_test_batch),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[checkpoint, early_stop],
        verbose=1
    )

    # Załaduj najlepsze wagi
    anfis_model.model.load_weights(checkpoint_path)

    # Ewaluacja
    train_loss, train_acc = anfis_model.model.evaluate(X_train_batch, y_train_batch, verbose=0)
    test_loss, test_acc = anfis_model.model.evaluate(X_test_batch, y_test_batch, verbose=0)

    print(f"\n{'=' * 60}")
    print("WYNIKI FINALNE:")
    print(f"{'=' * 60}")
    print(f"Train Accuracy: {train_acc:.4f}")
    print(f"Test Accuracy:  {test_acc:.4f}")
    print(f"Train Loss:     {train_loss:.4f}")
    print(f"Test Loss:      {test_loss:.4f}")

    # Zapisz wyniki
    results = {
        'n_memb': n_memb,
        'train_accuracy': float(train_acc),
        'test_accuracy': float(test_acc),
        'train_loss': float(train_loss),
        'test_loss': float(test_loss),
        'history': {
            'accuracy': [float(x) for x in history.history['accuracy']],
            'val_accuracy': [float(x) for x in history.history['val_accuracy']],
            'loss': [float(x) for x in history.history['loss']],
            'val_loss': [float(x) for x in history.history['val_loss']]
        }
    }

    with open(f'results/anfis_{n_memb}memb_results.json', 'w') as f:
        json.dump(results, f, indent=4)

    # Wizualizacja treningu
    plot_training_history(history, n_memb)

    return anfis_model, history, results


def plot_training_history(history, n_memb):
    """Rysuje wykresy treningu"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy
    axes[0].plot(history.history['accuracy'], label='Train', linewidth=2)
    axes[0].plot(history.history['val_accuracy'], label='Validation', linewidth=2)
    axes[0].set_xlabel('Epoka', fontsize=12)
    axes[0].set_ylabel('Dokładność', fontsize=12)
    axes[0].set_title(f'ANFIS ({n_memb} funkcje) - Dokładność', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)

    # Loss
    axes[1].plot(history.history['loss'], label='Train', linewidth=2)
    axes[1].plot(history.history['val_loss'], label='Validation', linewidth=2)
    axes[1].set_xlabel('Epoka', fontsize=12)
    axes[1].set_ylabel('Strata', fontsize=12)
    axes[1].set_title(f'ANFIS ({n_memb} funkcje) - Funkcja straty', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'results/anfis_{n_memb}memb_training.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Wykres zapisany: results/anfis_{n_memb}memb_training.png")


if __name__ == "__main__":
    os.makedirs('results', exist_ok=True)
    os.makedirs('models', exist_ok=True)

    # Trenuj modele z różną liczbą funkcji przynależności
    membership_functions = [2, 3]

    all_results = {}

    for n_memb in membership_functions:
        try:
            model, history, results = train_anfis_model(
                n_memb=n_memb,
                epochs=20,
                batch_size=32
            )
            all_results[f'anfis_{n_memb}memb'] = results
        except Exception as e:
            print(f"\n❌ Błąd dla n_memb={n_memb}: {e}")

    # Podsumowanie
    print(f"\n{'=' * 60}")
    print("PODSUMOWANIE WSZYSTKICH MODELI ANFIS")
    print(f"{'=' * 60}")
    for name, res in all_results.items():
        print(f"\n{name}:")
        print(f"  Test Accuracy: {res['test_accuracy']:.4f}")
        print(f"  Train Accuracy: {res['train_accuracy']:.4f}")