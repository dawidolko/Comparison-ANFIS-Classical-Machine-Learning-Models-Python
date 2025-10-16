import numpy as np
import tensorflow as tf
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle
import json
import matplotlib.pyplot as plt
import os

# Ustaw seed
np.random.seed(42)
tf.random.set_seed(42)


def train_neural_network():
    """Trenuje klasyczną sieć neuronową"""
    print("\n" + "=" * 60)
    print("TRENING KLASYCZNEJ SIECI NEURONOWEJ")
    print("=" * 60 + "\n")

    # Wczytaj dane
    X_train = np.load('data/X_train.npy')
    y_train = np.load('data/y_train.npy')
    X_test = np.load('data/X_test.npy')
    y_test = np.load('data/y_test.npy')

    # Zbuduj model
    nn_model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    nn_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    print("Architektura sieci neuronowej:")
    nn_model.summary()

    # Callback
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        'models/nn_best.keras',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )

    # Trening
    print("\nRozpoczynam trening...\n")
    history = nn_model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=50,
        batch_size=32,
        callbacks=[checkpoint, early_stop],
        verbose=1
    )

    # Ewaluacja
    train_loss, train_acc = nn_model.evaluate(X_train, y_train, verbose=0)
    test_loss, test_acc = nn_model.evaluate(X_test, y_test, verbose=0)

    print(f"\n{'=' * 60}")
    print("WYNIKI FINALNE NN:")
    print(f"{'=' * 60}")
    print(f"Train Accuracy: {train_acc:.4f}")
    print(f"Test Accuracy:  {test_acc:.4f}")

    # Zapisz wyniki
    results = {
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

    with open('results/nn_results.json', 'w') as f:
        json.dump(results, f, indent=4)

    # Wykres
    plot_training_history(history, "Neural Network")

    return nn_model, results


def train_svm():
    """Trenuje Support Vector Machine"""
    print("\n" + "=" * 60)
    print("TRENING SUPPORT VECTOR MACHINE (SVM)")
    print("=" * 60 + "\n")

    # Wczytaj dane
    X_train = np.load('data/X_train.npy')
    y_train = np.load('data/y_train.npy')
    X_test = np.load('data/X_test.npy')
    y_test = np.load('data/y_test.npy')

    # Trenuj SVM z RBF kernel
    print("Trenuję SVM z RBF kernel...")
    print("(To może potrwać kilka minut...)\n")

    svm_model = SVC(
        kernel='rbf',
        C=1.0,
        gamma='scale',
        random_state=42,
        verbose=True
    )

    svm_model.fit(X_train, y_train)

    # Ewaluacja
    train_acc = svm_model.score(X_train, y_train)
    test_acc = svm_model.score(X_test, y_test)

    print(f"\n{'=' * 60}")
    print("WYNIKI FINALNE SVM:")
    print(f"{'=' * 60}")
    print(f"Train Accuracy: {train_acc:.4f}")
    print(f"Test Accuracy:  {test_acc:.4f}")

    # Zapisz model i wyniki
    with open('models/svm_model.pkl', 'wb') as f:
        pickle.dump(svm_model, f)

    results = {
        'train_accuracy': float(train_acc),
        'test_accuracy': float(test_acc)
    }

    with open('results/svm_results.json', 'w') as f:
        json.dump(results, f, indent=4)

    return svm_model, results


def train_random_forest():
    """Trenuje Random Forest"""
    print("\n" + "=" * 60)
    print("TRENING RANDOM FOREST")
    print("=" * 60 + "\n")

    # Wczytaj dane
    X_train = np.load('data/X_train.npy')
    y_train = np.load('data/y_train.npy')
    X_test = np.load('data/X_test.npy')
    y_test = np.load('data/y_test.npy')

    # Trenuj Random Forest
    print("Trenuję Random Forest z 200 drzewami...\n")

    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )

    rf_model.fit(X_train, y_train)

    # Ewaluacja
    train_acc = rf_model.score(X_train, y_train)
    test_acc = rf_model.score(X_test, y_test)

    print(f"\n{'=' * 60}")
    print("WYNIKI FINALNE Random Forest:")
    print(f"{'=' * 60}")
    print(f"Train Accuracy: {train_acc:.4f}")
    print(f"Test Accuracy:  {test_acc:.4f}")

    # Zapisz model i wyniki
    with open('models/rf_model.pkl', 'wb') as f:
        pickle.dump(rf_model, f)

    results = {
        'train_accuracy': float(train_acc),
        'test_accuracy': float(test_acc)
    }

    with open('results/rf_results.json', 'w') as f:
        json.dump(results, f, indent=4)

    return rf_model, results


def plot_training_history(history, model_name):
    """Rysuje wykresy treningu dla NN"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy
    axes[0].plot(history.history['accuracy'], label='Train', linewidth=2)
    axes[0].plot(history.history['val_accuracy'], label='Validation', linewidth=2)
    axes[0].set_xlabel('Epoka', fontsize=12)
    axes[0].set_ylabel('Dokładność', fontsize=12)
    axes[0].set_title(f'{model_name} - Dokładność', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)

    # Loss
    axes[1].plot(history.history['loss'], label='Train', linewidth=2)
    axes[1].plot(history.history['val_loss'], label='Validation', linewidth=2)
    axes[1].set_xlabel('Epoka', fontsize=12)
    axes[1].set_ylabel('Strata', fontsize=12)
    axes[1].set_title(f'{model_name} - Funkcja straty', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'results/nn_training.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Wykres zapisany: results/nn_training.png")


if __name__ == "__main__":
    os.makedirs('results', exist_ok=True)
    os.makedirs('models', exist_ok=True)

    # Trenuj wszystkie modele
    all_results = {}

    # 1. Neural Network
    try:
        nn_model, nn_results = train_neural_network()
        all_results['neural_network'] = nn_results
    except Exception as e:
        print(f"\n❌ Błąd podczas treningu NN: {e}")

    # 2. SVM
    try:
        svm_model, svm_results = train_svm()
        all_results['svm'] = svm_results
    except Exception as e:
        print(f"\n❌ Błąd podczas treningu SVM: {e}")

    # 3. Random Forest
    try:
        rf_model, rf_results = train_random_forest()
        all_results['random_forest'] = rf_results
    except Exception as e:
        print(f"\n❌ Błąd podczas treningu RF: {e}")

    # Podsumowanie
    print(f"\n{'=' * 70}")
    print("PODSUMOWANIE WSZYSTKICH MODELI PORÓWNAWCZYCH")
    print(f"{'=' * 70}")

    for name, res in all_results.items():
        print(f"\n{name.upper()}:")
        print(f"  Train Accuracy: {res['train_accuracy']:.4f}")
        print(f"  Test Accuracy:  {res['test_accuracy']:.4f}")

    print("\n✓ Wszystkie modele wytrenowane!")