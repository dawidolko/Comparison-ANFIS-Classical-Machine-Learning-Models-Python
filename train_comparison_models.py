import os, json, random, time, pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    average_precision_score, confusion_matrix,
    classification_report
)

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)
os.environ["TF_DETERMINISTIC_OPS"] = "1"

RESULTS_DIR = "results"
MODELS_DIR  = "models"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR,  exist_ok=True)

def now_suffix():
    return time.strftime("%Y%m%d-%H%M%S")

def ensure_column_vector(y):
    y = np.asarray(y)
    if y.ndim == 1:
        return y.reshape(-1, 1)
    return y

def evaluate_and_save(y_true, y_pred_labels, y_pred_proba=None,
                      out_json='results.json', out_txt=None):
    y_true = np.asarray(y_true).ravel()
    y_pred_labels = np.asarray(y_pred_labels).ravel()

    res = {
        "accuracy": float(accuracy_score(y_true, y_pred_labels)),
        "f1": float(f1_score(y_true, y_pred_labels)),
        "confusion_matrix": confusion_matrix(y_true, y_pred_labels).tolist(),
        "report": classification_report(y_true, y_pred_labels, output_dict=True)
    }
    if y_pred_proba is not None:
        y_pred_proba = np.asarray(y_pred_proba).ravel()
        res["roc_auc"] = float(roc_auc_score(y_true, y_pred_proba))
        res["pr_auc"]  = float(average_precision_score(y_true, y_pred_proba))

    with open(out_json, "w") as f:
        json.dump(res, f, indent=2)

    if out_txt is not None:
        with open(out_txt, "w") as f:
            f.write(classification_report(y_true, y_pred_labels))

    return res

def class_weight_from_labels(y):
    y = np.asarray(y).ravel()
    pos = np.sum(y == 1)
    neg = np.sum(y == 0)
    if pos == 0 or neg == 0:
        return None
    w_pos = neg / max(pos, 1)
    return {0: 1.0, 1: float(w_pos)}

def plot_training_history(history, model_name, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history.history['accuracy'], label='Train', linewidth=2)
    if 'val_accuracy' in history.history:
        axes[0].plot(history.history['val_accuracy'], label='Validation', linewidth=2)
    axes[0].set_xlabel('Epoka'); axes[0].set_ylabel('Dokładność')
    axes[0].set_title(f'{model_name} - Dokładność'); axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(history.history['loss'], label='Train', linewidth=2)
    if 'val_loss' in history.history:
        axes[1].plot(history.history['val_loss'], label='Validation', linewidth=2)
    axes[1].set_xlabel('Epoka'); axes[1].set_ylabel('Strata')
    axes[1].set_title(f'{model_name} - Funkcja straty'); axes[1].legend(); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Wykres zapisany: {out_path}")

def load_data():
    X_train = np.load('data/X_train.npy')
    y_train = np.load('data/y_train.npy')
    X_test  = np.load('data/X_test.npy')
    y_test  = np.load('data/y_test.npy')
    y_train = ensure_column_vector(y_train).astype(np.float32)
    y_test  = ensure_column_vector(y_test).astype(np.float32)
    return X_train, y_train, X_test, y_test

def train_neural_network():
    print("\n" + "="*60)
    print("TRENING: NEURAL NETWORK (MLP)")
    print("="*60)

    X_train, y_train, X_test, y_test = load_data()

    scaler = StandardScaler().fit(X_train)
    with open(os.path.join(MODELS_DIR, 'scaler_nn.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    X_train_s = scaler.transform(X_train)
    X_test_s  = scaler.transform(X_test)

    nn_model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(X_train_s.shape[1],)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    nn_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    nn_model.summary()

    ckpt_path = os.path.join(MODELS_DIR, 'nn_best.keras')
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(ckpt_path, monitor='val_loss', save_best_only=True, mode='min', verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5, verbose=1)
    ]

    cw = class_weight_from_labels(y_train)

    print("\nRozpoczynam trening (validation_split=0.2)...\n")
    history = nn_model.fit(
        X_train_s, y_train,
        validation_split=0.2,
        epochs=50,
        batch_size=32,
        callbacks=callbacks,
        class_weight=cw,
        verbose=1
    )

    train_loss, train_acc = nn_model.evaluate(X_train_s, y_train, verbose=0)
    y_proba = nn_model.predict(X_test_s, verbose=0).ravel()
    y_pred  = (y_proba >= 0.5).astype(int)

    metrics = evaluate_and_save(
        y_test, y_pred, y_proba,
        out_json=os.path.join(RESULTS_DIR, 'nn_results.json'),
        out_txt=os.path.join(RESULTS_DIR, 'nn_report.txt')
    )

    metrics['train_accuracy'] = float(train_acc)
    metrics['train_loss']     = float(train_loss)
    metrics['test_accuracy']  = float(metrics.get('accuracy', np.nan))

    with open(os.path.join(RESULTS_DIR, 'nn_results.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    plot_training_history(history, "Neural Network",
                          os.path.join(RESULTS_DIR, f"nn_training_{now_suffix()}.png"))

    nn_model.save(ckpt_path, overwrite=True)
    print("✓ Zapisano model NN i metryki.")
    return nn_model, metrics


def train_svm():
    print("\n" + "="*60)
    print("TRENING: SVM (RBF)")
    print("="*60)

    X_train, y_train, X_test, y_test = load_data()
    y_train_r = y_train.ravel()
    y_test_r  = y_test.ravel()

    svm_clf = make_pipeline(
        StandardScaler(),
        SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=SEED)
    )
    svm_clf.fit(X_train, y_train_r)

    y_proba = svm_clf.predict_proba(X_test)[:, 1]
    y_pred  = (y_proba >= 0.5).astype(int)

    metrics = evaluate_and_save(
        y_test_r, y_pred, y_proba,
        out_json=os.path.join(RESULTS_DIR, 'svm_results.json'),
        out_txt=os.path.join(RESULTS_DIR, 'svm_report.txt')
    )

    metrics['train_accuracy'] = float(svm_clf.score(X_train, y_train_r))
    metrics['test_accuracy']  = float(metrics.get('accuracy', np.nan))

    with open(os.path.join(RESULTS_DIR, 'svm_results.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    with open(os.path.join(MODELS_DIR, 'svm_model.pkl'), 'wb') as f:
        pickle.dump(svm_clf, f)

    print("✓ Zapisano model SVM i metryki.")
    return svm_clf, metrics


def train_random_forest():
    print("\n" + "="*60)
    print("TRENING: RANDOM FOREST")
    print("="*60)

    X_train, y_train, X_test, y_test = load_data()
    y_train_r = y_train.ravel()
    y_test_r  = y_test.ravel()

    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        class_weight='balanced',
        oob_score=True,
        random_state=SEED,
        n_jobs=-1
    )
    rf.fit(X_train, y_train_r)

    y_proba = rf.predict_proba(X_test)[:, 1]
    y_pred  = (y_proba >= 0.5).astype(int)

    metrics = evaluate_and_save(
        y_test_r, y_pred, y_proba,
        out_json=os.path.join(RESULTS_DIR, 'rf_results.json'),
        out_txt=os.path.join(RESULTS_DIR, 'rf_report.txt')
    )

    metrics['train_accuracy'] = float(rf.score(X_train, y_train_r))
    metrics['test_accuracy']  = float(metrics.get('accuracy', np.nan))
    metrics['oob_score']      = float(rf.oob_score_) if hasattr(rf, "oob_score_") else None

    with open(os.path.join(RESULTS_DIR, 'rf_results.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    with open(os.path.join(MODELS_DIR, 'rf_model.pkl'), 'wb') as f:
        pickle.dump(rf, f)
    np.save(os.path.join(RESULTS_DIR, 'rf_feature_importances.npy'), rf.feature_importances_)

    print(f"✓ Zapisano model RF i metryki. OOB score: {metrics['oob_score']:.4f}")
    return rf, metrics

if __name__ == "__main__":
    all_results = {}

    try:
        nn_model, nn_metrics = train_neural_network()
        all_results['neural_network'] = nn_metrics
    except Exception as e:
        print(f"❌ Błąd NN: {e}")

    try:
        svm_model, svm_metrics = train_svm()
        all_results['svm'] = svm_metrics
    except Exception as e:
        print(f"❌ Błąd SVM: {e}")

    try:
        rf_model, rf_metrics = train_random_forest()
        all_results['random_forest'] = rf_metrics
    except Exception as e:
        print(f"❌ Błąd RF: {e}")

    print("\n" + "="*70)
    print("PODSUMOWANIE MODELI (TEST):")
    print("="*70)
    for name, res in all_results.items():
        acc = res.get('accuracy')
        f1  = res.get('f1')
        auc = res.get('roc_auc', None)
        pr  = res.get('pr_auc',  None)
        print(f"\n{name.upper()}:")
        print(f"  Acc:    {acc:.4f}")
        print(f"  F1:     {f1:.4f}")
        if auc is not None:
            print(f"  ROC-AUC:{auc:.4f} | PR-AUC:{pr:.4f}")
