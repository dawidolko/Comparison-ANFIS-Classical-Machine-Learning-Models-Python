import os
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

SEED = 42

FEATURE_COLUMNS_BASE = [
    'fixed acidity',
    'volatile acidity',
    'citric acid',
    'residual sugar',
    'chlorides',
    'free sulfur dioxide',
    'total sulfur dioxide',
    'density',
    'pH',
    'sulphates',
    'alcohol'
]

def _class_weight_from_labels(y):
    y = np.asarray(y).ravel()
    n0 = np.sum(y == 0)
    n1 = np.sum(y == 1)
    if n0 == 0 or n1 == 0:
        return None
    return {0: 1.0, 1: float(n0 / n1)}

def load_and_preprocess_data(add_type_feature=True, use_stratify_by_type=True):
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)

    red = pd.read_csv('data/winequality-red.csv', sep=';')
    white = pd.read_csv('data/winequality-white.csv', sep=';')

    red['type'] = 0
    white['type'] = 1

    df = pd.concat([red, white], axis=0, ignore_index=True)
    before = len(df)
    df = df.drop_duplicates(ignore_index=True)
    after = len(df)

    print(f"Całkowita liczba próbek: {after} (usunięto duplikatów: {before - after})")

    df['quality_binary'] = (df['quality'] > 5).astype(int)

    n0 = int((df['quality_binary'] == 0).sum())
    n1 = int((df['quality_binary'] == 1).sum())
    print("\n=== ROZKŁAD KLAS ===")
    print(f"Zła jakość (0): {n0}")
    print(f"Dobra jakość (1): {n1}")

    feature_cols = FEATURE_COLUMNS_BASE.copy()
    if add_type_feature:
        feature_cols.append('type')

    X = df[feature_cols].to_numpy(dtype=np.float32)
    y = df['quality_binary'].to_numpy(dtype=np.int32)

    if use_stratify_by_type:
        strat = list(zip(df['quality_binary'].to_numpy(), df['type'].to_numpy()))
    else:
        strat = y

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=SEED, stratify=strat
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train).astype(np.float32)
    X_test_s  = scaler.transform(X_test).astype(np.float32)

    with open('models/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    print(f"\nRozmiar zbioru treningowego: {X_train_s.shape}")
    print(f"Rozmiar zbioru testowego:    {X_test_s.shape}")

    class_weight = _class_weight_from_labels(y_train)

    meta = {
        "seed": SEED,
        "feature_columns": feature_cols,
        "label_rule": "quality_binary = (quality > 5)",
        "class_distribution": {"train": {
            "n0": int(np.sum(y_train == 0)),
            "n1": int(np.sum(y_train == 1))
        }, "test": {
            "n0": int(np.sum(y_test == 0)),
            "n1": int(np.sum(y_test == 1))
        }},
        "scaler": {
            "mean": scaler.mean_.tolist(),
            "scale": scaler.scale_.tolist()
        },
        "add_type_feature": add_type_feature,
        "use_stratify_by_type": use_stratify_by_type
    }
    with open('results/data_meta.json', 'w') as f:
        json.dump(meta, f, indent=2)

    return X_train_s, X_test_s, y_train, y_test, feature_cols, class_weight, meta


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, features, class_weight, meta = load_and_preprocess_data()

    np.save('data/X_train.npy', X_train.astype(np.float32))
    np.save('data/X_test.npy',  X_test.astype(np.float32))
    np.save('data/y_train.npy', y_train.astype(np.int32))
    np.save('data/y_test.npy',  y_test.astype(np.int32))

    with open('data/feature_columns.json', 'w') as f:
        json.dump(features, f, indent=2)

    if class_weight is not None:
        with open('data/class_weight.json', 'w') as f:
            json.dump(class_weight, f, indent=2)

    print("\n✓ Dane przetworzone i zapisane!")
