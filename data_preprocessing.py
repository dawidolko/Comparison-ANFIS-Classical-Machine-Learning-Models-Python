import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import json


def load_and_preprocess_data():
    """
    Ładuje i przetwarza dane Wine Quality (czerwone + białe wino).
    
    Tworzy trzy zestawy danych:
    - 'all': połączone czerwone i białe wino
    - 'red': tylko czerwone wino
    - 'white': tylko białe wino
    
    Dla każdego zestawu:
    - Binaryzuje jakość (quality > 5 = 1, inaczej 0)
    - Normalizuje cechy (StandardScaler)
    - Dzieli na zbiór treningowy i testowy (80/20)
    - Zapisuje jako pliki .npy i scaler jako .pkl
    
    Returns:
        Dict ze statystykami każdego zestawu danych
    """
    csv_red = 'data/wine-quality/winequality-red.csv' if os.path.exists('data/wine-quality/winequality-red.csv') else 'data/winequality-red.csv'
    csv_white = 'data/wine-quality/winequality-white.csv' if os.path.exists('data/wine-quality/winequality-white.csv') else 'data/winequality-white.csv'
    
    red_wine = pd.read_csv(csv_red, sep=';')
    white_wine = pd.read_csv(csv_white, sep=';')
    
    red_wine['wine_type'] = 'red'
    white_wine['wine_type'] = 'white'
    wine_data = pd.concat([red_wine, white_wine], axis=0, ignore_index=True)
    
    feature_columns = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
                      'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
                      'pH', 'sulphates', 'alcohol']
    
    wine_data['quality_binary'] = (wine_data['quality'] > 5).astype(int)
    
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    datasets_info = {}
    for dataset_name in ['all', 'red', 'white']:
        if dataset_name == 'all':
            df = wine_data
        elif dataset_name == 'red':
            df = wine_data[wine_data['wine_type'] == 'red']
        else:
            df = wine_data[wine_data['wine_type'] == 'white']
        
        X = df[feature_columns].values
        y = df['quality_binary'].values
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        if dataset_name == 'all':
            np.save('data/X_train.npy', X_train)
            np.save('data/X_test.npy', X_test)
            np.save('data/y_train.npy', y_train)
            np.save('data/y_test.npy', y_test)
            with open('models/scaler.pkl', 'wb') as f:
                pickle.dump(scaler, f)
        else:
            np.save(f'data/X_train_{dataset_name}.npy', X_train)
            np.save(f'data/X_test_{dataset_name}.npy', X_test)
            np.save(f'data/y_train_{dataset_name}.npy', y_train)
            np.save(f'data/y_test_{dataset_name}.npy', y_test)
            with open(f'models/scaler_{dataset_name}.pkl', 'wb') as f:
                pickle.dump(scaler, f)
        
        datasets_info[dataset_name] = {
            'n_samples': int(len(df)),
            'n_train': int(len(X_train)),
            'n_test': int(len(X_test)),
            'n_features': int(len(feature_columns)),
            'class_0': int((y == 0).sum()),
            'class_1': int((y == 1).sum())
        }
        
        print(f"Dataset {dataset_name}: {len(df)} samples, train={len(X_train)}, test={len(X_test)}")
    
    with open('data/datasets_summary.json', 'w') as f:
        json.dump(datasets_info, f, indent=2)
    
    print("\n✓ Dane dla all, red, white zapisane!")
    return datasets_info


def load_and_preprocess_concrete():
    """
    Ładuje i przetwarza dane Concrete Strength (wytrzymałość betonu).
    
    Proces:
    - Wczytuje dane z CSV (8 cech + 1 cel: wytrzymałość w MPa)
    - Normalizuje cechy (StandardScaler)
    - Dzieli na treningowy/testowy (80/20)
    - Zapisuje do data/concrete-strength/*.npy
    - Zapisuje scaler do models/concrete-strength/scaler.pkl
    
    Returns:
        Dict z informacjami o zestawie lub None jeśli brak pliku
    """
    csv_path = 'data/concrete-strength/Concrete_Data.csv'
    if not os.path.exists(csv_path):
        print(f"⚠ Brak {csv_path}")
        return None
    
    df = pd.read_csv(csv_path)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    os.makedirs('data/concrete-strength', exist_ok=True)
    os.makedirs('models/concrete-strength', exist_ok=True)
    
    np.save('data/concrete-strength/X_train.npy', X_train)
    np.save('data/concrete-strength/X_test.npy', X_test)
    np.save('data/concrete-strength/y_train.npy', y_train)
    np.save('data/concrete-strength/y_test.npy', y_test)
    
    with open('models/concrete-strength/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    info = {'n_samples': len(df), 'n_train': len(X_train), 'n_test': len(X_test), 'n_features': X.shape[1]}
    print(f"Concrete: {len(df)} samples, train={len(X_train)}, test={len(X_test)}")
    print("✓ Dane concrete zapisane!")
    return info


if __name__ == "__main__":
    load_and_preprocess_data()
    load_and_preprocess_concrete()