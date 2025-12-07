import pandas as pd  # Biblioteka do operacji na ramkach danych (DataFrames)
import numpy as np  # Biblioteka do operacji na tablicach numerycznych
import os  # Biblioteka do operacji na systemie plików
from sklearn.model_selection import train_test_split  # Funkcja do podziału danych na treningowe/testowe
from sklearn.preprocessing import StandardScaler  # Klasa do standaryzacji danych (mean=0, std=1)
import pickle  # Biblioteka do serializacji obiektów Pythona
import json  # Biblioteka do obsługi plików JSON


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
    csv_red = 'data/wine-quality/winequality-red.csv' if os.path.exists('data/wine-quality/winequality-red.csv') else 'data/winequality-red.csv'  # Określa ścieżkę do pliku czerwonego wina
    csv_white = 'data/wine-quality/winequality-white.csv' if os.path.exists('data/wine-quality/winequality-white.csv') else 'data/winequality-white.csv'  # Określa ścieżkę do pliku białego wina
    
    red_wine = pd.read_csv(csv_red, sep=';')  # Ładuje dane czerwonego wina z CSV (separator ;)
    white_wine = pd.read_csv(csv_white, sep=';')  # Ładuje dane białego wina z CSV (separator ;)
    
    red_wine['wine_type'] = 'red'  # Dodaje kolumnę z typem wina dla czerwonego
    white_wine['wine_type'] = 'white'  # Dodaje kolumnę z typem wina dla białego
    wine_data = pd.concat([red_wine, white_wine], axis=0, ignore_index=True)  # Łączy oba zbiory danych w jeden DataFrame
    
    feature_columns = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
                      'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
                      'pH', 'sulphates', 'alcohol']  # Lista nazw kolumn z cechami wina (11 cech)
    
    wine_data['quality_binary'] = (wine_data['quality'] > 5).astype(int)  # Tworzy binarną etykietę: 1 jeśli quality > 5, 0 w przeciwnym razie
    
    os.makedirs('data', exist_ok=True)  # Tworzy katalog data jeśli nie istnieje
    os.makedirs('models', exist_ok=True)  # Tworzy katalog models jeśli nie istnieje
    
    datasets_info = {}  # Inicjalizuje słownik do przechowywania statystyk zbiorów
    for dataset_name in ['all', 'red', 'white']:  # Iteruje przez wszystkie 3 zestawy danych
        if dataset_name == 'all':  # Jeśli to połączone dane
            df = wine_data  # Używa pełnego zbioru (czerwone + białe)
        elif dataset_name == 'red':  # Jeśli to tylko czerwone wino
            df = wine_data[wine_data['wine_type'] == 'red']  # Filtruje tylko wiersze z czerwonym winem
        else:  # Jeśli to tylko białe wino
            df = wine_data[wine_data['wine_type'] == 'white']  # Filtruje tylko wiersze z białym winem
        
        X = df[feature_columns].values  # Pobiera macierz cech (11 kolumn) jako tablice numpy
        y = df['quality_binary'].values  # Pobiera wektor etykiet binarnych jako tablicę numpy
        
        X_train, X_test, y_train, y_test = train_test_split(  # Dzieli dane na treningowe i testowe
            X, y, test_size=0.2, random_state=42, stratify=y  # 80% trening, 20% test, stratyfikacja zachowuje proporcje klas
        )
        
        scaler = StandardScaler()  # Tworzy obiekt scalera do standaryzacji
        X_train = scaler.fit_transform(X_train)  # Dopasowuje scaler do danych treningowych i transformuje (mean=0, std=1)
        X_test = scaler.transform(X_test)  # Transformuje dane testowe używając już dopasowanego scalera
        
        if dataset_name == 'all':  # Jeśli to połączone dane
            np.save('data/X_train.npy', X_train)  # Zapisuje dane treningowe cech do pliku .npy
            np.save('data/X_test.npy', X_test)  # Zapisuje dane testowe cech do pliku .npy
            np.save('data/y_train.npy', y_train)  # Zapisuje etykiety treningowe do pliku .npy
            np.save('data/y_test.npy', y_test)  # Zapisuje etykiety testowe do pliku .npy
            with open('models/scaler.pkl', 'wb') as f:  # Otwiera plik do zapisu scalera
                pickle.dump(scaler, f)  # Serializuje i zapisuje scaler do pliku
        else:  # Dla czerwonego lub białego wina osobno
            np.save(f'data/X_train_{dataset_name}.npy', X_train)  # Zapisuje dane treningowe z sufiksem nazwy zestawu
            np.save(f'data/X_test_{dataset_name}.npy', X_test)  # Zapisuje dane testowe z sufiksem nazwy zestawu
            np.save(f'data/y_train_{dataset_name}.npy', y_train)  # Zapisuje etykiety treningowe z sufiksem
            np.save(f'data/y_test_{dataset_name}.npy', y_test)  # Zapisuje etykiety testowe z sufiksem
            with open(f'models/scaler_{dataset_name}.pkl', 'wb') as f:  # Otwiera plik do zapisu scalera z sufiksem
                pickle.dump(scaler, f)  # Serializuje i zapisuje scaler do pliku
        
        datasets_info[dataset_name] = {  # Tworzy słownik ze statystykami dla bieżącego zestawu
            'n_samples': int(len(df)),  # Całkowita liczba przykładów
            'n_train': int(len(X_train)),  # Liczba przykładów treningowych
            'n_test': int(len(X_test)),  # Liczba przykładów testowych
            'n_features': int(len(feature_columns)),  # Liczba cech (11)
            'class_0': int((y == 0).sum()),  # Liczba przykładów klasy 0 (quality <= 5)
            'class_1': int((y == 1).sum())  # Liczba przykładów klasy 1 (quality > 5)
        }
        
        print(f"Dataset {dataset_name}: {len(df)} samples, train={len(X_train)}, test={len(X_test)}")  # Wypisuje statystyki dla bieżącego zestawu
    
    with open('data/datasets_summary.json', 'w') as f:  # Otwiera plik JSON do zapisu podsumowania
        json.dump(datasets_info, f, indent=2)  # Zapisuje słownik ze statystykami do JSON z wczęciami
    
    print("\n✓ Dane dla all, red, white zapisane!")  # Informuje o pomyślnym zapisie wszystkich zbiorów
    return datasets_info  # Zwraca słownik ze statystykami


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
    csv_path = 'data/concrete-strength/Concrete_Data.csv'  # Definiuje ścieżkę do pliku CSV z danymi betonu
    if not os.path.exists(csv_path):  # Sprawdza czy plik istnieje
        print(f"⚠ Brak {csv_path}")  # Informuje o braku pliku
        return None  # Zwraca None jeśli plik nie istnieje
    
    df = pd.read_csv(csv_path)  # Ładuje dane betonu z CSV (8 cech + 1 kolumna wyjściowa)
    X = df.iloc[:, :-1].values  # Pobiera wszystkie kolumny oprócz ostatniej jako cechy
    y = df.iloc[:, -1].values  # Pobiera ostatnią kolumnę jako wartości docelowe (wytrzymałość w MPa)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # Dzieli dane na treningowe (80%) i testowe (20%)
    
    scaler = StandardScaler()  # Tworzy obiekt scalera do standaryzacji
    X_train = scaler.fit_transform(X_train)  # Dopasowuje scaler do danych treningowych i transformuje
    X_test = scaler.transform(X_test)  # Transformuje dane testowe używając dopasowanego scalera
    
    os.makedirs('data/concrete-strength', exist_ok=True)  # Tworzy katalog dla danych betonu jeśli nie istnieje
    os.makedirs('models/concrete-strength', exist_ok=True)  # Tworzy katalog dla modeli betonu jeśli nie istnieje
    
    np.save('data/concrete-strength/X_train.npy', X_train)  # Zapisuje dane treningowe cech do pliku .npy
    np.save('data/concrete-strength/X_test.npy', X_test)  # Zapisuje dane testowe cech do pliku .npy
    np.save('data/concrete-strength/y_train.npy', y_train)  # Zapisuje wartości treningowe do pliku .npy
    np.save('data/concrete-strength/y_test.npy', y_test)  # Zapisuje wartości testowe do pliku .npy
    
    with open('models/concrete-strength/scaler.pkl', 'wb') as f:  # Otwiera plik do zapisu scalera
        pickle.dump(scaler, f)  # Serializuje i zapisuje scaler do pliku
    
    info = {'n_samples': len(df), 'n_train': len(X_train), 'n_test': len(X_test), 'n_features': X.shape[1]}  # Tworzy słownik ze statystykami
    print(f"Concrete: {len(df)} samples, train={len(X_train)}, test={len(X_test)}")  # Wypisuje statystyki
    print("✓ Dane concrete zapisane!")  # Informuje o pomyślnym zapisie
    return info  # Zwraca statystyki


if __name__ == "__main__":  # Sprawdza czy skrypt jest uruchamiany bezpośrednio
    load_and_preprocess_data()  # Uruchamia przetwarzanie danych wina
    load_and_preprocess_concrete()