"""
Moduł pomocniczy do ładowania scalerów dla różnych wariantów modeli  
Obsługuje zarówno scaler 11D (bez typu wina), jak i 12D (z typem wina)
"""

import pickle  # Biblioteka do serializacji i deserializacji obiektów Pythona
import os  # Biblioteka do operacji na systemie plików
from typing import Tuple, Optional  # Typy do adnotacji typu (krotka i opcjonalna wartość)
from sklearn.preprocessing import StandardScaler  # Klasa do standaryzacji danych (mean=0, std=1)


def load_scalers(dataset: str = 'wine') -> Tuple[Optional[StandardScaler], Optional[StandardScaler]]:
    """
    Załadowano scalery dla podanego zbioru danych.

    Args:
        dataset: 'wine' (domyślnie) lub 'concrete' – określono na tej podstawie ścieżki plików

    Returns:
        Zwrócono krotkę: (scaler_11D, scaler_12D)
    """
    scaler_11 = None  # Inicjalizuje zmienną dla scalera 11D jako None
    scaler_12 = None  # Inicjalizuje zmienną dla scalera 12D jako None

    # Zdefiniowano ścieżki domyślne dla zbioru wine
    if dataset == 'wine':  # Sprawdza czy zbiór to wine
        scaler_path_11 = os.path.join('models', 'scaler.pkl')  # Ścieżka do scalera 11D dla wine (bez typu wina)
        scaler_path_12 = os.path.join('models', 'scaler_nn.pkl')  # Ścieżka do scalera 12D dla wine (z typem wina)
    else:  # Dla pozostałych zbiorów (np. concrete)
        # Dla pozostałych zbiorów (np. concrete) przyjęto strukturę: models/<dataset>/
        scaler_path_11 = os.path.join('models', dataset, 'scaler.pkl')  # Ścieżka do scalera 11D dla innych zbiorów
        scaler_path_12 = os.path.join('models', dataset, 'scaler_nn.pkl')  # Ścieżka do scalera 12D dla innych zbiorów

    if os.path.exists(scaler_path_11):  # Sprawdza czy plik scalera 11D istnieje
        try:  # Próbuje załadować scaler z pliku
            with open(scaler_path_11, 'rb') as f:  # Otwiera plik w trybie binarnym do odczytu
                scaler_11 = pickle.load(f)  # Deserializuje obiekt StandardScaler z pliku
            print(f"✓ Załadowano scaler 11D: {scaler_path_11}")  # Informuje o pomyślnym załadowaniu
        except Exception as e:  # Łapie wszelkie wyjątki podczas ładowania
            print(f"⚠ Wystąpił błąd podczas ładowania scaler 11D: {e}")  # Wypisuje komunikat o błędzie

    if os.path.exists(scaler_path_12):  # Sprawdza czy plik scalera 12D istnieje
        try:  # Próbuje załadować scaler z pliku
            with open(scaler_path_12, 'rb') as f:  # Otwiera plik w trybie binarnym do odczytu
                scaler_12 = pickle.load(f)  # Deserializuje obiekt StandardScaler z pliku
            print(f"✓ Załadowano scaler 12D: {scaler_path_12}")  # Informuje o pomyślnym załadowaniu
        except Exception as e:  # Łapie wszelkie wyjątki podczas ładowania
            print(f"⚠ Wystąpił błąd podczas ładowania scaler 12D: {e}")  # Wypisuje komunikat o błędzie

    # W przypadku braku obu scalerów nie wyświetlano ostrzeżenia na poziomie modułu  
    # (powiadomienie powinno zostać wygenerowane przez interfejs użytkownika w odpowiednim kontekście)

    return scaler_11, scaler_12  # Zwraca obiekty scalerów (11D, 12D) lub None jeśli nie znaleziono


def get_scaler_11d(dataset: str = 'wine') -> Optional[StandardScaler]:
    """
    Załadowano wyłącznie scaler 11D (przeznaczony dla modelu ANFIS).
    
    Returns:
        Zwrócono obiekt StandardScaler lub None, jeśli plik nie istnieje.
    """
    if dataset == 'wine':  # Sprawdza czy zbiór to wine
        scaler_path = os.path.join('models', 'scaler.pkl')  # Ustawia ścieżkę dla wine
    else:  # Dla innych zbiorów
        scaler_path = os.path.join('models', dataset, 'scaler.pkl')  # Ustawia ścieżkę w podkatalogu dataset
    if os.path.exists(scaler_path):  # Sprawdza czy plik istnieje
        try:  # Próbuje załadować scaler
            with open(scaler_path, 'rb') as f:  # Otwiera plik w trybie binarnym
                return pickle.load(f)  # Zwraca zdeserializowany obiekt StandardScaler
        except Exception as e:  # Łapie wyjątki podczas ładowania
            print(f"⚠ Wystąpił błąd podczas ładowania scaler 11D: {e}")  # Wypisuje błąd
    return None  # Zwraca None jeśli plik nie istnieje lub wystąpił błąd


def get_scaler_12d(dataset: str = 'wine') -> Optional[StandardScaler]:
    """
    Załadowano wyłącznie scaler 12D (przeznaczony dla modeli NN/SVM/RF).
    
    Returns:
        Zwrócono obiekt StandardScaler lub None, jeśli plik nie istnieje.
    """
    if dataset == 'wine':  # Sprawdza czy zbiór to wine
        scaler_path = os.path.join('models', 'scaler_nn.pkl')  # Ustawia ścieżkę dla wine (scaler z typem wina)
    else:  # Dla innych zbiorów
        scaler_path = os.path.join('models', dataset, 'scaler_nn.pkl')  # Ustawia ścieżkę w podkatalogu dataset
    if os.path.exists(scaler_path):  # Sprawdza czy plik istnieje
        try:  # Próbuje załadować scaler
            with open(scaler_path, 'rb') as f:  # Otwiera plik w trybie binarnym
                return pickle.load(f)  # Zwraca zdeserializowany obiekt StandardScaler
        except Exception as e:  # Łapie wyjątki podczas ładowania
            print(f"⚠ Wystąpił błąd podczas ładowania scaler 12D: {e}")  # Wypisuje błąd
    return None  # Zwraca None jeśli plik nie istnieje lub wystąpił błąd