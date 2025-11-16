"""
Moduł pomocniczy do ładowania scalerów dla różnych wariantów modeli  
Obsługuje zarówno scaler 11D (bez typu wina), jak i 12D (z typem wina)
"""

import pickle
import os
from typing import Tuple, Optional
from sklearn.preprocessing import StandardScaler


def load_scalers(dataset: str = 'wine') -> Tuple[Optional[StandardScaler], Optional[StandardScaler]]:
    """
    Załadowano scalery dla podanego zbioru danych.

    Args:
        dataset: 'wine' (domyślnie) lub 'concrete' – określono na tej podstawie ścieżki plików

    Returns:
        Zwrócono krotkę: (scaler_11D, scaler_12D)
    """
    scaler_11 = None
    scaler_12 = None

    # Zdefiniowano ścieżki domyślne dla zbioru wine
    if dataset == 'wine':
        scaler_path_11 = os.path.join('models', 'scaler.pkl')
        scaler_path_12 = os.path.join('models', 'scaler_nn.pkl')
    else:
        # Dla pozostałych zbiorów (np. concrete) przyjęto strukturę: models/<dataset>/
        scaler_path_11 = os.path.join('models', dataset, 'scaler.pkl')
        scaler_path_12 = os.path.join('models', dataset, 'scaler_nn.pkl')

    if os.path.exists(scaler_path_11):
        try:
            with open(scaler_path_11, 'rb') as f:
                scaler_11 = pickle.load(f)
            print(f"✓ Załadowano scaler 11D: {scaler_path_11}")
        except Exception as e:
            print(f"⚠ Wystąpił błąd podczas ładowania scaler 11D: {e}")

    if os.path.exists(scaler_path_12):
        try:
            with open(scaler_path_12, 'rb') as f:
                scaler_12 = pickle.load(f)
            print(f"✓ Załadowano scaler 12D: {scaler_path_12}")
        except Exception as e:
            print(f"⚠ Wystąpił błąd podczas ładowania scaler 12D: {e}")

    # W przypadku braku obu scalerów nie wyświetlano ostrzeżenia na poziomie modułu  
    # (powiadomienie powinno zostać wygenerowane przez interfejs użytkownika w odpowiednim kontekście)

    return scaler_11, scaler_12


def get_scaler_11d(dataset: str = 'wine') -> Optional[StandardScaler]:
    """
    Załadowano wyłącznie scaler 11D (przeznaczony dla modelu ANFIS).
    
    Returns:
        Zwrócono obiekt StandardScaler lub None, jeśli plik nie istnieje.
    """
    if dataset == 'wine':
        scaler_path = os.path.join('models', 'scaler.pkl')
    else:
        scaler_path = os.path.join('models', dataset, 'scaler.pkl')
    if os.path.exists(scaler_path):
        try:
            with open(scaler_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"⚠ Wystąpił błąd podczas ładowania scaler 11D: {e}")
    return None


def get_scaler_12d(dataset: str = 'wine') -> Optional[StandardScaler]:
    """
    Załadowano wyłącznie scaler 12D (przeznaczony dla modeli NN/SVM/RF).
    
    Returns:
        Zwrócono obiekt StandardScaler lub None, jeśli plik nie istnieje.
    """
    if dataset == 'wine':
        scaler_path = os.path.join('models', 'scaler_nn.pkl')
    else:
        scaler_path = os.path.join('models', dataset, 'scaler_nn.pkl')
    if os.path.exists(scaler_path):
        try:
            with open(scaler_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"⚠ Wystąpił błąd podczas ładowania scaler 12D: {e}")
    return None