"""
Moduł pomocniczy do ładowania scalerów dla różnych wariantów modeli
Obsługuje zarówno scaler 11D (bez typu wina) jak i 12D (z typem wina)
"""

import pickle
import os
from typing import Tuple, Optional
from sklearn.preprocessing import StandardScaler


def load_scalers(dataset: str = 'wine') -> Tuple[Optional[StandardScaler], Optional[StandardScaler]]:
    """
    Ładuje scalery dla danego datasetu.

    Args:
        dataset: 'wine' (domyślnie) lub 'concrete' - wpływa na ścieżki plików

    Returns:
        Tuple[scaler_11D, scaler_12D]
    """
    scaler_11 = None
    scaler_12 = None

    # Domyślne ścieżki dla wine
    if dataset == 'wine':
        scaler_path_11 = os.path.join('models', 'scaler.pkl')
        scaler_path_12 = os.path.join('models', 'scaler_nn.pkl')
    else:
        # Dla innych datasetów (np. concrete) spodziewamy się katalogu models/<dataset>/
        scaler_path_11 = os.path.join('models', dataset, 'scaler.pkl')
        scaler_path_12 = os.path.join('models', dataset, 'scaler_nn.pkl')

    if os.path.exists(scaler_path_11):
        try:
            with open(scaler_path_11, 'rb') as f:
                scaler_11 = pickle.load(f)
            print(f"✓ Załadowano scaler 11D: {scaler_path_11}")
        except Exception as e:
            print(f"⚠ Błąd ładowania scaler 11D: {e}")

    if os.path.exists(scaler_path_12):
        try:
            with open(scaler_path_12, 'rb') as f:
                scaler_12 = pickle.load(f)
            print(f"✓ Załadowano scaler 12D: {scaler_path_12}")
        except Exception as e:
            print(f"⚠ Błąd ładowania scaler 12D: {e}")

    # Jeśli oba scalery są puste - nie wypisujemy globalnego ostrzeżenia tutaj
    # (GUI powinno pokazać informację w kontekście działania użytkownika)

    return scaler_11, scaler_12


def get_scaler_11d(dataset: str = 'wine') -> Optional[StandardScaler]:
    """
    Ładuje tylko scaler 11D (dla ANFIS)
    
    Returns:
        StandardScaler lub None jeśli nie istnieje
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
            print(f"⚠ Błąd ładowania scaler 11D: {e}")
    return None


def get_scaler_12d(dataset: str = 'wine') -> Optional[StandardScaler]:
    """
    Ładuje tylko scaler 12D (dla NN/SVM/RF)
    
    Returns:
        StandardScaler lub None jeśli nie istnieje
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
            print(f"⚠ Błąd ładowania scaler 12D: {e}")
    return None
