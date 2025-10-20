"""
Moduł pomocniczy do ładowania scalerów dla różnych wariantów modeli
Obsługuje zarówno scaler 11D (bez typu wina) jak i 12D (z typem wina)
"""

import pickle
import os
from typing import Tuple, Optional
from sklearn.preprocessing import StandardScaler


def load_scalers() -> Tuple[Optional[StandardScaler], Optional[StandardScaler]]:
    """
    Ładuje scalery dla danych 11D i 12D
    
    Returns:
        Tuple[scaler_11D, scaler_12D]:
            - scaler_11D: StandardScaler dla 11 cech (bez typu wina)
            - scaler_12D: StandardScaler dla 12 cech (z typem wina)
            Jeśli plik nie istnieje, zwraca None
    """
    scaler_11 = None
    scaler_12 = None
    
    # Główny scaler (11D - standardowy dla ANFIS)
    scaler_path_11 = 'models/scaler.pkl'
    if os.path.exists(scaler_path_11):
        try:
            with open(scaler_path_11, 'rb') as f:
                scaler_11 = pickle.load(f)
            print(f"✓ Załadowano scaler 11D: {scaler_path_11}")
        except Exception as e:
            print(f"⚠ Błąd ładowania scaler 11D: {e}")
    
    # Scaler dla Neural Network (12D - z typem wina)
    scaler_path_12 = 'models/scaler_nn.pkl'
    if os.path.exists(scaler_path_12):
        try:
            with open(scaler_path_12, 'rb') as f:
                scaler_12 = pickle.load(f)
            print(f"✓ Załadowano scaler 12D: {scaler_path_12}")
        except Exception as e:
            print(f"⚠ Błąd ładowania scaler 12D: {e}")
    
    # Fallback: jeśli nie ma scaler_12, użyj scaler_11 dla pierwszych 11 cech
    if scaler_11 is None and scaler_12 is None:
        print("⚠ Brak scalerów! Modele mogą działać niepoprawnie.")
    
    return scaler_11, scaler_12


def get_scaler_11d() -> Optional[StandardScaler]:
    """
    Ładuje tylko scaler 11D (dla ANFIS)
    
    Returns:
        StandardScaler lub None jeśli nie istnieje
    """
    scaler_path = 'models/scaler.pkl'
    if os.path.exists(scaler_path):
        try:
            with open(scaler_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"⚠ Błąd ładowania scaler 11D: {e}")
    return None


def get_scaler_12d() -> Optional[StandardScaler]:
    """
    Ładuje tylko scaler 12D (dla NN/SVM/RF)
    
    Returns:
        StandardScaler lub None jeśli nie istnieje
    """
    scaler_path = 'models/scaler_nn.pkl'
    if os.path.exists(scaler_path):
        try:
            with open(scaler_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"⚠ Błąd ładowania scaler 12D: {e}")
    return None
