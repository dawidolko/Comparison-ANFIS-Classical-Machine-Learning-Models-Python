"""
Moduł pomocniczy dla aplikacji Streamlit
Zawiera funkcje ładowania modeli i predykcji
"""

import os
import json
import numpy as np
import tensorflow as tf
import h5py
from typing import Tuple, Optional, Dict, Any
from anfis import ANFISModel


def load_anfis_model(weights_path: str) -> Optional[ANFISModel]:
    """
    Ładuje model ANFIS z pliku wag .h5 lub .weights.h5
    Automatycznie wykrywa n_input i n_memb z struktury pliku
    
    Args:
        weights_path: Ścieżka do pliku z wagami
    
    Returns:
        Załadowany model ANFISModel lub None w przypadku błędu
    """
    try:
        if not os.path.exists(weights_path):
            print(f"❌ Plik nie istnieje: {weights_path}")
            return None
        
        # Wykryj n_input i n_memb z pliku H5
        n_input = None
        n_memb = None
        
        with h5py.File(weights_path, 'r') as f:
            # Szukaj warstwy fuzzy_layer
            if 'fuzzy_layer' in f:
                # Próbuj znaleźć c (centres) i sigma
                if 'c:0' in f['fuzzy_layer']:
                    shape = f['fuzzy_layer']['c:0'].shape
                    n_memb, n_input = shape  # (n_memb, n_input)
                elif 'sigma:0' in f['fuzzy_layer']:
                    shape = f['fuzzy_layer']['sigma:0'].shape
                    n_memb, n_input = shape
            
            # Jeśli nie znaleziono, spróbuj przez vars
            if n_input is None and 'layers' in f and 'fuzzy_layer' in f['layers']:
                if 'vars' in f['layers']['fuzzy_layer']:
                    vars_group = f['layers']['fuzzy_layer']['vars']
                    if '0' in vars_group:
                        shape = vars_group['0'].shape
                        n_memb, n_input = shape
        
        if n_input is None or n_memb is None:
            print(f"❌ Nie można wykryć n_input/n_memb z {weights_path}")
            return None
        
        print(f"✓ Wykryto: n_input={n_input}, n_memb={n_memb}")
        
        # Utwórz model
        model = ANFISModel(n_input=int(n_input), n_memb=int(n_memb))
        
        # Zbuduj graf przez dummy forward pass
        dummy_input = tf.zeros((1, int(n_input)), dtype=tf.float32)
        _ = model.model(dummy_input)
        
        # Załaduj wagi
        model.model.load_weights(weights_path)
        print(f"✓ Model załadowany: {weights_path}")
        
        return model
        
    except Exception as e:
        print(f"❌ Błąd ładowania modelu: {e}")
        import traceback
        traceback.print_exc()
        return None


def load_results() -> Dict[str, Any]:
    """
    Ładuje wyniki wszystkich modeli z plików JSON
    
    Returns:
        Słownik z wynikami {nazwa_modelu: dane_JSON}
    """
    results = {}
    result_files = {
        'ANFIS (2 funkcje)': 'results/anfis_2memb_results.json',
        'ANFIS (3 funkcje)': 'results/anfis_3memb_results.json',
        'Neural Network': 'results/nn_results.json',
        'SVM': 'results/svm_results.json',
        'Random Forest': 'results/rf_results.json'
    }

    for name, path in result_files.items():
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    results[name] = json.load(f)
            except Exception as e:
                print(f"⚠ Błąd wczytywania {path}: {e}")

    return results
