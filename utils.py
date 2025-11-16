"""
Moduł pomocniczy dla aplikacji Streamlit
----------------------------------------
Zawiera funkcje:
 - ładowania modelu ANFIS z pliku wag
 - automatycznego wykrywania wyników modeli z katalogu results/
"""

import os
import json
import numpy as np
import tensorflow as tf
import h5py
from typing import Tuple, Optional, Dict, Any
from anfis import ANFISModel


# =====================================================
# 1️⃣  ŁADOWANIE MODELU ANFIS
# =====================================================
def load_anfis_model(weights_path: str) -> Optional[ANFISModel]:
    """
    Ładuje wytrenowany model ANFIS z pliku wag.
    
    Automatycznie wykrywa konfigurację modelu (liczbę wejść i funkcji przynależności)
    poprzez analizę struktury pliku HDF5.
    
    Args:
        weights_path: ścieżka do pliku z wagami (.h5, .weights.h5, .keras)
        
    Returns:
        Zrekonstruowany obiekt ANFISModel lub None w przypadku błędu
    """
    try:
        if not os.path.exists(weights_path):
            print(f"❌ Plik nie istnieje: {weights_path}")
            return None

        n_input = None
        n_memb = None

        with h5py.File(weights_path, "r") as f:
            # próbuj znaleźć parametry fuzzy_layer
            for key in f.keys():
                if "fuzzy_layer" in key:
                    grp = f[key]
                    if "c:0" in grp:
                        n_memb, n_input = grp["c:0"].shape
                        break
                    elif "sigma:0" in grp:
                        n_memb, n_input = grp["sigma:0"].shape
                        break

            # fallback — często w custom layers
            if n_input is None:
                for name in f.keys():
                    if "layer_with_weights" in name.lower():
                        try:
                            weights_shapes = [v.shape for v in f[name].values()]
                            for s in weights_shapes:
                                if len(s) == 2 and s[0] < 20:  # typowe MF
                                    n_memb, n_input = s
                                    break
                        except Exception:
                            pass

        if n_input is None or n_memb is None:
            print(f"⚠️ Nie udało się wykryć n_input/n_memb w {weights_path}")
            return None

        print(f"✓ Wykryto konfigurację: n_input={n_input}, n_memb={n_memb}")
        model = ANFISModel(n_input=int(n_input), n_memb=int(n_memb))
        _ = model.model(tf.zeros((1, int(n_input)), dtype=tf.float32))
        model.model.load_weights(weights_path)
        print(f"✓ Załadowano model z {weights_path}")
        return model

    except Exception as e:
        print(f"❌ Błąd podczas ładowania ANFIS: {e}")
        import traceback
        traceback.print_exc()
        return None


# =====================================================
# 2️⃣  ŁADOWANIE WYNIKÓW
# =====================================================
def load_results() -> Dict[str, Any]:
    """
    Ładuje wyniki wszystkich wytrenowanych modeli z katalogu results/.
    
    Obsługuje:
    - Modele ANFIS (klasyfikacja wine + regresja concrete)
    - Modele porównawcze (Neural Network, SVM, Random Forest)
    
    Wyniki są wczytywane z plików JSON i zwracane jako słownik.
    
    Returns:
        Dict[nazwa_modelu, dane_json] - słownik z wynikami wszystkich modeli
    """
    results = {}
    base_dir = "results"
    if not os.path.exists(base_dir):
        print("⚠️ Brak katalogu results/")
        return results

    # ręczne przypisania (dla kompatybilności GUI)
    manual_map = {
        "ANFIS (2 funkcje)": "anfis_all_2memb_results.json",
        "ANFIS (3 funkcje)": "anfis_all_3memb_results.json",
        "ANFIS Concrete (2)": "anfis_concrete_2memb_results.json",
        "ANFIS Concrete (3)": "anfis_concrete_3memb_results.json",
        "Neural Network": "nn_results.json",
        "SVM": "svm_results.json",
        "Random Forest": "rf_results.json"
    }

    # sprawdź wszystkie wpisy
    for name, filename in manual_map.items():
        path = os.path.join(base_dir, filename)
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    results[name] = json.load(f)
            except Exception as e:
                print(f"⚠️ Błąd przy wczytywaniu {filename}: {e}")

    # dodatkowo — autodiscover wszystkich wyników anfis_*.json
    for fn in os.listdir(base_dir):
        if fn.startswith("anfis_") and fn.endswith(".json"):
            full = os.path.join(base_dir, fn)
            try:
                key = fn.replace("_results.json", "").replace("anfis_", "ANFIS ").replace("_", " ")
                if key not in results:
                    with open(full, "r") as f:
                        results[key] = json.load(f)
            except Exception:
                pass

    print(f"✓ Załadowano {len(results)} zestawów wyników.")
    return results
