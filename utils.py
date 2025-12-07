"""
Moduł pomocniczy dla aplikacji Streamlit
----------------------------------------
Zawiera funkcje:
 - ładowania modelu ANFIS z pliku wag
 - automatycznego wykrywania wyników modeli z katalogu results/
"""

import os  # Biblioteka do operacji na systemie plików
import json  # Biblioteka do obsługi plików JSON
import numpy as np  # Biblioteka do operacji na tablicach numerycznych
import tensorflow as tf  # Framework do budowy i treningu sieci neuronowych
import h5py  # Biblioteka do odczytu plików HDF5 (format zapisu wag Keras)
from typing import Tuple, Optional, Dict, Any  # Typy do adnotacji (krotka, opcjonalna wartość, słownik, dowolny typ)
from anfis import ANFISModel  # Importuje klasę głównego modelu ANFIS


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
    try:  # Próbuje załadować model - obsługuje wszelkie wyjątki
        if not os.path.exists(weights_path):  # Sprawdza czy plik z wagami istnieje
            print(f"❌ Plik nie istnieje: {weights_path}")  # Wypisuje błąd jeśli nie znaleziono pliku
            return None  # Zwraca None jeśli plik nie istnieje

        n_input = None  # Inicjalizuje zmienną dla liczby cech wejściowych
        n_memb = None  # Inicjalizuje zmienną dla liczby funkcji przynależności

        with h5py.File(weights_path, "r") as f:  # Otwiera plik HDF5 w trybie odczytu
            # próba znalezienia parametru fuzzy_layer
            for key in f.keys():  # Iteruje przez wszystkie klucze w pliku HDF5
                if "fuzzy_layer" in key:  # Sprawdza czy klucz zawiera nazwę warstwy fuzzyfikacji
                    grp = f[key]  # Pobiera grupę danych dla znalezionej warstwy
                    if "c:0" in grp:  # Sprawdza czy istnieją centra funkcji gaussowskich
                        n_memb, n_input = grp["c:0"].shape  # Ekstrahuje wymiary: (liczba_MF, liczba_cech)
                        break  # Kończy pętlę po znalezieniu parametrów
                    elif "sigma:0" in grp:  # Jeśli nie ma centrów, sprawdza sigmy
                        n_memb, n_input = grp["sigma:0"].shape  # Ekstrahuje wymiary z wag sigma
                        break  # Kończy pętlę po znalezieniu parametrów

            # fallback — często w custom layers
            if n_input is None:  # Jeśli nie znaleziono parametrów w fuzzy_layer, próbuje alternatywnej metody
                for name in f.keys():  # Iteruje przez wszystkie klucze w pliku
                    if "layer_with_weights" in name.lower():  # Szuka warstw z wagami (zapis Keras)
                        try:  # Próbuje wyciągnąć kształty wag
                            weights_shapes = [v.shape for v in f[name].values()]  # Pobiera kształty wszystkich wag w warstwie
                            for s in weights_shapes:  # Iteruje przez kształty
                                if len(s) == 2 and s[0] < 20:  # typowe MF
                                    n_memb, n_input = s  # Ekstrahuje wymiary jeśli spełniają kryteria (2D, mała pierwsza dimensja)
                                    break  # Kończy pętlę po znalezieniu
                        except Exception:  # Łapie wszelkie błędy podczas ekstrakcji
                            pass  # Ignoruje błędy i kontynuuje

        if n_input is None or n_memb is None:  # Sprawdza czy udało się wykryć obydwa parametry
            print(f"⚠️ Nie udało się wykryć n_input/n_memb w {weights_path}")  # Informuje o niepowodzeniu detekcji
            return None  # Zwraca None jeśli nie wykryto parametrów

        print(f"✓ Wykryto konfigurację: n_input={n_input}, n_memb={n_memb}")  # Informuje o wykrytych parametrach
        model = ANFISModel(n_input=int(n_input), n_memb=int(n_memb))  # Tworzy nową instancję modelu ANFIS z wykrytymi parametrami
        _ = model.model(tf.zeros((1, int(n_input)), dtype=tf.float32))  # Inicjalizuje model przez forward pass z zerowymi danymi
        model.model.load_weights(weights_path)  # Ładuje zapisane wagi do modelu
        print(f"✓ Załadowano model z {weights_path}")  # Informuje o pomyślnym załadowaniu
        return model  # Zwraca załadowany model

    except Exception as e:  # Łapie wszelkie wyjątki podczas ładowania modelu
        print(f"❌ Błąd podczas ładowania ANFIS: {e}")  # Wypisuje komunikat o błędzie
        import traceback  # Importuje moduł do drukowania stack trace
        traceback.print_exc()  # Wypisuje pełny stack trace dla debugowania
        return None  # Zwraca None w przypadku błędu


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
    results = {}  # Inicjalizuje pusty słownik do przechowywania wyników
    base_dir = "results"  # Definiuje katalog z plikami wyników
    if not os.path.exists(base_dir):  # Sprawdza czy katalog results/ istnieje
        print("⚠️ Brak katalogu results/")  # Informuje o braku katalogu
        return results  # Zwraca pusty słownik

    # ręczne przypisania (dla kompatybilności GUI)
    manual_map = {  # Słownik mapujący czytelne nazwy modeli na nazwy plików JSON
        "ANFIS (2 funkcje)": "anfis_all_2memb_results.json",  # ANFIS dla wine z 2 funkcjami przynależności
        "ANFIS (3 funkcje)": "anfis_all_3memb_results.json",  # ANFIS dla wine z 3 funkcjami przynależności
        "ANFIS Concrete (2)": "anfis_concrete_2memb_results.json",  # ANFIS dla betonu z 2 funkcjami
        "ANFIS Concrete (3)": "anfis_concrete_3memb_results.json",  # ANFIS dla betonu z 3 funkcjami
        "Neural Network": "nn_results.json",  # Wyniki sieci neuronowej
        "SVM": "svm_results.json",  # Wyniki Support Vector Machine
        "Random Forest": "rf_results.json"  # Wyniki Random Forest
    }

    # sprawdzenie wszystkich wpisów
    for name, filename in manual_map.items():  # Iteruje przez wszystkie zdefiniowane mapowania
        path = os.path.join(base_dir, filename)  # Tworzy pełną ścieżkę do pliku
        if os.path.exists(path):  # Sprawdza czy plik istnieje
            try:  # Próbuje załadować plik JSON
                with open(path, "r") as f:  # Otwiera plik w trybie odczytu
                    results[name] = json.load(f)  # Wczytuje dane JSON i zapisuje pod czytelnym kluczem
            except Exception as e:  # Łapie błędy podczas wczytywania
                print(f"⚠️ Błąd przy wczytywaniu {filename}: {e}")  # Informuje o błędzie

    # autodiscover wszystkich wyników anfis_*.json
    for fn in os.listdir(base_dir):  # Iteruje przez wszystkie pliki w katalogu results/
        if fn.startswith("anfis_") and fn.endswith(".json"):  # Filtruje pliki ANFIS w formacie JSON
            full = os.path.join(base_dir, fn)  # Tworzy pełną ścieżkę do pliku
            try:  # Próbuje załadować plik
                key = fn.replace("_results.json", "").replace("anfis_", "ANFIS ").replace("_", " ")  # Generuje czytelny klucz z nazwy pliku
                if key not in results:  # Sprawdza czy klucz nie został już dodany przez manual_map
                    with open(full, "r") as f:  # Otwiera plik w trybie odczytu
                        results[key] = json.load(f)  # Wczytuje dane JSON
            except Exception:  # Łapie wszelkie błędy
                pass  # Ignoruje błędy i kontynuuje

    print(f"✓ Załadowano {len(results)} zestawów wyników.")  # Informuje o liczbie załadowanych wyników
    return results  # Zwraca słownik z wynikami wszystkich modeli
