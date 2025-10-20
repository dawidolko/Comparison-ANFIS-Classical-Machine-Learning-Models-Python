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


def load_anfis_model(weights_path: str,
                     X11: np.ndarray,
                     X12: np.ndarray,
                     default_memb: int = 3,
                     verbose: bool = True) -> Tuple[Optional[float], str]:
    """
    Ładuje model ANFIS i wykonuje predykcję
    
    Args:
        weights_path: Ścieżka do pliku z wagami .h5
        X11: Dane wejściowe 11D (znormalizowane)
        X12: Dane wejściowe 12D (znormalizowane)
        default_memb: Domyślna liczba funkcji przynależności
        verbose: Czy wyświetlać szczegółowe informacje
    
    Returns:
        Tuple[prawdopodobieństwo, diagnostyka_string]
    """
    info = []

    try:
        if not os.path.exists(weights_path):
            info.append(f"[path] Brak pliku: {weights_path}")
            return None, "\n".join(info)

        info.append(f"[path] OK: {weights_path}")

        n_input = None
        n_memb = None

        with h5py.File(weights_path, "r") as hf:
            datasets = []

            def visitor(name, obj):
                if isinstance(obj, h5py.Dataset) and len(getattr(obj, "shape", ())) == 2:
                    datasets.append((name, tuple(obj.shape)))

            hf.visititems(visitor)

            info.append(f"[h5] Znalezione datasety 2D: {[(n, s) for n, s in datasets][:6]}{' ...' if len(datasets) > 6 else ''}")

            candidates = [(n, s) for n, s in datasets
                          if any(k in n.lower() for k in ("centre", "center", "centres", "centers"))]

            if not candidates:
                for n, (a, b) in datasets:
                    if (a in (11, 12) and b <= 8):
                        n_input, n_memb = a, b
                        info.append(f"[infer] z {n}: n_input={n_input}, n_memb={n_memb}")
                        break
                    if (b in (11, 12) and a <= 8):
                        n_input, n_memb = b, a
                        info.append(f"[infer] z {n}: n_input={n_input}, n_memb={n_memb}")
                        break
            else:
                picked = None
                for n, (a, b) in candidates:
                    if a in (11, 12) or b in (11, 12):
                        picked = (n, (a, b))
                        break
                if picked is None:
                    picked = candidates[0]
                n, (a, b) = picked
                if a >= b:
                    n_input, n_memb = a, b
                else:
                    n_input, n_memb = b, a
                info.append(f"[centres] {n} -> n_input={n_input}, n_memb={n_memb}")

        if n_input is None or n_memb is None:
            info.append("[shape] Nie udało się wywnioskować (n_input, n_memb) z pliku.")
            return None, "\n".join(info)

        if n_memb <= 0 or n_memb > 32:
            info.append(f"[shape] Podejrzana liczba MF: {n_memb} – wymuszam default {default_memb}")
            n_memb = default_memb

        info.append(f"[shape] Final: n_input={n_input}, n_memb={n_memb}")
        try:
            anfis_model = ANFISModel(n_input=int(n_input), n_memb=int(n_memb or default_memb))
        except Exception as e:
            info.append(f"[build] Błąd konstruktora ANFISModel: {e!r}")
            return None, "\n".join(info)

        try:
            dummy = tf.zeros((1, int(n_input)), dtype=tf.float32)
            _ = anfis_model.model(dummy)
            info.append("[build] Model wywołany na dummy (ok).")
        except Exception as e:
            info.append(f"[build] Błąd przy inicjalizacji grafu: {e!r}")
            return None, "\n".join(info)

        try:
            with h5py.File(weights_path, "r") as hf:
                Wf0 = np.array(hf["layers/fuzzy_layer/vars/0"])
                Wf1 = np.array(hf["layers/fuzzy_layer/vars/1"])
                Wd0 = np.array(hf["layers/defuzz_layer/vars/0"])
                Wd1 = np.array(hf["layers/defuzz_layer/vars/1"])

            try:
                fuzzy = anfis_model.model.get_layer("fuzzy")
            except Exception:
                fuzzy = next((lyr for lyr in anfis_model.model.layers
                              if "fuzzy" in lyr.name.lower()), None)
            try:
                defuzz = anfis_model.model.get_layer("defuzz")
            except Exception:
                defuzz = next((lyr for lyr in anfis_model.model.layers
                               if "defuzz" in lyr.name.lower()), None)

            if fuzzy is None:
                info.append("[weights] Nie znalazłem warstwy 'fuzzy'.")
                return None, "\n".join(info)
            if defuzz is None:
                info.append("[weights] Nie znalazłem warstwy 'defuzz'.")
                return None, "\n".join(info)

            fuzzy_cur = fuzzy.get_weights()
            fuzzy_shapes = [w.shape for w in fuzzy_cur]
            info.append(f"[weights] fuzzy oczekuje: {fuzzy_shapes}")
            new_fuzzy = None
            if len(fuzzy_shapes) == 2:
                if fuzzy_shapes[0] == Wf0.shape and fuzzy_shapes[1] == Wf1.shape:
                    new_fuzzy = [Wf0, Wf1]
                elif fuzzy_shapes[0] == Wf1.shape and fuzzy_shapes[1] == Wf0.shape:
                    new_fuzzy = [Wf1, Wf0]
            if new_fuzzy is None:
                info.append("[weights] Nie dopasowałem wag fuzzy (kształty się nie zgadzają).")
                return None, "\n".join(info)
            fuzzy.set_weights(new_fuzzy)
            info.append("[weights] Ustawiono wagi FUZZY.")

            defuzz_cur = defuzz.get_weights()
            defuzz_shapes = [w.shape for w in defuzz_cur]
            info.append(f"[weights] defuzz oczekuje: {defuzz_shapes}")

            b_row = Wd0
            b_vec = Wd0.reshape(-1, )
            A = Wd1
            A_T = Wd1.T

            new_defuzz = None
            candidates = [
                [A, b_vec],
                [b_vec, A],
                [A_T, b_vec],
                [b_vec, A_T],
            ]

            for cand in candidates:
                if [w.shape for w in cand] == defuzz_shapes:
                    new_defuzz = cand
                    break

            if new_defuzz is None:
                info.append("[weights] Nie dopasowałem wag defuzz (kształty nie pasują po konwersji).")
                return None, "\n".join(info)

            defuzz.set_weights(new_defuzz)
            info.append("[weights] Ustawiono wagi DEFUZZ (A i b dopasowane).")

        except Exception as e:
            info.append(f"[weights] Ręczne ładowanie wag z H5 nieudane: {e!r}")
            return None, "\n".join(info)

        X = X12 if int(n_input) == 12 else X11
        if not isinstance(X, np.ndarray):
            X = np.asarray(X)
        if X.ndim != 2 or X.shape[1] != int(n_input):
            info.append(f"[predict] Zły kształt X: {X.shape}, oczekiwano (N,{n_input})")
            return None, "\n".join(info)
        if X.dtype != np.float32:
            X = X.astype(np.float32)
            info.append(f"[predict] Rzutuję X na float32")

        try:
            y = anfis_model.model.predict(X, verbose=0)
            prob = float(np.ravel(y)[0])
            info.append(f"[predict] OK: prob={prob:.6f}")
            return prob, "\n".join(info)
        except Exception as e:
            info.append(f"[predict] Błąd predykcji: {e!r}")
            return None, "\n".join(info)

    except Exception as e:
        info.append(f"[fatal] Nieoczekiwany wyjątek: {e!r}")
        return None, "\n".join(info)


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
