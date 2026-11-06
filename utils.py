"""
Helper module for the Streamlit app
----------------------------------------
Contains functions for:
 - loading the ANFIS model from a weights file
 - automatically detecting model results in the results/ directory
"""

import os
import json
import numpy as np
import tensorflow as tf
import h5py
from typing import Tuple, Optional, Dict, Any
from anfis import ANFISModel


# =====================================================
# 1.  LOADING THE ANFIS MODEL
# =====================================================
def load_anfis_model(weights_path: str) -> Optional[ANFISModel]:
    """
    Load a trained ANFIS model from a weights file.

    Automatically detects the model configuration (number of inputs and
    membership functions) by analyzing the HDF5 file structure.

    Args:
        weights_path: path to the weights file (.h5, .weights.h5, .keras)

    Returns:
        Reconstructed ANFISModel object, or None on error
    """
    try:
        if not os.path.exists(weights_path):
            print(f"❌ File does not exist: {weights_path}")
            return None

        n_input = None
        n_memb = None

        with h5py.File(weights_path, "r") as f:
            # try to find the fuzzy_layer parameter
            for key in f.keys():
                if "fuzzy_layer" in key:
                    grp = f[key]
                    if "c:0" in grp:
                        n_memb, n_input = grp["c:0"].shape
                        break
                    elif "sigma:0" in grp:
                        n_memb, n_input = grp["sigma:0"].shape
                        break

            # fallback - often in custom layers
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
            print(f"⚠️ Could not detect n_input/n_memb in {weights_path}")
            return None

        print(f"✓ Detected configuration: n_input={n_input}, n_memb={n_memb}")
        model = ANFISModel(n_input=int(n_input), n_memb=int(n_memb))
        _ = model.model(tf.zeros((1, int(n_input)), dtype=tf.float32))
        model.model.load_weights(weights_path)
        print(f"✓ Loaded model from {weights_path}")
        return model

    except Exception as e:
        print(f"❌ Error loading ANFIS: {e}")
        import traceback
        traceback.print_exc()
        return None


# =====================================================
# 2.  LOADING RESULTS
# =====================================================
def load_results() -> Dict[str, Any]:
    """
    Load the results of all trained models from the results/ directory.

    Supports:
    - ANFIS models (wine classification + concrete regression)
    - Comparison models (Neural Network, SVM, Random Forest)

    Results are read from JSON files and returned as a dict.

    Returns:
        Dict[model_name, json_data] - dict with the results of all models
    """
    results = {}
    base_dir = "results"
    if not os.path.exists(base_dir):
        print("⚠️ Missing results/ directory")
        return results

    # manual assignments (for GUI compatibility)
    manual_map = {
        "ANFIS (2 funkcje)": "anfis_all_2memb_results.json",
        "ANFIS (3 funkcje)": "anfis_all_3memb_results.json",
        "ANFIS Concrete (2)": "anfis_concrete_2memb_results.json",
        "ANFIS Concrete (3)": "anfis_concrete_3memb_results.json",
        "Neural Network": "nn_results.json",
        "SVM": "svm_results.json",
        "Random Forest": "rf_results.json"
    }

    for name, filename in manual_map.items():
        path = os.path.join(base_dir, filename)
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    results[name] = json.load(f)
            except Exception as e:
                print(f"⚠️ Error loading {filename}: {e}")

    # autodiscover all anfis_*.json results
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

    print(f"✓ Loaded {len(results)} result sets.")
    return results
