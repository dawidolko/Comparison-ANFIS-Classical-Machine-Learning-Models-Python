# scaller.py
import os
import pickle

try:
    import streamlit as _st_mod
except Exception:
    _st_mod = None

def _get_cache():
    if _st_mod is None:
        return lambda f: f
    for name in ("cache_resource", "experimental_singleton", "cache_data", "experimental_memo"):
        deco = getattr(_st_mod, name, None)
        if callable(deco):
            return deco
    return lambda f: f

_cache = _get_cache()

@_cache
def load_scalers():
    s11 = None
    s12 = None
    for p in ("models/scaler_12.pkl", "models/standard_scaler_12.pkl", "models/scaler.pkl"):
        if os.path.exists(p):
            with open(p, "rb") as f:
                s12 = pickle.load(f)
            break
    for p in ("models/scaler_11.pkl", "models/standard_scaler_11.pkl"):
        if os.path.exists(p):
            with open(p, "rb") as f:
                s11 = pickle.load(f)
            break
    return s11, s12
