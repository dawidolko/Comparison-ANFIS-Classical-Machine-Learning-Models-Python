"""
Streamlit GUI dla projektu Wine Quality Classification using ANFIS
Interaktywna aplikacja do wizualizacji wyników i predykcji

Uruchomienie: streamlit run app.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
import tensorflow as tf
from PIL import Image

# Importy z modułów projektu
from utils import load_anfis_model, load_results
from scaller import load_scalers

st.set_page_config(
    page_title="ANFIS Wine Quality",
    page_icon="🍷",
    layout="wide",
    initial_sidebar_state="expanded"
)

def show_home():
    st.title("🍷 Wine Quality Classification using ANFIS")
    # reszta funkcji
    pass

def show_results():
    pass

def show_anfis():
    pass

def show_data_exploration():
    pass

def show_prediction():
    pass

def sidebar():
    st.sidebar.title("🍷 Nawigacja")
    pages = {
        "🏠 Strona główna": show_home,
        "📊 Wyniki modeli": show_results,
    }
    selection = st.sidebar.radio("Wybierz stronę:", list(pages.keys()))
    return pages[selection]

def main():
    page = sidebar()
    page()

if __name__ == "__main__":
    main()
