"""
Streamlit GUI dla projektu Wine Quality Classification using ANFIS
Interaktywna aplikacja do wizualizacji wyników i predykcji

Uruchomienie: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import os
import pickle
import tensorflow as tf
from anfis import ANFISModel


# Konfiguracja strony
st.set_page_config(
    page_title="ANFIS Wine Quality",
    page_icon="🍷",
    layout="wide",
    initial_sidebar_state="expanded"
)


def load_results():
    """Wczytuje wyniki wszystkich modeli"""
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
            with open(path, 'r') as f:
                results[name] = json.load(f)

    return results


def show_home():
    """Strona główna"""
    st.title("🍷 Wine Quality Classification using ANFIS")
    st.markdown("---")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.header("📋 O projekcie")
        st.markdown("""
        Projekt porównuje algorytm **ANFIS (Adaptive Neuro-Fuzzy Inference System)**
        z klasycznymi metodami uczenia maszynowego w zadaniu klasyfikacji jakości wina.

        ### Główne cele:
        - ✅ Implementacja algorytmu ANFIS w TensorFlow/Keras
        - ✅ Porównanie ANFIS z klasycznymi modelami (NN, SVM, Random Forest)
        - ✅ Analiza interpretowalności modelu rozmytego
        - ✅ Wizualizacja wyuczonych funkcji przynależności

        ### Dataset:
        - **UCI Wine Quality Dataset**
        - 6497 próbek (czerwone i białe wino)
        - 11 cech fizyczno-chemicznych
        - Klasyfikacja binarna: dobra jakość (>5) vs zła jakość (≤5)
        """)

    with col2:
        st.header("👥 Autorzy")
        st.markdown("""
        - Dawid Olko
        - Piotr Smoła
        - Jakub Opar
        - Michał Pilecki

        **Kierunek:** Informatyka
        **Przedmiot:** Systemy rozmyte
        **Prowadzący:** mgr inż. Marcin Mrukowicz
        **Rok:** 2025/2026
        """)

    st.markdown("---")

    # Statystyki datasetu
    if os.path.exists('data/X_train.npy'):
        X_train = np.load('data/X_train.npy')
        X_test = np.load('data/X_test.npy')
        y_train = np.load('data/y_train.npy')
        y_test = np.load('data/y_test.npy')

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Próbki treningowe", f"{len(X_train)}")
        with col2:
            st.metric("Próbki testowe", f"{len(X_test)}")
        with col3:
            st.metric("Liczba cech", f"{X_train.shape[1]}")
        with col4:
            good_quality = np.sum(y_train == 1) + np.sum(y_test == 1)
            st.metric("Dobra jakość", f"{good_quality}")


def show_results():
    """Strona z wynikami modeli"""
    st.title("📊 Wyniki porównania modeli")
    st.markdown("---")

    results = load_results()

    if not results:
        st.error("❌ Nie znaleziono wyników! Uruchom najpierw: python main.py")
        return

    # Tabela porównawcza
    st.header("🏆 Ranking modeli")

    df_data = []
    for name, res in results.items():
        df_data.append({
            'Model': name,
            'Test Accuracy': f"{res['test_accuracy']*100:.2f}%",
            'Train Accuracy': f"{res['train_accuracy']*100:.2f}%",
            'Overfitting': f"{(res['train_accuracy'] - res['test_accuracy'])*100:.2f}%"
        })

    df = pd.DataFrame(df_data)
    df = df.sort_values('Test Accuracy', ascending=False).reset_index(drop=True)
    df.index = df.index + 1

    st.dataframe(df, use_container_width=True)

    # Wykresy porównawcze
    st.markdown("---")
    st.header("📈 Wizualizacje")

    col1, col2 = st.columns(2)

    with col1:
        if os.path.exists('results/all_models_comparison.png'):
            st.subheader("Porównanie dokładności")
            img = Image.open('results/all_models_comparison.png')
            st.image(img, use_column_width=True)

    with col2:
        if os.path.exists('results/overfitting_analysis.png'):
            st.subheader("Analiza overfittingu")
            img = Image.open('results/overfitting_analysis.png')
            st.image(img, use_column_width=True)


def show_anfis():
    """Strona z wynikami ANFIS"""
    st.title("🧠 ANFIS - Adaptive Neuro-Fuzzy Inference System")
    st.markdown("---")

    # Teoria ANFIS
    with st.expander("📚 Czym jest ANFIS?", expanded=False):
        st.markdown("""
        **ANFIS** to hybrydowy model łączący:
        - **Logikę rozmytą** - interpretowalne reguły IF-THEN
        - **Sieci neuronowe** - uczenie parametrów za pomocą propagacji wstecznej

        ### Architektura ANFIS:
        1. **FuzzyLayer** - fuzzyfikacja (gaussowska funkcja przynależności)
        2. **RuleLayer** - generowanie reguł rozmytych (T-norma AND)
        3. **NormLayer** - normalizacja wag reguł
        4. **DefuzzLayer** - defuzzyfikacja (Takagi-Sugeno)
        5. **SummationLayer** - agregacja wyników

        ### Funkcja przynależności:
        ```
        μ(x) = exp(-(x - c)² / 2σ²)
        ```
        gdzie `c` - centrum, `σ` - szerokość
        """)

    # Wyniki ANFIS
    results = load_results()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("ANFIS z 2 funkcjami przynależności")
        if 'ANFIS (2 funkcje)' in results:
            res = results['ANFIS (2 funkcje)']
            st.metric("Test Accuracy", f"{res['test_accuracy']*100:.2f}%")
            st.metric("Train Accuracy", f"{res['train_accuracy']*100:.2f}%")
            st.metric("Liczba reguł", "2048")

            if os.path.exists('results/anfis_2memb_training.png'):
                img = Image.open('results/anfis_2memb_training.png')
                st.image(img, use_column_width=True)

    with col2:
        st.subheader("ANFIS z 3 funkcjami przynależności")
        if 'ANFIS (3 funkcje)' in results:
            res = results['ANFIS (3 funkcje)']
            st.metric("Test Accuracy", f"{res['test_accuracy']*100:.2f}%")
            st.metric("Train Accuracy", f"{res['train_accuracy']*100:.2f}%")
            st.metric("Liczba reguł", "177,147")

            if os.path.exists('results/anfis_3memb_training.png'):
                img = Image.open('results/anfis_3memb_training.png')
                st.image(img, use_column_width=True)

    # Funkcje przynależności
    st.markdown("---")
    st.header("📉 Wyuczone funkcje przynależności")

    tab1, tab2 = st.tabs(["2 funkcje", "3 funkcje"])

    with tab1:
        if os.path.exists('results/membership_functions_2memb.png'):
            img = Image.open('results/membership_functions_2memb.png')
            st.image(img, use_column_width=True)
        else:
            st.warning("Brak wizualizacji funkcji przynależności dla 2 funkcji")

    with tab2:
        if os.path.exists('results/membership_functions_3memb.png'):
            img = Image.open('results/membership_functions_3memb.png')
            st.image(img, use_column_width=True)
        else:
            st.warning("Brak wizualizacji funkcji przynależności dla 3 funkcji")


def show_data_exploration():
    """Strona z eksploracją danych"""
    st.title("🔍 Eksploracja danych")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        if os.path.exists('results/quality_distribution.png'):
            st.subheader("Rozkład jakości wina")
            img = Image.open('results/quality_distribution.png')
            st.image(img, use_column_width=True)

    with col2:
        if os.path.exists('results/correlation_matrix.png'):
            st.subheader("Macierz korelacji")
            img = Image.open('results/correlation_matrix.png')
            st.image(img, use_column_width=True)

    # Pokaż przykładowe dane
    if os.path.exists('data/winequality-red.csv'):
        st.markdown("---")
        st.header("📋 Przykładowe dane")

        red_wine = pd.read_csv('data/winequality-red.csv', sep=';')
        st.subheader("Czerwone wino (pierwsze 10 wierszy)")
        st.dataframe(red_wine.head(10), use_container_width=True)

        st.subheader("Statystyki opisowe")
        st.dataframe(red_wine.describe(), use_container_width=True)


def show_prediction():
    """Strona z predykcją"""
    st.title("🔮 Predykcja jakości wina")
    st.markdown("---")

    st.info("⚠️ Funkcja predykcji - wprowadź parametry wina aby przewidzieć jego jakość")

    # Formularz
    col1, col2 = st.columns(2)

    with col1:
        fixed_acidity = st.slider("Fixed acidity", 4.0, 16.0, 7.0, 0.1)
        volatile_acidity = st.slider("Volatile acidity", 0.1, 1.6, 0.5, 0.01)
        citric_acid = st.slider("Citric acid", 0.0, 1.0, 0.3, 0.01)
        residual_sugar = st.slider("Residual sugar", 0.5, 20.0, 2.5, 0.1)
        chlorides = st.slider("Chlorides", 0.01, 0.2, 0.08, 0.001)
        free_sulfur_dioxide = st.slider("Free sulfur dioxide", 1.0, 80.0, 15.0, 1.0)

    with col2:
        total_sulfur_dioxide = st.slider("Total sulfur dioxide", 6.0, 300.0, 50.0, 1.0)
        density = st.slider("Density", 0.99, 1.01, 0.996, 0.0001)
        pH = st.slider("pH", 2.8, 4.0, 3.3, 0.01)
        sulphates = st.slider("Sulphates", 0.3, 2.0, 0.6, 0.01)
        alcohol = st.slider("Alcohol", 8.0, 15.0, 10.0, 0.1)

    if st.button("🍷 Przewiduj jakość wina", type="primary"):
        # Przygotuj dane wejściowe
        input_data = np.array([[
            fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
            chlorides, free_sulfur_dioxide, total_sulfur_dioxide,
            density, pH, sulphates, alcohol
        ]])

        # Wczytaj scaler i znormalizuj
        if os.path.exists('models/scaler.pkl'):
            with open('models/scaler.pkl', 'rb') as f:
                scaler = pickle.load(f)
            input_scaled = scaler.transform(input_data)

            # Predykcja dla różnych modeli
            st.markdown("---")
            st.header("📊 Wyniki predykcji")

            col1, col2, col3 = st.columns(3)

            # ANFIS 3 funkcje
            if os.path.exists('models/anfis_best_3memb.weights.h5'):
                with col1:
                    st.subheader("ANFIS (3 funkcje)")
                    anfis_model = ANFISModel(n_input=11, n_memb=3, batch_size=32)
                    anfis_model.model.load_weights('models/anfis_best_3memb.weights.h5')

                    # Padding do batch size
                    input_padded = np.repeat(input_scaled, 32, axis=0)
                    pred = anfis_model(input_padded)[0][0]

                    quality = "✅ DOBRA" if pred > 0.5 else "❌ ZŁA"
                    st.metric("Przewidywana jakość", quality)
                    st.metric("Prawdopodobieństwo", f"{pred*100:.2f}%")

            # Neural Network
            if os.path.exists('models/nn_best.keras'):
                with col2:
                    st.subheader("Neural Network")
                    nn_model = tf.keras.models.load_model('models/nn_best.keras')
                    pred = nn_model.predict(input_scaled, verbose=0)[0][0]

                    quality = "✅ DOBRA" if pred > 0.5 else "❌ ZŁA"
                    st.metric("Przewidywana jakość", quality)
                    st.metric("Prawdopodobieństwo", f"{pred*100:.2f}%")

            # SVM
            if os.path.exists('models/svm_model.pkl'):
                with col3:
                    st.subheader("SVM")
                    with open('models/svm_model.pkl', 'rb') as f:
                        svm_model = pickle.load(f)
                    pred = svm_model.predict(input_scaled)[0]
                    prob = svm_model.decision_function(input_scaled)[0]

                    quality = "✅ DOBRA" if pred == 1 else "❌ ZŁA"
                    st.metric("Przewidywana jakość", quality)
                    st.metric("Funkcja decyzyjna", f"{prob:.2f}")

        else:
            st.error("❌ Brak wytrenowanych modeli! Uruchom najpierw: python main.py")


# Sidebar
def sidebar():
    st.sidebar.title("🍷 Nawigacja")
    st.sidebar.markdown("---")

    pages = {
        "🏠 Strona główna": show_home,
        "📊 Wyniki modeli": show_results,
        "🧠 ANFIS": show_anfis,
        "🔍 Eksploracja danych": show_data_exploration,
        "🔮 Predykcja": show_prediction
    }

    selection = st.sidebar.radio("Wybierz stronę:", list(pages.keys()))

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📚 Info")
    st.sidebar.info("""
    Projekt zaliczeniowy z przedmiotu **Systemy rozmyte**.

    Porównanie ANFIS z klasycznymi modelami ML.
    """)

    return pages[selection]


# Główna aplikacja
def main():
    page = sidebar()
    page()


if __name__ == "__main__":
    main()
