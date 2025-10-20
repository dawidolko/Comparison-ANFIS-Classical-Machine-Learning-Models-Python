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
    """Strona główna z informacjami o projekcie"""
    st.title("🍷 Wine Quality Classification using ANFIS")
    st.markdown("### Porównanie ANFIS z Klasycznymi Modelami Machine Learning")
    
    st.markdown("""
    ---
    ## 📊 O Projekcie
    
    Projekt porównuje **ANFIS (Adaptive Neuro-Fuzzy Inference System)** z klasycznymi metodami 
    uczenia maszynowego w zadaniu klasyfikacji jakości wina.
    
    ### 🎯 Modele porównywane:
    - **ANFIS** (2 funkcje przynależności) - Fuzzy Logic + Neural Networks
    - **ANFIS** (3 funkcje przynależności) - Rozszerzona wersja
    - **Neural Network** - Klasyczna sieć neuronowa (Dense layers)
    - **SVM** - Support Vector Machine z RBF kernel
    - **Random Forest** - Ensemble 300 drzew decyzyjnych
    
    ### 📈 Dataset: UCI Wine Quality
    - **6497 próbek** wina (czerwone + białe)
    - **11 cech fizykochemicznych** (kwasowość, alkohol, pH, etc.)
    - **Klasyfikacja binarna**: Dobra jakość (>5) vs Zła jakość (≤5)
    """)
    
    # Statystyki projektu
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📦 Próbek", "6497")
    with col2:
        st.metric("🔢 Cech", "11")
    with col3:
        st.metric("🤖 Modeli", "5")
    with col4:
        st.metric("📊 Wykresów", "8+")
    
    st.markdown("""
    ---
    ## 🧠 Co to jest ANFIS?
    
    **ANFIS** łączy zalety dwóch światów:
    - **Logika rozmyta** → Interpretowalne reguły IF-THEN
    - **Sieci neuronowe** → Automatyczne uczenie parametrów
    
    ### Architektura ANFIS (5 warstw):
    1. **Fuzzy Layer** - Fuzzyfikacja (funkcje przynależności Gaussa)
    2. **Rule Layer** - Generowanie reguł rozmytych (T-norma = AND)
    3. **Norm Layer** - Normalizacja wag reguł
    4. **Defuzz Layer** - Defuzzyfikacja (Takagi-Sugeno-Kang)
    5. **Summation Layer** - Agregacja wyniku
    
    ### Przykład reguły rozmytej:
    ```
    JEŚLI alkohol jest WYSOKI AND kwasowość jest NISKA
    TO jakość wina jest DOBRA
    ```
    
    ---
    ## 👥 Autorzy
    - **Dawid Olko**
    - **Piotr Smoła**
    - **Jakub Opar**
    - **Michał Pilecki**
    
    **Prowadzący:** mgr inż. Marcin Mrukowicz  
    **Przedmiot:** Systemy rozmyte  
    **Rok akademicki:** 2025/2026
    """)


def show_results():
    """Strona z wynikami wszystkich modeli"""
    st.title("📊 Wyniki Modeli")
    st.markdown("### Porównanie dokładności wszystkich modeli")
    
    try:
        # Ładowanie wyników
        results = load_results()
        
        # Tworzenie DataFrame do wyświetlenia
        data = []
        for model_name, metrics in results.items():
            data.append({
                'Model': model_name,
                'Train Accuracy': f"{metrics.get('train_accuracy', 0)*100:.2f}%",
                'Test Accuracy': f"{metrics.get('test_accuracy', 0)*100:.2f}%",
                'Overfitting': f"{(metrics.get('train_accuracy', 0) - metrics.get('test_accuracy', 0))*100:.2f}%"
            })
        
        df = pd.DataFrame(data)
        df = df.sort_values('Test Accuracy', ascending=False).reset_index(drop=True)
        df.index = df.index + 1  # Numeracja od 1
        
        st.dataframe(df, use_container_width=True)
        
        # Wykresy porównawcze
        st.markdown("---")
        st.markdown("### 📈 Wykresy Porównawcze")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if os.path.exists('results/all_models_comparison.png'):
                st.image('results/all_models_comparison.png', 
                        caption='Porównanie Train vs Test Accuracy')
            else:
                st.warning("Wykres porównania nie został jeszcze wygenerowany")
        
        with col2:
            if os.path.exists('results/overfitting_analysis.png'):
                st.image('results/overfitting_analysis.png',
                        caption='Analiza overfittingu')
            else:
                st.warning("Wykres overfittingu nie został jeszcze wygenerowany")
        
        # Analiza wyników
        st.markdown("---")
        st.markdown("### 🎯 Kluczowe Obserwacje")
        
        best_model = df.iloc[0]['Model']
        best_acc = df.iloc[0]['Test Accuracy']
        
        st.success(f"🏆 **Najlepszy model:** {best_model} ({best_acc})")
        
        st.markdown("""
        **Wnioski:**
        - ✅ ANFIS osiąga konkurencyjną dokładność względem klasycznych modeli
        - ✅ ANFIS oferuje **interpretowalność** (reguły rozmyte)
        - ⚠️ Większa liczba funkcji przynależności = lepsza dokładność
        - 🎯 Random Forest najlepszy, ale problem z overfittingiem
        """)
        
    except FileNotFoundError as e:
        st.error(f"❌ Brak plików z wynikami! Uruchom najpierw: `python main.py`")
        st.info("💡 Pipeline projektu musi się wykonać przed wyświetleniem wyników")


def show_anfis():
    """Strona z szczegółami o ANFIS"""
    st.title("🧠 ANFIS - Szczegóły")
    st.markdown("### Adaptive Neuro-Fuzzy Inference System")
    
    # Teoria
    with st.expander("📖 Teoria - Co to jest ANFIS?", expanded=True):
        st.markdown("""
        **ANFIS** to hybrydowy model inteligentny łączący:
        - **Sieci neuronowe** - automatyczne uczenie się z danych
        - **Logikę rozmytą** - interpretowalne reguły IF-THEN
        
        ### Funkcja przynależności Gaussa:
        ```
        μ(x) = exp(-(x - c)² / (2σ²))
        ```
        gdzie:
        - `c` - centrum (środek funkcji)
        - `σ` - odchylenie standardowe (szerokość)
        
        ### Reguły rozmyte (Takagi-Sugeno):
        ```
        Rᵢ: JEŚLI x₁ jest A₁ᵢ AND x₂ jest A₂ᵢ AND ... xₙ jest Aₙᵢ
            TO yᵢ = p₀ + p₁x₁ + p₂x₂ + ... + pₙxₙ
        ```
        """)
    
    # Wizualizacje funkcji przynależności
    st.markdown("---")
    st.markdown("### 📉 Wyuczone Funkcje Przynależności")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### ANFIS (2 funkcje)")
        if os.path.exists('results/membership_functions_2memb.png'):
            st.image('results/membership_functions_2memb.png')
        else:
            st.warning("Brak wizualizacji dla 2 funkcji")
    
    with col2:
        st.markdown("#### ANFIS (3 funkcje)")
        if os.path.exists('results/membership_functions_3memb.png'):
            st.image('results/membership_functions_3memb.png')
        else:
            st.warning("Brak wizualizacji dla 3 funkcji")
    
    # Wykresy treningu
    st.markdown("---")
    st.markdown("### 📊 Historia Treningu")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if os.path.exists('results/anfis_2memb_training.png'):
            st.image('results/anfis_2memb_training.png',
                    caption='Trening ANFIS (2 funkcje)')
    
    with col2:
        if os.path.exists('results/anfis_3memb_training.png'):
            st.image('results/anfis_3memb_training.png',
                    caption='Trening ANFIS (3 funkcje)')


def show_data_exploration():
    """Strona z eksploracją danych"""
    st.title("📈 Eksploracja Danych")
    st.markdown("### Analiza datasetu Wine Quality")
    
    # Wczytanie danych
    try:
        df_red = pd.read_csv('data/winequality-red.csv', sep=';')
        df_white = pd.read_csv('data/winequality-white.csv', sep=';')
        
        df_red['type'] = 0  # czerwone
        df_white['type'] = 1  # białe
        df = pd.concat([df_red, df_white], ignore_index=True)
        
        st.markdown(f"**Łączna liczba próbek:** {len(df)}")
        st.markdown(f"**Czerwone wino:** {len(df_red)} | **Białe wino:** {len(df_white)}")
        
        # Podgląd danych
        st.markdown("---")
        st.markdown("### 🔍 Podgląd Danych")
        st.dataframe(df.head(10), use_container_width=True)
        
        # Statystyki
        st.markdown("---")
        st.markdown("### 📊 Statystyki Opisowe")
        st.dataframe(df.describe(), use_container_width=True)
        
        # Wykresy
        st.markdown("---")
        st.markdown("### 📉 Wizualizacje")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if os.path.exists('results/quality_distribution.png'):
                st.image('results/quality_distribution.png',
                        caption='Rozkład jakości wina')
        
        with col2:
            if os.path.exists('results/correlation_matrix.png'):
                st.image('results/correlation_matrix.png',
                        caption='Macierz korelacji cech')
        
    except FileNotFoundError:
        st.error("❌ Brak plików CSV! Sprawdź folder data/")


def show_prediction():
    """Strona z predykcją jakości wina"""
    st.title("🍷 Predykcja Jakości Wina")
    st.markdown("### Wprowadź parametry wina, aby przewidzieć jego jakość")
    
    st.markdown("""
    Użyj sliderów poniżej, aby ustawić parametry fizykochemiczne wina.
    Modele ANFIS przewidzą, czy wino jest dobrej jakości (>5) czy złej (≤5).
    """)
    
    # Slidery dla cech
    st.markdown("---")
    st.markdown("### 🎚️ Parametry Wina")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fixed_acidity = st.slider("Fixed Acidity", 3.8, 15.9, 7.0, 0.1)
        volatile_acidity = st.slider("Volatile Acidity", 0.08, 1.58, 0.3, 0.01)
        citric_acid = st.slider("Citric Acid", 0.0, 1.66, 0.3, 0.01)
        residual_sugar = st.slider("Residual Sugar", 0.6, 65.8, 5.0, 0.1)
        chlorides = st.slider("Chlorides", 0.009, 0.611, 0.05, 0.001)
        free_sulfur = st.slider("Free Sulfur Dioxide", 1.0, 289.0, 30.0, 1.0)
    
    with col2:
        total_sulfur = st.slider("Total Sulfur Dioxide", 6.0, 440.0, 100.0, 1.0)
        density = st.slider("Density", 0.987, 1.039, 0.995, 0.001)
        ph = st.slider("pH", 2.72, 4.01, 3.2, 0.01)
        sulphates = st.slider("Sulphates", 0.22, 2.0, 0.5, 0.01)
        alcohol = st.slider("Alcohol", 8.0, 14.9, 10.0, 0.1)
    
    # Przycisk predykcji
    if st.button("🔮 Przewiduj Jakość", type="primary"):
        try:
            # Przygotowanie danych
            input_data = np.array([[
                fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
                chlorides, free_sulfur, total_sulfur, density, ph, sulphates, alcohol
            ]])
            
            # Ładowanie scalera
            scaler_11, _ = load_scalers()
            if scaler_11 is None:
                st.error("❌ Brak scalera! Uruchom `python main.py`")
                return
            
            # Skalowanie danych
            input_scaled = scaler_11.transform(input_data)
            
            # Predykcje
            st.markdown("---")
            st.markdown("### 🎯 Wyniki Predykcji")
            
            col1, col2, col3 = st.columns(3)
            
            pred_2 = None
            pred_3 = None
            
            # ANFIS 2 funkcje
            with col1:
                try:
                    model_2 = load_anfis_model('models/anfis_best_2memb.weights.h5')
                    if model_2 is not None:
                        pred_2 = model_2.model.predict(input_scaled, verbose=0)[0][0]
                        quality_2 = "DOBRA" if pred_2 > 0.5 else "ZŁA"
                        color_2 = "green" if pred_2 > 0.5 else "red"
                        
                        st.markdown(f"**ANFIS (2 f.)**")
                        st.markdown(f":{color_2}[{quality_2}]")
                        st.progress(float(pred_2))
                        st.caption(f"Pewność: {pred_2*100:.1f}%")
                    else:
                        st.error("Nie można załadować modelu")
                except Exception as e:
                    st.error(f"Błąd ANFIS 2: {e}")
            
            # ANFIS 3 funkcje
            with col2:
                try:
                    model_3 = load_anfis_model('models/anfis_best_3memb.weights.h5')
                    if model_3 is not None:
                        pred_3 = model_3.model.predict(input_scaled, verbose=0)[0][0]
                        quality_3 = "DOBRA" if pred_3 > 0.5 else "ZŁA"
                        color_3 = "green" if pred_3 > 0.5 else "red"
                        
                        st.markdown(f"**ANFIS (3 f.)**")
                        st.markdown(f":{color_3}[{quality_3}]")
                        st.progress(float(pred_3))
                        st.caption(f"Pewność: {pred_3*100:.1f}%")
                    else:
                        st.error("Nie można załadować modelu")
                except Exception as e:
                    st.error(f"Błąd ANFIS 3: {e}")
            
            # Wynik końcowy
            with col3:
                if pred_2 is not None and pred_3 is not None:
                    avg_pred = (pred_2 + pred_3) / 2
                    final_quality = "DOBRA" if avg_pred > 0.5 else "ZŁA"
                    final_color = "green" if avg_pred > 0.5 else "red"
                    
                    st.markdown(f"**Średnia**")
                    st.markdown(f":{final_color}[{final_quality}]")
                    st.progress(float(avg_pred))
                    st.caption(f"Pewność: {avg_pred*100:.1f}%")
                else:
                    st.warning("Brak wyników do uśrednienia")
            
        except Exception as e:
            st.error(f"❌ Błąd podczas predykcji: {e}")
            st.info("💡 Upewnij się, że pipeline został wykonany: `python main.py`")

def sidebar():
    """Boczny panel nawigacji"""
    st.sidebar.title("🍷 Nawigacja")
    st.sidebar.markdown("---")
    
    pages = {
        "🏠 Strona główna": show_home,
        "📊 Wyniki modeli": show_results,
        "🧠 ANFIS - Szczegóły": show_anfis,
        "📈 Eksploracja danych": show_data_exploration,
        "🔮 Predykcja": show_prediction,
    }
    
    selection = st.sidebar.radio("Wybierz stronę:", list(pages.keys()))
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ℹ️ Informacje")
    st.sidebar.info("""
    **Projekt:** Wine Quality ANFIS  
    **Wersja:** 1.1.0  
    **Autorzy:** D. Olko, P. Smoła, J. Opar, M. Pilecki
    """)
    
    return pages[selection]

def main():
    page = sidebar()
    page()

if __name__ == "__main__":
    main()
