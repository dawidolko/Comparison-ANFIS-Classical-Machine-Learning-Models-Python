"""
Streamlit GUI dla projektu Wine Quality Classification using ANFIS
Interaktywna aplikacja do wizualizacji wyników i predykcji

Uruchomienie: streamlit run app.py
"""
import h5py
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
from scaller import load_scalers

# Konfiguracja strony
st.set_page_config(
    page_title="ANFIS Wine Quality",
    page_icon="🍷",
    layout="wide",
    initial_sidebar_state="expanded"
)

def _load_anfis(weights_path: str,
                X11: np.ndarray,
                X12: np.ndarray,
                default_memb: int = 3,
                verbose: bool = True) -> tuple[float | None, str]:
    """
    Próbuj ładować wagi ANFIS i zwróć (prob, debug_info).
    prob = None -> nieudane, a debug_info powie gdzie poległo.
    """
    info = []

    try:
        if not os.path.exists(weights_path):
            info.append(f"[path] Brak pliku: {weights_path}")
            return None, "\n".join(info)

        info.append(f"[path] OK: {weights_path}")

        # 1) Wykryj (n_input, n_memb) z pliku H5
        n_input = None
        n_memb = None

        import h5py
        with h5py.File(weights_path, "r") as hf:
            datasets = []

            def visitor(name, obj):
                if isinstance(obj, h5py.Dataset) and len(getattr(obj, "shape", ())) == 2:
                    datasets.append((name, tuple(obj.shape)))

            hf.visititems(visitor)

            info.append(f"[h5] Znalezione datasety 2D: {[(n, s) for n, s in datasets][:6]}{' ...' if len(datasets) > 6 else ''}")

            # spróbuj znaleźć coś z 'centre/center/centres/centers'
            candidates = [(n, s) for n, s in datasets
                          if any(k in n.lower() for k in ("centre", "center", "centres", "centers"))]

            if not candidates:
                # jak nie ma 'centres', weź dataset gdzie JEDEN wymiar to 11 lub 12, a drugi <= 8
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
                # preferuj te, które mają wymiar 11/12
                picked = None
                for n, (a, b) in candidates:
                    if a in (11, 12) or b in (11, 12):
                        picked = (n, (a, b))
                        break
                if picked is None:
                    picked = candidates[0]
                n, (a, b) = picked
                # załóż że większy wymiar = wejścia
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

        # 2) Zbuduj model
        try:
            anfis_model = ANFISModel(n_input=int(n_input), n_memb=int(n_memb or default_memb))
        except Exception as e:
            info.append(f"[build] Błąd konstruktora ANFISModel: {e!r}")
            return None, "\n".join(info)

        # Subclassed model: zainicjalizuj graf jednym przejściem
        try:
            dummy = tf.zeros((1, int(n_input)), dtype=tf.float32)
            _ = anfis_model.model(dummy)  # wywołanie forward
            info.append("[build] Model wywołany na dummy (ok).")
        except Exception as e:
            info.append(f"[build] Błąd przy inicjalizacji grafu: {e!r}")
            return None, "\n".join(info)

        # 3) Ładuj wagi – najpierw by_name (dla legacy H5); jeśli Keras 3 marudzi, fallback bez by_name
        try:
            anfis_model.model.load_weights(weights_path, by_name=True, skip_mismatch=True)
            info.append("[weights] load_weights(..., by_name=True, skip_mismatch=True) – OK")
        except (TypeError, ValueError) as e:
            info.append(f"[weights] by_name niedostępne ({e!r}) – próbuję bez by_name")
            try:
                # Keras 3 wspiera skip_mismatch bez by_name
                anfis_model.model.load_weights(weights_path, skip_mismatch=True)
                info.append("[weights] load_weights(..., skip_mismatch=True) – OK (bez by_name)")
            except Exception as e2:
                info.append(f"[weights] Fallback bez by_name też padł: {e2!r}")
                return None, "\n".join(info)
        except Exception as e:
            info.append(f"[weights] Inny błąd load_weights: {e!r}")
            return None, "\n".join(info)


        # 4) Predykcja – wybierz dopasowany wariant i dopilnuj dtype/kształtu
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
    st.title("🔮 Predykcja jakości wina")
    st.markdown("---")

    st.info("⚠️ Wprowadź parametry wina, aby przewidzieć jego jakość…")

    wine_type = st.radio("Rodzaj wina", ["Czerwone", "Białe"], horizontal=True)
    wine_type_val = 0 if wine_type == "Czerwone" else 1

    col1, col2 = st.columns(2)
    with col1:
        fixed_acidity      = st.slider("Fixed acidity",      4.0, 16.0, 7.0, 0.1)
        volatile_acidity   = st.slider("Volatile acidity",   0.1, 1.6, 0.5, 0.01)
        citric_acid        = st.slider("Citric acid",        0.0, 1.0, 0.3, 0.01)
        residual_sugar     = st.slider("Residual sugar",     0.5, 20.0, 2.5, 0.1)
        chlorides          = st.slider("Chlorides",          0.01, 0.2, 0.08, 0.001)
        free_SO2           = st.slider("Free sulfur dioxide", 1.0, 80.0, 15.0, 1.0)
    with col2:
        total_SO2          = st.slider("Total sulfur dioxide", 6.0, 300.0, 50.0, 1.0)
        density            = st.slider("Density",            0.99, 1.01, 0.996, 0.0001)
        pH                 = st.slider("pH",                 2.8, 4.0, 3.3, 0.01)
        sulphates          = st.slider("Sulphates",          0.3, 2.0, 0.6, 0.01)
        alcohol            = st.slider("Alcohol",            8.0, 15.0, 10.0, 0.1)

    if st.button("🍷 Przewiduj jakość wina", type="primary"):
        # surowe wejście 12D
        X12 = np.array([[
            fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
            chlorides, free_SO2, total_SO2, density, pH, sulphates,
            alcohol, wine_type_val
        ]], dtype=np.float32)

        # wariant 11D (bez typu)
        X11 = X12[:, :11].copy()

        scaler_11, scaler_12 = load_scalers()

        X12_scaled = scaler_12.transform(X12) if scaler_12 is not None else X12

        if scaler_11 is not None:
            X11_scaled = scaler_11.transform(X11)
        elif scaler_12 is not None:
            X11_scaled = X12_scaled[:, :11]  # obetnij przeskalowane 12D
        else:
            X11_scaled = X11

        st.markdown("---")
        st.header("📊 Wyniki predykcji")
        col1, col2, col3 = st.columns(3)

        # ─── ANFIS ───
        with col1:
            st.subheader("ANFIS")
            prob, dbg = _load_anfis(
                weights_path="models/anfis_best_3memb.weights.h5",
                X11=X11_scaled, X12=X12_scaled,
                default_memb=3, verbose=True
            )
            if prob is None:
                st.error("🚫 Nie udało się załadować wag ANFIS.")
                with st.expander("Pokaż diagnostykę ANFIS", expanded=False):
                    st.code(dbg or "(brak informacji)", language="text")
            else:
                quality = "✅ DOBRA" if prob > 0.5 else "❌ ZŁA"
                st.metric("Przewidywana jakość", quality)
                st.metric("Prawdopodobieństwo", f"{prob * 100:.2f}%")
                with st.expander("Szczegóły ładowania ANFIS", expanded=False):
                    st.code(dbg, language="text")

        # ─── Neural Network ─── (najczęściej trenowana na 12 cechach)
        with col2:
            st.subheader("Neural Network")
            try:
                nn = tf.keras.models.load_model("models/nn_best.keras", compile=False)
                prob = float(nn.predict(X12_scaled, verbose=0).ravel()[0])
                quality = "✅ DOBRA" if prob > 0.5 else "❌ ZŁA"
                st.metric("Przewidywana jakość", quality)
                st.metric("Prawdopodobieństwo", f"{prob*100:.2f}%")
            except Exception as e:
                st.error(f"NN błąd: {e}")

        # ─── SVM ─── (również zwykle 12 cech)
        with col3:
            st.subheader("SVM")
            try:
                with open("models/svm_model.pkl", "rb") as f:
                    svm = pickle.load(f)
                if hasattr(svm, "predict_proba"):
                    prob = float(svm.predict_proba(X12_scaled)[0, 1])
                    st.metric("Przewidywana jakość", "✅ DOBRA" if prob >= 0.5 else "❌ ZŁA")
                    st.metric("Prawdopodobieństwo", f"{prob*100:.2f}%")
                else:
                    y = float(svm.predict(X12_scaled)[0])
                    st.metric("Przewidywana jakość", "✅ DOBRA" if y >= 0.5 else "❌ ZŁA")
            except Exception as e:
                st.error(f"SVM błąd: {e}")

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
