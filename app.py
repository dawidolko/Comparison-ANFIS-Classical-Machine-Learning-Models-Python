import streamlit as st
import json
import os
from PIL import Image
import pandas as pd

# -------------------------------------------------------------
# Application configuration
# -------------------------------------------------------------
st.set_page_config(page_title="ANFIS Comparison", page_icon="🤖", layout="wide")


# -------------------------------------------------------------
# Helper functions
# -------------------------------------------------------------
def load_json_safe(path: str):
    """
    Safely load a JSON file.

    Args:
        path: path to the JSON file

    Returns:
        Dict with the data, or None on error or if the file is missing
    """
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None


def display_image_if_exists(path: str, caption: str = None):
    """
    Show an image in Streamlit if the file exists.

    Args:
        path: path to the image file
        caption: optional caption under the image

    Returns:
        True if the image was shown, False otherwise
    """
    if os.path.exists(path):
        st.image(Image.open(path), use_column_width=True, caption=caption)
        return True
    return False


# -------------------------------------------------------------
# Home page
# -------------------------------------------------------------
def show_home():
    """
    Show the home page of the Streamlit app.

    Contains:
    - Project descriptions (Wine Quality, Concrete Strength)
    - ANFIS model architecture
    - Preprocessing information
    - Compared models
    """
    st.title("🤖 ANFIS - Adaptive Neuro-Fuzzy Inference System")
    st.markdown("### Porównanie ANFIS z klasycznymi metodami ML")

    st.markdown("""
    ---
    ## 📊 Dwa problemy:

    ### 1. 🍷 Wine Quality Classification (UCI Dataset)
    - **3 datasety**: all (6497), red (1599), white (4898) próbek
    - **11 cech**: kwasowość, alkohol, pH, siarczan, chlorki, itp.
    - **Zadanie**: Klasyfikacja binarna — dobra (>5) vs zła (≤5) jakość

    ### 2. 🏗️ Concrete Strength Prediction
    - **1030 próbek** betonu
    - **8 cech**: cement, woda, kruszywo, wiek, itp.
    - **Zadanie**: Predykcja wytrzymałości na ściskanie (MPa)

    ---
    ## 🧠 Architektura ANFIS:

    1. **Fuzzy Layer** — Fuzzyfikacja wejść funkcjami Gaussa  
       μ(x) = exp(-(x-c)² / σ²)
    2. **Rule Layer** — Kombinacje reguł (AND / iloczyn)
    3. **Norm Layer** — Normalizacja wag
    4. **Defuzz Layer** — Model Takagi–Sugeno (TSK-1)
    5. **Summation Layer** — Suma ważona reguł

    ---
    ## 📦 Preprocessing:

    **Wine Quality:**
    - Binaryzacja jakości >5 → 1, ≤5 → 0  
    - Podział 80/20 (stratyfikowany)
    - StandardScaler per dataset

    **Concrete:**
    - Normalizacja cech
    - Podział 80/20
    - StandardScaler

    ---
    ## 🎯 Modele porównywane:
    - **ANFIS** (2/3 MF)
    - **Neural Network**
    - **SVM (RBF)**
    - **Random Forest**
    """)


# -------------------------------------------------------------
# ANFIS results section
# -------------------------------------------------------------
def show_anfis_results():
    st.title("📊 ANFIS — Wyniki Treningu")

    col1, col2 = st.columns(2)
    with col1:
        problem = st.selectbox("Wybierz problem:", ["Wine Quality", "Concrete Strength"], key="problem_select")
    with col2:
        if problem == "Wine Quality":
            dataset = st.selectbox("Dataset:", ["all", "red", "white"], key="wine_dataset")
        else:
            dataset = "concrete"
            st.info("Dataset: Concrete (1030 próbek)")

    col3, col4 = st.columns(2)
    with col3:
        n_memb = st.selectbox("Liczba funkcji przynależności:", [2, 3], key="n_memb")
    with col4:
        # Wine: 11 features, Concrete: 8 features -> number of rules = n_memb^features
        n_features = 11 if dataset != "concrete" else 8
        n_rules = n_memb ** n_features
        st.metric("Liczba reguł", f"{n_rules:,}".replace(",", " "))

    results_file = f"results/anfis_{dataset}_{n_memb}memb_results.json"
    cv_file = f"results/anfis_{dataset}_{n_memb}memb_cv.json"
    train_img = f"results/anfis_{dataset}_{n_memb}memb_training.png"
    mf_img = f"results/membership_functions_{dataset}_{n_memb}memb.png"

    is_classification = (dataset != "concrete")

    if is_classification:
        fit_img = f"results/anfis_{dataset}_{n_memb}memb_confmat_train.png"
        fit_title = "📊 Macierz pomyłek (zbiór treningowy)"
        report_file = f"results/anfis_{dataset}_{n_memb}memb_class_report_train.txt"
    else:
        fit_img = f"results/anfis_{dataset}_{n_memb}memb_diag_train.png"
        fit_title = "📊 Diagnostyka modelu (zbiór treningowy)"
        report_file = None

    results = load_json_safe(results_file)
    if not results:
        st.warning(f"⚠ Brak wyników dla dataset={dataset}, n_memb={n_memb}")
        st.info("Uruchom: `./setup.sh` lub `train_anfis.py`, aby wygenerować wyniki.")
        return

    st.markdown("---")
    st.subheader("📈 Statystyki treningu")

    col1, col2, col3, col4 = st.columns(4)
    if dataset == "concrete":
        col1.metric("Train MAE", f"{results.get('train_mae', 0):.4f}")
        col2.metric("Test MAE", f"{results.get('test_mae', 0):.4f}")
    else:
        col1.metric("Train Accuracy", f"{results.get('train_accuracy', 0):.4f}")
        col2.metric("Test Accuracy", f"{results.get('test_accuracy', 0):.4f}")
    col3.metric("Train Loss", f"{results.get('train_loss', 0):.4f}")
    col4.metric("Test Loss", f"{results.get('test_loss', 0):.4f}")

    st.markdown("---")
    st.subheader("📉 Krzywe uczenia")
    display_image_if_exists(train_img)

    st.markdown("---")
    st.subheader(fit_title)
    display_image_if_exists(fit_img)

    # Show the text report for classification
    if is_classification and report_file and os.path.exists(report_file):
        with st.expander("📝 Szczegółowy raport klasyfikacyjny (trening)"):
            with open(report_file, "r") as f:
                st.text(f.read())

    st.markdown("---")
    st.subheader("🔧 Funkcje przynależności (Gaussian MF)")
    display_image_if_exists(mf_img)

    # Wyniki cross-walidacji
    cv_data = load_json_safe(cv_file)
    if cv_data:
        st.markdown("---")
        st.subheader("✅ Cross-Walidacja (5-fold)")
        col1, col2 = st.columns(2)

        if dataset == "concrete":
            metric_name = cv_data.get("metric_type", "mae").upper()
            col1.metric(f"Mean {metric_name}", f"{cv_data.get('mean_mae', 0):.4f}")
            col2.metric(f"Std {metric_name}", f"± {cv_data.get('std_mae', 0):.4f}")
        else:
            col1.metric("Mean Accuracy", f"{cv_data.get('mean_accuracy', 0):.4f}")
            col2.metric("Std Accuracy", f"± {cv_data.get('std_accuracy', 0):.4f}")

        if "folds" in cv_data:
            fold_df = pd.DataFrame(cv_data["folds"])
            st.markdown("**Wyniki dla każdego folda:**")
            st.dataframe(fold_df, use_container_width=True)


# -------------------------------------------------------------
# ANFIS rules section
# -------------------------------------------------------------
def show_rules():
    st.title("📜 Reguły ANFIS i Historia Uczenia")

    col1, col2 = st.columns(2)
    with col1:
        problem = st.selectbox("Wybierz problem:", ["Wine Quality", "Concrete Strength"], key="rules_problem")
    with col2:
        if problem == "Wine Quality":
            dataset = st.selectbox("Dataset:", ["all", "red", "white"], key="rules_dataset")
        else:
            dataset = "concrete"
            st.info("Dataset: Concrete")

    n_memb = st.selectbox("Liczba MF:", [2, 3], key="rules_memb")

    rules_file = f"results/anfis_{dataset}_{n_memb}memb_rules.json"
    results_file = f"results/anfis_{dataset}_{n_memb}memb_results.json"

    rules_data = load_json_safe(rules_file)
    results = load_json_safe(results_file)

    if rules_data:
        st.markdown("---")
        st.subheader("📊 Statystyki reguł")
        c1, c2, c3 = st.columns(3)
        c1.metric("Łączna liczba reguł", rules_data.get("n_rules_total", 0))
        c2.metric("Pokazanych reguł", rules_data.get("rules_listed", 0))
        c3.metric("Liczba cech", rules_data.get("n_features", 0))

        if rules_data.get("approx_top_rule_frequency"):
            st.markdown("---")
            st.subheader("🔥 Top 10 Najczęściej Aktywowanych Reguł")
            freq = rules_data["approx_top_rule_frequency"]
            df = pd.DataFrame(sorted(freq.items(), key=lambda x: x[1], reverse=True)[:10],
                              columns=["Rule Index", "Activations"])
            st.bar_chart(df.set_index("Rule Index"))

        st.markdown("---")
        st.subheader("📋 Przykładowe Reguły")
        st.code("IF cecha_1 IS MF[i1] AND cecha_2 IS MF[i2] AND ... THEN output = w0 + w1*x1 + ... + wn*xn")

        for rule in rules_data.get("rules", [])[:5]:
            with st.expander(f"Reguła #{rule['rule_index']}"):
                st.write(f"**Indeksy MF:** {rule['membership_indices']}")
                st.write(f"**Bias:** {rule['consequent']['bias']:.4f}")
                st.write(f"**Wagi:** {[f'{w:.4f}' for w in rule['consequent']['weights'][:5]]}...")

        st.download_button(
            label="📥 Pobierz wszystkie reguły (JSON)",
            data=json.dumps(rules_data, indent=2),
            file_name=os.path.basename(rules_file),
            mime="application/json"
        )

    if results and results.get("history"):
        st.markdown("---")
        st.subheader("📈 Historia Uczenia")
        hist = results["history"]

        df = pd.DataFrame({
            "Epoch": list(range(1, max(len(hist.get("loss", [])), len(hist.get("val_loss", []))) + 1)),
            "Train Loss": hist.get("loss", []),
            "Val Loss": hist.get("val_loss", []),
            "Train Metric": hist.get("accuracy", hist.get("mae", [])),
            "Val Metric": hist.get("val_accuracy", hist.get("val_mae", []))
        })
        st.dataframe(df, use_container_width=True)


# -------------------------------------------------------------
# Model comparison section
# -------------------------------------------------------------
def show_comparison():
    st.title("📊 Porównanie Modeli")
    st.markdown("### ANFIS vs Neural Network vs SVM vs Random Forest")

    problem = st.radio("Wybierz problem:", ["Wine Quality", "Concrete Strength"], horizontal=True)

    if problem == "Wine Quality":
        display_image_if_exists("results/model_comparison_bar_wine.png", "Porównanie modeli — Wine Quality")
        display_image_if_exists("results/overfitting_analysis_wine.png", "Analiza overfittingu — Wine Quality")
    else:
        display_image_if_exists("results/model_comparison_bar_concrete.png", "Porównanie modeli — Concrete Strength")
        display_image_if_exists("results/overfitting_analysis_concrete.png", "Analiza overfittingu — Concrete Strength")


# -------------------------------------------------------------
# Data analysis
# -------------------------------------------------------------
def show_data_analysis():
    st.title("📈 Analiza Danych")

    problem = st.selectbox("Wybierz problem:", ["Wine Quality", "Concrete Strength"], key="analysis_problem")

    if problem == "Wine Quality":
        st.markdown("### UCI Wine Quality Dataset — Eksploracja")

        imgs = [
            "results/wine_class_distribution.png",
            "results/wine_correlation.png",
            "results/wine_feature_distributions.png",
            "results/wine_pairplot.png"
        ]

    else:
        st.markdown("### Concrete Strength Dataset — Eksploracja")

        imgs = [
            "results/concrete_target_distribution.png",      
            "results/concrete_correlation.png",
            "results/concrete_feature_distributions.png",    
            "results/concrete_pairplot.png"                  
        ]

    # --- plot display ---
    missing = []
    for img_path in imgs:
        if os.path.exists(img_path):
            st.markdown("---")
            display_image_if_exists(img_path)
        else:
            missing.append(img_path)
            print(f"[WARN] Missing plot file: {img_path}")  # log do konsoli

    if missing:
        st.warning(f"Brakuje {len(missing)} wykresów: {', '.join(os.path.basename(m) for m in missing)}")



# -------------------------------------------------------------
# Main navigation
# -------------------------------------------------------------
def main():
    st.sidebar.title("📂 Nawigacja")
    st.sidebar.markdown("### Wybierz sekcję:")

    page = st.sidebar.radio(
        "",
        ["🏠 Home", "📊 ANFIS - Wyniki", "📜 Reguły i Historia", "🆚 Porównanie Modeli", "📈 Analiza Danych"],
        label_visibility="collapsed"
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### ℹ️ Informacje")
    st.sidebar.info("""
    **Projekt:** Comparison ANFIS vs ML Models  
    **Datasety:**  
    - Wine Quality (UCI)  
    - Concrete Strength  

    **Autorzy:**  
    - Dawid Olko  
    - Piotr Smoła  
    - Jakub Opar  
    - Michał Pilecki
    """)

    if page == "🏠 Home":
        show_home()
    elif page == "📊 ANFIS - Wyniki":
        show_anfis_results()
    elif page == "📜 Reguły i Historia":
        show_rules()
    elif page == "🆚 Porównanie Modeli":
        show_comparison()
    elif page == "📈 Analiza Danych":
        show_data_analysis()


if __name__ == "__main__":
    main()
