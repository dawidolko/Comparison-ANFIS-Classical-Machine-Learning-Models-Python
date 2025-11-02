import streamlit as st
import json
import os
from PIL import Image
import pandas as pd

st.set_page_config(page_title="ANFIS Comparison", page_icon="🤖", layout="wide")


def load_json_safe(path):
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return None


def show_home():
    st.title("🤖 ANFIS - Adaptive Neuro-Fuzzy Inference System")
    st.markdown("### Porównanie ANFIS z klasycznymi metodami ML")
    
    st.markdown("""
    ---
    ## 📊 Dwa problemy:
    
    ### 1. 🍷 Wine Quality Classification (UCI Dataset)
    - **3 datasety**: all (6497), red (1599), white (4898) próbek
    - **11 cech**: kwasowość, alkohol, pH, siarczan, chlorki, itp.
    - **Zadanie**: Klasyfikacja binarna - dobra (>5) vs zła (≤5) jakość
    
    ### 2. 🏗️ Concrete Strength Prediction
    - **1030 próbek** betonu
    - **8 cech**: cement, woda, kruszywo, wiek, itp.
    - **Zadanie**: Predykcja wytrzymałości na ściskanie (MPa)
    
    ---
    ## 🧠 Architektura ANFIS (5 warstw):
    
    1. **Fuzzy Layer** - Fuzzyfikacja wejść funkcjami Gaussa
       - μ(x) = exp(-(x-c)²/σ²)
       - Każda cecha ma 2 lub 3 funkcje przynależności
    
    2. **Rule Layer** - Generowanie reguł rozmytych (T-norma AND)
       - Liczba reguł = n_memb ^ n_features
       - Np. 11 cech × 2 MF = 2048 reguł
    
    3. **Norm Layer** - Normalizacja wag reguł
    
    4. **Defuzz Layer** - Defuzzyfikacja Takagi-Sugeno-Kang
       - f_i = w_0 + w_1*x_1 + ... + w_n*x_n
    
    5. **Summation Layer** - Suma ważona wszystkich reguł
    
    ---
    ## 📦 Preprocessing:
    
    **Wine Quality:**
    - Binaryzacja jakości: >5 → 1, ≤5 → 0
    - Podział 80/20 stratyfikowany
    - StandardScaler osobno dla każdego datasetu
    
    **Concrete:**
    - Normalizacja wszystkich cech
    - Podział 80/20
    - StandardScaler
    
    **Dlaczego ważne?** ANFIS działa w znormalizowanej przestrzeni [-3, 3]
    
    ---
    ## 🎯 Modele porównywane:
    - **ANFIS** (2/3 funkcje przynależności)
    - **Neural Network** (Dense layers)
    - **SVM** (RBF kernel)
    - **Random Forest** (300 drzew)
    """)


def show_anfis_results():
    st.title("📊 ANFIS - Wyniki Treningu")
    
    col1, col2 = st.columns(2)
    with col1:
        problem = st.selectbox("Wybierz problem:", ['Wine Quality', 'Concrete Strength'], key='problem_select')
    with col2:
        if problem == 'Wine Quality':
            dataset = st.selectbox("Dataset:", ['all', 'red', 'white'], key='wine_dataset')
        else:
            dataset = 'concrete'
            st.info("Dataset: Concrete (1030 próbek)")
    
    col3, col4 = st.columns(2)
    with col3:
        n_memb = st.selectbox("Liczba funkcji przynależności:", [2, 3], key='n_memb')
    with col4:
        st.metric("Liczba reguł", f"{n_memb ** (11 if dataset != 'concrete' else 8)}")
    
    results_file = f'results/anfis_{dataset}_{n_memb}memb_results.json'
    cv_file = f'results/anfis_{dataset}_{n_memb}memb_cv.json'
    train_img = f'results/anfis_{dataset}_{n_memb}memb_training.png'
    fit_img = f'results/anfis_{dataset}_{n_memb}memb_fit_train.png'
    mf_img = f'results/membership_functions_{dataset}_{n_memb}memb.png'
    rules_file = f'results/anfis_{dataset}_{n_memb}memb_rules.json'
    
    results = load_json_safe(results_file)
    if results:
        st.markdown("---")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Train Accuracy", f"{results['train_accuracy']:.4f}")
        col2.metric("Test Accuracy", f"{results['test_accuracy']:.4f}")
        col3.metric("Train Loss", f"{results['train_loss']:.4f}")
        col4.metric("Test Loss", f"{results['test_loss']:.4f}")
        
        st.markdown("---")
        st.subheader("📈 Krzywe Uczenia (Accuracy + Loss)")
        if os.path.exists(train_img):
            st.image(Image.open(train_img), use_column_width=True)
        
        st.markdown("---")
        st.subheader("📊 Dopasowanie Modelu na Danych Treningowych")
        if os.path.exists(fit_img):
            st.image(Image.open(fit_img), use_column_width=True)
        
        st.markdown("---")
        st.subheader("🔧 Funkcje Przynależności (Gaussian MF)")
        if os.path.exists(mf_img):
            st.image(Image.open(mf_img), use_column_width=True)
        
        cv_data = load_json_safe(cv_file)
        if cv_data:
            st.markdown("---")
            st.subheader("✅ Cross-Walidacja (5-fold Stratified)")
            col1, col2 = st.columns(2)
            col1.metric("Mean Accuracy", f"{cv_data['mean_accuracy']:.4f}")
            col2.metric("Std Accuracy", f"± {cv_data['std_accuracy']:.4f}")
            
            st.markdown("**Wyniki dla każdego folda:**")
            fold_df = pd.DataFrame(cv_data['folds'])
            st.dataframe(fold_df, use_container_width=True)
    else:
        st.warning(f"⚠ Brak wyników dla dataset={dataset}, n_memb={n_memb}")
        st.info("Uruchom: ./setup.sh aby wygenerować wszystkie wyniki")


def show_rules():
    st.title("📜 Reguły ANFIS i Historia Uczenia")
    
    col1, col2 = st.columns(2)
    with col1:
        problem = st.selectbox("Wybierz problem:", ['Wine Quality', 'Concrete Strength'], key='rules_problem')
    with col2:
        if problem == 'Wine Quality':
            dataset = st.selectbox("Dataset:", ['all', 'red', 'white'], key='rules_dataset')
        else:
            dataset = 'concrete'
            st.info("Dataset: Concrete")
    
    n_memb = st.selectbox("Liczba MF:", [2, 3], key='rules_memb')
    
    rules_file = f'results/anfis_{dataset}_{n_memb}memb_rules.json'
    results_file = f'results/anfis_{dataset}_{n_memb}memb_results.json'
    
    rules_data = load_json_safe(rules_file)
    results = load_json_safe(results_file)
    
    if rules_data:
        st.markdown("---")
        st.subheader("📊 Statystyki Reguł")
        col1, col2, col3 = st.columns(3)
        col1.metric("Łączna liczba reguł", rules_data['n_rules_total'])
        col2.metric("Pokazanych reguł", rules_data['rules_listed'])
        col3.metric("Liczba cech", rules_data['n_features'])
        
        if rules_data.get('approx_top_rule_frequency'):
            st.markdown("---")
            st.subheader("🔥 Top 10 Najczęściej Aktywowanych Reguł")
            freq = rules_data['approx_top_rule_frequency']
            top10 = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:10]
            df = pd.DataFrame(top10, columns=['Rule Index', 'Activations'])
            st.bar_chart(df.set_index('Rule Index'))
        
        st.markdown("---")
        st.subheader("📋 Przykładowe Reguły")
        st.markdown(f"**Interpretacja:** Każda reguła ma postać:")
        st.code("""
IF cecha_1 IS MF[i1] AND cecha_2 IS MF[i2] AND ... AND cecha_n IS MF[in]
THEN output = w0 + w1*x1 + w2*x2 + ... + wn*xn
        """)
        
        for i, rule in enumerate(rules_data['rules'][:5]):
            with st.expander(f"Reguła #{rule['rule_index']}"):
                st.write(f"**Indeksy MF:** {rule['membership_indices']}")
                st.write(f"**Bias:** {rule['consequent']['bias']:.4f}")
                st.write(f"**Wagi:** {[f'{w:.4f}' for w in rule['consequent']['weights'][:5]]}...")
        
        st.download_button(
            label="📥 Pobierz wszystkie reguły (JSON)",
            data=json.dumps(rules_data, indent=2),
            file_name=f'anfis_{dataset}_{n_memb}memb_rules.json',
            mime='application/json'
        )
    
    if results and results.get('history'):
        st.markdown("---")
        st.subheader("📈 Historia Uczenia (Szczegóły)")
        history = results['history']
        
        epochs = list(range(1, len(history['accuracy']) + 1))
        df = pd.DataFrame({
            'Epoch': epochs,
            'Train Accuracy': history['accuracy'],
            'Val Accuracy': history['val_accuracy'],
            'Train Loss': history['loss'],
            'Val Loss': history['val_loss']
        })
        
        st.dataframe(df, use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Najlepsza Val Accuracy", f"{max(history['val_accuracy']):.4f}")
            st.metric("Epoch", epochs[history['val_accuracy'].index(max(history['val_accuracy']))])
        with col2:
            st.metric("Najlepsza Val Loss", f"{min(history['val_loss']):.4f}")
            st.metric("Epoch", epochs[history['val_loss'].index(min(history['val_loss']))])


def show_comparison():
    st.title("📊 Porównanie Modeli")
    st.markdown("### ANFIS vs Neural Network vs SVM vs Random Forest")
    
    img_bar = 'results/model_comparison_bar.png'
    img_overfit = 'results/overfitting_analysis.png'
    
    if os.path.exists(img_bar):
        st.image(Image.open(img_bar), use_column_width=True)
    else:
        st.warning("⚠ Brak wykresów porównawczych - uruchom ./setup.sh")
    
    if os.path.exists(img_overfit):
        st.markdown("---")
        st.subheader("🔍 Analiza Overfittingu")
        st.image(Image.open(img_overfit), use_column_width=True)


def show_data_analysis():
    st.title("📈 Analiza Danych")
    
    problem = st.selectbox("Wybierz problem:", ['Wine Quality', 'Concrete Strength'], key='analysis_problem')
    
    if problem == 'Wine Quality':
        imgs = [
            'results/wine_class_distribution.png',
            'results/wine_correlation.png',
            'results/wine_feature_distributions.png',
            'results/wine_pairplot.png'
        ]
        st.markdown("### UCI Wine Quality Dataset - Eksploracja")
    else:
        imgs = [
            'results/concrete_distribution.png',
            'results/concrete_correlation.png'
        ]
        st.markdown("### Concrete Strength Dataset - Eksploracja")
    
    found = False
    for img_path in imgs:
        if os.path.exists(img_path):
            st.image(Image.open(img_path), use_column_width=True)
            st.markdown("---")
            found = True
    
    if not found:
        st.warning("⚠ Brak wykresów analizy danych - uruchom ./setup.sh")


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
