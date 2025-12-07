import streamlit as st  # Biblioteka do tworzenia aplikacji webowych
import json  # Biblioteka do operacji na JSON
import os  # Biblioteka do operacji na systemie plików
from PIL import Image  # Biblioteka do ładowania i przetwarzania obrazów
import pandas as pd  # Biblioteka do operacji na ramkach danych

# -------------------------------------------------------------
# Konfiguracja aplikacji
# -------------------------------------------------------------
st.set_page_config(page_title="ANFIS Comparison", page_icon="🤖", layout="wide")  # Ustawia tytuł, ikonę i szeroki layout aplikacji Streamlit


# -------------------------------------------------------------
# Funkcje pomocnicze
# -------------------------------------------------------------
def load_json_safe(path: str):  # Funkcja bezpiecznie wczytująca plik JSON
    """
    Bezpiecznie wczytuje plik JSON.  # Opis funkcji
    
    Args:  # Sekcja argumentów
        path: ścieżka do pliku JSON  # Parametr: ścieżka do pliku
        
    Returns:  # Sekcja zwracanych wartości
        Dict z danymi lub None w przypadku błędu lub braku pliku  # Co funkcja zwraca
    """
    if os.path.exists(path):  # Sprawdza czy plik istnieje
        try:  # Próbuje wczytać plik
            with open(path, "r", encoding="utf-8") as f:  # Otwiera plik w trybie odczytu z kodowaniem UTF-8
                return json.load(f)  # Wczytuje i zwraca dane JSON
        except Exception:  # Łapie wszelkie błędy
            return None  # Zwraca None w przypadku błędu
    return None  # Zwraca None jeśli plik nie istnieje


def display_image_if_exists(path: str, caption: str = None):  # Funkcja wyświetlająca obrazek jeśli istnieje
    """
    Wyświetla obrazek w Streamlit jeśli plik istnieje.  # Opis funkcji
    
    Args:  # Sekcja argumentów
        path: ścieżka do pliku graficznego  # Parametr: ścieżka do obrazka
        caption: opcjonalny podpis pod obrazkiem  # Parametr: opcjonalny opis
        
    Returns:  # Sekcja zwracanych wartości
        True jeśli obrazek został wyświetlony, False w przeciwnym razie  # Co funkcja zwraca
    """
    if os.path.exists(path):  # Sprawdza czy plik istnieje
        st.image(Image.open(path), use_column_width=True, caption=caption)  # Otwiera i wyświetla obrazek z pełną szerokością kolumny i opcjonalnym podpisem
        return True  # Zwraca True jeśli obrazek został wyświetlony
    return False  # Zwraca False jeśli plik nie istnieje


# -------------------------------------------------------------
# Strona główna
# -------------------------------------------------------------
def show_home():  # Funkcja wyświetlająca stronę główną aplikacji
    """
    Wyświetla stronę główną aplikacji Streamlit.  # Opis funkcji
    
    Zawiera:  # Lista zawartości
    - Opis projektów (Wine Quality, Concrete Strength)  # Opis zadań
    - Architekturę modelu ANFIS  # Schemat warstw
    - Informacje o preprocessingu  # Przetwarzanie danych
    - Porównywane modele  # Lista modeli ML
    """
    st.title("🤖 ANFIS - Adaptive Neuro-Fuzzy Inference System")  # Wyświetla tytuł główny aplikacji
    st.markdown("### Porównanie ANFIS z klasycznymi metodami ML")  # Wyświetla podtytuł

    st.markdown("""  # Rozpoczyna wieloliniowy markdown z opisem projektów
    ---
    ## 📊 Dwa problemy:  # Nagłówek sekcji

    ### 1. 🍷 Wine Quality Classification (UCI Dataset)  # Pierwszy problem - klasyfikacja wina
    - **3 datasety**: all (6497), red (1599), white (4898) próbek  # Liczba próbek w każdym datasecie
    - **11 cech**: kwasowość, alkohol, pH, siarczan, chlorki, itp.  # Lista cech wina
    - **Zadanie**: Klasyfikacja binarna — dobra (>5) vs zła (≤5) jakość  # Opis zadania klasyfikacji

    ### 2. 🏗️ Concrete Strength Prediction  # Drugi problem - predykcja wytrzymałości betonu
    - **1030 próbek** betonu  # Liczba próbek w datasecie
    - **8 cech**: cement, woda, kruszywo, wiek, itp.  # Lista cech betonu
    - **Zadanie**: Predykcja wytrzymałości na ściskanie (MPa)  # Opis zadania regresji

    ---
    ## 🧠 Architektura ANFIS:  # Nagłówek sekcji architektury

    1. **Fuzzy Layer** — Fuzzyfikacja wejść funkcjami Gaussa  # Warstwa 1: funkcje przynależności
       μ(x) = exp(-(x-c)² / σ²)  # Wzor matematyczny funkcji Gaussa
    2. **Rule Layer** — Kombinacje reguł (AND / iloczyn)  # Warstwa 2: aktywacja reguł
    3. **Norm Layer** — Normalizacja wag  # Warstwa 3: normalizacja
    4. **Defuzz Layer** — Model Takagi–Sugeno (TSK-1)  # Warstwa 4: konsekwenty liniowe
    5. **Summation Layer** — Suma ważona reguł  # Warstwa 5: agregacja

    ---
    ## 📦 Preprocessing:  # Nagłówek sekcji preprocessing

    **Wine Quality:**  # Preprocessing dla wina
    - Binaryzacja jakości >5 → 1, ≤5 → 0  # Konwersja na problem binarny
    - Podział 80/20 (stratyfikowany)  # Proporcje train/test
    - StandardScaler per dataset  # Normalizacja cech

    **Concrete:**  # Preprocessing dla betonu
    - Normalizacja cech  # Standaryzacja wartości cech
    - Podział 80/20  # Proporcje train/test
    - StandardScaler  # Narzędzie do normalizacji

    ---
    ## 🎯 Modele porównywane:  # Nagłówek sekcji modeli
    - **ANFIS** (2/3 MF)  # Model ANFIS z 2 lub 3 funkcjami przynależności
    - **Neural Network**  # Sieć neuronowa
    - **SVM (RBF)**  # Support Vector Machine z jądrem RBF
    - **Random Forest**  # Las losowy
    """)  # Kończy wieloliniowy markdown


# -------------------------------------------------------------
# Sekcja wyników ANFIS
# -------------------------------------------------------------
def show_anfis_results():  # Funkcja wyświetlająca wyniki treningu ANFIS
    st.title("📊 ANFIS — Wyniki Treningu")  # Wyświetla tytuł sekcji

    col1, col2 = st.columns(2)  # Tworzy dwie kolumny dla widgetów
    with col1:  # W pierwszej kolumnie
        problem = st.selectbox("Wybierz problem:", ["Wine Quality", "Concrete Strength"], key="problem_select")  # Widget wyboru problemu
    with col2:  # W drugiej kolumnie
        if problem == "Wine Quality":  # Jeśli wybrano Wine Quality
            dataset = st.selectbox("Dataset:", ["all", "red", "white"], key="wine_dataset")  # Widget wyboru datasetu wine
        else:  # W przeciwnym razie (Concrete)
            dataset = "concrete"  # Ustawia dataset na concrete
            st.info("Dataset: Concrete (1030 próbek)")  # Wyświetla informację o datasecie

    col3, col4 = st.columns(2)  # Tworzy dwie kolumny dla liczby MF i reguł
    with col3:  # W trzeciej kolumnie
        n_memb = st.selectbox("Liczba funkcji przynależności:", [2, 3], key="n_memb")  # Widget wyboru liczby MF
    with col4:  # W czwartej kolumnie
        # Wine: 11 featureów, Concrete: 8 featureów → liczba reguł = n_memb^features
        n_features = 11 if dataset != "concrete" else 8  # Ustawia liczbę cech zależnie od datasetu
        n_rules = n_memb ** n_features  # Oblicza liczbę reguł jako potęgę n_memb^n_features
        st.metric("Liczba reguł", f"{n_rules:,}".replace(",", " "))  # Wyświetla metryki liczby reguł z formatowaniem

    # Ścieżki do plików
    results_file = f"results/anfis_{dataset}_{n_memb}memb_results.json"  # Ścieżka do pliku wyników JSON
    cv_file = f"results/anfis_{dataset}_{n_memb}memb_cv.json"  # Ścieżka do pliku cross-validation JSON
    train_img = f"results/anfis_{dataset}_{n_memb}memb_training.png"  # Ścieżka do wykresu krzywych uczenia
    mf_img = f"results/membership_functions_{dataset}_{n_memb}memb.png"  # Ścieżka do wykresu funkcji przynależności

    # Rodzaj zadania
    is_classification = (dataset != "concrete")  # Sprawdza czy zadanie to klasyfikacja (wine) czy regresja (concrete)

    if is_classification:  # Jeśli zadanie to klasyfikacja
        fit_img = f"results/anfis_{dataset}_{n_memb}memb_confmat_train.png"  # Ścieżka do macierzy pomyłek
        fit_title = "📊 Macierz pomyłek (zbiór treningowy)"  # Tytuł sekcji dla klasyfikacji
        report_file = f"results/anfis_{dataset}_{n_memb}memb_class_report_train.txt"  # Ścieżka do raportu klasyfikacyjnego
    else:  # W przeciwnym razie (regresja)
        fit_img = f"results/anfis_{dataset}_{n_memb}memb_diag_train.png"  # Ścieżka do wykresów diagnostycznych
        fit_title = "📊 Diagnostyka modelu (zbiór treningowy)"  # Tytuł sekcji dla regresji
        report_file = None  # Brak raportu tekstowego dla regresji

    # Ładowanie wyników
    results = load_json_safe(results_file)  # Wczytuje wyniki z pliku JSON
    if not results:  # Jeśli wyniki nie istnieją lub wystąpił błąd
        st.warning(f"⚠ Brak wyników dla dataset={dataset}, n_memb={n_memb}")  # Wyświetla ostrzeżenie
        st.info("Uruchom: `./setup.sh` lub `train_anfis.py`, aby wygenerować wyniki.")  # Wyświetla informację jak wygenerować wyniki
        return  # Zakończa funkcję

    st.markdown("---")  # Wyświetla separator poziomy
    st.subheader("📈 Statystyki treningu")  # Wyświetla nagłówek sekcji statystyk

    col1, col2, col3, col4 = st.columns(4)  # Tworzy cztery kolumny dla metryk
    if dataset == "concrete":  # Jeśli dataset to concrete (regresja)
        col1.metric("Train MAE", f"{results.get('train_mae', 0):.4f}")  # Wyświetla MAE treningowe
        col2.metric("Test MAE", f"{results.get('test_mae', 0):.4f}")  # Wyświetla MAE testowe
    else:  # W przeciwnym razie (klasyfikacja)
        col1.metric("Train Accuracy", f"{results.get('train_accuracy', 0):.4f}")  # Wyświetla accuracy treningowe
        col2.metric("Test Accuracy", f"{results.get('test_accuracy', 0):.4f}")  # Wyświetla accuracy testowe
    col3.metric("Train Loss", f"{results.get('train_loss', 0):.4f}")  # Wyświetla stratę treningową
    col4.metric("Test Loss", f"{results.get('test_loss', 0):.4f}")  # Wyświetla stratę testową

    st.markdown("---")  # Wyświetla separator poziomy
    st.subheader("📉 Krzywe uczenia")  # Wyświetla nagłówek sekcji krzywych uczenia
    display_image_if_exists(train_img)  # Wyświetla wykres krzywych uczenia jeśli istnieje

    st.markdown("---")  # Wyświetla separator poziomy
    st.subheader(fit_title)  # Wyświetla tytuł sekcji (macierz pomyłek lub diagnostyka)
    display_image_if_exists(fit_img)  # Wyświetla wykres macierzy pomyłek lub diagnostyki jeśli istnieje

    # Wyświetl raport tekstowy dla klasyfikacji
    if is_classification and report_file and os.path.exists(report_file):  # Jeśli klasyfikacja i raport istnieje
        with st.expander("📝 Szczegółowy raport klasyfikacyjny (trening)"):  # Tworzy rozwijaną sekcję
            with open(report_file, "r") as f:  # Otwiera plik raportu
                st.text(f.read())  # Wyświetla treść raportu jako tekst

    st.markdown("---")  # Wyświetla separator poziomy
    st.subheader("🔧 Funkcje przynależności (Gaussian MF)")  # Wyświetla nagłówek sekcji MF
    display_image_if_exists(mf_img)  # Wyświetla wykres funkcji przynależności jeśli istnieje

    # Wyniki cross-walidacji
    cv_data = load_json_safe(cv_file)  # Wczytuje dane cross-validation z pliku JSON
    if cv_data:  # Jeśli dane CV istnieją
        st.markdown("---")  # Wyświetla separator poziomy
        st.subheader("✅ Cross-Walidacja (5-fold)")  # Wyświetla nagłówek sekcji CV
        col1, col2 = st.columns(2)  # Tworzy dwie kolumny dla metryk CV

        if dataset == "concrete":  # Jeśli dataset to concrete (regresja)
            metric_name = cv_data.get("metric_type", "mae").upper()  # Pobiera typ metryki i konwertuje do wielkich liter
            col1.metric(f"Mean {metric_name}", f"{cv_data.get('mean_mae', 0):.4f}")  # Wyświetla średnią MAE
            col2.metric(f"Std {metric_name}", f"± {cv_data.get('std_mae', 0):.4f}")  # Wyświetla odchylenie standardowe MAE
        else:  # W przeciwnym razie (klasyfikacja)
            col1.metric("Mean Accuracy", f"{cv_data.get('mean_accuracy', 0):.4f}")  # Wyświetla średnią accuracy
            col2.metric("Std Accuracy", f"± {cv_data.get('std_accuracy', 0):.4f}")  # Wyświetla odchylenie standardowe accuracy

        if "folds" in cv_data:  # Jeśli dane zawierają wyniki foldów
            fold_df = pd.DataFrame(cv_data["folds"])  # Tworzy DataFrame z wyników każdego folda
            st.markdown("**Wyniki dla każdego folda:**")  # Wyświetla nagłówek tabeli
            st.dataframe(fold_df, use_container_width=True)  # Wyświetla tabelę z wynikami foldów


# -------------------------------------------------------------
# Sekcja reguł ANFIS
# -------------------------------------------------------------
def show_rules():  # Funkcja wyświetlająca reguły rozmyte ANFIS
    st.title("📜 Reguły ANFIS i Historia Uczenia")  # Wyświetla tytuł sekcji

    col1, col2 = st.columns(2)  # Tworzy dwie kolumny dla widgetów
    with col1:  # W pierwszej kolumnie
        problem = st.selectbox("Wybierz problem:", ["Wine Quality", "Concrete Strength"], key="rules_problem")  # Widget wyboru problemu
    with col2:  # W drugiej kolumnie
        if problem == "Wine Quality":  # Jeśli wybrano Wine Quality
            dataset = st.selectbox("Dataset:", ["all", "red", "white"], key="rules_dataset")  # Widget wyboru datasetu wine
        else:  # W przeciwnym razie (Concrete)
            dataset = "concrete"  # Ustawia dataset na concrete
            st.info("Dataset: Concrete")  # Wyświetla informację o datasecie

    n_memb = st.selectbox("Liczba MF:", [2, 3], key="rules_memb")  # Widget wyboru liczby funkcji przynależności

    rules_file = f"results/anfis_{dataset}_{n_memb}memb_rules.json"  # Ścieżka do pliku reguł JSON
    results_file = f"results/anfis_{dataset}_{n_memb}memb_results.json"  # Ścieżka do pliku wyników JSON

    rules_data = load_json_safe(rules_file)  # Wczytuje dane reguł z pliku JSON
    results = load_json_safe(results_file)  # Wczytuje wyniki z pliku JSON

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
# Sekcja porównania modeli
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
# Analiza danych
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

    # --- wyświetlanie wykresów ---
    missing = []
    for img_path in imgs:
        if os.path.exists(img_path):
            st.markdown("---")
            display_image_if_exists(img_path)
        else:
            missing.append(img_path)
            print(f"[WARN] Brak pliku wykresu: {img_path}")  # log do konsoli

    if missing:
        st.warning(f"Brakuje {len(missing)} wykresów: {', '.join(os.path.basename(m) for m in missing)}")



# -------------------------------------------------------------
# Nawigacja główna
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
