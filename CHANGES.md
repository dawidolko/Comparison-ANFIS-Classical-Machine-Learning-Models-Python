# 📋 LISTA ZMIAN W PROJEKCIE ANFIS

## ✅ ZAIMPLEMENTOWANE FUNKCJONALNOŚCI

### 1. **Wsparcie dla wielu datasetów (all / red / white)**
   - **Pliki**: `data_preprocessing.py`, `train_anfis.py`, `visualize_membership_functions.py`, `app.py`
   - **Zapisywane pliki**:
     - `data/X_train.npy`, `data/X_train_red.npy`, `data/X_train_white.npy`
     - `data/y_train.npy`, `data/y_train_red.npy`, `data/y_train_white.npy`
     - `models/scaler.pkl`, `models/scaler_red.pkl`, `models/scaler_white.pkl`
     - `data/datasets_summary.json`

### 2. **Wykresy krzywych uczenia**
   - **Plik**: `train_anfis.py` → `plot_training_history()`
   - **Zapisywane**: `results/anfis_{dataset}_{n_memb}memb_training.png`

### 3. **Wykres dopasowania na train**
   - **Plik**: `train_anfis.py` → `plot_fit_on_train()`
   - **Zapisywane**: `results/anfis_{dataset}_{n_memb}memb_fit_train.png`

### 4. **Ekstrakcja reguł ANFIS**
   - **Plik**: `train_anfis.py` → `extract_and_save_rules()`
   - **Zapisywane**: `results/anfis_{dataset}_{n_memb}memb_rules.json`

### 5. **Cross-walidacja 5-fold**
   - **Plik**: `train_anfis.py` → `cross_validate_anfis()`
   - **Zapisywane**: `results/anfis_{dataset}_{n_memb}memb_cv.json`

### 6. **Wizualizacja MF**
   - **Plik**: `visualize_membership_functions.py`
   - **Zapisywane**: `results/membership_functions_{dataset}_{n_memb}memb.png`

### 7. **GUI Streamlit z wyborem datasetu**
   - **Plik**: `app.py`
   - Dropdown dla dataset (all/red/white) i n_memb (2/3)
   - Dynamiczne ładowanie wykresów, reguł, CV

### 8. **Opis ANFIS + preprocessingu w GUI**
   - **Plik**: `app.py` → `show_home()`
   - Architektura 5 warstw, funkcje Gaussa, preprocessing

### 9. **Automatyzacja setup.sh i setup.bat**
   - Pełny pipeline: venv → instalacja → preprocessing → trening CV → wizualizacja → GUI

## 🚀 URUCHOMIENIE
```bash
./setup.sh    # macOS/Linux
setup.bat     # Windows
```
