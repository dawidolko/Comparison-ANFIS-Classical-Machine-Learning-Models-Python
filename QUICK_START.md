# 🚀 QUICK START - Szybkie Uruchomienie Projektu

## ✅ Wymagania

- Python 3.8+
- pip
- Git

## 📥 Instalacja i Uruchomienie (3 kroki)

### 1️⃣ Sklonuj repo

```bash
git clone -b dev https://github.com/dawidolko/Comparison-ANFIS-Classical-Machine-Learning-Models-Python.git
cd Comparison-ANFIS-Classical-Machine-Learning-Models-Python
```

### 2️⃣ Uruchom setup (automatyczna instalacja + trening + Streamlit)

**Linux/Mac:**

```bash
chmod +x setup.sh
./setup.sh
```

**Windows:**

```cmd
setup.bat
```

### 3️⃣ Gotowe! 🎉

Aplikacja Streamlit otworzy się automatycznie na:

- 🌐 http://localhost:8501

---

## 📋 Co robi `setup.sh`?

1. ✅ Sprawdza Python
2. 📦 Instaluje zależności (`pip install -r requirements.txt`)
3. 🍷 Uruchamia pipeline (6 kroków):
   - Eksploracja danych
   - Preprocessing
   - Trening ANFIS (2 i 3 funkcje)
   - Trening NN, SVM, Random Forest
   - Porównanie wyników
   - Wizualizacja funkcji przynależności
4. 🌐 Uruchamia Streamlit

**Czas wykonania:** ~10-15 minut (trening modeli)

---

## 📂 Po uruchomieniu zobaczysz:

```
models/
  ├── anfis_best_2memb.weights.h5
  ├── anfis_best_3memb.weights.h5
  ├── nn_best.keras
  ├── svm_model.pkl
  ├── rf_model.pkl
  ├── scaler.pkl
  └── scaler_nn.pkl

results/
  ├── *.png (wykresy)
  └── *.json (wyniki liczbowe)
```

---

## 🔧 Problemy?

### Brak Pythona?

```bash
# Mac
brew install python3

# Ubuntu/Debian
sudo apt install python3 python3-pip
```

### Błędy podczas instalacji pakietów?

```bash
pip3 install --upgrade pip
pip3 install -r requirements.txt
```

---

## 🎓 Struktura Aplikacji Streamlit

- 🏠 **Strona główna** - Metryki wszystkich modeli
- 📊 **Wyniki modeli** - Porównanie, wykresy, analiza
- 🧠 **ANFIS** - Teoria + wizualizacje funkcji przynależności
- 📈 **Eksploracja danych** - CSV, statystyki, korelacje
- 🍷 **Predykcja** - Interaktywne predykcje jakości wina

---

**Autor:** Dawid Olko  
**Projekt:** Systemy Rozmyte - Wine Quality Classification
