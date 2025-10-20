#!/bin/bash
# Setup script dla Linux/Mac - Wine Quality ANFIS Project
# Instaluje wszystkie zależności i uruchamia projekt

echo "========================================"
echo "Wine Quality ANFIS - Setup (Linux/Mac)"
echo "========================================"
echo ""

# Sprawdź czy Python jest zainstalowany
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python3 nie jest zainstalowany!"
    echo "Zainstaluj Python3 używając menedżera pakietów twojej dystrybucji"
    exit 1
fi

echo "[1/3] Sprawdzam wersję Pythona..."
python3 --version
echo ""

echo "[2/3] Instaluję zależności z requirements.txt..."
pip3 install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "[ERROR] Błąd podczas instalacji pakietów!"
    exit 1
fi
echo ""

echo "[3/3] Instalacja zakończona pomyślnie!"
echo ""

echo "========================================"
echo "[KROK 1] Uruchamianie projektu..."
echo "========================================"
echo ""

if [ $? -ne 0 ]; then
    echo ""
    echo "[ERROR] Projekt zakończył się błędem!"
    exit 1
fi

echo ""
echo "========================================"
echo "[KROK 2] Uruchamianie GUI Streamlit..."
echo "========================================"
echo ""
echo "Otwiera się okno przeglądarki: http://localhost:8501"
echo ""
echo "Aby zatrzymać serwer Streamlit, naciśnij Ctrl+C"
echo "========================================"
echo ""

streamlit run app.py
