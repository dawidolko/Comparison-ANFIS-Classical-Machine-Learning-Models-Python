#!/bin/bash
set -e
echo "======================================"
echo "INSTALACJA ŚRODOWISKA I GENEROWANIE DANYCH"
echo "======================================"
if [ ! -d "venv" ]; then
    echo "Tworzę venv..."
    python3 -m venv venv
fi
source venv/bin/activate
echo "Instaluję zależności..."
pip install --upgrade pip
pip install -r requirements.txt
echo ""
echo "======================================"
echo "PRZETWARZANIE DANYCH (all, red, white)"
echo "======================================"
python3 data_preprocessing.py
echo ""
echo "======================================"
echo "TRENING ANFIS - WSZYSTKIE DATASETY + CV"
echo "======================================"
python3 train_anfis.py --datasets concrete all red white --memb 2 3 --epochs 20 --cv
echo ""
echo "======================================"
echo "WIZUALIZACJA FUNKCJI PRZYNALEŻNOŚCI"
echo "======================================"
python3 visualize_membership_functions.py --datasets concrete all red white --memb 2 3
echo ""
echo "======================================"
echo "EKSPLORACJA DANYCH (WYKRESY)"
echo "======================================"
python3 data_exploration.py
echo ""
echo "======================================"
echo "PORÓWNANIE WSZYSTKICH MODELI"
echo "======================================"
python3 train_comparison_models.py
python3 compare_all_models.py
echo ""
echo "======================================"
echo "✅ WSZYSTKO WYGENEROWANE!"
echo "======================================"
echo ""
echo "Wygenerowane pliki:"
ls -1 results/anfis_*.png results/anfis_*.json 2>/dev/null | wc -l | xargs echo "  - ANFIS wyniki:"
ls -1 results/membership_functions_*.png 2>/dev/null | wc -l | xargs echo "  - Funkcje przynależności:"
echo ""
echo "======================================"
echo "URUCHAMIAM STREAMLIT GUI"
echo "======================================"
streamlit run app.py
