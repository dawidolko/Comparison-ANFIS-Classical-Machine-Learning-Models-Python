#!/bin/bash
# Setup script dla Linux/Mac - Wine Quality ANFIS Project
# Instaluje wszystkie zależności i uruchamia cały pipeline projektu

echo "============================================================"
echo "🍷 Wine Quality ANFIS - Pełna Instalacja i Uruchomienie"
echo "============================================================"
echo ""

# Sprawdź czy Python jest zainstalowany
if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null; then
    echo "❌ [ERROR] Python nie jest zainstalowany!"
    echo "Zainstaluj Python używając menedżera pakietów twojej dystrybucji"
    exit 1
fi

# Jeśli nie jesteśmy w wirtualnym środowisku - utwórz i aktywuj .venv
if [ -z "$VIRTUAL_ENV" ]; then
    # Preferuj pyenv python 3.11.9 jeśli jest zainstalowany (użyty wcześniej podczas testów)
    if [ -x "/home/jakub/.pyenv/versions/3.11.9/bin/python" ]; then
        PYTHON_BIN="/home/jakub/.pyenv/versions/3.11.9/bin/python"
    else
        PYTHON_BIN=$(command -v python3 || command -v python)
    fi

    echo "Utworzę/aktywuję wirtualne środowisko .venv używając: $PYTHON_BIN"
    if [ ! -d ".venv" ]; then
        $PYTHON_BIN -m venv .venv
    fi
    # Aktywuj venv dla tego skryptu
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

echo "✓ [1/5] Sprawdzam wersję Pythona (w venv jeśli aktywne)..."
python --version
echo ""

echo "📦 [2/5] Instaluję zależności z requirements.txt..."
echo "To może potrwać kilka minut..."
pip3 install -r requirements.txt --quiet
if [ $? -ne 0 ]; then
    echo "❌ [ERROR] Błąd podczas instalacji pakietów!"
    exit 1
fi
echo "✓ Wszystkie pakiety zainstalowane pomyślnie!"
echo ""

echo "============================================================"
echo "🚀 [3/5] URUCHAMIANIE PIPELINE'U PROJEKTU"
echo "============================================================"
echo ""
echo "Wykonuję kolejno:"
echo "  1️⃣  Eksploracja danych (data_exploration.py)"
echo "  2️⃣  Preprocessing danych (data_preprocessing.py)"
echo "  3️⃣  Trening ANFIS (train_anfis.py)"
echo "  4️⃣  Trening modeli porównawczych (train_comparison_models.py)"
echo "  5️⃣  Porównanie wyników (compare_all_models.py)"
echo "  6️⃣  Wizualizacja funkcji przynależności (visualize_membership_functions.py)"
echo ""
echo "⏳ To może potrwać 10-15 minut (trenowanie modeli)..."
echo "============================================================"
echo ""

python3 main.py

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ [ERROR] Pipeline zakończył się błędem!"
    echo "Sprawdź logi powyżej, aby znaleźć przyczynę."
    exit 1
fi

echo ""
echo "============================================================"
echo "✅ [4/5] PIPELINE ZAKOŃCZONY POMYŚLNIE!"
echo "============================================================"
echo ""
echo "📂 Wygenerowane pliki:"
echo "  • data/*.npy - przetworzone dane treningowe/testowe"
echo "  • models/*.h5, *.keras, *.pkl - wytrenowane modele"
echo "  • results/*.png - wykresy i wizualizacje"
echo "  • results/*.json - wyniki liczbowe"
echo ""
echo "============================================================"
echo "🌐 [5/5] URUCHAMIANIE INTERFEJSU STREAMLIT"
echo "============================================================"
echo ""
echo "🍷 Aplikacja webowa będzie dostępna pod adresem:"
echo "   👉 http://localhost:8501"
echo ""
echo "📌 Aplikacja otworzy się automatycznie w przeglądarce"
echo "📌 Aby zatrzymać serwer Streamlit, naciśnij Ctrl+C"
echo ""
echo "============================================================"
echo ""

# Czekaj 3 sekundy przed uruchomieniem Streamlit
sleep 3

streamlit run app.py
