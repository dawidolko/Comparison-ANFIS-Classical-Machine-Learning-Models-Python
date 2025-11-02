@echo off
echo ======================================
echo INSTALACJA I GENEROWANIE WSZYSTKIEGO
echo ======================================
if not exist "venv" (
    echo Tworzę venv...
    python -m venv venv
)
call venv\Scripts\activate.bat
echo Instaluję zależności...
python -m pip install --upgrade pip
pip install -r requirements.txt
echo.
echo ======================================
echo PRZETWARZANIE DANYCH
echo ======================================
python data_preprocessing.py
echo.
echo ======================================
echo TRENING ANFIS + CV
echo ======================================
python train_anfis.py --datasets all red white --memb 2 3 --epochs 20 --cv
echo.
echo ======================================
echo WIZUALIZACJA MF
echo ======================================
python visualize_membership_functions.py --datasets concrete all red white --memb 2 3
echo.
echo ======================================
echo EKSPLORACJA DANYCH
echo ======================================
python data_exploration.py
echo.
echo ======================================
echo POROWNANIE MODELI
echo ======================================
python train_comparison_models.py
python compare_all_models.py
echo.
echo ======================================
echo WSZYSTKO WYGENEROWANE!
echo ======================================
echo.
echo URUCHAMIAM STREAMLIT GUI
echo ======================================
streamlit run app.py
