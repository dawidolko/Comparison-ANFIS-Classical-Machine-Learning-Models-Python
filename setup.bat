@echo off
echo ======================================
echo SETUP: Environment ^& Data Generation
echo ======================================

REM Create virtual environment if it doesn't exist
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Install dependencies
echo Installing dependencies...
python -m pip install --upgrade pip
pip install -r requirements.txt

echo.
echo ======================================
echo STEP 1: Data Preprocessing
echo ======================================
python data_preprocessing.py

echo.
echo ======================================
echo STEP 2: ANFIS Training (All Datasets + Cross-Validation)
echo ======================================
python train_anfis.py --datasets concrete all red white --memb 2 3 --epochs 20 --cv

echo.
echo ======================================
echo STEP 3: Membership Functions Visualization
echo ======================================
python visualize_membership_functions.py --datasets concrete all red white --memb 2 3

echo.
echo ======================================
echo STEP 4: Data Exploration (Plots)
echo ======================================
python data_exploration.py

echo.
echo ======================================
echo STEP 5: Model Comparison
echo ======================================
python train_comparison_models.py
python compare_all_models.py

echo.
echo ======================================
echo ALL FILES GENERATED SUCCESSFULLY!
echo ======================================
echo.
echo Generated files:
dir /b results\anfis_*.png results\anfis_*.json 2>nul | find /c /v "" && echo   - ANFIS results
dir /b results\membership_functions_*.png 2>nul | find /c /v "" && echo   - Membership functions
dir /b results\wine_*.png results\concrete_*.png 2>nul | find /c /v "" && echo   - Data exploration
dir /b results\model_comparison_*.png 2>nul | find /c /v "" && echo   - Model comparison

echo.
echo ======================================
echo LAUNCHING STREAMLIT GUI
echo ======================================
echo.
echo Open your browser at: http://localhost:8501
echo.
streamlit run app.py
