REM filepath: c:\Users\poczt\Desktop\Comparison-ANFIS-Classical-Machine-Learning-Models-Python\setup.bat
@echo off
echo ======================================
echo SETUP: Environment ^& Data Generation
echo ======================================

REM Create virtual environment if it doesn't exist
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment
        pause
        exit /b 1
    )
)

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Install dependencies
echo Installing dependencies...
python -m pip install --upgrade pip
pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo ======================================
    echo ERROR: Failed to install dependencies!
    echo ======================================
    echo.
    echo Try manually:
    echo   venv\Scripts\activate
    echo   pip install tensorflow numpy pandas scikit-learn matplotlib seaborn streamlit pillow
    echo.
    pause
    exit /b 1
)

echo.
echo ======================================
echo STEP 1: Data Preprocessing
echo ======================================
python data_preprocessing.py
if errorlevel 1 (
    echo WARNING: Data preprocessing failed
    pause
)

echo.
echo ======================================
echo STEP 2: ANFIS Training (All Datasets + Cross-Validation)
echo ======================================
python train_anfis.py --datasets concrete all red white --memb 2 3 --epochs 20 --cv
if errorlevel 1 echo WARNING: ANFIS training failed

echo.
echo ======================================
echo STEP 3: Membership Functions Visualization
echo ======================================
python visualize_membership_functions.py --datasets concrete all red white --memb 2 3
if errorlevel 1 echo WARNING: Visualization failed

echo.
echo ======================================
echo STEP 4: Data Exploration (Plots)
echo ======================================
python data_exploration.py
if errorlevel 1 echo WARNING: Data exploration failed

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
dir /b results\anfis_*.png results\anfis_*.json 2>nul | find /c /v ""
dir /b results\membership_functions_*.png 2>nul | find /c /v ""
dir /b results\wine_*.png results\concrete_*.png 2>nul | find /c /v ""
dir /b results\model_comparison_*.png 2>nul | find /c /v ""

echo.
echo ======================================
echo LAUNCHING STREAMLIT GUI
echo ======================================
echo.
echo Open your browser at: http://localhost:8501
echo.
streamlit run app.py