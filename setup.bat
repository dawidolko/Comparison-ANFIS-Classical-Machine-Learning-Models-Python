@echo off
REM Setup script dla Windows - Wine Quality ANFIS Project
REM Instaluje wszystkie zależności i uruchamia projekt

echo ========================================
echo Wine Quality ANFIS - Setup (Windows)
echo ========================================
echo.

REM Sprawdź czy Python jest zainstalowany
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python nie jest zainstalowany!
    echo Pobierz Python z: https://www.python.org/downloads/
    pause
    exit /b 1
)

echo [1/3] Sprawdzam wersje Pythona...
python --version
echo.

echo [2/3] Instaluje zaleznosci z requirements.txt...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo [ERROR] Blad podczas instalacji pakietow!
    pause
    exit /b 1
)
echo.

echo [3/3] Instalacja zakonczona pomyslnie!
echo.

echo ========================================
echo [KROK 1] Uruchamianie projektu...
echo ========================================
echo.

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Projekt zakonczyl sie bledem!
    pause
    exit /b 1
)

echo.
echo ========================================
echo [KROK 2] Uruchamianie GUI Streamlit...
echo ========================================
echo.
echo Otwiera sie okno przegladarki: http://localhost:8501
echo.
echo Aby zatrzymac serwer Streamlit, nacisnij Ctrl+C
echo ========================================
echo.

streamlit run app.py

pause
