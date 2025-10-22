@echo off
REM Setup script dla Windows - Wine Quality ANFIS Project
REM Instaluje wszystkie zależności i uruchamia cały pipeline projektu

echo ============================================================
echo 🍷 Wine Quality ANFIS - Pełna Instalacja i Uruchomienie
echo ============================================================
echo.

REM Sprawdź czy Python jest zainstalowany
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo X [ERROR] Python nie jest zainstalowany!
    echo Pobierz Python z https://www.python.org/downloads/
    pause
    exit /b 1
)

echo OK [1/5] Sprawdzam wersję Pythona...
python --version
echo.

echo PAKIETY [2/5] Instaluję zależności z requirements.txt...
echo Aby uniknąć problemów z instalacją do systemowego Pythona, upewnij się, że używasz virtualenv.
if "%VIRTUAL_ENV%"=="" (
    echo Brak aktywnego venv. Tworzenie .venv...
    python -m venv .venv
    call .venv\Scripts\activate
)
echo To może potrwać kilka minut...
python -m pip install -r requirements.txt --quiet
if %errorlevel% neq 0 (
    echo X [ERROR] Błąd podczas instalacji pakietów!
    pause
    exit /b 1
)
echo OK Wszystkie pakiety zainstalowane pomyślnie!
echo.

echo ============================================================
echo PIPELINE [3/5] URUCHAMIANIE PIPELINE'U PROJEKTU
echo ============================================================
echo.
echo Wykonuję kolejno:
echo   1  Eksploracja danych
echo   2  Preprocessing danych
echo   3  Trening ANFIS
echo   4  Trening modeli porównawczych
echo   5  Porównanie wyników
echo   6  Wizualizacja funkcji przynależności
echo.
echo Czas: ~10-15 minut (trenowanie modeli)...
echo ============================================================
echo.

python main.py

if %errorlevel% neq 0 (
    echo.
    echo X [ERROR] Pipeline zakończył się błędem!
    echo Sprawdź logi powyżej, aby znaleźć przyczynę.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo OK [4/5] PIPELINE ZAKOŃCZONY POMYŚLNIE!
echo ============================================================
echo.
echo Wygenerowane pliki:
echo   OK data/*.npy - przetworzone dane
echo   OK models/*.h5, *.keras, *.pkl - modele
echo   OK results/*.png - wykresy
echo   OK results/*.json - wyniki
echo.
echo ============================================================
echo WEB [5/5] URUCHAMIANIE INTERFEJSU STREAMLIT
echo ============================================================
echo.
echo Aplikacja dostępna pod: http://localhost:8501
echo.
echo Aby zatrzymać, naciśnij Ctrl+C
echo ============================================================
echo.

timeout /t 3 /nobreak >nul

streamlit run app.py

pause
