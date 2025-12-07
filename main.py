"""
GŁÓWNY SKRYPT PROJEKTU - Klasyfikacja jakości wina z wykorzystaniem ANFIS
Orkiestruje cały pipeline projektu: od danych do wyników

Autorzy: Dawid Olko, Piotr Smoła, Jakub Opar, Michał Pilecki
Uruchomienie: python main.py
"""

import subprocess  # Biblioteka do uruchamiania procesów systemowych (skryptów)
import sys  # Biblioteka do interakcji z interpreterem Pythona
from datetime import datetime  # Klasa do operacji na dacie i czasie
import time  # Biblioteka do pomiaru czasu i opóźnień


def print_progress_bar(current, total, description, bar_length=50):  # Funkcja wyświetlająca pasek postępu w terminalu
    """Wyświetlono pasek postępu w terminalu"""
    percent = 100 * (current / float(total))  # Oblicza procent wykonania
    filled = int(bar_length * current // total)  # Oblicza liczbę wypełnionych znaków paska
    bar = '█' * filled + '░' * (bar_length - filled)  # Tworzy pasek z wypełnionych i pustych znaków
    
    print(f'\r🍷 Postęp: [{bar}] {percent:.1f}% - {description}', end='', flush=True)  # Wypisuje pasek postępu (\r nadpisuje linię)
    if current == total:  # Sprawdza czy zakończono wszystkie kroki
        print()  # Dodaje nową linię po zakończeniu


def run_script(script_name, description, step_num, total_steps):  # Funkcja uruchamiająca skrypt Pythona i monitorująca postęp
    """Uruchomiono skrypt Pythona i wyświetlono postęp wykonania"""
    
    print(f"\n{'='*80}")  # Wypisuje separator
    print(f"  📊KROK {step_num}/{total_steps}: {description}")  # Wypisuje numer kroku i opis
    print(f"{'='*80}\n")  # Wypisuje separator kończący nagłówek
    
    # Wyświetlono postęp przed rozpoczęciem kroku
    print_progress_bar(step_num - 1, total_steps, f"Rozpoczynanie: {description}")  # Wyświetla pasek postępu przed krokiem
    
    start_time = time.time()  # Zapisuje czas rozpoczęcia
    result = subprocess.run([sys.executable, script_name], capture_output=False)  # Uruchamia skrypt Pythona
    elapsed = time.time() - start_time  # Oblicza czas wykonania

    if result.returncode != 0:  # Sprawdza czy skrypt zakończył się błędem
        print(f"\n❌ [BŁĄD] Skrypt {script_name} został zakończony z błędem!")  # Informuje o błędzie
        return False  # Zwraca False sygnalizując niepowodzenie

    # Wyświetlono postęp po zakończeniu kroku
    print_progress_bar(step_num, total_steps, f"✅ Zakończono w {elapsed:.1f}s")  # Wyświetla pasek postępu po zakończeniu
    print(f"\n[INFO] {description} – ZAKOŃCZONO (czas: {elapsed:.1f}s)\n")  # Wypisuje informację o pomyślnym zakończeniu
    return True  # Zwraca True sygnalizując sukces


def main():  # Główna funkcja orkiestrująca cały pipeline projektu
    """Wykonano główny pipeline projektu"""

    print("\n" + "=" * 80)  # Wypisuje separator
    print("  🍷 PROJEKT: Porównanie ANFIS z klasycznymi modelami ML")  # Wypisuje tytuł projektu
    print("  Klasyfikacja jakości wina")  # Wypisuje opis projektu
    print("=" * 80)  # Wypisuje separator
    print(f"\n⏰ Rozpoczęto: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")  # Wypisuje czas rozpoczęcia

    # Zdefiniowano listę kroków do wykonania
    steps = [  # Lista kroków pipeline jako krotki (skrypt, opis)
        ("data_exploration.py", "Eksploracja danych"),  # Krok 1: Analiza i wizualizacja danych
        ("data_preprocessing.py", "Przetwarzanie danych"),  # Krok 2: Normalizacja i podział danych
        ("train_anfis.py", "Trenowanie modeli ANFIS"),  # Krok 3: Trening modeli ANFIS (2 i 3 MF)
        ("train_comparison_models.py", "Trenowanie modeli porównawczych"),  # Krok 4: Trening NN, SVM, RF
        ("visualize_membership_functions.py", "Wizualizacja funkcji przynależności"),  # Krok 5: Wykresy funkcji gaussowskich
        ("compare_all_models.py", "Porównanie wszystkich modeli"),  # Krok 6: Porównanie wyników wszystkich modeli
    ]
    
    total_steps = len(steps)  # Oblicza całkowitą liczbę kroków
    start_time = time.time()  # Zapisuje czas rozpoczęcia całego pipeline

    print(f"📋 Liczba kroków: {total_steps}")  # Wypisuje liczbę kroków do wykonania
    print(f"⏱️  Szacowany czas wykonania: ~10–15 minut\n")  # Informuje o szacowanym czasie
    
    # Wyświetlono pasek postępu na początku
    print_progress_bar(0, total_steps, "Przygotowanie...")  # Wyświetla początkowy pasek postępu
    print()  # Dodaje nową linię

    # Wykonano wszystkie kroki sekwencyjnie
    for idx, (script, description) in enumerate(steps, 1):  # Iteruje przez wszystkie kroki zaczynając od 1
        success = run_script(script, description, idx, total_steps)  # Uruchamia skrypt i sprawdza sukces
        if not success and script in ["data_preprocessing.py", "train_anfis.py"]:  # Sprawdza czy wystąpił błąd krytyczny
            # Zidentyfikowano błąd krytyczny – przerwano wykonanie
            print(f"\n❌ [BŁĄD] KRYTYCZNY w {script}. Przerwano pipeline.")  # Informuje o błędzie krytycznym
            sys.exit(1)  # Kończy program z kodem błędu
    
    total_elapsed = time.time() - start_time  # Oblicza całkowity czas wykonania

    # Wygenerowano podsumowanie wykonania
    print("\n" + "=" * 80)  # Wypisuje separator
    print("  ✅ PROJEKT ZOSTAŁ ZAKOŃCZONY POMYŚLNIE!")  # Wypisuje komunikat o sukcesie
    print("=" * 80)  # Wypisuje separator
    
    # Sformatowano całkowity czas wykonania
    minutes, seconds = divmod(int(total_elapsed), 60)  # Konwertuje sekundy na minuty i sekundy
    time_str = f"{minutes} min {seconds} s" if minutes > 0 else f"{seconds} s"  # Formatuje string czasu
    
    print(f"\n⏱️  Całkowity czas wykonania: {time_str}")  # Wypisuje całkowity czas
    print("\n📂 Wygenerowane zasoby:")  # Nagłówek sekcji zasobów
    print("  ✓ data/       – Zbiory danych (CSV, NPY)")  # Katalog z danymi
    print("  ✓ models/     – Wytrenowane modele (.keras, .pkl)")  # Katalog z modelami
    print("  ✓ results/    – Wykresy (PNG) oraz wyniki (JSON)")  # Katalog z wynikami
    print("\n🚀 Zalecane następne kroki:")  # Nagłówek sekcji zaleć
    print("  1. Przejrzyj wykresy w katalogu results/")  # Zalecenie 1
    print("  2. Uruchom aplikację GUI: streamlit run app.py")  # Zalecenie 2
    print(f"\n⏰ Zakończono: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")  # Wypisuje czas zakończenia
    print("=" * 80 + "\n")  # Wypisuje separator końcowy


if __name__ == "__main__":  # Sprawdza czy skrypt jest uruchamiany bezpośrednio
    try:  # Próbuje wykonać główny pipeline
        main()  # Uruchamia funkcję main
    except KeyboardInterrupt:  # Łapie przerwanie użytkownika (Ctrl+C)
        print("\n\n[INFO] Wykonanie zostało przerwane przez użytkownika (Ctrl+C)")  # Informuje o przerwaniu
        sys.exit(0)  # Kończy program z kodem sukcesu
    except Exception as e:  # Łapie wszelkie inne wyjątki
        print(f"\n\n[ERROR] WYSTĄPIŁ BŁĄD KRYTYCZNY: {e}")  # Wypisuje komunikat o błędzie
        import traceback  # Importuje moduł do drukowania stack trace

        traceback.print_exc()  # Wypisuje pełny stack trace
        sys.exit(1)