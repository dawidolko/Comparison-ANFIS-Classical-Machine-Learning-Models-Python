"""
GŁÓWNY SKRYPT PROJEKTU - Klasyfikacja jakości wina z wykorzystaniem ANFIS
Orkiestruje cały pipeline projektu: od danych do wyników

Autorzy: Dawid Olko, Piotr Smoła, Jakub Opar, Michał Pilecki
Uruchomienie: python main.py
"""

import subprocess
import sys
from datetime import datetime
import time


def print_progress_bar(current, total, description, bar_length=50):
    """Wyświetlono pasek postępu w terminalu"""
    percent = 100 * (current / float(total))
    filled = int(bar_length * current // total)
    bar = '█' * filled + '░' * (bar_length - filled)
    
    print(f'\r🍷 Postęp: [{bar}] {percent:.1f}% - {description}', end='', flush=True)
    if current == total:
        print()


def run_script(script_name, description, step_num, total_steps):
    """Uruchomiono skrypt Pythona i wyświetlono postęp wykonania"""
    
    print(f"\n{'='*80}")
    print(f"  📊 KROK {step_num}/{total_steps}: {description}")
    print(f"{'='*80}\n")
    
    # Wyświetlono postęp przed rozpoczęciem kroku
    print_progress_bar(step_num - 1, total_steps, f"Rozpoczynanie: {description}")
    
    start_time = time.time()
    result = subprocess.run([sys.executable, script_name], capture_output=False)
    elapsed = time.time() - start_time

    if result.returncode != 0:
        print(f"\n❌ [BŁĄD] Skrypt {script_name} został zakończony z błędem!")
        return False

    # Wyświetlono postęp po zakończeniu kroku
    print_progress_bar(step_num, total_steps, f"✅ Zakończono w {elapsed:.1f}s")
    print(f"\n[INFO] {description} – ZAKOŃCZONO (czas: {elapsed:.1f}s)\n")
    return True


def main():
    """Wykonano główny pipeline projektu"""

    print("\n" + "=" * 80)
    print("  🍷 PROJEKT: Porównanie ANFIS z klasycznymi modelami ML")
    print("  Klasyfikacja jakości wina")
    print("=" * 80)
    print(f"\n⏰ Rozpoczęto: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Zdefiniowano listę kroków do wykonania
    steps = [
        ("data_exploration.py", "Eksploracja danych"),
        ("data_preprocessing.py", "Przetwarzanie danych"),
        ("train_anfis.py", "Trenowanie modeli ANFIS"),
        ("train_comparison_models.py", "Trenowanie modeli porównawczych"),
        ("visualize_membership_functions.py", "Wizualizacja funkcji przynależności"),
        ("compare_all_models.py", "Porównanie wszystkich modeli"),
    ]
    
    total_steps = len(steps)
    start_time = time.time()

    print(f"📋 Liczba kroków: {total_steps}")
    print(f"⏱️  Szacowany czas wykonania: ~10–15 minut\n")
    
    # Wyświetlono pasek postępu na początku
    print_progress_bar(0, total_steps, "Przygotowanie...")
    print()

    # Wykonano wszystkie kroki sekwencyjnie
    for idx, (script, description) in enumerate(steps, 1):
        success = run_script(script, description, idx, total_steps)
        if not success and script in ["data_preprocessing.py", "train_anfis.py"]:
            # Zidentyfikowano błąd krytyczny – przerwano wykonanie
            print(f"\n❌ [BŁĄD] KRYTYCZNY w {script}. Przerwano pipeline.")
            sys.exit(1)
    
    total_elapsed = time.time() - start_time

    # Wygenerowano podsumowanie wykonania
    print("\n" + "=" * 80)
    print("  ✅ PROJEKT ZOSTAŁ ZAKOŃCZONY POMYŚLNIE!")
    print("=" * 80)
    
    # Sformatowano całkowity czas wykonania
    minutes, seconds = divmod(int(total_elapsed), 60)
    time_str = f"{minutes} min {seconds} s" if minutes > 0 else f"{seconds} s"
    
    print(f"\n⏱️  Całkowity czas wykonania: {time_str}")
    print("\n📂 Wygenerowane zasoby:")
    print("  ✓ data/       – Zbiory danych (CSV, NPY)")
    print("  ✓ models/     – Wytrenowane modele (.keras, .pkl)")
    print("  ✓ results/    – Wykresy (PNG) oraz wyniki (JSON)")
    print("\n🚀 Zalecane następne kroki:")
    print("  1. Przejrzyj wykresy w katalogu results/")
    print("  2. Uruchom aplikację GUI: streamlit run app.py")
    print(f"\n⏰ Zakończono: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[INFO] Wykonanie zostało przerwane przez użytkownika (Ctrl+C)")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n[ERROR] WYSTĄPIŁ BŁĄD KRYTYCZNY: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)