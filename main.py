"""
GŁÓWNY SKRYPT PROJEKTU - Wine Quality Classification using ANFIS
Orkiestruje cały pipeline projektu: od danych do wyników

Autorzy: Dawid Olko, Piotr Smoła, Jakub Opar, Michał Pilecki
Uruchomienie: python main.py
"""

import subprocess
import sys
from datetime import datetime
import time


def print_progress_bar(current, total, description, bar_length=50):
    """Wyświetla pasek postępu w terminalu"""
    percent = 100 * (current / float(total))
    filled = int(bar_length * current // total)
    bar = '█' * filled + '░' * (bar_length - filled)
    
    print(f'\r🍷 Postęp: [{bar}] {percent:.1f}% - {description}', end='', flush=True)
    if current == total:
        print()  # Nowa linia po zakończeniu


def run_script(script_name, description, step_num, total_steps):
    """Uruchamia skrypt Pythona i pokazuje postęp"""
    
    print(f"\n{'='*80}")
    print(f"  📊 KROK {step_num}/{total_steps}: {description}")
    print(f"{'='*80}\n")
    
    # Pokazanie aktualnego postępu przed rozpoczęciem
    print_progress_bar(step_num - 1, total_steps, f"Rozpoczynam: {description}")
    
    start_time = time.time()
    result = subprocess.run([sys.executable, script_name], capture_output=False)
    elapsed = time.time() - start_time

    if result.returncode != 0:
        print(f"\n❌ [ERROR] Skrypt {script_name} zakończył się błędem!")
        return False

    # Pokazanie postępu po zakończeniu
    print_progress_bar(step_num, total_steps, f"✅ Zakończono w {elapsed:.1f}s")
    print(f"\n[INFO] {description} - ZAKOŃCZONE (czas: {elapsed:.1f}s)\n")
    return True


def main():
    """Główna funkcja - wykonuje cały pipeline projektu"""

    print("\n" + "=" * 80)
    print("  🍷 PROJEKT: Porównanie ANFIS z Klasycznymi Modelami ML")
    print("  Wine Quality Classification")
    print("=" * 80)
    print(f"\n⏰ Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Lista kroków do wykonania
    steps = [
        ("data_exploration.py", "Eksploracja danych"),
        ("data_preprocessing.py", "Przetwarzanie danych"),
        ("train_anfis.py", "Trening modeli ANFIS"),
        ("train_comparison_models.py", "Trening modeli porównawczych"),
        ("visualize_membership_functions.py", "Wizualizacja funkcji przynależności"),
        ("compare_all_models.py", "Porównanie wszystkich modeli"),
    ]
    
    total_steps = len(steps)
    start_time = time.time()

    print(f"📋 Całkowita liczba kroków: {total_steps}")
    print(f"⏱️  Szacowany czas: ~10-15 minut\n")
    
    # Pokazanie paska postępu na początku
    print_progress_bar(0, total_steps, "Przygotowanie...")
    print()

    # Wykonaj wszystkie kroki
    for idx, (script, description) in enumerate(steps, 1):
        success = run_script(script, description, idx, total_steps)
        if not success and script in ["data_preprocessing.py", "train_anfis.py"]:
            # Krytyczne skrypty - przerwij jeżeli błąd
            print(f"\n❌ [ERROR] KRYTYCZNY w {script}. Przerywam wykonywanie.")
            sys.exit(1)
    
    total_elapsed = time.time() - start_time

    # Podsumowanie
    print("\n" + "=" * 80)
    print("  ✅ PROJEKT ZAKOŃCZONY POMYŚLNIE!")
    print("=" * 80)
    
    # Formatowanie czasu wykonania
    minutes, seconds = divmod(int(total_elapsed), 60)
    time_str = f"{minutes}m {seconds}s" if minutes > 0 else f"{seconds}s"
    
    print(f"\n⏱️  Całkowity czas wykonania: {time_str}")
    print("\n📂 Wygenerowane pliki:")
    print("  ✓ data/       - Zbiory danych (CSV, NPY)")
    print("  ✓ models/     - Wytrenowane modele (.keras, .pkl)")
    print("  ✓ results/    - Wykresy (PNG) i wyniki (JSON)")
    print("\n🚀 Kolejne kroki:")
    print("  1. Sprawdź wykresy w folderze results/")
    print("  2. Uruchom GUI: streamlit run app.py")
    print(f"\n⏰ Koniec: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[INFO] Przerwano przez użytkownika (Ctrl+C)")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n[ERROR] BŁĄD KRYTYCZNY: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
