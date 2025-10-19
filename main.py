"""
GŁÓWNY SKRYPT PROJEKTU - Wine Quality Classification using ANFIS
Orkiestruje cały pipeline projektu: od danych do wyników

Autorzy: Dawid Olko, Piotr Smoła, Jakub Opar, Michał Pilecki
Uruchomienie: python main.py
"""

import subprocess
import sys
from datetime import datetime


def run_script(script_name, description):
    print(f"\n{'='*80}")
    print(f"  {description}")
    print(f"{'='*80}\n")

    result = subprocess.run([sys.executable, script_name], capture_output=False)

    if result.returncode != 0:
        print(f"\n[ERROR] Skrypt {script_name} zakończył się błędem!")
        return False

    print(f"\n[INFO] {description} - ZAKOŃCZONE\n")
    return True


def main():
    """Główna funkcja - wykonuje cały pipeline projektu"""

    print("\n" + "=" * 80)
    print("  PROJEKT: Porównanie ANFIS z Klasycznymi Modelami ML")
    print("  Wine Quality Classification")
    print("=" * 80)
    print(f"\nStart: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Lista kroków do wykonania
    steps = [
        ("data_exploration.py", "KROK 1/6: Eksploracja danych"),
        ("data_preprocessing.py", "KROK 2/6: Przetwarzanie danych"),
        ("train_anfis.py", "KROK 3/6: Trening modeli ANFIS"),
        ("train_comparison_models.py", "KROK 4/6: Trening modeli porównawczych"),
        (
            "visualize_membership_functions.py",
            "KROK 5/6: Wizualizacja funkcji przynależności",
        ),
        ("compare_all_models.py", "KROK 6/6: Porównanie wszystkich modeli"),
    ]

    # Wykonaj wszystkie kroki
    for script, description in steps:
        success = run_script(script, description)
        if not success and script in ["data_preprocessing.py", "train_anfis.py"]:
            # Krytyczne skrypty - przerwij jeżeli błąd
            print(f"\n[ERROR] KRYTYCZNY w {script}. Przerywam wykonywanie.")
            sys.exit(1)

    # Podsumowanie
    print("\n" + "=" * 80)
    print("  [INFO] PROJEKT ZAKOŃCZONY POMYŚLNIE!")
    print("=" * 80)
    print("\nWygenerowane pliki:")
    print("  => data/       - Zbiory danych (CSV, NPY)")
    print("  => models/     - Wytrenowane modele (.keras, .pkl)")
    print("  => results/    - Wykresy (PNG) i wyniki (JSON)")
    print("\nKolejne kroki:")
    print("  1. Sprawdź wykresy w folderze results/")
    print("  2. Uruchom GUI: streamlit run app.py")
    print(f"\nKoniec: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
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
