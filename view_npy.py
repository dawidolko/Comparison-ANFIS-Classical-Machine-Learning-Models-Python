#!/usr/bin/env python3

# python view_npy.py results/mf_sigmas_before_all_2memb.npy

"""
Prosty skrypt do przeglądania plików .npy
Użycie: python view_npy.py <ścieżka_do_pliku.npy>
        python view_npy.py  (bez argumentów = pokaż wszystkie z results/)
"""
import numpy as np
import sys
import os

def show_npy(filepath):
    """Wyświetla zawartość pliku .npy"""
    data = np.load(filepath)
    print(f"\n{'='*60}")
    print(f"📁 {filepath}")
    print(f"{'='*60}")
    print(f"Shape: {data.shape}")
    print(f"Dtype: {data.dtype}")
    print(f"Min: {data.min():.4f}, Max: {data.max():.4f}, Mean: {data.mean():.4f}")
    print(f"\nZawartość:")
    print(data)
    print()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Podano konkretny plik
        filepath = sys.argv[1]
        if os.path.exists(filepath):
            show_npy(filepath)
        else:
            print(f"❌ Plik nie istnieje: {filepath}")
    else:
        # Pokaż listę dostępnych plików
        print("\n📂 Dostępne pliki .npy w results/:")
        print("-" * 40)
        npy_files = sorted([f for f in os.listdir('results') if f.endswith('.npy')])
        for i, f in enumerate(npy_files, 1):
            print(f"  {i:2}. {f}")
        
        print("\n💡 Użycie:")
        print("   python view_npy.py results/mf_centers_after_all_2memb.npy")
        print("   python view_npy.py data/X_train.npy")
