#!/usr/bin/env python3

# python view_npy.py results/mf_sigmas_before_all_2memb.npy

"""
Simple script for browsing .npy files
Usage: python view_npy.py <path_to_file.npy>
       python view_npy.py  (no arguments = show all from results/)
"""
import numpy as np
import sys
import os

def show_npy(filepath):
    """Display the contents of a .npy file"""
    data = np.load(filepath)
    print(f"\n{'='*60}")
    print(f"📁 {filepath}")
    print(f"{'='*60}")
    print(f"Shape: {data.shape}")
    print(f"Dtype: {data.dtype}")
    print(f"Min: {data.min():.4f}, Max: {data.max():.4f}, Mean: {data.mean():.4f}")
    print(f"\nContents:")
    print(data)
    print()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # A specific file was given
        filepath = sys.argv[1]
        if os.path.exists(filepath):
            show_npy(filepath)
        else:
            print(f"❌ File does not exist: {filepath}")
    else:
        # Show the list of available files
        print("\n📂 Available .npy files in results/:")
        print("-" * 40)
        npy_files = sorted([f for f in os.listdir('results') if f.endswith('.npy')])
        for i, f in enumerate(npy_files, 1):
            print(f"  {i:2}. {f}")
        
        print("\n💡 Usage:")
        print("   python view_npy.py results/mf_centers_after_all_2memb.npy")
        print("   python view_npy.py data/X_train.npy")
