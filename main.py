"""
MAIN PROJECT SCRIPT - Wine quality classification using ANFIS
Orchestrates the whole project pipeline: from data to results

Authors: Dawid Olko, Piotr Smoła, Jakub Opar, Michał Pilecki
Usage: python main.py
"""

import subprocess
import sys
from datetime import datetime
import time


def print_progress_bar(current, total, description, bar_length=50):
    """Show a progress bar in the terminal"""
    percent = 100 * (current / float(total))
    filled = int(bar_length * current // total)
    bar = '█' * filled + '░' * (bar_length - filled)
    
    print(f'\r🍷 Progress: [{bar}] {percent:.1f}% - {description}', end='', flush=True)
    if current == total:
        print()


def run_script(script_name, description, step_num, total_steps):
    """Run a Python script and show its progress"""
    
    print(f"\n{'='*80}")
    print(f"  📊 STEP {step_num}/{total_steps}: {description}")
    print(f"{'='*80}\n")
    
    # Show progress before the step starts
    print_progress_bar(step_num - 1, total_steps, f"Starting: {description}")
    
    start_time = time.time()
    result = subprocess.run([sys.executable, script_name], capture_output=False)
    elapsed = time.time() - start_time

    if result.returncode != 0:
        print(f"\n❌ [ERROR] Script {script_name} exited with an error!")
        return False

    # Show progress after the step finishes
    print_progress_bar(step_num, total_steps, f"✅ Finished in {elapsed:.1f}s")
    print(f"\n[INFO] {description} - FINISHED (time: {elapsed:.1f}s)\n")
    return True


def main():
    """Run the main project pipeline"""

    print("\n" + "=" * 80)
    print("  🍷 PROJECT: Comparing ANFIS with classical ML models")
    print("  Wine quality classification")
    print("=" * 80)
    print(f"\n⏰ Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    steps = [
        ("data_exploration.py", "Data exploration"),
        ("data_preprocessing.py", "Data preprocessing"),
        ("train_anfis.py", "Training ANFIS models"),
        ("train_comparison_models.py", "Training comparison models"),
        ("visualize_membership_functions.py", "Membership function visualisation"),
        ("compare_all_models.py", "Comparing all models"),
    ]
    
    total_steps = len(steps)
    start_time = time.time()

    print(f"📋 Steps: {total_steps}")
    print(f"⏱️  Estimated run time: ~10-15 minutes\n")
    
    print_progress_bar(0, total_steps, "Przygotowanie...")
    print()

    # Run all steps sequentially
    for idx, (script, description) in enumerate(steps, 1):
        success = run_script(script, description, idx, total_steps)
        if not success and script in ["data_preprocessing.py", "train_anfis.py"]:
            # Critical error: abort execution
            print(f"\n❌ [ERROR] CRITICAL in {script}. Pipeline aborted.")
            sys.exit(1)
    
    total_elapsed = time.time() - start_time

    # Print the run summary
    print("\n" + "=" * 80)
    print("  ✅ PROJECT FINISHED SUCCESSFULLY!")
    print("=" * 80)
    
    # Format the total run time
    minutes, seconds = divmod(int(total_elapsed), 60)
    time_str = f"{minutes} min {seconds} s" if minutes > 0 else f"{seconds} s"
    
    print(f"\n⏱️  Total run time: {time_str}")
    print("\n📂 Generated assets:")
    print("  ✓ data/       - Datasets (CSV, NPY)")
    print("  ✓ models/     - Trained models (.keras, .pkl)")
    print("  ✓ results/    - Plots (PNG) and results (JSON)")
    print("\n🚀 Recommended next steps:")
    print("  1. Review the plots in results/")
    print("  2. Start the GUI app: streamlit run app.py")
    print(f"\n⏰ Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[INFO] Execution interrupted by the user (Ctrl+C)")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n[ERROR] CRITICAL ERROR: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)