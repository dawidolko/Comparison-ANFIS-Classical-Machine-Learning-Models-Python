import json
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, List

RESULTS_DIR = "results"

def _last_from_history(d: Dict[str, Any], key: str):
    try:
        hist = d.get("history", {})
        arr = hist.get(key, None)
        if isinstance(arr, list) and len(arr) > 0:
            return arr[-1]
    except Exception:
        pass
    return None

def _coerce_float(x):
    try:
        return float(x)
    except Exception:
        return None

def normalize_result(name: str, raw: Dict[str, Any]) -> Dict[str, Any]:
    train_candidates = [
        "train_accuracy", "accuracy_train", "train_acc", "acc_train"
    ]
    test_candidates = [
        "test_accuracy", "accuracy_test", "test_acc", "acc_test",
        "accuracy"
    ]

    train_acc = None
    for k in train_candidates:
        if k in raw:
            train_acc = _coerce_float(raw[k]); break
    if train_acc is None:
        v = _last_from_history(raw, "accuracy")
        train_acc = _coerce_float(v) if v is not None else None

    test_acc = None
    for k in test_candidates:
        if k in raw:
            test_acc = _coerce_float(raw[k]); break
    if test_acc is None:
        v = _last_from_history(raw, "val_accuracy")
        test_acc = _coerce_float(v) if v is not None else None

    f1      = _coerce_float(raw.get("f1", None))
    roc_auc = _coerce_float(raw.get("roc_auc", None))
    pr_auc  = _coerce_float(raw.get("pr_auc", None))

    return {
        "name": name,
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "has_train": train_acc is not None
    }

def safe_load(name: str, path: str):
    if not os.path.exists(path):
        print(f"⚠️  Pomijam {name}: brak pliku {path}")
        return None
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"⚠️  Nie udało się wczytać {name} ({path}): {e}")
        return None

def load_all_results() -> List[Dict[str, Any]]:
    files = {
        "ANFIS (2 funkcje)": os.path.join(RESULTS_DIR, "anfis_2memb_results.json"),
        "ANFIS (3 funkcje)": os.path.join(RESULTS_DIR, "anfis_3memb_results.json"),
        "Neural Network":   os.path.join(RESULTS_DIR, "nn_results.json"),
        "SVM":              os.path.join(RESULTS_DIR, "svm_results.json"),
        "Random Forest":    os.path.join(RESULTS_DIR, "rf_results.json"),
    }

    results: List[Dict[str, Any]] = []
    for model_name, path in files.items():
        raw = safe_load(model_name, path)
        if raw is None:
            continue
        results.append(normalize_result(model_name, raw))

    return results

def plot_comparison_bar_chart(results: List[Dict[str, Any]]):
    models = [r["name"] for r in results]
    train = [r["train_accuracy"] * 100 if r["train_accuracy"] is not None else np.nan for r in results]
    test  = [r["test_accuracy"]  * 100 if r["test_accuracy"]  is not None else np.nan for r in results]

    x = np.arange(len(models))
    width = 0.35
    fig, ax = plt.subplots(figsize=(12, 7))

    bars1 = ax.bar(x - width/2, train, width, label="Train Accuracy", edgecolor="black", alpha=0.85)
    bars2 = ax.bar(x + width/2, test,  width, label="Test Accuracy",  edgecolor="black", alpha=0.85)

    ax.set_xlabel("Model", fontsize=14, fontweight="bold")
    ax.set_ylabel("Dokładność (%)", fontsize=14, fontweight="bold")
    ax.set_title("Porównanie dokładności modeli", fontsize=16, fontweight="bold", pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.legend(fontsize=12)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    vals = [v for v in (train + test) if not np.isnan(v)]
    if vals:
        lo = max(0, min(vals) - 5)
        hi = min(100, max(vals) + 5)
        ax.set_ylim([lo, hi])

    for bars in (bars1, bars2):
        for bar in bars:
            h = bar.get_height()
            if not np.isnan(h):
                ax.text(bar.get_x() + bar.get_width()/2., h + 0.5, f"{h:.1f}%", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = os.path.join(RESULTS_DIR, "all_models_comparison.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"✓ Wykres zapisany: {out}")

def plot_overfitting_analysis(results: List[Dict[str, Any]]):
    r = [x for x in results if x["has_train"] and x["test_accuracy"] is not None]
    if not r:
        print("ℹ️  Brak modeli z kompletem (train & test) — pomijam analizę overfittingu.")
        return

    models = [x["name"] for x in r]
    train  = [x["train_accuracy"] * 100 for x in r]
    test   = [x["test_accuracy"]  * 100 for x in r]
    gap    = [t - s for t, s in zip(train, test)]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["green" if g < 5 else "orange" if g < 10 else "red" for g in gap]
    bars = ax.barh(models, gap, color=colors, alpha=0.8, edgecolor="black")

    ax.set_xlabel("Różnica Train - Test (%)", fontsize=13, fontweight="bold")
    ax.set_title("Overfitting (mniej = lepiej)", fontsize=15, fontweight="bold", pad=15)
    ax.axvline(x=5,  color="green",  linestyle="--", alpha=0.5, label="<5% OK")
    ax.axvline(x=10, color="orange", linestyle="--", alpha=0.5, label="<10% średni")
    ax.legend(fontsize=11)
    ax.grid(axis="x", alpha=0.3)

    for i, (bar, val) in enumerate(zip(bars, gap)):
        ax.text(val + 0.5, i, f"{val:.2f}%", va="center", fontsize=10, fontweight="bold")

    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "overfitting_analysis.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"✓ Wykres zapisany: {out}")

def plot_auc_pr(results: List[Dict[str, Any]]):
    r = [x for x in results if (x["roc_auc"] is not None) or (x["pr_auc"] is not None)]
    if not r:
        print("ℹ️  Brak modeli z ROC-AUC/PR-AUC — pomijam wykres AUC.")
        return

    models  = [x["name"] for x in r]
    roc_auc = [x["roc_auc"] if x["roc_auc"] is not None else 0.0 for x in r]
    pr_auc  = [x["pr_auc"]  if x["pr_auc"]  is not None else 0.0 for x in r]

    x = np.arange(len(models))
    width = 0.35
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width/2, roc_auc, width, label="ROC-AUC", edgecolor="black", alpha=0.85)
    ax.bar(x + width/2, pr_auc,  width, label="PR-AUC",  edgecolor="black", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.set_ylim([0.0, 1.0])
    ax.set_ylabel("Wartość AUC", fontsize=13, fontweight="bold")
    ax.set_title("Porównanie ROC-AUC i PR-AUC", fontsize=15, fontweight="bold")
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.legend()

    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "auc_pr_comparison.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"✓ Wykres zapisany: {out}")

def create_summary_table(results: List[Dict[str, Any]]):
    print("\n" + "="*90)
    print("SZCZEGÓŁOWA TABELA PORÓWNAWCZA")
    print("="*90)
    print(f"{'Model':<25} {'Train Acc':<12} {'Test Acc':<12} {'F1':<10} {'ROC-AUC':<10} {'PR-AUC':<10} {'Gap':<8}")
    print("-"*90)

    def score_key(r):
        return -1.0 if r["test_accuracy"] is None else r["test_accuracy"]

    rows = []
    for r in sorted(results, key=score_key, reverse=True):
        train = f"{r['train_accuracy']*100:>6.2f}%" if r["train_accuracy"] is not None else "  —   "
        test  = f"{r['test_accuracy']*100:>6.2f}%"  if r["test_accuracy"]  is not None else "  —   "
        f1    = f"{r['f1']:.4f}"      if r["f1"]      is not None else " — "
        roc   = f"{r['roc_auc']:.4f}" if r["roc_auc"] is not None else " — "
        pr    = f"{r['pr_auc']:.4f}"  if r["pr_auc"]  is not None else " — "
        gap   = " — "
        if r["train_accuracy"] is not None and r["test_accuracy"] is not None:
            gap = f"{(r['train_accuracy'] - r['test_accuracy'])*100:>5.2f}%"

        print(f"{r['name']:<25} {train:<12} {test:<12} {f1:<10} {roc:<10} {pr:<10} {gap:<8}")

        rows.append({
            "model": r["name"],
            "train_accuracy": r["train_accuracy"],
            "test_accuracy": r["test_accuracy"],
            "f1": r["f1"],
            "roc_auc": r["roc_auc"],
            "pr_auc": r["pr_auc"],
            "gap": (r["train_accuracy"] - r["test_accuracy"]) if (r["train_accuracy"] is not None and r["test_accuracy"] is not None) else None
        })

    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(RESULTS_DIR, "summary_table.csv")
    try:
        import pandas as pd
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        print(f"\n✓ Zapisano tabelę: {csv_path}")
    except Exception:
        import csv
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n✓ Zapisano tabelę (csv, bez pandas): {csv_path}")

    print("="*90)

if __name__ == "__main__":
    print("Ładuję wyniki…")
    results = load_all_results()

    if not results:
        print("❗ Nie znaleziono żadnych wyników w folderze 'results/'.")
    else:
        print("\n1) Wykres porównawczy (Acc)…")
        plot_comparison_bar_chart(results)

        print("\n2) Analiza overfittingu…")
        plot_overfitting_analysis(results)

        print("\n3) Porównanie AUC (jeśli dostępne)…")
        plot_auc_pr(results)

        print("\n4) Tabela podsumowująca + CSV…")
        create_summary_table(results)

        print("\n✓ Gotowe!")
