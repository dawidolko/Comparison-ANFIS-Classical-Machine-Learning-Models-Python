import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def load_all_results():
    """Wczytuje wyniki wszystkich modeli"""
    results = {}

    # ANFIS modele
    with open('results/anfis_2memb_results.json', 'r') as f:
        results['ANFIS (2 funkcje)'] = json.load(f)

    with open('results/anfis_3memb_results.json', 'r') as f:
        results['ANFIS (3 funkcje)'] = json.load(f)

    # Inne modele
    with open('results/nn_results.json', 'r') as f:
        results['Neural Network'] = json.load(f)

    with open('results/svm_results.json', 'r') as f:
        results['SVM'] = json.load(f)

    with open('results/rf_results.json', 'r') as f:
        results['Random Forest'] = json.load(f)

    return results


def plot_comparison_bar_chart(results):
    """Tworzy wykres słupkowy porównujący wszystkie modele"""
    models = list(results.keys())
    train_acc = [results[m]['train_accuracy'] * 100 for m in models]
    test_acc = [results[m]['test_accuracy'] * 100 for m in models]

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 7))

    bars1 = ax.bar(x - width / 2, train_acc, width, label='Train Accuracy',
                   color='steelblue', alpha=0.8, edgecolor='black')
    bars2 = ax.bar(x + width / 2, test_acc, width, label='Test Accuracy',
                   color='coral', alpha=0.8, edgecolor='black')

    ax.set_xlabel('Model', fontsize=14, fontweight='bold')
    ax.set_ylabel('Dokładność (%)', fontsize=14, fontweight='bold')
    ax.set_title('Porównanie dokładności wszystkich modeli',
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha='right')
    ax.legend(fontsize=12)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim([60, 100])

    # Dodaj wartości na słupkach
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + 0.5,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig('results/all_models_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("\n✓ Wykres zapisany: results/all_models_comparison.png")


def plot_overfitting_analysis(results):
    """Analiza overfittingu - różnica Train vs Test"""
    models = list(results.keys())
    train_acc = [results[m]['train_accuracy'] * 100 for m in models]
    test_acc = [results[m]['test_accuracy'] * 100 for m in models]
    overfitting = [train - test for train, test in zip(train_acc, test_acc)]

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ['green' if x < 5 else 'orange' if x < 10 else 'red' for x in overfitting]
    bars = ax.barh(models, overfitting, color=colors, alpha=0.7, edgecolor='black')

    ax.set_xlabel('Różnica Train - Test (%)', fontsize=13, fontweight='bold')
    ax.set_title('Analiza Overfittingu (im mniejsze, tym lepiej)',
                 fontsize=15, fontweight='bold', pad=15)
    ax.axvline(x=5, color='green', linestyle='--', alpha=0.5, label='Dobry (<5%)')
    ax.axvline(x=10, color='orange', linestyle='--', alpha=0.5, label='Średni (<10%)')
    ax.legend(fontsize=11)
    ax.grid(axis='x', alpha=0.3)

    # Dodaj wartości
    for i, (bar, val) in enumerate(zip(bars, overfitting)):
        ax.text(val + 0.5, i, f'{val:.2f}%', va='center', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig('results/overfitting_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✓ Wykres zapisany: results/overfitting_analysis.png")


def create_summary_table(results):
    """Tworzy tabelę podsumowującą"""
    print("\n" + "=" * 80)
    print("SZCZEGÓŁOWA TABELA PORÓWNAWCZA")
    print("=" * 80)
    print(f"{'Model':<25} {'Train Acc':<12} {'Test Acc':<12} {'Gap':<10} {'Ranking'}")
    print("-" * 80)

    # Sortuj po test accuracy
    sorted_models = sorted(results.items(),
                           key=lambda x: x[1]['test_accuracy'],
                           reverse=True)

    for rank, (model, res) in enumerate(sorted_models, 1):
        train = res['train_accuracy'] * 100
        test = res['test_accuracy'] * 100
        gap = train - test

        medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else "  "

        print(f"{model:<25} {train:>6.2f}%     {test:>6.2f}%     "
              f"{gap:>5.2f}%    {medal} #{rank}")

    print("=" * 80)


if __name__ == "__main__":
    print("Ładuję wyniki wszystkich modeli...")
    results = load_all_results()

    print("\n1. Tworzę wykres porównawczy...")
    plot_comparison_bar_chart(results)

    print("\n2. Tworzę analizę overfittingu...")
    plot_overfitting_analysis(results)

    print("\n3. Generuję tabelę podsumowującą...")
    create_summary_table(results)

    print("\n" + "=" * 80)
    print("✓ WSZYSTKO GOTOWE!")
    print("=" * 80)