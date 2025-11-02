import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from anfis import ANFISModel
import os
import argparse


def visualize_membership_functions(n_memb=2, dataset='all'):
    print(f"\nWizualizacja MF: dataset={dataset}, n_memb={n_memb}")
    
    model_path = f'models/anfis_{dataset}_best_{n_memb}memb.weights.h5'
    
    if dataset == 'concrete':
        X_train = np.load('data/concrete-strength/X_train.npy')
    elif dataset == 'all':
        X_train = np.load('data/X_train.npy')
    else:
        X_train = np.load(f'data/X_train_{dataset}.npy')
    
    if not os.path.exists(model_path):
        print(f"✗ Model {model_path} nie istnieje!")
        return
    
    n_features = X_train.shape[1]
    anfis_model = ANFISModel(n_input=n_features, n_memb=n_memb, batch_size=32)
    anfis_model.model.load_weights(model_path)
    anfis_model.update_weights()
    centers, sigmas = anfis_model.get_membership_functions()
    
    # Nazwy cech zależą od datasetu
    if dataset == 'concrete':
        feature_names = ['cement', 'blast furnace slag', 'fly ash', 'water',
                        'superplasticizer', 'coarse aggregate', 'fine aggregate', 'age']
        important_features = list(range(min(6, n_features)))  # pierwszych 6 lub mniej jeśli dataset ma mniej
    else:
        feature_names = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
                        'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
                        'pH', 'sulphates', 'alcohol']
        important_features = [10, 1, 8, 9, 0, 7]
    
    # Upewnij się że important_features nie wykraczają poza zakres
    important_features = [f for f in important_features if f < n_features]
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    x_range = np.linspace(-3, 3, 300)
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    for idx, feat_idx in enumerate(important_features):
        c = centers[:, feat_idx]
        sigma = sigmas[:, feat_idx]
        for i in range(n_memb):
            mu = np.exp(-(x_range - c[i])**2 / (2 * sigma[i]**2))
            axes[idx].plot(x_range, mu, color=colors[i % len(colors)], linewidth=2,
                          label=f'MF {i+1}')
        axes[idx].set_title(feature_names[feat_idx])
        axes[idx].set_xlabel('Normalized value')
        axes[idx].set_ylabel('Membership μ(x)')
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)
        axes[idx].set_ylim([-0.05, 1.05])
    
    plt.suptitle(f'ANFIS Membership Functions ({dataset}, {n_memb} MF)')
    plt.tight_layout()
    output_path = f'results/membership_functions_{dataset}_{n_memb}memb.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Zapisano {output_path}")


if __name__ == "__main__":
    os.makedirs('results', exist_ok=True)
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+', default=['all'], choices=['concrete', 'all', 'red', 'white'])
    parser.add_argument('--memb', nargs='+', type=int, default=[2, 3])
    args = parser.parse_args()
    
    for dataset in args.datasets:
        for n_memb in args.memb:
            try:
                visualize_membership_functions(n_memb, dataset)
            except Exception as e:
                print(f"✗ Błąd dla dataset={dataset}, n_memb={n_memb}: {e}")
    
    print("\n✓ Wizualizacja MF zakończona!")
