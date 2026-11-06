# ANFIS vs Classical Machine Learning

> 🤖 **A model that can be read out loud** — an Adaptive Neuro-Fuzzy Inference System compared against classical models on the same split, and asked to explain itself

**ANFIS vs Classical ML** puts an Adaptive Neuro-Fuzzy Inference System next to SVM and Random Forest on two real datasets — wine quality (classification) and concrete compressive strength (regression) — and measures them on identical splits with a single seed.

The point is not only the score. ANFIS is built here as a Keras model whose fuzzy layer keeps its membership functions accessible, so after training it can export its rules as JSON and plot the shapes it learned. A Random Forest gives you a number; this gives you *"if acidity is low and alcohol is high, then …"*.

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20-FF6F00?logo=tensorflow&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5%2B-F7931E?logo=scikitlearn&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-2.2%2B-150458?logo=pandas&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-app-FF4B4B?logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🎯 Key Features

- **ANFIS implemented, not imported** — `anfis.py` builds the five-layer architecture as Keras layers: a `FuzzyLayer` holding the membership parameters, rule firing strengths, normalisation, the consequent layer and the output. The parameters stay reachable after training, which is what makes the next two points possible.
- **Rules exported as JSON** — `to_rules_json()` turns the trained network into readable rules, so the model can be inspected rather than trusted.
- **Membership functions plotted** — `visualize_membership_functions.py` draws the shapes the network learned for each input, which is where a fuzzy model stops being a black box.
- **Both task types on the same footing** — wine quality as classification (SVC, RandomForestClassifier), concrete strength as regression (SVR, RandomForestRegressor), each against ANFIS on the same split.
- **Scaling inside the pipeline** — the classical baselines use `make_pipeline(StandardScaler(), …)`, so a scaler can never see the test fold; a comparison built otherwise would flatter the baseline.
- **One seed, one split, reproducible** — a single `SEED` drives every random component, and the dataset summary (`data/datasets_summary.json`) records exactly how many rows went where.
- **A Streamlit walkthrough** — `app.py` presents the datasets, the results, the plots and the learned rules as a browsable report.
- **One command runs the lot** — `main.py` orchestrates exploration, preprocessing, ANFIS training, baseline training and the comparison, with a progress bar per step.

---

## 📊 The Datasets

| Dataset                  | Task           | Rows  | Features | Note                                              |
| ------------------------ | -------------- | ----- | -------- | ------------------------------------------------- |
| **Wine Quality** (red + white) | Classification | 6 497 | 11       | Binarised quality; 2 384 / 4 113 class split.     |
| **Concrete Strength**    | Regression     | 1 030 | 8        | Compressive strength in MPa.                      |

Red and white wine are also evaluated separately, so the effect of the harder, smaller red subset is visible rather than averaged away.

---

## 🧩 The ANFIS Architecture

```
Input ─▶ FuzzyLayer          membership degrees per feature (Gaussian, learned μ and σ)
      ─▶ Rule layer          firing strength of every rule combination
      ─▶ Normalisation       relative strength of each rule
      ─▶ Consequent layer    a linear function per rule (Sugeno)
      ─▶ Output              weighted sum
```

`n_memb` controls how many fuzzy sets each input is divided into, so the rule count is `n_memb ^ n_input` — which is also why the number of inputs is kept small deliberately.

---

## 🎬 Demo

[![Watch the demo on YouTube](https://img.youtube.com/vi/O3X3MsWPVi8/hqdefault.jpg)](https://youtu.be/O3X3MsWPVi8)

---

## 🛠️ Technology Stack

| Technology       | Version | Role                                                      |
| ---------------- | ------- | --------------------------------------------------------- |
| **TensorFlow**   | 2.20    | The ANFIS layers and training loop.                       |
| **scikit-learn** | 1.5+    | SVM and Random Forest baselines, splits, metrics.         |
| **pandas**       | 2.2+    | Loading and preparing both datasets.                      |
| **NumPy**        | 2.0+    | Arrays and the saved intermediate `.npy` files.           |
| **Matplotlib**   | 3.9+    | Membership function plots and comparison charts.          |
| **Streamlit**    | —       | The walkthrough application.                              |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or newer
- pip

### 1. Clone the repository

```bash
git clone https://github.com/dawidolko/Comparison-ANFIS-Classical-Machine-Learning-Models-Python.git
cd Comparison-ANFIS-Classical-Machine-Learning-Models-Python
```

### 2. Install dependencies

```bash
# macOS / Linux
./setup.sh

# Windows
setup.bat

# or by hand
pip install -r requirements.txt
```

### 3. Run the whole study

```bash
python main.py
```

This walks the full pipeline: exploration → preprocessing → ANFIS training → baseline training → comparison, writing into `results/` and `models/`.

### 4. Or run the steps separately

```bash
python data_exploration.py             # dataset summary and distributions
python data_preprocessing.py           # splits, scaling, saved .npy arrays
python train_anfis.py                  # train the fuzzy model
python train_comparison_models.py      # SVM and Random Forest baselines
python compare_all_models.py           # the comparison tables
python visualize_membership_functions.py # what the fuzzy sets learned
```

### 5. Browse the results

```bash
streamlit run app.py
```

---

## 📁 Project Structure

```
Comparison-ANFIS-Classical-Machine-Learning-Models-Python/
├── anfis.py                          # ANFISModel and FuzzyLayer — the architecture
├── main.py                           # runs the full pipeline, step by step
├── data_exploration.py               # distributions and the dataset summary
├── data_preprocessing.py             # splits, scaling, .npy arrays
├── train_anfis.py                    # ANFIS training
├── train_comparison_models.py        # SVM and Random Forest, scaler inside the pipeline
├── compare_all_models.py             # metric tables across every model
├── visualize_membership_functions.py # the learned fuzzy sets
├── app.py                            # Streamlit walkthrough
├── utils.py  scaller.py  view_npy.py # helpers
├── data/
│   ├── wine-quality/                 # red and white CSVs
│   ├── concrete-strength/            # Concrete_Data.csv
│   └── datasets_summary.json         # exact row counts per split
├── models/                           # trained models
├── results/                          # metrics, tables, plots
└── docs/                             # theory notes, documentation, defence notes
```

---

## 📄 License

MIT © [Dawid Olko](https://dawidolko.pl)
