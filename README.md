# ANFIS-Wine-Quality-Classification  

> 🚀 **ANFIS vs Classical Machine Learning** - Comprehensive comparison of Adaptive Neuro-Fuzzy Inference System with traditional ML algorithms on real-world datasets

## 📋 Description

Welcome to the **ANFIS Wine Quality Classification** repository! This comprehensive project compares ANFIS (Adaptive Neuro-Fuzzy Inference System) with classical machine learning algorithms on two real-world datasets: Wine Quality Classification and Concrete Compressive Strength Prediction. The system demonstrates the power of neuro-fuzzy systems by combining the learning capabilities of neural networks with the interpretability of fuzzy logic systems.

Built with TensorFlow/Keras and featuring automated pipeline execution, this project showcases best practices in fuzzy systems implementation, machine learning model comparison, and scientific experimentation. The system includes comprehensive data exploration, cross-validation, membership function visualization, and interactive Streamlit GUI for real-time predictions.

## 📁 Repository Structure

```

ANFIS-Wine-Quality-Classification/
├── 📁 data/ # Raw datasets
│ ├── 📁 wine-quality/
│ │ ├── 🍷 winequality-red.csv # Red wine dataset
│ │ └── 🍷 winequality-white.csv # White wine dataset
│ └── 📁 concrete-strength/
│ └── 🏗️ Concrete_Data.csv # Concrete strength dataset
├── 📁 models/ # Trained model weights (generated)
│ ├── anfis_2memb.weights.h5
│ ├── anfis_3memb.weights.h5
│ ├── neural_network.pkl
│ ├── svm_model.pkl
│ └── random_forest.pkl
├── 📁 results/ # Generated plots and metrics
│ ├── 📊 all_models_comparison.png
│ ├── 📈 overfitting_analysis.png
│ ├── 🧠 anfis_2memb_training.png
│ ├── 🧠 anfis_3memb_training.png
│ ├── 📉 membership_functions_visualization.png
│ ├── 📊 quality_distribution.png
│ ├── 🔥 correlation_matrix.png
│ └── \*.json (numerical results)
├── 🧠 anfis.py # ANFIS core implementation
├── 📊 data_exploration.py # Exploratory data analysis
├── 🔄 data_preprocessing.py # Data preparation and normalization
├── 🏋️ train_anfis.py # ANFIS model training
├── 🤖 train_comparison_models.py # Train NN, SVM, Random Forest
├── 📈 compare_all_models.py # Results comparison and visualization
├── 📉 visualize_membership_functions.py # Membership function plots
├── 🛠️ utils.py # Helper functions (NEW v1.1.0)
├── 📐 scaller.py # Scaler management (NEW v1.1.0)
├── 🎯 app.py # Streamlit web interface
├── 🚀 main.py # Main automated pipeline
├── 📋 requirements.txt # Python dependencies
├── 🔧 setup.sh # Linux/macOS setup script
├── 🔧 setup.bat # Windows setup script
├── 📖 MANUAL_INSTRUCTION.md # Detailed installation guide
└── 📖 README.md # Project documentation

```

## 🚀 Quick Start

### Prerequisites

- **Python** 3.8-3.12 (tested on 3.12)
- **pip** package manager
- **4GB RAM** minimum
- **~1GB disk space** for dependencies and datasets

### One-Command Automated Setup

#### Linux/macOS:

```bash
chmod +x setup.sh
./setup.sh
```

#### Windows:

```bash
setup.bat
```

### What the Setup Script Does:

1. ✅ Creates virtual environment
2. ✅ Installs all dependencies
3. ✅ Downloads and preprocesses datasets
4. ✅ Trains ANFIS models (2 & 3 membership functions)
5. ✅ Performs 5-fold cross-validation
6. ✅ Visualizes membership functions
7. ✅ Generates data exploration plots
8. ✅ Trains comparison models (NN, SVM, RF)
9. ✅ Creates comparison charts
10. ✅ Launches Streamlit GUI at [http://localhost:8501](http://localhost:8501)

**⏱️ Estimated time:** 15-30 minutes (CPU-dependent)

## ⚙️ System Requirements

### **Essential Tools:**

- **Python** 3.8-3.12 (Python 3.13+ not compatible with TensorFlow 2.17)
- **pip** package manager
- **4GB RAM** minimum (8GB recommended)
- **1GB disk space** for dependencies

### **Required Python Libraries:**

```
tensorflow==2.17.0
numpy==1.26.4
pandas==2.2.3
scikit-learn==1.5.2
matplotlib==3.9.2
seaborn==0.12.2
streamlit==1.39.0
h5py==3.12.1
pillow==11.0.0
```

### **Manual Installation:**

```bash
# Install dependencies
pip install -r requirements.txt

# Run automated pipeline
python main.py

# Or run individual steps (see Manual Execution section)
```

### **Development Environment:**

- **Code Editor** (VS Code, PyCharm, Jupyter Notebook)
- **Python Debugger** for development
- **Git** for version control

## ✨ Key Features

### **🧠 ANFIS Implementation**

- **5-Layer Takagi-Sugeno-Kang Architecture:**
  1. **Fuzzy Layer** - Gaussian membership functions with learned parameters
  2. **Rule Layer** - Fuzzy rule generation (T-norm multiplication)
  3. **Norm Layer** - Rule weight normalization
  4. **Defuzz Layer** - TSK-type defuzzification with linear consequents
  5. **Summation Layer** - Weighted output aggregation

- **Configurable Membership Functions:**
  - 2 membership functions: 2,048 fuzzy rules
  - 3 membership functions: 177,147 fuzzy rules

- **Advanced Training:**
  - Nadam optimizer (learning rate: 0.001)
  - Early stopping (patience: 10 epochs)
  - Model checkpointing (saves best weights)
  - 20 training epochs

### **📊 Two Real-World Datasets**

#### **1. Wine Quality Classification 🍷**

- **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/wine+quality)
- **Samples:** 6,497 (1,599 red + 4,898 white wines)
- **Features:** 11 physicochemical properties
  - Fixed acidity, volatile acidity, citric acid
  - Residual sugar, chlorides
  - Free/total sulfur dioxide
  - Density, pH, sulphates, alcohol
- **Task:** Binary classification (quality > 5 vs ≤ 5)
- **Variants:** Combined (all), red only, white only

#### **2. Concrete Compressive Strength 🏗️**

- **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Concrete+Compressive+Strength)
- **Samples:** 1,030
- **Features:** 8 concrete components
  - Cement, blast furnace slag, fly ash
  - Water, superplasticizer
  - Coarse/fine aggregate, age (days)
- **Task:** Regression (predict compressive strength in MPa)

### **🤖 Model Comparison**

| Model              | Type           | Configuration                       |
| ------------------ | -------------- | ----------------------------------- |
| **ANFIS**          | Neuro-Fuzzy    | 2 & 3 Gaussian membership functions |
| **Neural Network** | Deep Learning  | 16→Dropout→8→Dropout→1 architecture |
| **SVM**            | Kernel Methods | RBF kernel, C=1.0                   |
| **Random Forest**  | Ensemble       | 200 trees, max_depth=15             |

### **📈 Comprehensive Evaluation**

- **Cross-Validation:** 5-fold stratified (classification) / standard (regression)
- **Metrics:** Accuracy, MAE, MSE, train-test gap
- **Overfitting Analysis:** Train-test performance comparison
- **Statistical Significance:** Multiple random seeds

### **📊 Rich Visualizations**

- Training curves (accuracy/loss over epochs)
- Prediction scatter plots (predicted vs actual)
- Membership function plots for top features
- Correlation heatmaps
- Feature distribution histograms
- Model comparison bar charts
- Overfitting analysis plots
- Publication-ready 300 DPI PNG exports

### **🎯 Interactive GUI**

Streamlit web interface with:

- 🏠 **Dashboard** - Project overview and statistics
- 📊 **Model Results** - Comparison and ranking tables
- 🧠 **ANFIS Theory** - Architecture explanations and visualizations
- 📈 **Data Exploration** - Dataset insights and distributions
- 🍷 **Real-Time Prediction** - Interactive wine quality prediction

### **🔧 Modular Architecture**

- Separate modules for each functionality
- Clean separation of concerns (v1.1.0)
- Reusable utility functions
- Easy to extend and modify
- Well-documented code with Polish docstrings

## 🛠️ Technologies Used

- **TensorFlow 2.17** - Deep learning framework
- **Keras** - High-level neural networks API
- **NumPy** - Numerical computing
- **Pandas** - Data manipulation and analysis
- **Scikit-learn** - Machine learning algorithms
- **Matplotlib** - Data visualization
- **Seaborn** - Statistical data visualization
- **Streamlit** - Interactive web applications
- **H5py** - HDF5 file format for model weights

## 📖 Step-by-Step Manual Execution

### Step 1: Data Exploration 📊

```bash
python data_exploration.py
```

**What it does:**

- Downloads Wine Quality dataset (red + white)
- Combines datasets (6,497 samples total)
- Analyzes quality distribution (scale 3-9)
- Checks for missing values and feature correlations
- Generates visualizations

**Output:**

- ✅ `quality_distribution.png` - Quality distribution histogram
- ✅ `correlation_matrix.png` - Feature correlation heatmap

### Step 2: Data Preprocessing 🔄

```bash
python data_preprocessing.py
```

**What it does:**

- Transforms problem into binary classification:
  - **Class 0** (poor quality): quality ≤ 5
  - **Class 1** (good quality): quality > 5
- Selects 11 most important features
- Splits data: 80% training, 20% testing
- Applies StandardScaler normalization (critical for ANFIS!)
- Saves processed data to `.npy` files

**Output:**

- ✅ 5,197 training samples
- ✅ 1,300 test samples
- ✅ Class distribution: 2,384 poor / 4,113 good quality

### Step 3: ANFIS Training 🧠

```bash
python train_anfis.py
```

**What it does:**

- Trains 2 ANFIS models:
  - **ANFIS with 2 membership functions** (2,048 rules)
  - **ANFIS with 3 membership functions** (177,147 rules)
- Each model trains for 20 epochs
- Uses Nadam optimizer + binary cross-entropy
- Saves best model weights (ModelCheckpoint)
- Early stopping after 15 epochs without improvement
- Generates training plots for each model

**ANFIS Architecture:**

1. **FuzzyLayer** - Gaussian membership functions (μ(x) = exp(-(x-c)²/(2σ²)))
2. **RuleLayer** - Fuzzy rule generation (T-norm AND operation)
3. **NormLayer** - Rule weight normalization
4. **DefuzzLayer** - TSK defuzzification (linear combination)
5. **SummationLayer** - Output aggregation

**Results:**

- ✅ ANFIS (2 functions): Test Accuracy = **69.06%**
- ✅ ANFIS (3 functions): Test Accuracy = **76.48%**
- ✅ Models saved in `models/`
- ✅ Training plots in `results/`

**Execution time:** ~2 minutes

### Step 4: Comparison Models Training 🤖

```bash
python train_comparison_models.py
```

**What it does:**
Trains 3 classical machine learning models:

#### **Neural Network (NN)**

- Architecture: 16 → Dropout(0.3) → 8 → Dropout(0.2) → 1
- Activation functions: ReLU + Sigmoid
- Optimizer: Adam
- 50 epochs with early stopping

#### **Support Vector Machine (SVM)**

- Kernel: RBF (Radial Basis Function)
- Hyperparameters: C=1.0, gamma='scale'
- Trained on full dataset

#### **Random Forest**

- 200 decision trees
- max_depth=15
- Parallel training (n_jobs=-1)

**Results:**

- ✅ Neural Network: Test Accuracy = **75.69%**
- ✅ SVM: Test Accuracy = **77.85%**
- ✅ Random Forest: Test Accuracy = **83.23%** 🏆
- ✅ All models saved in `models/`

**Execution time:** ~5-10 minutes

### Step 5: Model Comparison 📈

```bash
python compare_all_models.py
```

**What it does:**

- Loads results from all 5 models
- Generates 2 comparison plots:
  - **all_models_comparison.png** - Train vs Test bar chart
  - **overfitting_analysis.png** - Train-Test gap analysis
- Displays detailed ranking table

**Final Rankings:**

```
🥇 #1: Random Forest     - 83.23% (overfitting: 14.46% ⚠️)
🥈 #2: SVM               - 77.85% (minimal overfitting: 1.47%)
🥉 #3: ANFIS (3 functions)- 76.48% (slight overfitting: 4.59%)
   #4: Neural Network    - 75.69% (minimal overfitting: 1.76%)
   #5: ANFIS (2 functions)- 69.06% (no overfitting: 0.75%)
```

### Step 6: Membership Function Visualization 📉

```bash
python visualize_membership_functions.py
```

**What it does:**

- Loads ANFIS model weights
- Plots Gaussian membership functions for 6 most important features
- Saves visualization to `membership_functions_visualization.png`

### Step 7: Launch Interactive GUI 🎯

```bash
streamlit run app.py
```

**Features:**

- Real-time wine quality prediction
- Model comparison dashboard
- ANFIS architecture explanations
- Data exploration tools
- Interactive visualizations

Access at: [http://localhost:8501](http://localhost:8501)

## 📊 Results Analysis

### Final Model Comparison

| Ranking | Model          | Test Accuracy | Train Accuracy | Overfitting | Interpretability    |
| ------- | -------------- | ------------- | -------------- | ----------- | ------------------- |
| 🥇      | Random Forest  | **83.23%**    | 97.69%         | 14.46% ⚠️   | ❌ Black box        |
| 🥈      | SVM            | **77.85%**    | 79.31%         | 1.47% ✅    | ❌ Black box        |
| 🥉      | ANFIS (3 MF)   | **76.48%**    | 81.08%         | 4.59% ✅    | ✅ **Fuzzy rules!** |
| 4       | Neural Network | **75.69%**    | 77.45%         | 1.76% ✅    | ❌ Black box        |
| 5       | ANFIS (2 MF)   | **69.06%**    | 69.81%         | 0.75% ✅    | ✅ **Fuzzy rules!** |

### Key Insights

✅ **ANFIS is Competitive!**

- ANFIS (3 functions) achieves 76.48% - only 6.75% below best model
- Better than classical Neural Network (75.69%)
- Minimal overfitting (4.59%)

✅ **ANFIS Provides Interpretability!**

- Visualized membership functions show learned patterns
- Identifiable fuzzy rules (e.g., "IF alcohol HIGH AND acidity LOW THEN quality GOOD")
- Other models are "black boxes"

⚠️ **Random Forest Overfits**

- Highest test accuracy (83.23%)
- But severe overfitting (14.46%)
- Train accuracy = 97.69% (nearly perfect fit to training data)

🔬 **3 Membership Functions >> 2 Membership Functions**

- +7.42% accuracy improvement (76.48% vs 69.06%)
- More rules = better data representation
- Computational trade-off: 2,048 rules vs 177,147 rules

## 🔬 Fuzzy Logic Elements in ANFIS

### Gaussian Membership Function

```
μ(x) = exp(-(x - c)² / (2σ²))
```

**Parameters (learned during training):**

- `c` - center of function
- `σ` - width/spread of function

### Fuzzy Rules Example

```
Rule 1: IF alcohol is HIGH AND acidity is LOW
        THEN quality is GOOD

Rule 2: IF alcohol is LOW AND acidity is HIGH
        THEN quality is POOR
```

### Takagi-Sugeno Defuzzification

```
Output = Σ(wᵢ × (aᵢx₁ + bᵢx₂ + ... + cᵢ))
```

where `wᵢ` are normalized rule weights

## 🆕 Version 1.1.0 Changes

### ✅ Optimizations Implemented

1. **🖼️ Fixed Matplotlib Blocking**
   - Added `matplotlib.use('Agg')` to all plotting scripts
   - Removed all `plt.show()` calls - plots save automatically
   - **Effect:** Pipeline executes without stopping for windows!

2. **📦 Business Logic Separation**
   - Created `utils.py` - ANFIS model and results loading functions
   - Created `scaller.py` - centralized scaler management
   - **Effect:** `app.py` contains only Streamlit UI code

3. **🚫 Extended .gitignore**
   - Ignores generated files (_.npy, _.h5, _.pkl, _.png)
   - **Effect:** Repository clean of binary artifacts

4. **📚 Complete Documentation**
   - `CHANGELOG.md` - detailed technical changes
   - `MANUAL_INSTRUCTION.md` - step-by-step installation guide
   - **Backward compatible:** All changes maintain compatibility ✅

## 🧰 Troubleshooting

### Issue: Streamlit doesn't launch automatically

**Solution:** Manually run `streamlit run app.py` after setup completes

### Issue: TensorFlow installation fails

**Solution:**

- Ensure Python 3.8-3.12 (TensorFlow 2.17 not compatible with Python 3.13+)
- Try: `pip install tensorflow==2.17.0 --no-cache-dir`

### Issue: Out of memory during training

**Solution:** Reduce batch size in `train_anfis.py` (line 95: `batch_size=16`)

### Issue: Matplotlib backend errors

**Solution:**

- Install: `pip install python3-tk` (Linux)
- Or use backend: `export MPLBACKEND=Agg` before running scripts

### Issue: Dataset download fails

**Solution:** Manually download datasets from UCI ML Repository and place in `data/` directory

## 🎓 Conclusions

1. **ANFIS combines best of both worlds:**
   - Learning like neural networks
   - Interpretation like expert systems

2. **3 membership functions significantly better than 2:**
   - +7.42% accuracy (76.48% vs 69.06%)
   - More rules = better data representation

3. **ANFIS vs Classical Models:**
   - Random Forest best but overfits
   - SVM solid choice (77.85%, minimal overfitting)
   - **ANFIS excellent compromise:** good accuracy + interpretability

4. **Wine Quality Problem:**
   - 11 numerical features, 6,497 samples
   - Class imbalance (37% poor / 63% good quality)
   - All models achieve >75% accuracy

## 🤝 Contributing

Contributions are highly welcomed! Here's how you can help:

- 🐛 **Report bugs** - Found an issue? Let us know!
- 💡 **Suggest improvements** - Have ideas for better features?
- 🔧 **Submit pull requests** - Share your enhancements and solutions
- 📖 **Improve documentation** - Help make the project clearer

Feel free to open issues or reach out through GitHub for any questions or suggestions.

## 👨‍💻 Authors

Created by:

- **Dawid Olko** - Project Lead
- **Piotr Smoła** - ML Implementation
- **Jakub Opar** - Data Analysis
- **Michał Pilecki** - Visualization

Course: Fuzzy Systems  
Supervisor: mgr inż. Marcin Mrukowicz  
Rzeszów University of Technology, 2025/2026

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---
