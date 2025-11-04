# 📘 Manual Installation Instructions

This guide provides step-by-step instructions for setting up and running the ANFIS comparison project from scratch.

---

## 📋 Table of Contents

1. [System Requirements](#system-requirements)
2. [Installation Methods](#installation-methods)
   - [Method 1: Automated Setup (Recommended)](#method-1-automated-setup-recommended)
   - [Method 2: Manual Installation](#method-2-manual-installation)
3. [Verifying Installation](#verifying-installation)
4. [Running Individual Components](#running-individual-components)
5. [Troubleshooting](#troubleshooting)

---

## 💻 System Requirements

### Minimum Requirements
- **OS**: Linux, macOS, or Windows 10+
- **Python**: 3.8 - 3.12 (⚠️ Python 3.13+ not supported due to TensorFlow compatibility)
- **RAM**: 4 GB
- **Disk Space**: 1 GB free space
- **Internet**: Required for downloading dependencies

### Recommended Requirements
- **RAM**: 8 GB or more
- **CPU**: Multi-core processor (4+ cores)
- **GPU**: CUDA-compatible GPU (optional, speeds up TensorFlow training)

### Software Dependencies
- **Python 3.8+** with pip
- **Git** (for cloning repository)
- **bash** (Linux/macOS) or **cmd/PowerShell** (Windows)

---

## 🚀 Installation Methods

### Method 1: Automated Setup (Recommended)

This is the **fastest and easiest** way to get started.

#### Step 1: Clone Repository

```bash
git clone https://github.com/dawidolko/Comparison-ANFIS-Classical-Machine-Learning-Models-Python.git
cd Comparison-ANFIS-Classical-Machine-Learning-Models-Python
```

#### Step 2: Run Setup Script

**Linux/macOS:**
```bash
chmod +x setup.sh
./setup.sh
```

**Windows:**
```cmd
setup.bat
```

#### Step 3: Wait for Completion

The script will automatically:
- Create virtual environment
- Install all dependencies (TensorFlow, scikit-learn, Streamlit, etc.)
- Preprocess datasets (Wine Quality + Concrete Strength)
- Train ANFIS models with 2 & 3 membership functions
- Perform 5-fold cross-validation
- Generate membership function visualizations
- Create data exploration plots
- Train comparison models (Neural Network, SVM, Random Forest)
- Generate model comparison charts
- Launch Streamlit GUI

**⏱️ Estimated Time**: 15-30 minutes (varies by CPU speed)

#### Step 4: Access GUI

Once complete, your browser should automatically open:
```
http://localhost:8501
```

If not, manually navigate to that URL.

---

### Method 2: Manual Installation

For users who want more control over the installation process.

#### Step 1: Clone Repository

```bash
git clone https://github.com/dawidolko/Comparison-ANFIS-Classical-Machine-Learning-Models-Python.git
cd Comparison-ANFIS-Classical-Machine-Learning-Models-Python
```

#### Step 2: Create Virtual Environment

**Linux/macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows (cmd):**
```cmd
python -m venv venv
venv\Scripts\activate.bat
```

**Windows (PowerShell):**
```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

#### Step 3: Upgrade pip

```bash
python -m pip install --upgrade pip
```

#### Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

**Expected packages** (see `requirements.txt` for exact versions):
- `tensorflow==2.17.0`
- `numpy==1.26.4`
- `pandas==2.2.3`
- `scikit-learn==1.5.2`
- `matplotlib==3.9.2`
- `seaborn==0.12.2`
- `streamlit==1.39.0`
- `Pillow==10.4.0`

#### Step 5: Data Preprocessing

```bash
python data_preprocessing.py
```

**Expected Output:**
```
Dataset all: 6497 samples, train=5197, test=1300
Dataset red: 1599 samples, train=1279, test=320
Dataset white: 4898 samples, train=3918, test=980
✓ Dane dla all, red, white zapisane!

Concrete: 1030 samples, train=824, test=206
✓ Dane concrete zapisane!
```

**Generated Files** (in `data/`):
- `X_train.npy`, `X_test.npy`, `y_train.npy`, `y_test.npy` (all wines)
- `X_train_red.npy`, `X_test_red.npy`, `y_train_red.npy`, `y_test_red.npy`
- `X_train_white.npy`, `X_test_white.npy`, `y_train_white.npy`, `y_test_white.npy`
- `scaler_all.pkl`, `scaler_red.pkl`, `scaler_white.pkl`
- `concrete-strength/X_train.npy`, `X_test.npy`, `y_train.npy`, `y_test.npy`
- `concrete-strength/scaler.pkl`

#### Step 6: Train ANFIS Models

```bash
python train_anfis.py --datasets concrete all red white --memb 2 3 --epochs 20 --cv
```

**Arguments:**
- `--datasets`: Which datasets to train on (concrete, all, red, white)
- `--memb`: Number of membership functions (2, 3)
- `--epochs`: Maximum training epochs (20)
- `--cv`: Enable 5-fold cross-validation

**Training Time**:
- Per model: 1-5 minutes
- Total (4 datasets × 2 MF configs): ~10-20 minutes

**Generated Files** (in `results/`):
- `anfis_{dataset}_{memb}memb_results.json`
- `anfis_{dataset}_{memb}memb_training.png`
- `anfis_{dataset}_{memb}memb_fit_train.png`
- `anfis_{dataset}_{memb}memb_rules.json`
- `anfis_{dataset}_{memb}memb_cv.json`

#### Step 7: Visualize Membership Functions

```bash
python visualize_membership_functions.py --datasets concrete all red white --memb 2 3
```

**Generated Files**:
- `membership_functions_{dataset}_{memb}memb.png`

#### Step 8: Data Exploration

```bash
python data_exploration.py
```

**Generated Files**:
- `wine_class_distribution.png`
- `wine_correlation.png`
- `wine_feature_distributions.png`
- `wine_pairplot.png`
- `concrete_distribution.png`
- `concrete_correlation.png`

#### Step 9: Train Comparison Models

```bash
python train_comparison_models.py
```

**Models Trained**:
- Neural Network (3-layer MLP)
- SVM (RBF kernel)
- Random Forest (300 trees)

**Generated Files**:
- `nn_results.json`, `nn_best.keras`
- `svm_results.json`
- `rf_results.json`

#### Step 10: Generate Comparison Plots

```bash
python compare_all_models.py
```

**Generated Files**:
- `model_comparison_bar.png`
- `overfitting_analysis.png`

#### Step 11: Launch Streamlit GUI

```bash
streamlit run app.py
```

Your browser will open at `http://localhost:8501`

---

## ✅ Verifying Installation

### Check Python Version

```bash
python --version
# Should output: Python 3.8.x - 3.12.x
```

### Check Installed Packages

```bash
pip list | grep -E "(tensorflow|scikit-learn|streamlit)"
```

**Expected Output:**
```
scikit-learn    1.5.2
streamlit       1.39.0
tensorflow      2.17.0
```

### Verify Dataset Files

```bash
ls data/wine-quality/
ls data/concrete-strength/
```

**Expected:**
```
winequality-red.csv
winequality-white.csv
Concrete_Data.csv
```

### Test ANFIS Import

```bash
python -c "from anfis import ANFISModel; print('✓ ANFIS module OK')"
```

---

## 🎮 Running Individual Components

### Train Specific Dataset

```bash
# Only concrete with 2 membership functions
python train_anfis.py --datasets concrete --memb 2 --epochs 20

# Only red wine with 3 membership functions
python train_anfis.py --datasets red --memb 3 --epochs 20 --cv
```

### Visualize Specific Model

```bash
python visualize_membership_functions.py --datasets all --memb 2
```

### Skip Training, Only Launch GUI

If you've already trained models:
```bash
streamlit run app.py
```

---

## 🐛 Troubleshooting

### Issue 1: Python Version Incompatible

**Error:**
```
ERROR: Could not find a version that satisfies the requirement tensorflow==2.17.0
```

**Solution:**
```bash
# Check Python version
python --version

# If 3.13+, downgrade to 3.12 or use pyenv/conda
conda install python=3.12
```

---

### Issue 2: TensorFlow GPU Not Detected

**Error:**
```
Could not load dynamic library 'libcudart.so.11.0'
```

**Solution:**
TensorFlow 2.17 requires CUDA 11.x. Either:
1. Install CUDA 11.8: https://developer.nvidia.com/cuda-11-8-0-download-archive
2. Use CPU-only version (already installed by default)

GPU is optional - training will work fine on CPU, just slower.

---

### Issue 3: Streamlit Port Already in Use

**Error:**
```
OSError: [Errno 48] Address already in use
```

**Solution:**
```bash
# Kill existing Streamlit process
pkill -f streamlit

# Or use different port
streamlit run app.py --server.port 8502
```

---

### Issue 4: Out of Memory During Training

**Error:**
```
ResourceExhaustedError: OOM when allocating tensor
```

**Solution:**
Edit `train_anfis.py` line 95:
```python
batch_size = 16  # Reduce from 32
```

---

### Issue 5: Missing Dataset Files

**Error:**
```
FileNotFoundError: [Errno 2] No such file or directory: 'data/wine-quality/winequality-red.csv'
```

**Solution:**
Ensure you cloned the full repository including `data/` folder:
```bash
git pull origin main
ls data/wine-quality/
ls data/concrete-strength/
```

---

### Issue 6: Virtual Environment Not Activating

**Windows PowerShell Error:**
```
cannot be loaded because running scripts is disabled
```

**Solution:**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
venv\Scripts\Activate.ps1
```

---

## 📞 Getting Help

If you encounter issues not covered here:

1. **Check Logs**: Training scripts print detailed error messages
2. **GitHub Issues**: https://github.com/dawidolko/Comparison-ANFIS-Classical-Machine-Learning-Models-Python/issues
3. **Stack Overflow**: Tag with `tensorflow`, `anfis`, `fuzzy-logic`

---

## 🎓 Next Steps

After successful installation:

1. **Explore GUI**: Navigate through 5 tabs (Home, ANFIS Results, Rules, Comparison, Data Analysis)
2. **Experiment**: Try different membership function counts
3. **Customize**: Modify hyperparameters in training scripts
4. **Extend**: Add your own datasets to `data_preprocessing.py`

---

## 📚 Additional Resources

- **ANFIS Paper**: https://ieeexplore.ieee.org/document/256541
- **TensorFlow Docs**: https://www.tensorflow.org/guide
- **Streamlit Docs**: https://docs.streamlit.io
- **UCI ML Repository**: https://archive.ics.uci.edu/ml/index.php

---

**Happy Learning! 🚀**
