#!/bin/bash
set -e

echo "======================================"
echo "SETUP: Environment & Data Generation"
echo "======================================"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "======================================"
echo "STEP 1: Data Preprocessing"
echo "======================================"
python3 data_preprocessing.py

echo ""
echo "======================================"
echo "STEP 2: ANFIS Training (All Datasets + Cross-Validation)"
echo "======================================"
python3 train_anfis.py --datasets concrete all red white --memb 2 3 --epochs 20 --cv

echo ""
echo "======================================"
echo "STEP 3: Membership Functions Visualization"
echo "======================================"
python3 visualize_membership_functions.py --datasets concrete all red white --memb 2 3

echo ""
echo "======================================"
echo "STEP 4: Data Exploration (Plots)"
echo "======================================"
python3 data_exploration.py

echo ""
echo "======================================"
echo "STEP 5: Model Comparison"
echo "======================================"
python3 train_comparison_models.py
python3 compare_all_models.py

echo ""
echo "======================================"
echo "✅ ALL FILES GENERATED SUCCESSFULLY!"
echo "======================================"
echo ""
echo "Generated files:"
ls -1 results/anfis_*.png results/anfis_*.json 2>/dev/null | wc -l | xargs echo "  - ANFIS results:"
ls -1 results/membership_functions_*.png 2>/dev/null | wc -l | xargs echo "  - Membership functions:"
ls -1 results/wine_*.png results/concrete_*.png 2>/dev/null | wc -l | xargs echo "  - Data exploration:"
ls -1 results/model_comparison_*.png 2>/dev/null | wc -l | xargs echo "  - Model comparison:"

echo ""
echo "======================================"
echo "🚀 LAUNCHING STREAMLIT GUI"
echo "======================================"
echo ""
echo "Open your browser at: http://localhost:8501"
echo ""
streamlit run app.py
