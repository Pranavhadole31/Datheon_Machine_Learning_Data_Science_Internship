# Datheon Task 1 — Iris Classification Benchmark

A reproducible multi-class classification benchmark comparing Logistic Regression and Decision Tree on the Iris dataset.

## Contents
- `datheon_task1_iris_classification.ipynb`: end-to-end notebook with EDA, preprocessing, model training, metrics, confusion matrices, and decision boundaries.
- `requirements.txt`: Python dependencies.

## Run locally
```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
pip install -r requirements.txt
jupyter notebook datheon_task1_iris_classification.ipynb
```

Run all cells from top to bottom. The notebook downloads the supplied Iris CSV and falls back to Scikit-Learn's bundled Iris data if the download is unavailable.

## Methods
- Stratified 80/20 train-test split
- StandardScaler in a Scikit-Learn Pipeline
- Logistic Regression and Decision Tree
- Accuracy, macro precision, macro recall, macro F1-score
- Classification reports and confusion matrix heatmaps
- Decision-boundary plots using petal length and petal width

## Notes
The test set is used for evaluation, not tuning. For more robust performance estimates, add cross-validation and hyperparameter selection using training data only.
