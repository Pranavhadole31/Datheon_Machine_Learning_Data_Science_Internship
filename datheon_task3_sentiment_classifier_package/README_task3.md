# Datheon Task 3 — Customer Churn Classifier

Random Forest and XGBoost classification for telecom customer churn.

## Run locally
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook customer_churn_classifier.ipynb
```

## Methods
- Convert `TotalCharges` to numeric and handle missing values with a pipeline
- One-hot encode categorical columns
- Stratified train/test split
- Random Forest class weighting and XGBoost `scale_pos_weight`
- Five-fold stratified cross-validation
- ROC-AUC, ROC curves, confusion matrices, classification report
- Top 10 feature importance visualization

The notebook reports measured scores. ROC-AUC above 0.80 is a target, not a guaranteed result. Run all cells before reporting results.
