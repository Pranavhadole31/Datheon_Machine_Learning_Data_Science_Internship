# Datheon Task 2 — California Housing Regression

An end-to-end benchmark comparing OLS, Ridge, and Lasso for median house-value prediction.

## Files
- `datheon_task2_california_housing_regression.ipynb`: EDA, feature engineering, model selection, and residual analysis.
- `requirements.txt`: Python dependencies.

## Run locally
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook datheon_task2_california_housing_regression.ipynb
```

The notebook tries the supplied CSV first and falls back to Scikit-Learn's built-in California Housing dataset if needed. It prints which source was loaded.

## Methods
- Median/mode imputation and one-hot encoding
- Polynomial and interaction features
- Standardization in a leakage-safe pipeline
- OLS baseline; Ridge and Lasso alpha tuning with 5-fold GridSearchCV
- Held-out RMSE, MAE, R²
- Actual-vs-predicted, residual, coefficient, and CV plots

Interpret results from the executed notebook; do not assume a model is best without checking the metrics.
