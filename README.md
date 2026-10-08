# Credit Card Fraud Detection: ML Classification, Calibration & Cost-Aware Risk Estimation

MSc Digital Finance and AI dissertation project — Loughborough University.

An end-to-end machine learning pipeline for detecting credit card fraud, correcting
probability miscalibration introduced by class-imbalance handling, and estimating
the financial cost of fraud for risk-management purposes.

## Overview

Fraud detection is a severe class-imbalance problem — in this dataset, only 0.17%
of transactions are fraudulent. This project trains and compares three models across
an interpretability–performance spectrum, corrects the probability distortion caused
by undersampling, and turns calibrated probabilities into a cost-based expected-loss
estimate.

## Key Findings

- **No single model wins outright.** Logistic Regression, Random Forest, and XGBoost
  each lead on different metrics (precision, recall, F1, AUC). Paired t-tests on
  stratified 5-fold CV AUC found no significant difference (p = 0.29 to 0.92).
- **Undersampling distorts predicted probabilities.** Applying the Dal Pozzolo et al.
  (2015) correction reduces Brier score by 95.0% to 96.8% across all three models.
- **Expected loss is concentration-prone.** Total expected loss on the test set is
  34,217.60 monetary units (the dataset's currency is unconfirmed). 72.7% comes from
  false alarms on genuine transactions, and one genuine transaction (25,691.16) makes
  up 51.5% of the total.

## Dataset

[ULB/Worldline Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
(Kaggle) — 284,807 European card transactions over ~48 hours, 492 confirmed fraud
cases. Features `V1`–`V28` are PCA-anonymised; `Amount` and `Time` are the only
interpretable columns.

**Note:** `creditcard.csv` is not included in this repository. Download it from the
Kaggle link above and place it in the project root before running the notebook.

## Tech Stack

Python · pandas · scikit-learn · XGBoost · SHAP · LIME · SciPy · matplotlib

## How to Run

1. Clone this repository
2. Download `creditcard.csv` from Kaggle (link above) into the project root
3. Install dependencies:
```
pip install pandas scikit-learn xgboost matplotlib numpy shap lime scipy
```
4. Open `fraud_pipeline.ipynb` in Jupyter and run all cells top to bottom
