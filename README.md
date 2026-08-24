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
  each lead on different metrics (precision, recall, F1, AUC) — there is no
  statistically clean winner, confirmed via stratified 5-fold cross-validation.
- **Undersampling distorts predicted probabilities.** Applying the Dal Pozzolo et al.
  (2015) correction reduces Brier score by ~95–97% across all three models.
- **Expected loss is concentration-prone.** A single high-value false positive
  accounts for over half of the total expected loss (~£34,200) across the test set —
  a key limitation of summed cost proxies.

## Dataset

[ULB/Worldline Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
(Kaggle) — 284,807 European card transactions over ~48 hours, 492 confirmed fraud
cases. Features `V1`–`V28` are PCA-anonymised; `Amount` and `Time` are the only
interpretable columns.

**Note:** `creditcard.csv` is not included in this repository. Download it from the
Kaggle link above and place it in the project root before running the notebook.

## Tech Stack

Python · pandas · scikit-learn · XGBoost · matplotlib

## How to Run

1. Clone this repository
2. Download `creditcard.csv` from Kaggle (link above) into the project root
3. Install dependencies:
```
pip install pandas scikit-learn xgboost matplotlib numpy
```
4. Open `fraud_pipeline.ipynb` in Jupyter and run all cells top to bottom
