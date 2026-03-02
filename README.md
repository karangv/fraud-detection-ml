# Credit Card Fraud Detection

Machine learning classification project for detecting fraudulent credit card transactions using XGBoost, SMOTE, and cost-sensitive threshold optimization.

## Overview

This project addresses the challenge of fraud detection in highly imbalanced datasets where only 0.17% of transactions are fraudulent (577:1 imbalance ratio). Through advanced techniques including synthetic oversampling and gradient boosting, the model achieves 92% fraud detection rate with 84% precision.

## Dataset

**Source:** [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)

- **Transactions:** 284,807
- **Features:** 30 (Time, V1-V28 PCA components, Amount)
- **Frauds:** 492 (0.173%)
- **Imbalance Ratio:** 577:1

## Project Structure

The project is organized into phases covering the complete ML workflow:

### Phase 1: Problem Framing & EDA
**File:** `problem_framing.ipynb`, `eda_&_preprocessing.ipynb`

- Dataset exploration and feature analysis
- Class imbalance quantification (577:1 ratio)
- Feature correlation analysis (V17, V14, V12 identified as top predictors)
- Demonstrated why accuracy is misleading for imbalanced data

### Phase 2: Baseline Modeling
**File:** `baselinemodeling.ipynb`

- Logistic Regression and Decision Tree baselines
- Stratified train-test split (80/20)
- Confusion matrix analysis
- Proved that 99.9% accuracy can coexist with poor fraud detection

**Baseline Results:**
- Logistic Regression: 61% recall, 88% precision
- Decision Tree: 80% recall, 81% precision

### Phase 3: Imbalance Handling
**File:** `imbalance_handling.ipynb`

- Class weighting
- Random undersampling
- SMOTE (Synthetic Minority Over-sampling Technique)
- SMOTE + Tomek Links

**Best Technique:** SMOTE (PR-AUC: 0.85, Recall: 91%)

### Phase 4-6: Advanced Optimization
**File:** `advanced_models.ipynb`

**Phase 4: XGBoost Training**
- SMOTE oversampling applied to training set
- Hyperparameter tuning across 54 configurations
- Cross-validation for robust evaluation
- Feature importance analysis

**Phase 5: Threshold Optimization**
- Systematic sweep of 99 thresholds (0.01 to 0.99)
- F1-optimal threshold identification
- High-recall threshold analysis
- Precision-recall trade-off visualization

**Phase 6: Cost-Sensitive Evaluation**
- Business cost matrix: $100 per missed fraud, $10 per false alarm
- Cost-optimal threshold calculation
- ROI quantification and business impact analysis

## Results

### Final Model Performance

| Metric | Value |
|--------|-------|
| **Recall (Fraud Detection Rate)** | 92% |
| **Precision** | 84% |
| **F1-Score** | 88% |
| **PR-AUC** | 0.90 |
| **False Alarm Rate** | 0.03% |

### Confusion Matrix (Optimized Threshold)
```
              Predicted
           Normal   Fraud
Actual N   56,847    17
       F        7    91
```

**Interpretation:**
- Catching 91 out of 98 frauds (92.9% detection rate)
- Only 7 frauds missed (7.1% miss rate)
- 17 false alarms (0.03% of normal transactions)

### Business Impact

**Cost Analysis:**
- Missed frauds: 7 × $100 = $700
- False alarms: 17 × $10 = $170
- Total cost: $870 per 56,962 transactions

**Baseline Comparison:**
- Baseline cost (Phase 2): $3,940
- Optimized cost: $870
- **Savings: $3,070 per test set**
- **Projected: $53,900 annual savings per 1M transactions**

## Key Techniques

### Imbalance Handling
- **SMOTE:** Balances training data from 577:1 to 1:1 ratio
- **scale_pos_weight:** XGBoost parameter weighting fraud samples higher
- Combined approach for optimal minority class learning

### Model Optimization
- Tested 54 hyperparameter combinations
- Selection prioritized recall with minimum 70% precision
- 5-fold stratified cross-validation

### Threshold Optimization
- Default 0.5 threshold is arbitrary for imbalanced data
- Optimal threshold (0.30-0.35) maximizes F1-score
- Cost-optimal threshold minimizes business losses

## Technologies

**Languages & Libraries:**
- Python 3.8+
- XGBoost 1.7+
- scikit-learn 1.3+
- imbalanced-learn 0.11+
- Pandas, NumPy
- Matplotlib, Seaborn

**ML Techniques:**
- Gradient Boosting (XGBoost)
- Synthetic oversampling (SMOTE)
- Cross-validation
- Hyperparameter optimization
- Threshold tuning

## Installation

```bash
pip install numpy pandas scikit-learn xgboost imbalanced-learn matplotlib seaborn jupyter
```

## Usage

1. **Download Dataset**
   ```bash
   # Download creditcard.csv from Kaggle
   # Place in project directory
   ```

2. **Run Notebooks Sequentially**
   ```bash
   jupyter notebook
   # Open and run:
   # 1. problem_framing.ipynb
   # 2. eda_&_preprocessing.ipynb
   # 3. baselinemodeling.ipynb
   # 4. imbalance_handling.ipynb
   # 5. advanced_models.ipynb (Phases 4-6)
   ```

## Key Learnings

### 1. Accuracy is Misleading
A model predicting all transactions as "Normal" achieves 99.83% accuracy but catches zero frauds. PR-AUC is the appropriate metric for imbalanced classification.

### 2. Imbalance Requires Special Handling
Default ML algorithms fail on 577:1 imbalance. SMOTE combined with XGBoost's scale_pos_weight dramatically improves minority class detection.

### 3. Threshold Optimization Matters
Default 0.5 threshold ignores class imbalance and business costs. Optimizing threshold from 0.5 to 0.35 improves performance and reduces business losses.

### 4. Business Costs Drive Decisions
Cost-sensitive evaluation reveals that optimizing for metrics alone misses business value. Incorporating $100 FN vs $10 FP costs changes optimal threshold selection.

## Future Enhancements

- Probability calibration (Platt scaling, isotonic regression)
- SHAP explainability for model interpretability
- FastAPI deployment for real-time predictions
- Streamlit dashboard for business users
- Model monitoring and drift detection

## Author

Karan - [GitHub](https://github.com/karangv)

## Acknowledgments

- Dataset: [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- Original dataset: ULB Machine Learning Group
