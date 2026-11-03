# Credit Card Fraud Detection

Detects fraudulent credit card transactions with XGBoost, SMOTE and a decision threshold tuned against a business cost matrix. Raw data goes in; a trained model, test metrics and a scoring command come out.

## Overview

Only 0.17% of transactions in this dataset are fraud (a 577:1 imbalance). On the held-out test set, the model catches 80.6% of fraud with 91.9% precision at the threshold chosen on the validation split. PR-AUC is 0.867.

## Dataset

[Kaggle Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud): 284,807 transactions, 492 of them fraud. There are 30 features: Time, Amount and the PCA components V1 to V28.

## Running the pipeline

You need Python 3.12 or newer.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

Put `creditcard.csv` in `data/`. If it isn't there, kagglehub downloads it.

```bash
make train evaluate
make predict INPUT=new.csv OUTPUT=scored.csv
make test
```

`make train` took about 27 minutes on 12 cores. To try a smaller run, shrink `param_grid` in `config.yaml`. Windows doesn't ship with make (`winget install ezwinports.make` adds it), and the commands behind the targets work on their own:

```bash
python -m fraud.train    --config config.yaml
python -m fraud.evaluate --config config.yaml
python -m fraud.predict  --config config.yaml --input new.csv --output scored.csv
```

`predict` needs the 30 feature columns, in any order, and ignores any other column, including `Class`. It writes `fraud_probability` and `is_fraud` for each input row, in the same order.

`config.yaml` holds the data path, seed, split sizes, parameter grid, selection metric, threshold strategy (`cost` or `f1`), cost matrix and artifact directory.

Training writes to `artifacts/`, which git ignores:

| File | Contents |
|------|----------|
| `model.joblib` | Fitted pipeline: StandardScaler, SMOTE, XGBoost |
| `threshold.json` | Chosen threshold, strategy, cost matrix, validation metrics at that threshold |
| `cv_results.csv` | Every config tried, ranked by CV score |
| `metrics.json` | Test metrics at 0.5 and at the chosen threshold, written by `evaluate` |

### How the data is used

The data is split 60/20/20 with stratification and a fixed seed. Each split has one job:

| Split | Used for |
|-------|----------|
| Train | 5-fold grid search over 54 configs, then fitting the final model |
| Validation | Choosing the decision threshold |
| Test | Final metrics, in `evaluate` only |

SMOTE sits inside the pipeline, so it only runs on training folds. It never touches validation, test or inference data. If you change the config because of what `metrics.json` says, the test split no longer gives an unbiased estimate.

## Project layout

```
config.yaml
src/fraud/    config, data, costs, pipeline, train, evaluate, predict
tests/        pytest suite on synthetic data, no download needed
notebooks/    the original analysis
```

The notebooks cover problem framing, EDA, baseline models and a comparison of imbalance techniques. They are kept as a record of how the approach was chosen, and the numbers printed in them come from an earlier method. The results below come from the pipeline.

## Results

Test set: 56,962 transactions, 98 of them fraud. PR-AUC is 0.867 and ROC-AUC is 0.981 at any threshold.

| Metric | Chosen threshold (0.94) | Default threshold (0.5) |
|--------|-------------------------|-------------------------|
| Recall | 80.6% | 84.7% |
| Precision | 91.9% | 78.3% |
| F1 | 85.9% | 81.4% |
| False-alarm rate | 0.012% | 0.040% |
| Total cost | $1,891 | $1,647 |

Confusion matrix at the chosen threshold:

```
              Predicted
           Normal   Fraud
Actual N   56,857       7
       F       19      79
```

Confusion matrix at the default threshold:

```
              Predicted
           Normal   Fraud
Actual N   56,841      23
       F       15      83
```

### Business impact

Each missed fraud costs $100 and each false alarm costs $10. Each caught fraud earns back $1.

| Threshold | Missed frauds | False alarms | Caught frauds | Total |
|-----------|---------------|--------------|---------------|-------|
| 0.94 (chosen) | 19 × $100 = $1,900 | 7 × $10 = $70 | 79 × -$1 = -$79 | $1,891 |
| 0.5 (default) | 15 × $100 = $1,500 | 23 × $10 = $230 | 83 × -$1 = -$83 | $1,647 |

The tuned threshold lowered false alarms but cost $244 more than the default on this test set, or about $4,284 per million transactions. It was the cheapest threshold on validation and didn't hold up on test. With about 100 frauds in each split, one extra missed fraud moves the total by $101, so this comparison is noisy in both directions.

## Key techniques

SMOTE balances the training data from 577:1 to 1:1. XGBoost's `scale_pos_weight` then adds further weight to the fraud class.

Configs are ranked by mean PR-AUC across the CV folds. Ties go to the lower standard deviation, then to the smaller parameter values. The best config is `scale_pos_weight=3.0`, `max_depth=8`, `learning_rate=0.1`, with a CV PR-AUC of 0.855. The grid doesn't separate configs by much: all 54 score between 0.842 and 0.855, and the standard deviation across folds is about 0.08.

The threshold sweep tries 0.01 to 0.99 in steps of 0.01 on the validation split and keeps the lowest-cost value. Set `threshold_strategy: f1` to maximize F1 instead.

## Technologies

Python, XGBoost, scikit-learn, imbalanced-learn, pandas and NumPy, with matplotlib and seaborn in the notebooks. `requirements.txt` has the exact versions.

## Key learnings

### Accuracy is misleading

A model that calls every transaction normal scores 99.83% accuracy and catches no fraud. PR-AUC is a better measure on data this imbalanced.

### Imbalance needs handling

SMOTE and `scale_pos_weight` both make fraud count for more during training. SMOTE has to stay inside the training folds, because synthetic rows in a validation fold would inflate its score.

### Higher precision doesn't always mean lower cost

Moving from 0.5 to 0.94 raised precision from 78.3% to 91.9%, and F1 from 81.4% to 85.9%. It also missed 4 more frauds, which outweighed the 16 fewer false alarms at these costs.

### A threshold needs enough positives behind it

Choosing the threshold on validation gives an honest test score, and that score can come back worse than the default. About 100 validation frauds aren't enough to tell two thresholds apart reliably.

## Author

Karan - [GitHub](https://github.com/karangv)

## Acknowledgments

- Dataset: [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- Original dataset: ULB Machine Learning Group
