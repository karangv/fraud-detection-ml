# Credit Card Fraud Detection

Detects fraudulent credit card transactions with XGBoost, SMOTE and a decision threshold tuned against a business cost matrix. Raw data goes in; a trained model, test metrics and a scoring command come out.

## Overview

Only 0.17% of transactions in this dataset are fraud (a 577:1 imbalance). On the held-out test set, the model catches 78.9% of fraud with 67.6% precision at the threshold chosen by cost. PR-AUC is 0.804.

## Dataset

[Kaggle Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud): 284,807 transactions, 492 of them fraud. There are 30 features: Time, Amount and the PCA components V1 to V28.

The pipeline drops 1,081 exact duplicate rows before splitting, which leaves 283,726 rows and 473 frauds.

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

`make train` took about 20 minutes on 12 cores. To try a smaller run, shrink `param_grid` in `config.yaml`. Windows doesn't ship with make (`winget install ezwinports.make` adds it), and the commands behind the targets work on their own:

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
| `threshold.json` | Chosen threshold, strategy, cost matrix, out-of-fold metrics at that threshold |
| `cv_results.csv` | Every config tried, ranked by CV score |
| `metrics.json` | Test metrics at 0.5 and at the chosen threshold, written by `evaluate` |

### How the data is used

The data is split 60/20/20 with stratification and a fixed seed:

| Rows | Used for |
|------|----------|
| Train | 5-fold grid search over 54 configs |
| Train + validation | Choosing the decision threshold from out-of-fold scores, then fitting the final model |
| Test | Final metrics, in `evaluate` only |

Out-of-fold scoring means each row gets its score from a model that didn't train on it, so the threshold can use all 378 frauds in train and validation. SMOTE sits inside the pipeline, so it only runs on training folds and never touches held-out, test or inference data. If you change the config because of what `metrics.json` says, the test split no longer gives an unbiased estimate.

## Project layout

```
config.yaml
src/fraud/    config, data, costs, pipeline, train, evaluate, predict
tests/        pytest suite on synthetic data, no download needed
notebooks/    the original analysis
```

The notebooks cover problem framing, EDA, baseline models and a comparison of imbalance techniques. They are kept as a record of how the approach was chosen, and the numbers printed in them come from an earlier method. The results below come from the pipeline.

## Results

Test set: 56,746 transactions, 95 of them fraud. PR-AUC is 0.804 and ROC-AUC is 0.972 at any threshold.

| Metric | Chosen threshold (0.27) | Default threshold (0.5) |
|--------|-------------------------|-------------------------|
| Recall | 78.9% | 77.9% |
| Precision | 67.6% | 75.5% |
| F1 | 72.8% | 76.7% |
| False-alarm rate | 0.064% | 0.042% |
| Total cost | $2,285 | $2,266 |

Confusion matrix at the chosen threshold:

```
              Predicted
           Normal   Fraud
Actual N   56,615      36
       F       20      75
```

Confusion matrix at the default threshold:

```
              Predicted
           Normal   Fraud
Actual N   56,627      24
       F       21      74
```

### Business impact

Each missed fraud costs $100 and each false alarm costs $10. Each caught fraud earns back $1.

| Threshold | Missed frauds | False alarms | Caught frauds | Total |
|-----------|---------------|--------------|---------------|-------|
| 0.27 (chosen) | 20 × $100 = $2,000 | 36 × $10 = $360 | 75 × -$1 = -$75 | $2,285 |
| 0.5 (default) | 21 × $100 = $2,100 | 24 × $10 = $240 | 74 × -$1 = -$74 | $2,266 |

The two thresholds roughly tie. The tuned one catches 1 more fraud and raises 12 more false alarms, for a total of $19 more than the default on this test set, or about $335 per million transactions. One missed fraud moves the total by $101, so tuning the threshold shows no saving here.

## Key techniques

SMOTE balances the training data from 577:1 to 1:1. XGBoost's `scale_pos_weight` then adds further weight to the fraud class.

Configs are ranked by mean PR-AUC across the CV folds. Ties go to the lower standard deviation, then to the smaller parameter values. The best config is `scale_pos_weight=6.0`, `max_depth=8`, `learning_rate=0.12`, with a CV PR-AUC of 0.837. The grid doesn't separate configs by much: all 54 score between 0.825 and 0.837, and the standard deviation across folds is about 0.07.

The threshold sweep tries 0.01 to 0.99 in steps of 0.01 on the out-of-fold scores and keeps the lowest-cost value. Set `threshold_strategy: f1` to maximize F1 instead.

## Technologies

Python, XGBoost, scikit-learn, imbalanced-learn, pandas and NumPy, with matplotlib and seaborn in the notebooks. `requirements.txt` has the exact versions.

## Key learnings

### Accuracy is misleading

A model that calls every transaction normal scores 99.83% accuracy and catches no fraud. PR-AUC is a better measure on data this imbalanced.

### Imbalance needs handling

SMOTE and `scale_pos_weight` both make fraud count for more during training. SMOTE has to stay inside the training folds, because synthetic rows in a validation fold would inflate its score.

### Duplicates flatter a random split

A row copied into both train and test gets scored by a model that has already seen it. This dataset has 1,081 exact duplicates, so they are removed before splitting.

### Threshold tuning has to earn its place

Tuned on out-of-fold scores, the cost-optimal threshold (0.27) came within $19 of the default 0.5 on test, and its F1 was lower at 72.8% against 76.7%. At these costs, the threshold moves the total far less than the model's missed frauds do: those alone account for $2,000 of the $2,285.

## Author

Karan - [GitHub](https://github.com/karangv)

## Acknowledgments

- Dataset: [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- Original dataset: ULB Machine Learning Group
