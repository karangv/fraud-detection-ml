from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from fraud.data import FEATURE_COLUMNS, TARGET_COLUMN

SMALL_XGB = {"n_estimators": 5, "max_depth": 2, "n_jobs": 1, "eval_metric": "aucpr"}


def make_transactions(n: int = 1000, fraud_rate: float = 0.05, seed: int = 0) -> pd.DataFrame:
    """Fake transactions where frauds sit about two standard deviations away."""
    rng = np.random.default_rng(seed)
    y = np.zeros(n, dtype=int)
    y[: int(n * fraud_rate)] = 1
    rng.shuffle(y)
    df = pd.DataFrame(rng.normal(size=(n, len(FEATURE_COLUMNS))), columns=FEATURE_COLUMNS)
    df[FEATURE_COLUMNS] += y[:, None] * 2.0
    df[TARGET_COLUMN] = y
    return df


@pytest.fixture
def transactions() -> pd.DataFrame:
    return make_transactions()


@pytest.fixture
def config_path(tmp_path: Path, transactions: pd.DataFrame) -> Path:
    """A config with a tiny grid, pointing at a synthetic CSV in tmp_path."""
    transactions.to_csv(tmp_path / "transactions.csv", index=False)
    path = tmp_path / "config.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "data": {"path": "transactions.csv", "kaggle_dataset": "unused"},
                "seed": 42,
                "artifact_dir": "artifacts",
                "split": {"train": 0.6, "val": 0.2, "test": 0.2},
                "cv": {"folds": 2, "n_jobs": 1},
                "selection": {"metric": "pr_auc", "min_precision": 0.7},
                "smote_k_neighbors": 3,
                "xgb_params": {"n_estimators": 5, "n_jobs": 1, "eval_metric": "aucpr"},
                "param_grid": {"max_depth": [2, 3], "learning_rate": [0.1]},
                "threshold_strategy": "cost",
                "costs": {"fn": 100, "fp": 10, "tp": -1, "tn": 0},
            }
        )
    )
    return path
