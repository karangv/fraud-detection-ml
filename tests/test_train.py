import json

import numpy as np
import pandas as pd
import pytest

from fraud.config import load_config
from fraud.costs import threshold_metrics
from fraud.data import FEATURE_COLUMNS, split_data
from fraud.evaluate import evaluate
from fraud.pipeline import load_artifacts
from fraud.train import rank_configs, recall_at_precision, train


def test_rank_prefers_higher_score():
    results = pd.DataFrame(
        {"max_depth": [6, 8], "mean_cv_score": [0.80, 0.85], "std_cv_score": [0.01, 0.01]}
    )
    assert rank_configs(results, ["max_depth"]).loc[0, "max_depth"] == 8


def test_rank_breaks_score_ties_on_std():
    results = pd.DataFrame(
        {"max_depth": [6, 8], "mean_cv_score": [0.85, 0.85], "std_cv_score": [0.03, 0.01]}
    )
    assert rank_configs(results, ["max_depth"]).loc[0, "max_depth"] == 8


def test_rank_breaks_full_ties_on_smallest_params_in_any_input_order():
    rows = [
        {"max_depth": 8, "learning_rate": 0.1},
        {"max_depth": 6, "learning_rate": 0.12},
        {"max_depth": 6, "learning_rate": 0.08},
    ]
    results = pd.DataFrame(rows).assign(mean_cv_score=0.85, std_cv_score=0.01)
    params = ["max_depth", "learning_rate"]
    first = rank_configs(results, params)
    second = rank_configs(results.iloc[::-1], params)
    assert first.loc[0, "learning_rate"] == 0.08
    assert first["rank"].tolist() == [1, 2, 3]
    pd.testing.assert_frame_equal(first, second)


def test_rank_treats_float_noise_as_a_tie():
    results = pd.DataFrame(
        {
            "max_depth": [8, 6],
            "mean_cv_score": [0.85 + 1e-12, 0.85],
            "std_cv_score": [0.01, 0.01],
        }
    )
    assert rank_configs(results, ["max_depth"]).loc[0, "max_depth"] == 6


def test_recall_at_precision():
    y_true = np.array([1, 1, 0, 1, 0])
    y_proba = np.array([0.9, 0.8, 0.7, 0.6, 0.1])
    assert recall_at_precision(y_true, y_proba, 0.7) == 1.0
    assert recall_at_precision(y_true, y_proba, 0.9) == pytest.approx(2 / 3)


@pytest.fixture
def trained(config_path):
    cfg = load_config(config_path)
    train(cfg)
    return cfg


def test_train_writes_artifacts_but_not_test_metrics(trained):
    names = {p.name for p in trained.artifact_dir.iterdir()}
    assert names == {"model.joblib", "threshold.json", "cv_results.csv"}


def test_cv_results_lists_every_config(trained):
    results = pd.read_csv(trained.artifact_dir / "cv_results.csv")
    assert len(results) == 2
    assert results["rank"].tolist() == [1, 2]
    assert results["mean_cv_score"].is_monotonic_decreasing


def test_threshold_is_chosen_on_the_validation_split(trained, transactions):
    saved = json.loads((trained.artifact_dir / "threshold.json").read_text())
    assert saved["strategy"] == "cost"
    assert saved["cost_matrix"] == trained.costs
    assert saved["threshold"] == saved["val_metrics"]["threshold"]

    _, val, _ = split_data(transactions, trained.split, trained.seed)
    matrix = saved["val_metrics"]["confusion_matrix"]
    assert sum(matrix.values()) == len(val)
    assert matrix["tp"] + matrix["fn"] == val["Class"].sum()


def test_test_rows_have_no_effect_on_training(trained, transactions):
    artifacts = trained.artifact_dir
    cv_before = (artifacts / "cv_results.csv").read_text()
    threshold_before = (artifacts / "threshold.json").read_text()

    # Class is untouched, so the split stays the same and only the test features change.
    _, _, test = split_data(transactions, trained.split, trained.seed)
    tampered = transactions.copy()
    tampered.loc[test.index, FEATURE_COLUMNS] = 1000.0
    tampered.to_csv(trained.data_path, index=False)
    train(trained)

    assert (artifacts / "cv_results.csv").read_text() == cv_before
    assert (artifacts / "threshold.json").read_text() == threshold_before


def test_evaluate_writes_metrics(trained):
    metrics = evaluate(trained)
    on_disk = json.loads((trained.artifact_dir / "metrics.json").read_text())
    assert on_disk == metrics

    assert metrics["n_test"] == 200
    assert metrics["n_fraud"] == 10
    assert 0 <= metrics["pr_auc"] <= 1
    assert 0 <= metrics["roc_auc"] <= 1

    default = metrics["at_default_threshold"]
    chosen = metrics["at_chosen_threshold"]
    assert default["threshold"] == 0.5
    assert sum(default["confusion_matrix"].values()) == 200
    assert sum(chosen["confusion_matrix"].values()) == 200
    assert metrics["savings_vs_default"] == default["total_cost"] - chosen["total_cost"]
    assert metrics["savings_per_million_transactions"] == metrics["savings_vs_default"] * 5000


def test_evaluate_scores_the_test_split(trained, transactions):
    metrics = evaluate(trained)

    model, saved = load_artifacts(trained.artifact_dir)
    _, _, test = split_data(transactions, trained.split, trained.seed)
    proba = model.predict_proba(test[FEATURE_COLUMNS])[:, 1]
    expected = threshold_metrics(test["Class"], proba, saved["threshold"], saved["cost_matrix"])

    assert metrics["at_chosen_threshold"] == expected


def test_evaluate_without_a_model_fails_clearly(config_path):
    with pytest.raises(FileNotFoundError, match="fraud.train"):
        evaluate(load_config(config_path))
