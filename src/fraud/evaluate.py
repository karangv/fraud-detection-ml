"""Score the test split with the saved model and write metrics.json."""

import argparse
import json
from typing import Any

from sklearn.metrics import average_precision_score, roc_auc_score

from fraud.config import Config, load_config
from fraud.costs import threshold_metrics
from fraud.data import FEATURE_COLUMNS, TARGET_COLUMN, load_dataset, split_data
from fraud.pipeline import load_artifacts

DEFAULT_THRESHOLD = 0.5


def evaluate(cfg: Config) -> dict[str, Any]:
    model, saved = load_artifacts(cfg.artifact_dir)
    costs = saved["cost_matrix"]

    _, _, test = split_data(load_dataset(cfg), cfg.split, cfg.seed)
    y_test = test[TARGET_COLUMN]
    proba = model.predict_proba(test[FEATURE_COLUMNS])[:, 1]

    at_default = threshold_metrics(y_test, proba, DEFAULT_THRESHOLD, costs)
    at_chosen = threshold_metrics(y_test, proba, saved["threshold"], costs)
    savings = at_default["total_cost"] - at_chosen["total_cost"]

    metrics = {
        "n_test": len(test),
        "n_fraud": int(y_test.sum()),
        "pr_auc": float(average_precision_score(y_test, proba)),
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "threshold_strategy": saved["strategy"],
        "cost_matrix": costs,
        "at_default_threshold": at_default,
        "at_chosen_threshold": at_chosen,
        "savings_vs_default": savings,
        "savings_per_million_transactions": savings * 1_000_000 / len(test),
    }
    (cfg.artifact_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args(argv)
    print(json.dumps(evaluate(load_config(args.config)), indent=2))


if __name__ == "__main__":
    main()
