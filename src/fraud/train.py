"""Grid search on train, pick a threshold on val, save the artifacts.

The test split is never used here. fraud.evaluate scores it afterwards.
"""

import argparse
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import make_scorer, precision_recall_curve
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from fraud.config import Config, load_config
from fraud.costs import sweep_thresholds
from fraud.data import FEATURE_COLUMNS, TARGET_COLUMN, load_dataset, split_data
from fraud.pipeline import MODEL_FILE, THRESHOLD_FILE, build_pipeline


def recall_at_precision(y_true: np.ndarray, y_proba: np.ndarray, min_precision: float) -> float:
    """Best recall reachable at any threshold with precision >= min_precision."""
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    return float(recall[precision >= min_precision].max(initial=0.0))


def _scorer(cfg: Config):
    if cfg.selection_metric == "pr_auc":
        return "average_precision"
    return make_scorer(
        recall_at_precision, response_method="predict_proba", min_precision=cfg.min_precision
    )


def rank_configs(results: pd.DataFrame, param_names: list[str]) -> pd.DataFrame:
    """Best config first: highest score, then lowest std, then smallest parameter values."""
    # Scores that differ only by float noise count as ties.
    ranked = results.assign(
        mean_cv_score=results["mean_cv_score"].round(8),
        std_cv_score=results["std_cv_score"].round(8),
    )
    keys = ["mean_cv_score", "std_cv_score", *sorted(param_names)]
    ascending = [False] + [True] * (len(keys) - 1)
    ranked = ranked.sort_values(keys, ascending=ascending, kind="stable", ignore_index=True)
    return ranked.assign(rank=range(1, len(ranked) + 1))


def grid_search(X: pd.DataFrame, y: pd.Series, cfg: Config) -> pd.DataFrame:
    """Cross-validate every config in the grid and return them ranked."""
    search = GridSearchCV(
        build_pipeline(cfg.xgb_params, cfg.seed, cfg.smote_k_neighbors),
        param_grid={f"xgb__{name}": values for name, values in cfg.param_grid.items()},
        scoring=_scorer(cfg),
        cv=StratifiedKFold(cfg.cv_folds, shuffle=True, random_state=cfg.seed),
        refit=False,
        n_jobs=cfg.cv_n_jobs,
        verbose=1,
    )
    search.fit(X, y)
    results = pd.DataFrame(search.cv_results_["params"]).rename(
        columns=lambda c: c.removeprefix("xgb__")
    )
    results["mean_cv_score"] = search.cv_results_["mean_test_score"]
    results["std_cv_score"] = search.cv_results_["std_test_score"]
    return rank_configs(results, list(cfg.param_grid))


def train(cfg: Config) -> None:
    train_df, val_df, _ = split_data(load_dataset(cfg), cfg.split, cfg.seed)
    X_train, y_train = train_df[FEATURE_COLUMNS], train_df[TARGET_COLUMN]
    X_val, y_val = val_df[FEATURE_COLUMNS], val_df[TARGET_COLUMN]
    print(f"train: {len(train_df):,} rows, {y_train.sum()} frauds")
    print(f"val:   {len(val_df):,} rows, {y_val.sum()} frauds")

    results = grid_search(X_train, y_train, cfg)
    best_params = {name: results.loc[0, name].item() for name in cfg.param_grid}
    print(f"best config: {best_params}, CV {cfg.selection_metric} {results.loc[0, 'mean_cv_score']:.4f}")

    model = build_pipeline({**cfg.xgb_params, **best_params}, cfg.seed, cfg.smote_k_neighbors)
    model.fit(X_train, y_train)

    val_proba = model.predict_proba(X_val)[:, 1]
    best, _ = sweep_thresholds(y_val, val_proba, cfg.costs, cfg.threshold_strategy)
    print(f"threshold ({cfg.threshold_strategy}): {best['threshold']:.2f}")

    cfg.artifact_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, cfg.artifact_dir / MODEL_FILE)
    results.to_csv(cfg.artifact_dir / "cv_results.csv", index=False)
    (cfg.artifact_dir / THRESHOLD_FILE).write_text(
        json.dumps(
            {
                "threshold": best["threshold"],
                "strategy": cfg.threshold_strategy,
                "cost_matrix": cfg.costs,
                "val_metrics": best,
            },
            indent=2,
        )
    )
    print(f"saved to {cfg.artifact_dir}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args(argv)
    train(load_config(args.config))


if __name__ == "__main__":
    main()
