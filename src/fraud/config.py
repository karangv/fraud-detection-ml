"""Load and validate config.yaml."""

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

SELECTION_METRICS = ("pr_auc", "recall_at_precision")
THRESHOLD_STRATEGIES = ("f1", "cost")


@dataclass(frozen=True)
class Config:
    data_path: Path
    kaggle_dataset: str
    seed: int
    artifact_dir: Path
    split: dict[str, float]
    cv_folds: int
    cv_n_jobs: int
    selection_metric: str
    min_precision: float
    smote_k_neighbors: int
    xgb_params: dict[str, Any]
    param_grid: dict[str, list]
    threshold_strategy: str
    costs: dict[str, float]


def load_config(path: str | Path) -> Config:
    """Read a config file. Relative paths inside it are relative to the file."""
    path = Path(path)
    raw = yaml.safe_load(path.read_text())
    try:
        cfg = Config(
            data_path=path.parent / raw["data"]["path"],
            kaggle_dataset=raw["data"]["kaggle_dataset"],
            seed=raw["seed"],
            artifact_dir=path.parent / raw["artifact_dir"],
            split=raw["split"],
            cv_folds=raw["cv"]["folds"],
            cv_n_jobs=raw["cv"]["n_jobs"],
            selection_metric=raw["selection"]["metric"],
            min_precision=raw["selection"]["min_precision"],
            smote_k_neighbors=raw["smote_k_neighbors"],
            xgb_params=raw["xgb_params"],
            param_grid=raw["param_grid"],
            threshold_strategy=raw["threshold_strategy"],
            costs=raw["costs"],
        )
    except KeyError as e:
        raise ValueError(f"{path}: missing setting {e}") from e
    _validate(cfg)
    return cfg


def _validate(cfg: Config) -> None:
    if set(cfg.split) != {"train", "val", "test"} or not math.isclose(sum(cfg.split.values()), 1.0):
        raise ValueError(f"split needs train, val and test summing to 1, got {cfg.split}")
    if min(cfg.split.values()) <= 0:
        raise ValueError(f"split sizes must be positive, got {cfg.split}")
    if cfg.cv_folds < 2:
        raise ValueError(f"cv.folds must be at least 2, got {cfg.cv_folds}")
    if cfg.selection_metric not in SELECTION_METRICS:
        raise ValueError(f"selection.metric must be one of {SELECTION_METRICS}")
    if not 0 < cfg.min_precision <= 1:
        raise ValueError(f"selection.min_precision must be in (0, 1], got {cfg.min_precision}")
    if cfg.threshold_strategy not in THRESHOLD_STRATEGIES:
        raise ValueError(f"threshold_strategy must be one of {THRESHOLD_STRATEGIES}")
    if not cfg.param_grid or not all(cfg.param_grid.values()):
        raise ValueError("param_grid needs at least one value per parameter")
    if set(cfg.costs) != {"fn", "fp", "tp", "tn"}:
        raise ValueError(f"costs needs fn, fp, tp and tn, got {sorted(cfg.costs)}")
