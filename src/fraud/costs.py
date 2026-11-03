"""Business cost of a decision threshold, and the sweep that picks one."""

from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
import pandas as pd

DEFAULT_THRESHOLDS = np.arange(1, 100) / 100


def total_cost(tp: int, fp: int, fn: int, tn: int, costs: Mapping[str, float]) -> float:
    """Sum of count times cost per cell. A negative cost, like TP = -1, is a reward."""
    return tp * costs["tp"] + fp * costs["fp"] + fn * costs["fn"] + tn * costs["tn"]


def threshold_metrics(
    y_true: Any, y_proba: Any, threshold: float, costs: Mapping[str, float]
) -> dict[str, Any]:
    """Precision, recall, F1, false-alarm rate, counts and cost at one threshold."""
    threshold = float(threshold)
    actual = np.asarray(y_true).astype(bool)
    flagged = np.asarray(y_proba) >= threshold

    tp = int((flagged & actual).sum())
    fp = int((flagged & ~actual).sum())
    fn = int((~flagged & actual).sum())
    tn = int((~flagged & ~actual).sum())

    return {
        "threshold": threshold,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "f1": 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0,
        "false_alarm_rate": fp / (fp + tn) if fp + tn else 0.0,
        "total_cost": total_cost(tp, fp, fn, tn, costs),
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }


def sweep_thresholds(
    y_true: Any,
    y_proba: Any,
    costs: Mapping[str, float],
    strategy: str = "cost",
    thresholds: Iterable[float] = DEFAULT_THRESHOLDS,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Return the best threshold's metrics and the full sweep. Ties go to the lowest threshold."""
    rows = [threshold_metrics(y_true, y_proba, t, costs) for t in sorted(thresholds)]
    table = pd.DataFrame(rows).drop(columns="confusion_matrix")
    if strategy == "cost":
        best = table["total_cost"].idxmin()
    elif strategy == "f1":
        best = table["f1"].idxmax()
    else:
        raise ValueError(f"unknown threshold strategy: {strategy}")
    return rows[best], table
