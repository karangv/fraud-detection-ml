import numpy as np
import pytest

from fraud.costs import sweep_thresholds, threshold_metrics, total_cost

COSTS = {"fn": 100, "fp": 10, "tp": -1, "tn": 0}


@pytest.mark.parametrize(
    "cell, expected",
    [("tp", -1), ("fp", 10), ("fn", 100), ("tn", 0)],
)
def test_total_cost_sign_of_each_cell(cell, expected):
    counts = {"tp": 0, "fp": 0, "fn": 0, "tn": 0, cell: 1}
    assert total_cost(**counts, costs=COSTS) == expected


def test_total_cost_matches_the_corrected_notebook_figure():
    # The notebook's default-threshold counts. Its sign bug reported 2062.
    assert total_cost(tp=82, fp=38, fn=16, tn=56826, costs=COSTS) == 1898


def test_threshold_metrics_by_hand():
    y_true = [1, 1, 1, 0, 0, 0, 0, 0]
    y_proba = [0.9, 0.8, 0.2, 0.7, 0.1, 0.1, 0.1, 0.1]
    m = threshold_metrics(y_true, y_proba, 0.5, COSTS)
    assert m["confusion_matrix"] == {"tn": 4, "fp": 1, "fn": 1, "tp": 2}
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["recall"] == pytest.approx(2 / 3)
    assert m["f1"] == pytest.approx(2 / 3)
    assert m["false_alarm_rate"] == pytest.approx(1 / 5)
    assert m["total_cost"] == 108


def test_threshold_metrics_when_nothing_is_flagged():
    m = threshold_metrics([1, 0], [0.1, 0.1], 0.5, COSTS)
    assert m["precision"] == 0.0
    assert m["f1"] == 0.0
    assert m["total_cost"] == 100


# Three frauds and three normals, scored so the two strategies disagree.
#   t=0.2 flags everything: tp=3 fp=3 fn=0  cost=27  f1=0.67
#   t=0.5 flags 0.9, 0.8, 0.7, 0.6: tp=2 fp=2 fn=1  cost=118  f1=0.57
#   t=0.75 flags 0.9, 0.8: tp=2 fp=0 fn=1  cost=98  f1=0.80
#   t=0.95 flags nothing: tp=0 fp=0 fn=3  cost=300  f1=0
Y_TRUE = np.array([1, 1, 1, 0, 0, 0])
Y_PROBA = np.array([0.9, 0.8, 0.3, 0.7, 0.6, 0.25])
THRESHOLDS = [0.2, 0.5, 0.75, 0.95]


def test_sweep_picks_lowest_cost():
    best, table = sweep_thresholds(Y_TRUE, Y_PROBA, COSTS, "cost", THRESHOLDS)
    assert best["threshold"] == 0.2
    assert best["total_cost"] == 27
    assert table["total_cost"].tolist() == [27, 118, 98, 300]


def test_sweep_picks_best_f1():
    best, _ = sweep_thresholds(Y_TRUE, Y_PROBA, COSTS, "f1", THRESHOLDS)
    assert best["threshold"] == 0.75
    assert best["f1"] == pytest.approx(0.8)


@pytest.mark.parametrize("strategy", ["cost", "f1"])
def test_sweep_ties_go_to_the_lowest_threshold(strategy):
    # Every threshold in (0.2, 0.9] gives the same confusion matrix.
    best, _ = sweep_thresholds([1, 0], [0.9, 0.2], COSTS, strategy, [0.8, 0.5, 0.3])
    assert best["threshold"] == 0.3


def test_sweep_default_grid_matches_the_notebook():
    _, table = sweep_thresholds(Y_TRUE, Y_PROBA, COSTS)
    assert len(table) == 99
    assert table["threshold"].iloc[0] == 0.01
    assert table["threshold"].iloc[-1] == 0.99


def test_sweep_rejects_unknown_strategy():
    with pytest.raises(ValueError):
        sweep_thresholds(Y_TRUE, Y_PROBA, COSTS, "accuracy")
