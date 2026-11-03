import numpy as np
import pandas as pd
import pytest

from conftest import SMALL_XGB, make_transactions
from fraud.config import load_config
from fraud.data import FEATURE_COLUMNS, TARGET_COLUMN
from fraud.pipeline import build_pipeline
from fraud.predict import main, score
from fraud.train import train


@pytest.fixture
def model(transactions):
    X, y = transactions[FEATURE_COLUMNS], transactions[TARGET_COLUMN]
    return build_pipeline(SMALL_XGB).fit(X, y)


def test_one_output_row_per_input_row(model):
    new = make_transactions(n=40, seed=1).drop(columns=TARGET_COLUMN)
    scored = score(new, model, 0.5)
    assert scored.columns.tolist() == ["fraud_probability", "is_fraud"]
    assert len(scored) == 40


def test_is_fraud_follows_the_threshold(model):
    new = make_transactions(n=40, seed=1)
    scored = score(new, model, 0.5)
    expected = (scored["fraud_probability"] >= 0.5).astype(int)
    pd.testing.assert_series_equal(scored["is_fraud"], expected, check_names=False)
    assert score(new, model, 1.0)["is_fraud"].sum() == 0


def test_missing_columns_are_rejected(model):
    new = make_transactions(n=5, seed=1).drop(columns=["V5", TARGET_COLUMN])
    with pytest.raises(ValueError, match="missing columns: V5"):
        score(new, model, 0.5)


def test_class_column_is_ignored(model):
    with_class = make_transactions(n=40, seed=1)
    flipped = with_class.assign(Class=1 - with_class[TARGET_COLUMN])
    without = with_class.drop(columns=TARGET_COLUMN)
    expected = score(without, model, 0.5)
    pd.testing.assert_frame_equal(score(with_class, model, 0.5), expected)
    pd.testing.assert_frame_equal(score(flipped, model, 0.5), expected)


def test_column_order_does_not_matter(model):
    new = make_transactions(n=40, seed=1).drop(columns=TARGET_COLUMN)
    shuffled = new[list(np.random.default_rng(0).permutation(FEATURE_COLUMNS))]
    pd.testing.assert_frame_equal(score(shuffled, model, 0.5), score(new, model, 0.5))


def test_cli_scores_a_csv(config_path, tmp_path):
    train(load_config(config_path))
    new = make_transactions(n=25, seed=1)
    new.to_csv(tmp_path / "new.csv", index=False)
    output = tmp_path / "out" / "scored.csv"

    main(["--config", str(config_path), "--input", str(tmp_path / "new.csv"), "--output", str(output)])

    scored = pd.read_csv(output)
    assert len(scored) == 25
    assert set(scored["is_fraud"]) <= {0, 1}
    assert scored["fraud_probability"].between(0, 1).all()


def test_cli_exits_on_missing_columns(config_path, tmp_path):
    train(load_config(config_path))
    make_transactions(n=5, seed=1).drop(columns="Amount").to_csv(tmp_path / "new.csv", index=False)
    output = tmp_path / "scored.csv"

    with pytest.raises(SystemExit, match="missing columns: Amount"):
        main(["--config", str(config_path), "--input", str(tmp_path / "new.csv"), "--output", str(output)])
    assert not output.exists()
