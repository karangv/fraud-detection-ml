import numpy as np
import pandas as pd
import pytest

from fraud.config import load_config
from fraud.data import FEATURE_COLUMNS, TARGET_COLUMN, load_dataset, split_data, validate_features

SPLIT = {"train": 0.6, "val": 0.2, "test": 0.2}


def test_feature_columns():
    assert len(FEATURE_COLUMNS) == 30
    assert FEATURE_COLUMNS[0] == "Time" and FEATURE_COLUMNS[-1] == "Amount"


def test_split_sizes(transactions):
    train, val, test = split_data(transactions, SPLIT, seed=1)
    assert (len(train), len(val), len(test)) == (600, 200, 200)


def test_split_is_stratified(transactions):
    for part in split_data(transactions, SPLIT, seed=1):
        assert part[TARGET_COLUMN].mean() == pytest.approx(0.05, abs=0.005)


def test_split_is_disjoint(transactions):
    train, val, test = split_data(transactions, SPLIT, seed=1)
    train_ids, val_ids, test_ids = set(train.index), set(val.index), set(test.index)
    assert not train_ids & val_ids
    assert not train_ids & test_ids
    assert not val_ids & test_ids
    assert train_ids | val_ids | test_ids == set(transactions.index)


def test_split_is_reproducible(transactions):
    first = split_data(transactions, SPLIT, seed=1)
    second = split_data(transactions, SPLIT, seed=1)
    other_seed = split_data(transactions, SPLIT, seed=2)
    for a, b in zip(first, second):
        pd.testing.assert_frame_equal(a, b)
    assert not first[2].index.equals(other_seed[2].index)


def test_validate_rejects_missing_columns(transactions):
    with pytest.raises(ValueError, match="V7, Amount"):
        validate_features(transactions.drop(columns=["V7", "Amount"]))


def test_validate_rejects_non_numeric(transactions):
    bad = transactions.assign(V3="oops")
    with pytest.raises(ValueError, match="non-numeric columns: V3"):
        validate_features(bad)


@pytest.mark.parametrize("value", [np.nan, np.inf])
def test_validate_rejects_non_finite(transactions, value):
    bad = transactions.copy()
    bad.loc[0, "V1"] = value
    with pytest.raises(ValueError, match="1 rows"):
        validate_features(bad)


def test_load_dataset_requires_binary_target(config_path, transactions):
    transactions.assign(Class=2).to_csv(config_path.parent / "transactions.csv", index=False)
    with pytest.raises(ValueError, match="Class"):
        load_dataset(load_config(config_path))


def test_load_dataset_requires_target(config_path, transactions):
    transactions.drop(columns="Class").to_csv(config_path.parent / "transactions.csv", index=False)
    with pytest.raises(ValueError, match="Class"):
        load_dataset(load_config(config_path))
