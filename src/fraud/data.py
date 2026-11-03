"""Load the transactions CSV, check its schema and split it."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from fraud.config import Config

FEATURE_COLUMNS = ["Time", *(f"V{i}" for i in range(1, 29)), "Amount"]
TARGET_COLUMN = "Class"


def validate_features(df: pd.DataFrame) -> None:
    """Raise ValueError unless all 30 features are present, numeric and finite."""
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns: {', '.join(missing)}")
    features = df[FEATURE_COLUMNS]
    non_numeric = features.select_dtypes(exclude="number").columns.tolist()
    if non_numeric:
        raise ValueError(f"non-numeric columns: {', '.join(non_numeric)}")
    bad_rows = int((~np.isfinite(features.to_numpy(dtype=float)).all(axis=1)).sum())
    if bad_rows:
        raise ValueError(f"{bad_rows} rows have NaN or infinite feature values")


def _dataset_path(cfg: Config) -> Path:
    if cfg.data_path.exists():
        return cfg.data_path
    import kagglehub

    print(f"{cfg.data_path} not found, using kagglehub")
    return Path(kagglehub.dataset_download(cfg.kaggle_dataset)) / "creditcard.csv"


def load_dataset(cfg: Config) -> pd.DataFrame:
    """Load the labelled dataset, check its schema and drop exact duplicate rows."""
    df = pd.read_csv(_dataset_path(cfg))
    validate_features(df)
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"missing columns: {TARGET_COLUMN}")
    if not df[TARGET_COLUMN].isin([0, 1]).all():
        raise ValueError(f"{TARGET_COLUMN} must only contain 0 and 1")
    # A copy on each side of a random split lets the model score rows it has already seen.
    deduped = df.drop_duplicates(ignore_index=True)
    print(f"dropped {len(df) - len(deduped):,} duplicate rows")
    return deduped


def split_data(
    df: pd.DataFrame, split: dict[str, float], seed: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified train, val and test frames. Same seed, same rows."""
    rest, test = train_test_split(
        df, test_size=split["test"], random_state=seed, stratify=df[TARGET_COLUMN]
    )
    train, val = train_test_split(
        rest,
        test_size=split["val"] / (split["train"] + split["val"]),
        random_state=seed,
        stratify=rest[TARGET_COLUMN],
    )
    return train, val, test
