"""Score a CSV of new transactions with the saved model."""

import argparse
import sys
from pathlib import Path

import pandas as pd
from imblearn.pipeline import Pipeline

from fraud.config import load_config
from fraud.data import FEATURE_COLUMNS, validate_features
from fraud.pipeline import load_artifacts


def score(df: pd.DataFrame, model: Pipeline, threshold: float) -> pd.DataFrame:
    """One output row per input row. Extra columns, including Class, are ignored."""
    validate_features(df)
    proba = model.predict_proba(df[FEATURE_COLUMNS])[:, 1]
    return pd.DataFrame(
        {"fraud_probability": proba, "is_fraud": (proba >= float(threshold)).astype(int)}
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    model, saved = load_artifacts(load_config(args.config).artifact_dir)
    df = pd.read_csv(args.input)
    try:
        scored = score(df, model, saved["threshold"])
    except ValueError as e:
        sys.exit(f"{args.input}: {e}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(output, index=False)
    print(f"{args.output}: {len(scored):,} rows, {scored['is_fraud'].sum()} flagged")


if __name__ == "__main__":
    main()
