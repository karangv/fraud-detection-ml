"""The model pipeline and its saved artifacts."""

import json
from pathlib import Path
from typing import Any

import joblib
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

MODEL_FILE = "model.joblib"
THRESHOLD_FILE = "threshold.json"


def build_pipeline(
    params: dict[str, Any], seed: int = 42, smote_k_neighbors: int = 5
) -> Pipeline:
    """Scale, oversample, classify. imblearn only runs SMOTE during fit."""
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("smote", SMOTE(random_state=seed, k_neighbors=smote_k_neighbors)),
            ("xgb", XGBClassifier(random_state=seed, **params)),
        ]
    )


def load_artifacts(artifact_dir: Path) -> tuple[Pipeline, dict[str, Any]]:
    """Load the fitted pipeline and the contents of threshold.json."""
    model_path = artifact_dir / MODEL_FILE
    threshold_path = artifact_dir / THRESHOLD_FILE
    if not model_path.exists() or not threshold_path.exists():
        raise FileNotFoundError(f"no trained model in {artifact_dir}, run fraud.train first")
    return joblib.load(model_path), json.loads(threshold_path.read_text())
