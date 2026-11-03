from pathlib import Path

import pytest
import yaml

from fraud.config import load_config


def rewrite(path: Path, **changes) -> Path:
    raw = yaml.safe_load(path.read_text())
    raw.update(changes)
    path.write_text(yaml.safe_dump(raw))
    return path


def test_loads_the_repo_config():
    cfg = load_config(Path(__file__).parents[1] / "config.yaml")
    assert cfg.costs == {"fn": 100, "fp": 10, "tp": -1, "tn": 0}
    grid_size = 1
    for values in cfg.param_grid.values():
        grid_size *= len(values)
    assert grid_size == 54


def test_paths_are_relative_to_the_config_file(config_path):
    cfg = load_config(config_path)
    assert cfg.data_path == config_path.parent / "transactions.csv"
    assert cfg.artifact_dir == config_path.parent / "artifacts"


@pytest.mark.parametrize(
    "changes",
    [
        {"split": {"train": 0.6, "val": 0.2, "test": 0.3}},
        {"threshold_strategy": "accuracy"},
        {"selection": {"metric": "accuracy", "min_precision": 0.7}},
        {"param_grid": {"max_depth": []}},
        {"costs": {"fn": 100}},
    ],
)
def test_rejects_bad_settings(config_path, changes):
    with pytest.raises(ValueError):
        load_config(rewrite(config_path, **changes))


def test_rejects_missing_setting(config_path):
    raw = yaml.safe_load(config_path.read_text())
    del raw["seed"]
    config_path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="seed"):
        load_config(config_path)
