from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedKFold, cross_val_score

from conftest import SMALL_XGB
from fraud.data import FEATURE_COLUMNS, TARGET_COLUMN
from fraud.pipeline import build_pipeline


def test_step_order():
    assert list(build_pipeline(SMALL_XGB).named_steps) == ["scale", "smote", "xgb"]


def test_fit_and_predict(transactions):
    X, y = transactions[FEATURE_COLUMNS], transactions[TARGET_COLUMN]
    model = build_pipeline(SMALL_XGB).fit(X, y)
    proba = model.predict_proba(X)[:, 1]
    assert proba[y == 1].mean() > proba[y == 0].mean()


def test_smote_does_not_change_row_count_at_predict_time(transactions):
    X, y = transactions[FEATURE_COLUMNS], transactions[TARGET_COLUMN]
    model = build_pipeline(SMALL_XGB).fit(X, y)
    assert len(model.predict(X)) == len(X)
    assert len(model.predict_proba(X[:7])) == 7


def test_smote_oversamples_during_fit(transactions, monkeypatch):
    X, y = transactions[FEATURE_COLUMNS], transactions[TARGET_COLUMN]
    calls = _record_smote(monkeypatch)
    build_pipeline(SMALL_XGB).fit(X, y)
    assert calls == [(len(X), 2 * (y == 0).sum())]


def test_smote_only_sees_training_folds_during_cv(transactions, monkeypatch):
    X, y = transactions[FEATURE_COLUMNS], transactions[TARGET_COLUMN]
    calls = _record_smote(monkeypatch)
    cv = StratifiedKFold(4, shuffle=True, random_state=0)
    cross_val_score(build_pipeline(SMALL_XGB), X, y, cv=cv, scoring="average_precision")
    assert [rows_in for rows_in, _ in calls] == [750, 750, 750, 750]


def _record_smote(monkeypatch) -> list[tuple[int, int]]:
    """Log (rows in, rows out) for each SMOTE call."""
    calls = []
    original = SMOTE.fit_resample

    def spy(self, X, y, **kwargs):
        X_out, y_out = original(self, X, y, **kwargs)
        calls.append((len(X), len(X_out)))
        return X_out, y_out

    monkeypatch.setattr(SMOTE, "fit_resample", spy)
    return calls
