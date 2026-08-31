from __future__ import annotations

from flight_delay_milp.models import build_models


def test_builds_the_eleven_paper_models() -> None:
    names = [
        "random_forest",
        "extra_trees",
        "xgboost",
        "lightgbm",
        "catboost",
        "knn",
        "gaussian_nb",
        "mlp",
        "logistic_regression",
        "linear_svm",
        "stacking",
    ]
    models = build_models(
        {"names": names, "selected": "stacking", "fast": True, "n_jobs": 1, "stack_folds": 3},
        42,
    )
    assert list(models) == names
