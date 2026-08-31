from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np

MODEL_NAMES = (
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
)


def _search_space(model_name: str) -> dict[str, list[Any]]:
    spaces: dict[str, dict[str, list[Any]]] = {
        "random_forest": {
            "n_estimators": [300, 600, 900, 1200],
            "max_depth": [None, 12, 24, 40],
            "min_samples_leaf": [1, 2, 4, 8],
            "max_features": ["sqrt", 0.5, 1.0],
        },
        "extra_trees": {
            "n_estimators": [300, 600, 900, 1200],
            "max_depth": [None, 12, 24, 40],
            "min_samples_leaf": [1, 2, 4, 8],
            "max_features": ["sqrt", 0.5, 1.0],
        },
        "xgboost": {
            "max_depth": [3, 4, 6, 8],
            "learning_rate": [0.02, 0.04, 0.07, 0.12, 0.20],
            "min_child_weight": [1, 3, 5, 10],
            "subsample": [0.7, 0.85, 1.0],
            "colsample_bytree": [0.7, 0.85, 1.0],
            "reg_alpha": [0.0, 0.01, 0.1, 1.0],
            "reg_lambda": [0.1, 0.3, 1.0, 3.0, 10.0],
        },
        "lightgbm": {
            "num_leaves": [15, 31, 63, 127],
            "max_depth": [-1, 6, 10, 16],
            "min_child_samples": [10, 20, 50, 100],
            "learning_rate": [0.02, 0.04, 0.07, 0.12, 0.20],
            "subsample": [0.7, 0.85, 1.0],
            "colsample_bytree": [0.7, 0.85, 1.0],
            "reg_alpha": [0.0, 0.01, 0.1, 1.0],
            "reg_lambda": [0.1, 0.3, 1.0, 3.0, 10.0],
        },
        "catboost": {
            "depth": [4, 6, 8, 10],
            "learning_rate": [0.02, 0.04, 0.07, 0.12, 0.20],
            "l2_leaf_reg": [0.1, 0.3, 1.0, 3.0, 10.0],
            "random_strength": [0.1, 0.3, 1.0, 3.0],
            "bagging_temperature": [0.0, 0.5, 1.0, 2.0],
        },
        "knn": {
            "n_neighbors": [5, 9, 15, 25, 41],
            "weights": ["uniform", "distance"],
            "p": [1, 2],
        },
        "gaussian_nb": {"var_smoothing": list(np.logspace(-12, -6, 13))},
        "mlp": {
            "hidden_layer_sizes": [(64,), (128,), (64, 32), (128, 64)],
            "alpha": [1e-6, 1e-5, 1e-4, 1e-3, 1e-2],
            "learning_rate_init": [3e-4, 1e-3, 3e-3],
            "activation": ["relu", "tanh"],
        },
        "logistic_regression": {"C": list(np.logspace(-3, 3, 13))},
        "linear_svm": {"C": list(np.logspace(-3, 3, 13))},
        "stacking": {"C": list(np.logspace(-3, 3, 13))},
    }
    try:
        return spaces[model_name]
    except KeyError as error:
        raise ValueError(f"Unknown model name: {model_name}") from error


def deterministic_candidates(model_name: str, seed: int, budget: int) -> list[dict[str, Any]]:
    if budget < 1:
        raise ValueError("Search budget must be positive.")
    space = _search_space(model_name)
    keys = list(space)
    pool = [dict(zip(keys, values, strict=True)) for values in itertools.product(*space.values())]
    if budget > len(pool):
        raise ValueError(
            f"Search budget {budget} exceeds {model_name} space size {len(pool)}."
        )
    if budget == len(pool):
        return pool
    indices = np.random.default_rng(seed).choice(len(pool), size=budget, replace=False)
    return [pool[int(index)] for index in indices]


def choose_one_standard_error(
    candidates: list[dict[str, Any]], folds: int
) -> dict[str, Any]:
    if not candidates:
        raise ValueError("No candidate results were provided.")
    if folds < 2:
        raise ValueError("At least two folds are required.")
    best = max(candidates, key=lambda item: float(item["mean_roc_auc"]))
    standard_error = float(best["std_roc_auc"]) / np.sqrt(folds)
    floor = float(best["mean_roc_auc"]) - standard_error
    eligible = [item for item in candidates if float(item["mean_roc_auc"]) >= floor]
    return min(
        eligible,
        key=lambda item: (-float(item["mean_roc_auc"]), float(item["mean_brier"])),
    ) if len(eligible) == 1 else min(eligible, key=lambda item: float(item["mean_brier"]))


def parameters_sha256(parameters: dict[str, Any]) -> str:
    encoded = json.dumps(parameters, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != serialized:
            raise FileExistsError(
                f"Refusing to overwrite checkpoint with different content: {path}"
            )
        return
    path.write_text(serialized, encoding="utf-8")


def require_tuning_freeze(run_dir: Path, parameters: dict[str, Any]) -> None:
    marker = run_dir / "TUNING_DONE.json"
    if not marker.is_file():
        raise RuntimeError("TUNING_DONE marker is required before test evaluation.")
    payload = json.loads(marker.read_text(encoding="utf-8"))
    if payload.get("status") != "complete":
        raise RuntimeError("TUNING_DONE marker is not complete.")
    if payload.get("parameters_sha256") != parameters_sha256(parameters):
        raise RuntimeError("TUNING_DONE parameter hash does not match frozen parameters.")
