from __future__ import annotations

import hashlib
import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier, early_stopping
from sklearn.base import BaseEstimator
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier

from .evaluation import classification_metrics
from .models import (
    one_hot_preprocessor,
    ordinal_preprocessor,
    positive_probability,
)

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
BOOSTING_NAMES = ("xgboost", "lightgbm", "catboost")


@dataclass
class PreprocessedModel:
    preprocessor: BaseEstimator
    estimator: BaseEstimator

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        return self.estimator.predict_proba(self.preprocessor.transform(features))


@dataclass
class TunedStackModel:
    base_models: dict[str, BaseEstimator]
    meta_model: LogisticRegression

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        meta_features = np.column_stack(
            [positive_probability(self.base_models[name], features) for name in BOOSTING_NAMES]
        )
        return self.meta_model.predict_proba(meta_features)


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


def summarize_candidate_folds(
    candidate: int, params: dict[str, Any], folds: list[dict[str, float]]
) -> dict[str, Any]:
    if not folds:
        raise ValueError("Candidate has no fold results.")
    roc_values = np.asarray([fold["roc_auc"] for fold in folds], dtype=float)
    result: dict[str, Any] = {
        "candidate": candidate,
        "params": params,
        "folds": folds,
        "mean_roc_auc": float(roc_values.mean()),
        "std_roc_auc": float(roc_values.std(ddof=1)) if len(roc_values) > 1 else 0.0,
        "mean_pr_auc": float(np.mean([fold["pr_auc"] for fold in folds])),
        "mean_brier": float(np.mean([fold["brier"] for fold in folds])),
    }
    if all("f1" in fold for fold in folds):
        result["mean_f1"] = float(np.mean([fold["f1"] for fold in folds]))
    iterations = [int(fold["best_iteration"]) for fold in folds if "best_iteration" in fold]
    if iterations:
        result["median_best_iteration"] = int(np.median(iterations))
    return result


def _fixed_model(
    model_name: str,
    params: dict[str, Any],
    seed: int,
    threads: int,
    iterations: int | None = None,
) -> BaseEstimator:
    def ordinal_pipeline(estimator: BaseEstimator, *, scale: bool = False) -> Pipeline:
        return Pipeline(
            [("preprocess", ordinal_preprocessor(scale=scale)), ("model", estimator)]
        )

    if model_name == "random_forest":
        return ordinal_pipeline(
            RandomForestClassifier(**params, random_state=seed, n_jobs=threads)
        )
    if model_name == "extra_trees":
        return ordinal_pipeline(ExtraTreesClassifier(**params, random_state=seed, n_jobs=threads))
    if model_name == "xgboost":
        return ordinal_pipeline(
            XGBClassifier(
                **params,
                n_estimators=int(iterations or 300),
                eval_metric="logloss",
                random_state=seed,
                n_jobs=threads,
            )
        )
    if model_name == "lightgbm":
        return ordinal_pipeline(
            LGBMClassifier(
                **params,
                n_estimators=int(iterations or 300),
                random_state=seed,
                n_jobs=threads,
                verbosity=-1,
            )
        )
    if model_name == "catboost":
        return ordinal_pipeline(
            CatBoostClassifier(
                **params,
                iterations=int(iterations or 300),
                random_seed=seed,
                verbose=False,
                thread_count=threads,
                allow_writing_files=False,
            )
        )
    if model_name == "knn":
        return ordinal_pipeline(KNeighborsClassifier(**params, n_jobs=threads), scale=True)
    if model_name == "gaussian_nb":
        return ordinal_pipeline(GaussianNB(**params), scale=True)
    if model_name == "mlp":
        return ordinal_pipeline(
            MLPClassifier(
                **params,
                max_iter=300,
                early_stopping=True,
                random_state=seed,
            ),
            scale=True,
        )
    if model_name == "logistic_regression":
        return Pipeline(
            [
                ("preprocess", one_hot_preprocessor()),
                (
                    "model",
                    LogisticRegression(**params, max_iter=1000, random_state=seed),
                ),
            ]
        )
    if model_name == "linear_svm":
        return Pipeline(
            [
                ("preprocess", one_hot_preprocessor()),
                (
                    "model",
                    CalibratedClassifierCV(
                        LinearSVC(**params, random_state=seed),
                        method="sigmoid",
                        cv=5,
                        n_jobs=threads,
                    ),
                ),
            ]
        )
    raise ValueError(f"Cannot build standalone model: {model_name}")


def _fit_boost_with_early_stopping(
    model_name: str,
    params: dict[str, Any],
    features: pd.DataFrame,
    target: pd.Series,
    seed: int,
    threads: int,
    max_boost_rounds: int,
    early_stopping_patience: int,
) -> tuple[PreprocessedModel, int]:
    x_fit, x_stop, y_fit, y_stop = train_test_split(
        features,
        target,
        test_size=0.10,
        random_state=seed,
        stratify=target,
    )
    preprocessor = ordinal_preprocessor()
    transformed_fit = preprocessor.fit_transform(x_fit, y_fit)
    transformed_stop = preprocessor.transform(x_stop)
    if model_name == "xgboost":
        estimator = XGBClassifier(
            **params,
            n_estimators=max_boost_rounds,
            early_stopping_rounds=early_stopping_patience,
            eval_metric="logloss",
            random_state=seed,
            n_jobs=threads,
        )
        estimator.fit(transformed_fit, y_fit, eval_set=[(transformed_stop, y_stop)], verbose=False)
        best_iteration = int(estimator.best_iteration) + 1
    elif model_name == "lightgbm":
        estimator = LGBMClassifier(
            **params,
            n_estimators=max_boost_rounds,
            random_state=seed,
            n_jobs=threads,
            verbosity=-1,
        )
        estimator.fit(
            transformed_fit,
            y_fit,
            eval_set=[(transformed_stop, y_stop)],
            callbacks=[early_stopping(early_stopping_patience, verbose=False)],
        )
        best_iteration = int(estimator.best_iteration_)
    elif model_name == "catboost":
        estimator = CatBoostClassifier(
            **params,
            iterations=max_boost_rounds,
            random_seed=seed,
            verbose=False,
            thread_count=threads,
            allow_writing_files=False,
        )
        estimator.fit(
            transformed_fit,
            y_fit,
            eval_set=(transformed_stop, y_stop),
            early_stopping_rounds=early_stopping_patience,
            verbose=False,
        )
        best_iteration = int(estimator.get_best_iteration()) + 1
    else:
        raise ValueError(f"Early stopping is not defined for: {model_name}")
    return PreprocessedModel(preprocessor, estimator), max(best_iteration, 1)


def _evaluate_candidate(
    model_name: str,
    candidate: int,
    params: dict[str, Any],
    features: pd.DataFrame,
    target: pd.Series,
    folds: int,
    seed: int,
    max_boost_rounds: int,
    early_stopping_patience: int,
) -> dict[str, Any]:
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    fold_results: list[dict[str, float]] = []
    for fold, (train_indices, validation_indices) in enumerate(splitter.split(features, target)):
        x_train = features.iloc[train_indices]
        y_train = target.iloc[train_indices]
        x_validation = features.iloc[validation_indices]
        y_validation = target.iloc[validation_indices]
        fold_seed = seed + candidate * 100 + fold
        if model_name in BOOSTING_NAMES:
            model, best_iteration = _fit_boost_with_early_stopping(
                model_name,
                params,
                x_train,
                y_train,
                fold_seed,
                1,
                max_boost_rounds,
                early_stopping_patience,
            )
        else:
            model = _fixed_model(model_name, params, fold_seed, 1)
            model.fit(x_train, y_train)
            best_iteration = None
        probability = positive_probability(model, x_validation)
        metrics = classification_metrics(y_validation, probability)
        if best_iteration is not None:
            metrics["best_iteration"] = float(best_iteration)
        fold_results.append(metrics)
    return summarize_candidate_folds(candidate, params, fold_results)


def evaluate_candidates(
    model_name: str,
    candidates: list[dict[str, Any]],
    features: pd.DataFrame,
    target: pd.Series,
    folds: int,
    seed: int,
    n_jobs: int,
    max_boost_rounds: int,
    early_stopping_patience: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    results = joblib.Parallel(n_jobs=n_jobs)(
        joblib.delayed(_evaluate_candidate)(
            model_name,
            candidate,
            params,
            features,
            target,
            folds,
            seed,
            max_boost_rounds,
            early_stopping_patience,
        )
        for candidate, params in enumerate(candidates)
    )
    return results, choose_one_standard_error(results, folds)


def fit_selected_model(
    model_name: str,
    selected: dict[str, Any],
    features: pd.DataFrame,
    target: pd.Series,
    seed: int,
    threads: int,
) -> BaseEstimator:
    iterations = selected.get("median_best_iteration") or selected.get("iterations")
    model = _fixed_model(model_name, dict(selected["params"]), seed, threads, iterations)
    model.fit(features, target)
    return model


def _meta_candidate_result(
    candidate: int,
    params: dict[str, Any],
    meta_features: np.ndarray,
    target: pd.Series,
    folds: int,
    seed: int,
) -> dict[str, Any]:
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    results: list[dict[str, float]] = []
    for train_indices, validation_indices in splitter.split(meta_features, target):
        model = LogisticRegression(C=float(params["C"]), max_iter=1000, random_state=seed)
        model.fit(meta_features[train_indices], target.iloc[train_indices])
        probability = model.predict_proba(meta_features[validation_indices])[:, 1]
        results.append(classification_metrics(target.iloc[validation_indices], probability))
    return summarize_candidate_folds(candidate, params, results)


def fit_stack_model(
    features: pd.DataFrame,
    target: pd.Series,
    base_specs: dict[str, dict[str, Any]],
    meta_candidates: list[dict[str, Any]],
    folds: int,
    seed: int,
    threads: int,
) -> tuple[TunedStackModel, dict[str, Any]]:
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    oof = np.zeros((len(features), len(BOOSTING_NAMES)), dtype=float)
    for fold, (train_indices, validation_indices) in enumerate(splitter.split(features, target)):
        for column, model_name in enumerate(BOOSTING_NAMES):
            model = fit_selected_model(
                model_name,
                base_specs[model_name],
                features.iloc[train_indices],
                target.iloc[train_indices],
                seed + fold,
                1,
            )
            oof[validation_indices, column] = positive_probability(
                model, features.iloc[validation_indices]
            )
    meta_results = [
        _meta_candidate_result(candidate, params, oof, target, folds, seed)
        for candidate, params in enumerate(meta_candidates)
    ]
    selected = choose_one_standard_error(meta_results, folds)
    meta_model = LogisticRegression(
        C=float(selected["params"]["C"]), max_iter=1000, random_state=seed
    )
    meta_model.fit(oof, target)
    base_models = {
        model_name: fit_selected_model(
            model_name, base_specs[model_name], features, target, seed, threads
        )
        for model_name in BOOSTING_NAMES
    }
    return TunedStackModel(base_models, meta_model), selected


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
