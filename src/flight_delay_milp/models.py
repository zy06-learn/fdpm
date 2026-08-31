from __future__ import annotations

from collections import OrderedDict

import numpy as np
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, StackingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier

from .data import CATEGORICAL_FEATURES, NUMERIC_FEATURES


def ordinal_preprocessor(scale: bool = False) -> Pipeline | ColumnTransformer:
    transformer = ColumnTransformer(
        [
            (
                "categorical",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                CATEGORICAL_FEATURES,
            ),
            ("numeric", SimpleImputer(strategy="median"), NUMERIC_FEATURES),
        ],
        sparse_threshold=0.0,
    )
    if scale:
        return Pipeline([("columns", transformer), ("scale", StandardScaler())])
    return transformer


def one_hot_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", min_frequency=2),
                CATEGORICAL_FEATURES,
            ),
            (
                "numeric",
                Pipeline(
                    [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
                ),
                NUMERIC_FEATURES,
            ),
        ]
    )


def _boosting_estimators(seed: int, fast: bool, n_jobs: int) -> tuple[BaseEstimator, ...]:
    rounds = 20 if fast else 300
    xgb = XGBClassifier(
        n_estimators=rounds,
        max_depth=4 if fast else 6,
        learning_rate=0.10 if fast else 0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=seed,
        n_jobs=n_jobs,
    )
    lgbm = LGBMClassifier(
        n_estimators=rounds,
        learning_rate=0.10 if fast else 0.05,
        num_leaves=15 if fast else 31,
        random_state=seed,
        n_jobs=n_jobs,
        verbosity=-1,
    )
    cat = CatBoostClassifier(
        iterations=rounds,
        depth=4 if fast else 6,
        learning_rate=0.10 if fast else 0.05,
        random_seed=seed,
        verbose=False,
        thread_count=n_jobs,
        allow_writing_files=False,
    )
    return xgb, lgbm, cat


def build_models(config: dict, seed: int) -> OrderedDict[str, BaseEstimator]:
    fast = bool(config.get("fast", False))
    n_jobs = int(config.get("n_jobs", -1))
    trees = 30 if fast else 300
    xgb, lgbm, cat = _boosting_estimators(seed, fast, n_jobs)

    def ordinal_pipeline(estimator: BaseEstimator, *, scale: bool = False) -> Pipeline:
        return Pipeline([("preprocess", ordinal_preprocessor(scale=scale)), ("model", estimator)])

    models: OrderedDict[str, BaseEstimator] = OrderedDict(
        [
            (
                "random_forest",
                ordinal_pipeline(
                    RandomForestClassifier(
                        n_estimators=trees, random_state=seed, n_jobs=n_jobs
                    )
                ),
            ),
            (
                "extra_trees",
                ordinal_pipeline(
                    ExtraTreesClassifier(n_estimators=trees, random_state=seed, n_jobs=n_jobs)
                ),
            ),
            ("xgboost", ordinal_pipeline(xgb)),
            ("lightgbm", ordinal_pipeline(lgbm)),
            ("catboost", ordinal_pipeline(cat)),
            (
                "knn",
                ordinal_pipeline(KNeighborsClassifier(n_neighbors=15, n_jobs=n_jobs), scale=True),
            ),
            ("gaussian_nb", ordinal_pipeline(GaussianNB(), scale=True)),
            (
                "mlp",
                ordinal_pipeline(
                    MLPClassifier(
                        hidden_layer_sizes=(32,) if fast else (64, 32),
                        max_iter=80 if fast else 250,
                        early_stopping=True,
                        random_state=seed,
                    ),
                    scale=True,
                ),
            ),
            (
                "logistic_regression",
                Pipeline(
                    [
                        ("preprocess", one_hot_preprocessor()),
                        (
                            "model",
                            LogisticRegression(max_iter=1000, random_state=seed),
                        ),
                    ]
                ),
            ),
            (
                "linear_svm",
                Pipeline(
                    [
                        ("preprocess", one_hot_preprocessor()),
                        (
                            "model",
                            CalibratedClassifierCV(
                                LinearSVC(random_state=seed),
                                method="sigmoid",
                                cv=3 if fast else 5,
                                n_jobs=n_jobs,
                            ),
                        ),
                    ]
                ),
            ),
        ]
    )

    # Stacking parallelizes its three base estimators. Keep each base learner
    # single-threaded to prevent nested all-core oversubscription.
    stack_xgb, stack_lgbm, stack_cat = _boosting_estimators(seed, fast, 1)
    stack = StackingClassifier(
        estimators=[
            ("xgboost", ordinal_pipeline(stack_xgb)),
            ("lightgbm", ordinal_pipeline(stack_lgbm)),
            ("catboost", ordinal_pipeline(stack_cat)),
        ],
        final_estimator=LogisticRegression(max_iter=1000, random_state=seed),
        cv=StratifiedKFold(
            n_splits=int(config.get("stack_folds", 5)), shuffle=True, random_state=seed
        ),
        stack_method="predict_proba",
        passthrough=False,
        n_jobs=n_jobs,
    )
    models["stacking"] = stack

    requested = list(config["names"])
    unknown = sorted(set(requested) - set(models))
    if unknown:
        raise ValueError(f"Unknown model names: {unknown}")
    return OrderedDict((name, models[name]) for name in requested)


def build_selected_model(config: dict, seed: int) -> BaseEstimator:
    selected = str(config["selected"])
    selected_config = dict(config)
    selected_config["names"] = [selected]
    return build_models(selected_config, seed)[selected]


def positive_probability(model: BaseEstimator, features) -> np.ndarray:
    probabilities = model.predict_proba(features)
    if probabilities.ndim != 2 or probabilities.shape[1] != 2:
        raise ValueError("Expected binary predict_proba output with two columns.")
    return np.asarray(probabilities[:, 1], dtype=float)
