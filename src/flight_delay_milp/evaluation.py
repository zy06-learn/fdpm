from __future__ import annotations

from collections import OrderedDict

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .models import positive_probability


def classification_metrics(
    target: np.ndarray | pd.Series, probability: np.ndarray, threshold: float = 0.5
) -> dict[str, float]:
    y_true = np.asarray(target, dtype=int)
    y_prob = np.asarray(probability, dtype=float)
    prediction = (y_prob >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "f1": float(f1_score(y_true, prediction, zero_division=0)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "accuracy": float(accuracy_score(y_true, prediction)),
        "precision": float(precision_score(y_true, prediction, zero_division=0)),
        "recall": float(recall_score(y_true, prediction, zero_division=0)),
    }


def fit_and_evaluate(
    models: OrderedDict[str, BaseEstimator],
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, BaseEstimator]]:
    rows: list[dict[str, object]] = []
    probabilities: dict[str, np.ndarray] = {}
    fitted: dict[str, BaseEstimator] = {}
    for name, model in models.items():
        model.fit(x_train, y_train)
        probability = positive_probability(model, x_test)
        rows.append({"model": name, **classification_metrics(y_test, probability, threshold)})
        probabilities[name] = probability
        fitted[name] = model
    metrics = pd.DataFrame(rows).sort_values("roc_auc", ascending=False).reset_index(drop=True)
    return metrics, probabilities, fitted


def bootstrap_metric_intervals(
    target: pd.Series | np.ndarray,
    probability: np.ndarray,
    threshold: float,
    samples: int,
    seed: int,
) -> pd.DataFrame:
    y_true = np.asarray(target, dtype=int)
    y_prob = np.asarray(probability, dtype=float)
    rng = np.random.default_rng(seed)
    collected: list[dict[str, float]] = []
    attempts = 0
    while len(collected) < samples and attempts < samples * 10:
        attempts += 1
        indices = rng.integers(0, len(y_true), size=len(y_true))
        if np.unique(y_true[indices]).size < 2:
            continue
        collected.append(classification_metrics(y_true[indices], y_prob[indices], threshold))
    if len(collected) != samples:
        raise RuntimeError("Could not draw enough two-class bootstrap samples.")
    frame = pd.DataFrame(collected)
    rows = []
    point = classification_metrics(y_true, y_prob, threshold)
    for metric in frame.columns:
        rows.append(
            {
                "metric": metric,
                "point": point[metric],
                "ci_low": float(frame[metric].quantile(0.025)),
                "ci_high": float(frame[metric].quantile(0.975)),
                "bootstrap_samples": samples,
            }
        )
    return pd.DataFrame(rows)
