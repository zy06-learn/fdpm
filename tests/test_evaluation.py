from __future__ import annotations

import numpy as np

from flight_delay_milp.evaluation import classification_metrics


def test_classification_metrics_for_perfect_probabilities() -> None:
    target = np.array([0, 0, 1, 1])
    probability = np.array([0.05, 0.20, 0.80, 0.95])
    metrics = classification_metrics(target, probability)
    assert metrics["roc_auc"] == 1.0
    assert metrics["pr_auc"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["accuracy"] == 1.0
    assert metrics["brier"] < 0.03
