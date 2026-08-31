from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from flight_delay_milp.data import feature_target, make_synthetic_bts, prepare_dataset
from flight_delay_milp.models import positive_probability
from flight_delay_milp.tuning import (
    MODEL_NAMES,
    choose_one_standard_error,
    deterministic_candidates,
    evaluate_candidates,
    fit_selected_model,
    fit_stack_model,
    require_tuning_freeze,
    summarize_candidate_folds,
    write_checkpoint,
)


def test_one_standard_error_prefers_brier_within_roc_band() -> None:
    candidates = [
        {"candidate": 0, "mean_roc_auc": 0.8300, "std_roc_auc": 0.0060, "mean_brier": 0.172},
        {"candidate": 1, "mean_roc_auc": 0.8280, "std_roc_auc": 0.0040, "mean_brier": 0.165},
        {"candidate": 2, "mean_roc_auc": 0.8200, "std_roc_auc": 0.0030, "mean_brier": 0.155},
    ]
    selected = choose_one_standard_error(candidates, folds=3)
    assert selected["candidate"] == 1


def test_candidate_sampling_is_deterministic_and_bounded() -> None:
    first = deterministic_candidates("random_forest", seed=42, budget=8)
    replay = deterministic_candidates("random_forest", seed=42, budget=8)
    different = deterministic_candidates("random_forest", seed=43, budget=8)
    assert first == replay
    assert first != different
    assert len(first) == 8
    assert len({json.dumps(item, sort_keys=True) for item in first}) == 8


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_every_paper_model_has_a_search_space(model_name: str) -> None:
    assert deterministic_candidates(model_name, seed=42, budget=1)


def test_checkpoint_is_idempotent_but_not_overwritable(tmp_path: Path) -> None:
    checkpoint = tmp_path / "outer_0.json"
    write_checkpoint(checkpoint, {"status": "complete", "score": 0.8})
    write_checkpoint(checkpoint, {"status": "complete", "score": 0.8})
    with pytest.raises(FileExistsError, match="different content"):
        write_checkpoint(checkpoint, {"status": "complete", "score": 0.9})


def test_test_evaluation_requires_matching_tuning_freeze(tmp_path: Path) -> None:
    params = {"random_forest": {"n_estimators": 300}}
    with pytest.raises(RuntimeError, match="TUNING_DONE"):
        require_tuning_freeze(tmp_path, params)

    marker = tmp_path / "TUNING_DONE.json"
    marker.write_text('{"status":"complete","parameters_sha256":"wrong"}\n', encoding="utf-8")
    with pytest.raises(RuntimeError, match="parameter hash"):
        require_tuning_freeze(tmp_path, params)


def _small_training_data():
    prepared, _ = prepare_dataset(
        make_synthetic_bts(500, 7),
        {
            "start_year": 2005,
            "start_month": 1,
            "end_year": 2025,
            "end_month": 1,
            "target_threshold": 0.20,
            "target_operator": "gt",
            "missing_arr_del15_policy": "zero",
        },
    )
    return feature_target(prepared)


def test_candidate_fold_summary_records_variance_and_best_iteration() -> None:
    summary = summarize_candidate_folds(
        candidate=2,
        params={"depth": 6},
        folds=[
            {"roc_auc": 0.80, "brier": 0.20, "pr_auc": 0.75, "best_iteration": 101},
            {"roc_auc": 0.84, "brier": 0.18, "pr_auc": 0.78, "best_iteration": 121},
        ],
    )
    assert summary["mean_roc_auc"] == pytest.approx(0.82)
    assert summary["std_roc_auc"] > 0
    assert summary["mean_brier"] == pytest.approx(0.19)
    assert summary["median_best_iteration"] == 111


def test_small_candidate_search_and_refit_produce_probabilities() -> None:
    features, target = _small_training_data()
    candidates = [{"var_smoothing": 1e-9}, {"var_smoothing": 1e-7}]
    results, selected = evaluate_candidates(
        "gaussian_nb",
        candidates,
        features,
        target,
        folds=2,
        seed=42,
        n_jobs=1,
        max_boost_rounds=20,
        early_stopping_patience=5,
    )
    assert len(results) == 2
    assert selected in results
    model = fit_selected_model(
        "gaussian_nb", selected, features, target, seed=42, threads=1
    )
    probability = positive_probability(model, features.iloc[:20])
    assert probability.shape == (20,)
    assert np.all((probability >= 0) & (probability <= 1))


def test_small_stack_uses_oof_meta_features() -> None:
    features, target = _small_training_data()
    base_specs = {
        "xgboost": {"params": deterministic_candidates("xgboost", 1, 1)[0], "iterations": 10},
        "lightgbm": {"params": deterministic_candidates("lightgbm", 2, 1)[0], "iterations": 10},
        "catboost": {"params": deterministic_candidates("catboost", 3, 1)[0], "iterations": 10},
    }
    model, meta_result = fit_stack_model(
        features,
        target,
        base_specs,
        meta_candidates=[{"C": 0.1}, {"C": 1.0}],
        folds=2,
        seed=42,
        threads=1,
    )
    probability = positive_probability(model, features.iloc[:20])
    assert probability.shape == (20,)
    assert meta_result["params"]["C"] in {0.1, 1.0}
