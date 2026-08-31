from __future__ import annotations

import json
from pathlib import Path

import pytest

from flight_delay_milp.tuning import (
    MODEL_NAMES,
    choose_one_standard_error,
    deterministic_candidates,
    require_tuning_freeze,
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

