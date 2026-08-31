from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from flight_delay_milp.cli import build_parser
from flight_delay_milp.config import load_config
from flight_delay_milp.tuning import MODEL_NAMES, parameters_sha256
from flight_delay_milp.tuning_run import indices_sha256, run_tuning_experiment


def _smoke_config(output_root: Path) -> dict:
    return {
        "run": {
            "kind": "tuning_smoke",
            "seed": 42,
            "output_root": str(output_root),
            "save_all_models": False,
        },
        "data": {
            "synthetic": True,
            "synthetic_rows": 600,
            "start_year": 2005,
            "start_month": 1,
            "end_year": 2025,
            "end_month": 12,
            "target_threshold": 0.15,
            "target_operator": "gt",
            "missing_arr_del15_policy": "zero",
            "test_size": 0.20,
            "temporal_train_end_year": 2023,
            "temporal_test_start_year": 2024,
        },
        "models": {
            "names": list(MODEL_NAMES),
            "selected": "stacking",
            "fast": True,
            "n_jobs": 1,
            "stack_folds": 2,
        },
        "evaluation": {
            "threshold": 0.5,
            "calibration_bins": 8,
            "bootstrap_samples": 10,
            "run_temporal_arm": False,
        },
        "optimization": {
            "penalty": 2000.0,
            "capacity": 0.60,
            "commercial_value_low": 1000,
            "commercial_value_high": 5000,
            "value_seed": 42,
            "mandatory_fraction": 0.05,
            "mandatory_seed": 42,
            "carrier_min_fraction": 0.40,
            "solver_time_limit_seconds": 10,
            "scenario_penalties": [2000.0],
            "scenario_capacities": [0.60],
        },
        "tuning": {
            "outer_folds": 2,
            "inner_folds": 2,
            "stack_oof_folds": 2,
            "n_jobs": 1,
            "model_threads": 1,
            "max_boost_rounds": 10,
            "early_stopping_patience": 3,
            "sample_rows": 400,
            "smoke_fast": True,
            "budgets": {name: 1 for name in MODEL_NAMES},
        },
    }


def test_tune_cli_accepts_resume_directory(tmp_path: Path) -> None:
    args = build_parser().parse_args(
        [
            "tune",
            "--config",
            "configs/tuning_smoke.yaml",
            "--input",
            "data/raw",
            "--resume-run",
            str(tmp_path),
        ]
    )
    assert args.command == "tune"
    assert args.resume_run == tmp_path


def test_test_index_hash_is_order_sensitive_and_deterministic() -> None:
    assert indices_sha256([4, 2, 8]) == indices_sha256([4, 2, 8])
    assert indices_sha256([4, 2, 8]) != indices_sha256([2, 4, 8])


def test_formal_tuning_config_matches_approved_protocol() -> None:
    config = load_config("configs/tuning.yaml")
    tuning = config["tuning"]
    assert (tuning["outer_folds"], tuning["inner_folds"]) == (3, 3)
    assert tuning["stack_oof_folds"] == 5
    assert (tuning["max_boost_rounds"], tuning["early_stopping_patience"]) == (2000, 50)
    assert (tuning["n_jobs"], tuning["model_threads"]) == (8, 1)
    assert tuning["budgets"] == {
        "random_forest": 16,
        "extra_trees": 16,
        "xgboost": 20,
        "lightgbm": 20,
        "catboost": 20,
        "knn": 20,
        "gaussian_nb": 13,
        "mlp": 16,
        "logistic_regression": 13,
        "linear_svm": 13,
        "stacking": 13,
    }


def test_tuning_smoke_freezes_params_without_scoring_test_and_resumes(tmp_path: Path) -> None:
    config = _smoke_config(tmp_path / "tuning-smoke")
    run_dir = run_tuning_experiment(config)

    marker_before = (run_dir / "SMOKE_DONE.json").read_text(encoding="utf-8")
    freeze = json.loads((run_dir / "TUNING_DONE.json").read_text(encoding="utf-8"))
    parameters = json.loads((run_dir / "best_parameters.json").read_text(encoding="utf-8"))
    metrics = run_dir / "nested_cv_metrics.csv"

    assert freeze["parameters_sha256"] == parameters_sha256(parameters)
    assert freeze["test_indices_sha256"]
    assert metrics.is_file()
    assert not (run_dir / "test_model_metrics.csv").exists()
    assert len(list((run_dir / "checkpoints" / "outer").rglob("*.json"))) == 22
    assert len(list((run_dir / "checkpoints" / "final").glob("*.json"))) == 11

    resumed = run_tuning_experiment(config, resume_run=run_dir)
    assert resumed == run_dir
    assert (run_dir / "SMOKE_DONE.json").read_text(encoding="utf-8") == marker_before

    freeze_before = (run_dir / "TUNING_DONE.json").read_text(encoding="utf-8")
    (run_dir / "SMOKE_DONE.json").unlink()
    resumed_partial = run_tuning_experiment(config, resume_run=run_dir)
    assert resumed_partial == run_dir
    assert (run_dir / "SMOKE_DONE.json").is_file()
    assert (run_dir / "TUNING_DONE.json").read_text(encoding="utf-8") == freeze_before


def test_tiny_formal_run_unlocks_one_test_evaluation_after_freeze(tmp_path: Path) -> None:
    config = _smoke_config(tmp_path / "tuning-formal")
    config["run"]["kind"] = "tuning"
    run_dir = run_tuning_experiment(config)

    freeze = json.loads((run_dir / "TUNING_DONE.json").read_text(encoding="utf-8"))
    metrics = pd.read_csv(run_dir / "test_model_metrics.csv")

    assert freeze["test_scoring_locked"] is False
    assert set(metrics["model"]) == set(MODEL_NAMES)
    assert len(list((run_dir / "checkpoints" / "test").glob("*.json"))) == 11
    assert (run_dir / "random_split_optimization_strategies.csv").is_file()
    assert (run_dir / "model_stacking.joblib").is_file()
    assert (run_dir / "DONE.json").is_file()
    assert not (run_dir / "SMOKE_DONE.json").exists()
