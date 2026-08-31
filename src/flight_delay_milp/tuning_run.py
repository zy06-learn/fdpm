from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import StratifiedKFold, train_test_split

from .data import (
    class_counts,
    load_raw_data,
    make_synthetic_bts,
    prepare_dataset,
    random_split_and_balance,
    temporal_split_and_balance,
    validation_report,
)
from .evaluation import bootstrap_metric_intervals, classification_metrics
from .models import positive_probability
from .plots import plot_calibration, plot_model_metrics
from .run import (
    _run_optimization,
    _save_json,
    configure_logging,
    create_run_directory,
    record_run_context,
    utc_now,
)
from .tuning import (
    BOOSTING_NAMES,
    MODEL_NAMES,
    deterministic_candidates,
    evaluate_candidates,
    fit_selected_model,
    fit_stack_model,
    parameters_sha256,
    require_tuning_freeze,
    write_checkpoint,
)

LOGGER = logging.getLogger(__name__)
STANDALONE_NAMES = tuple(name for name in MODEL_NAMES if name != "stacking")


def indices_sha256(indices: Any) -> str:
    values = np.asarray(list(indices), dtype=np.int64)
    return hashlib.sha256(values.tobytes()).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _smoke_candidate(model_name: str) -> dict[str, Any]:
    candidates: dict[str, dict[str, Any]] = {
        "random_forest": {
            "n_estimators": 20,
            "max_depth": 8,
            "min_samples_leaf": 2,
            "max_features": "sqrt",
        },
        "extra_trees": {
            "n_estimators": 20,
            "max_depth": 8,
            "min_samples_leaf": 2,
            "max_features": "sqrt",
        },
        "xgboost": {
            "max_depth": 3,
            "learning_rate": 0.1,
            "min_child_weight": 1,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "reg_alpha": 0.0,
            "reg_lambda": 1.0,
        },
        "lightgbm": {
            "num_leaves": 15,
            "max_depth": 6,
            "min_child_samples": 20,
            "learning_rate": 0.1,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "reg_alpha": 0.0,
            "reg_lambda": 1.0,
        },
        "catboost": {
            "depth": 4,
            "learning_rate": 0.1,
            "l2_leaf_reg": 1.0,
            "random_strength": 1.0,
            "bagging_temperature": 0.0,
        },
        "knn": {"n_neighbors": 9, "weights": "distance", "p": 2},
        "gaussian_nb": {"var_smoothing": 1e-9},
        "mlp": {
            "hidden_layer_sizes": (16,),
            "alpha": 1e-4,
            "learning_rate_init": 1e-3,
            "activation": "relu",
        },
        "logistic_regression": {"C": 1.0},
        "linear_svm": {"C": 1.0},
        "stacking": {"C": 1.0},
    }
    return candidates[model_name]


def _candidates(config: dict[str, Any], model_name: str, seed: int) -> list[dict[str, Any]]:
    tuning = config["tuning"]
    if tuning.get("smoke_fast"):
        return [_smoke_candidate(model_name)]
    return deterministic_candidates(
        model_name,
        seed,
        int(tuning["budgets"][model_name]),
    )


def _selected_spec(selected: dict[str, Any]) -> dict[str, Any]:
    spec = {"params": selected["params"]}
    if "median_best_iteration" in selected:
        spec["iterations"] = int(selected["median_best_iteration"])
    return spec


def _sample_training(
    features: pd.DataFrame,
    target: pd.Series,
    rows: int | None,
    seed: int,
) -> tuple[pd.DataFrame, pd.Series]:
    if rows is None or rows >= len(features):
        return features.reset_index(drop=True), target.reset_index(drop=True)
    sampled_features, _, sampled_target, _ = train_test_split(
        features,
        target,
        train_size=rows,
        random_state=seed,
        stratify=target,
    )
    return sampled_features.reset_index(drop=True), sampled_target.reset_index(drop=True)


def _search_one_model(
    config: dict[str, Any],
    model_name: str,
    features: pd.DataFrame,
    target: pd.Series,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    tuning = config["tuning"]
    return evaluate_candidates(
        model_name,
        _candidates(config, model_name, seed),
        features,
        target,
        folds=int(tuning["inner_folds"]),
        seed=seed,
        n_jobs=int(tuning["n_jobs"]),
        max_boost_rounds=int(tuning["max_boost_rounds"]),
        early_stopping_patience=int(tuning["early_stopping_patience"]),
    )


def _outer_checkpoint(
    run_dir: Path,
    config: dict[str, Any],
    outer_fold: int,
    model_name: str,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_validation: pd.DataFrame,
    y_validation: pd.Series,
    selected_by_model: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    path = run_dir / "checkpoints" / "outer" / f"fold_{outer_fold}" / f"{model_name}.json"
    if path.is_file():
        payload = _read_json(path)
        selected_by_model[model_name] = payload["selected"]
        LOGGER.info("Resume: outer fold %d %s", outer_fold, model_name)
        return payload

    seed = int(config["run"]["seed"]) + MODEL_NAMES.index(model_name)
    if model_name == "stacking":
        base_specs = {
            name: _selected_spec(selected_by_model[name]) for name in BOOSTING_NAMES
        }
        model, selected = fit_stack_model(
            x_train,
            y_train,
            base_specs,
            _candidates(config, model_name, seed),
            folds=int(config["tuning"]["stack_oof_folds"]),
            seed=seed,
            threads=int(config["tuning"]["model_threads"]),
        )
        candidate_results = [selected]
    else:
        candidate_results, selected = _search_one_model(
            config, model_name, x_train, y_train, seed
        )
        model = fit_selected_model(
            model_name,
            selected,
            x_train,
            y_train,
            seed=seed,
            threads=int(config["tuning"]["model_threads"]),
        )
    probability = positive_probability(model, x_validation)
    payload = {
        "status": "complete",
        "outer_fold": outer_fold,
        "model": model_name,
        "selected": selected,
        "candidate_results": candidate_results,
        "metrics": classification_metrics(
            y_validation,
            probability,
            float(config["evaluation"]["threshold"]),
        ),
    }
    write_checkpoint(path, payload)
    selected_by_model[model_name] = selected
    return payload


def _run_nested_cv(
    run_dir: Path,
    config: dict[str, Any],
    features: pd.DataFrame,
    target: pd.Series,
) -> pd.DataFrame:
    splitter = StratifiedKFold(
        n_splits=int(config["tuning"]["outer_folds"]),
        shuffle=True,
        random_state=int(config["run"]["seed"]),
    )
    rows: list[dict[str, Any]] = []
    for outer_fold, (train_indices, validation_indices) in enumerate(
        splitter.split(features, target)
    ):
        selected_by_model: dict[str, dict[str, Any]] = {}
        x_train = features.iloc[train_indices].reset_index(drop=True)
        y_train = target.iloc[train_indices].reset_index(drop=True)
        x_validation = features.iloc[validation_indices].reset_index(drop=True)
        y_validation = target.iloc[validation_indices].reset_index(drop=True)
        LOGGER.info("Outer fold %d/%d", outer_fold + 1, splitter.n_splits)
        for model_name in MODEL_NAMES:
            payload = _outer_checkpoint(
                run_dir,
                config,
                outer_fold,
                model_name,
                x_train,
                y_train,
                x_validation,
                y_validation,
                selected_by_model,
            )
            rows.append(
                {
                    "outer_fold": outer_fold,
                    "model": model_name,
                    **payload["metrics"],
                }
            )
    metrics = pd.DataFrame(rows).sort_values(["model", "outer_fold"]).reset_index(drop=True)
    metrics.to_csv(run_dir / "nested_cv_metrics.csv", index=False)
    summary = (
        metrics.groupby("model", as_index=False)
        .agg(
            mean_roc_auc=("roc_auc", "mean"),
            std_roc_auc=("roc_auc", "std"),
            mean_pr_auc=("pr_auc", "mean"),
            mean_brier=("brier", "mean"),
            mean_f1=("f1", "mean"),
        )
        .sort_values("mean_roc_auc", ascending=False)
    )
    summary.to_csv(run_dir / "nested_cv_summary.csv", index=False)
    return metrics


def _final_checkpoint(
    run_dir: Path,
    config: dict[str, Any],
    model_name: str,
    features: pd.DataFrame,
    target: pd.Series,
    parameters: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    path = run_dir / "checkpoints" / "final" / f"{model_name}.json"
    if path.is_file():
        payload = _read_json(path)
        parameters[model_name] = payload["spec"]
        LOGGER.info("Resume: final tuning %s", model_name)
        return payload

    seed = int(config["run"]["seed"]) + MODEL_NAMES.index(model_name)
    if model_name == "stacking":
        base_specs = {name: parameters[name] for name in BOOSTING_NAMES}
        _, selected = fit_stack_model(
            features,
            target,
            base_specs,
            _candidates(config, model_name, seed),
            folds=int(config["tuning"]["stack_oof_folds"]),
            seed=seed,
            threads=int(config["tuning"]["model_threads"]),
        )
        candidate_results = [selected]
    else:
        candidate_results, selected = _search_one_model(
            config, model_name, features, target, seed
        )
    spec = _selected_spec(selected)
    payload = {
        "status": "complete",
        "model": model_name,
        "selected": selected,
        "candidate_results": candidate_results,
        "spec": spec,
    }
    write_checkpoint(path, payload)
    parameters[model_name] = spec
    return payload


def _run_final_tuning(
    run_dir: Path,
    config: dict[str, Any],
    features: pd.DataFrame,
    target: pd.Series,
) -> dict[str, dict[str, Any]]:
    parameters: dict[str, dict[str, Any]] = {}
    for model_name in MODEL_NAMES:
        LOGGER.info("Final training-side tuning: %s", model_name)
        _final_checkpoint(run_dir, config, model_name, features, target, parameters)
    _save_json(run_dir / "best_parameters.json", parameters)
    return parameters


def _fit_frozen_model(
    model_name: str,
    parameters: dict[str, dict[str, Any]],
    features: pd.DataFrame,
    target: pd.Series,
    config: dict[str, Any],
):
    seed = int(config["run"]["seed"]) + MODEL_NAMES.index(model_name)
    if model_name == "stacking":
        model, _ = fit_stack_model(
            features,
            target,
            {name: parameters[name] for name in BOOSTING_NAMES},
            [parameters["stacking"]["params"]],
            folds=int(config["tuning"]["stack_oof_folds"]),
            seed=seed,
            threads=int(config["tuning"]["model_threads"]),
        )
        return model
    return fit_selected_model(
        model_name,
        parameters[model_name],
        features,
        target,
        seed=seed,
        threads=int(config["tuning"]["model_threads"]),
    )


def _atomic_save_array(path: Path, values: np.ndarray) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite predictions: {path}")
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("wb") as handle:
        np.save(handle, values)
    os.replace(temporary, path)


def _evaluate_frozen_test(
    run_dir: Path,
    config: dict[str, Any],
    parameters: dict[str, dict[str, Any]],
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], Any]:
    require_tuning_freeze(run_dir, parameters)
    probability_dir = run_dir / "test_probabilities"
    probability_dir.mkdir(exist_ok=True)
    rows: list[dict[str, Any]] = []
    probabilities: dict[str, np.ndarray] = {}
    selected_model = None
    for model_name in MODEL_NAMES:
        checkpoint = run_dir / "checkpoints" / "test" / f"{model_name}.json"
        prediction_path = probability_dir / f"{model_name}.npy"
        if checkpoint.is_file():
            if not prediction_path.is_file():
                raise RuntimeError(f"Test checkpoint lacks predictions: {checkpoint}")
            payload = _read_json(checkpoint)
            probability = np.load(prediction_path)
            model = None
            LOGGER.info("Resume: frozen test evaluation %s", model_name)
        else:
            model = _fit_frozen_model(model_name, parameters, x_train, y_train, config)
            probability = positive_probability(model, x_test)
            _atomic_save_array(prediction_path, probability)
            payload = {
                "status": "complete",
                "model": model_name,
                "prediction_sha256": hashlib.sha256(probability.tobytes()).hexdigest(),
                "metrics": classification_metrics(
                    y_test,
                    probability,
                    float(config["evaluation"]["threshold"]),
                ),
            }
            write_checkpoint(checkpoint, payload)
        if model_name == config["models"]["selected"]:
            selected_model = model
        rows.append({"model": model_name, **payload["metrics"]})
        probabilities[model_name] = probability
    metrics = pd.DataFrame(rows).sort_values("roc_auc", ascending=False).reset_index(drop=True)
    metrics.to_csv(run_dir / "test_model_metrics.csv", index=False)
    plot_model_metrics(metrics, run_dir)
    return metrics, probabilities, selected_model


def _record_resume(run_dir: Path) -> None:
    with (run_dir / "resume_events.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"resumed_at_utc": utc_now().isoformat()}) + "\n")


def _freeze_tuning(
    run_dir: Path,
    parameters: dict[str, dict[str, Any]],
    test_hash: str,
    test_scoring_locked: bool,
) -> None:
    marker = run_dir / "TUNING_DONE.json"
    if marker.is_file():
        require_tuning_freeze(run_dir, parameters)
        recorded = _read_json(marker)
        if recorded.get("test_indices_sha256") != test_hash:
            raise RuntimeError("TUNING_DONE test-index hash does not match the current split.")
        if bool(recorded.get("test_scoring_locked")) != test_scoring_locked:
            raise RuntimeError("TUNING_DONE test-lock state does not match the current run.")
        return
    write_checkpoint(
        marker,
        {
            "status": "complete",
            "completed_at_utc": utc_now().isoformat(),
            "parameters_sha256": parameters_sha256(parameters),
            "test_indices_sha256": test_hash,
            "test_scoring_locked": test_scoring_locked,
        },
    )


def _resolve_run_directory(
    config: dict[str, Any], resume_run: str | Path | None
) -> tuple[Path, bool]:
    if resume_run is None:
        run_dir = create_run_directory(config["run"]["output_root"], int(config["run"]["seed"]))
        return run_dir, False
    run_dir = Path(resume_run)
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Resume run directory does not exist: {run_dir}")
    recorded = yaml.safe_load((run_dir / "resolved_config.yaml").read_text(encoding="utf-8"))
    if recorded != config:
        raise ValueError("Resume configuration does not match resolved_config.yaml.")
    return run_dir, True


def run_tuning_experiment(
    config: dict[str, Any],
    input_path: str | Path | None = None,
    resume_run: str | Path | None = None,
) -> Path:
    run_dir, resumed = _resolve_run_directory(config, resume_run)
    marker_name = "SMOKE_DONE.json" if config["run"]["kind"] == "tuning_smoke" else "DONE.json"
    if resumed and (run_dir / marker_name).is_file():
        return run_dir

    configure_logging(run_dir)
    if resumed:
        _record_resume(run_dir)
    else:
        record_run_context(run_dir, config)
    started = utc_now()
    LOGGER.info("Tuning run directory: %s", run_dir.resolve())

    try:
        if config["data"].get("synthetic"):
            raw = make_synthetic_bts(
                int(config["data"]["synthetic_rows"]), int(config["run"]["seed"])
            )
        else:
            if input_path is None:
                raise ValueError("A BTS data path is required for tuning.")
            raw = load_raw_data(input_path)
        prepared, summary = prepare_dataset(raw, config["data"])
        _save_json(run_dir / "data_validation.json", validation_report(summary, config["data"]))
        x_train, y_train, x_test, y_test, test_indices = random_split_and_balance(
            prepared,
            float(config["data"]["test_size"]),
            int(config["run"]["seed"]),
        )
        tune_features, tune_target = _sample_training(
            x_train,
            y_train,
            config["tuning"].get("sample_rows"),
            int(config["run"]["seed"]),
        )
        test_hash = indices_sha256(test_indices)
        _save_json(
            run_dir / "split_contract.json",
            {
                "balanced_train_rows": len(x_train),
                "tuning_rows": len(tune_features),
                "test_rows": len(x_test),
                "tuning_class_counts": class_counts(tune_target),
                "test_class_counts": class_counts(y_test),
                "test_indices_sha256": test_hash,
                "test_scoring_locked": True,
            },
        )
        _run_nested_cv(run_dir, config, tune_features, tune_target)
        parameters = _run_final_tuning(run_dir, config, tune_features, tune_target)
        _freeze_tuning(
            run_dir,
            parameters,
            test_hash,
            test_scoring_locked=config["run"]["kind"] == "tuning_smoke",
        )

        if config["run"]["kind"] == "tuning_smoke":
            completed = utc_now()
            write_checkpoint(
                run_dir / marker_name,
                {
                    "status": "complete",
                    "kind": config["run"]["kind"],
                    "started_at_utc": started.isoformat(),
                    "completed_at_utc": completed.isoformat(),
                    "elapsed_seconds": (completed - started).total_seconds(),
                    "run_directory": str(run_dir.resolve()),
                },
            )
            return run_dir

        require_tuning_freeze(run_dir, parameters)
        _, probabilities, selected_model = _evaluate_frozen_test(
            run_dir,
            config,
            parameters,
            tune_features,
            tune_target,
            x_test,
            y_test,
        )
        selected_name = str(config["models"]["selected"])
        selected_probability = probabilities[selected_name]
        prediction_frame = x_test.copy()
        prediction_frame["target"] = y_test.to_numpy()
        for name, probability in probabilities.items():
            prediction_frame[f"probability_{name}"] = probability
        prediction_frame.to_csv(run_dir / "test_predictions.csv.gz", index=False)
        if selected_model is None:
            selected_model = _fit_frozen_model(
                selected_name, parameters, tune_features, tune_target, config
            )
        joblib.dump(selected_model, run_dir / f"model_{selected_name}.joblib")
        plot_calibration(
            y_test.to_numpy(),
            selected_probability,
            int(config["evaluation"]["calibration_bins"]),
            run_dir,
        )
        _run_optimization(
            run_dir,
            prepared.iloc[test_indices].reset_index(drop=True),
            y_test,
            selected_probability,
            config,
        )

        if config["evaluation"].get("run_temporal_arm"):
            tx_train, ty_train, tx_test, ty_test, temporal_indices = temporal_split_and_balance(
                prepared,
                int(config["data"]["temporal_train_end_year"]),
                int(config["data"]["temporal_test_start_year"]),
                int(config["run"]["seed"]),
            )
            temporal_model = _fit_frozen_model(
                selected_name, parameters, tx_train, ty_train, config
            )
            temporal_probability = positive_probability(temporal_model, tx_test)
            intervals = bootstrap_metric_intervals(
                ty_test,
                temporal_probability,
                float(config["evaluation"]["threshold"]),
                int(config["evaluation"]["bootstrap_samples"]),
                int(config["run"]["seed"]),
            )
            intervals.to_csv(run_dir / "temporal_metrics_with_bootstrap_ci.csv", index=False)
            temporal_predictions = tx_test.copy()
            temporal_predictions["target"] = ty_test.to_numpy()
            temporal_predictions["probability_stacking"] = temporal_probability
            temporal_predictions.to_csv(run_dir / "temporal_predictions.csv.gz", index=False)
            joblib.dump(temporal_model, run_dir / "model_stacking_temporal.joblib")
            _run_optimization(
                run_dir,
                prepared.loc[temporal_indices].reset_index(drop=True),
                ty_test,
                temporal_probability,
                config,
                prefix="temporal_split",
            )

        completed = utc_now()
        write_checkpoint(
            run_dir / marker_name,
            {
                "status": "complete",
                "kind": config["run"]["kind"],
                "started_at_utc": started.isoformat(),
                "completed_at_utc": completed.isoformat(),
                "elapsed_seconds": (completed - started).total_seconds(),
                "run_directory": str(run_dir.resolve()),
            },
        )
        return run_dir
    except Exception:
        failed = datetime.now(UTC)
        failure_path = run_dir / "FAILED.json"
        if failure_path.exists():
            failure_path = run_dir / f"FAILED_{failed.strftime('%Y%m%dT%H%M%S.%fZ')}.json"
        _save_json(
            failure_path,
            {
                "status": "failed",
                "kind": config["run"]["kind"],
                "started_at_utc": started.isoformat(),
                "failed_at_utc": failed.isoformat(),
                "elapsed_seconds": (failed - started).total_seconds(),
            },
        )
        LOGGER.exception("Tuning run failed; evidence preserved in %s", run_dir.resolve())
        raise
