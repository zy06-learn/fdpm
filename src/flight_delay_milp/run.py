from __future__ import annotations

import json
import logging
import os
import platform
import shlex
import subprocess
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml

from .data import (
    class_counts,
    load_raw_data,
    make_synthetic_bts,
    prepare_dataset,
    random_split_and_balance,
    temporal_split_and_balance,
    validation_report,
)
from .evaluation import bootstrap_metric_intervals, fit_and_evaluate
from .models import build_models, build_selected_model, positive_probability
from .optimize import (
    commercial_values,
    compare_strategies,
    decision_metrics,
    scenario_analysis,
    solve_milp,
)
from .plots import plot_calibration, plot_model_metrics, plot_scenarios, plot_strategy_costs

LOGGER = logging.getLogger(__name__)


def utc_now() -> datetime:
    return datetime.now(UTC)


def create_run_directory(root: str | Path, seed: int) -> Path:
    output_root = Path(root)
    output_root.mkdir(parents=True, exist_ok=True)
    stamp = utc_now().strftime("%Y%m%dT%H%M%S.%fZ")
    path = output_root / f"{stamp}-seed{seed}-{uuid.uuid4().hex[:8]}"
    path.mkdir(exist_ok=False)
    return path


def configure_logging(run_dir: Path) -> None:
    formatter = logging.Formatter("%(asctime)sZ %(levelname)s %(name)s: %(message)s")
    formatter.converter = __import__("time").gmtime
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()
    logging.captureWarnings(True)
    file_handler = logging.FileHandler(run_dir / "run.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    root.addHandler(file_handler)
    root.addHandler(stream_handler)


def _run_command(args: list[str], cwd: Path) -> str:
    try:
        return subprocess.check_output(args, cwd=cwd, text=True, stderr=subprocess.STDOUT).strip()
    except (subprocess.CalledProcessError, FileNotFoundError) as error:
        return f"unavailable: {error}"


def record_run_context(run_dir: Path, config: dict[str, Any]) -> None:
    repo_root = Path.cwd()
    (run_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    command = " ".join(shlex.quote(argument) for argument in sys.argv)
    (run_dir / "command.txt").write_text(f"cwd={repo_root}\ncommand={command}\n", encoding="utf-8")
    git_state = {
        "commit": _run_command(["git", "rev-parse", "HEAD"], repo_root),
        "status": _run_command(["git", "status", "--short"], repo_root),
        "diff_stat": _run_command(["git", "diff", "--stat"], repo_root),
    }
    (run_dir / "git_state.json").write_text(
        json.dumps(git_state, indent=2) + "\n", encoding="utf-8"
    )
    environment = {
        "recorded_at_utc": utc_now().isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
    }
    (run_dir / "environment.json").write_text(
        json.dumps(environment, indent=2) + "\n", encoding="utf-8"
    )
    freeze = _run_command([sys.executable, "-m", "pip", "freeze"], repo_root)
    (run_dir / "pip_freeze.txt").write_text(freeze + "\n", encoding="utf-8")


def _save_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8")


def _run_optimization(
    run_dir: Path,
    test_frame: pd.DataFrame,
    targets: pd.Series,
    probabilities: np.ndarray,
    config: dict,
    prefix: str = "random_split",
) -> pd.DataFrame:
    opt = config["optimization"]
    values = commercial_values(
        len(probabilities),
        int(opt["commercial_value_low"]),
        int(opt["commercial_value_high"]),
        int(opt["value_seed"]),
    )
    strategies, selections = compare_strategies(
        values,
        probabilities,
        targets.to_numpy(),
        float(opt["penalty"]),
        float(opt["capacity"]),
        int(config["run"]["seed"]),
    )
    capacity_only = solve_milp(
        values,
        probabilities,
        float(opt["penalty"]),
        float(opt["capacity"]),
        time_limit_seconds=int(opt["solver_time_limit_seconds"]),
    )
    selections["capacity_only_milp"] = capacity_only.selected
    strategies = pd.concat(
        [
            strategies,
            pd.DataFrame(
                [
                    {
                        **decision_metrics(
                            "capacity_only_milp",
                            capacity_only.selected,
                            values,
                            probabilities,
                            float(opt["penalty"]),
                        ),
                        "solver_status": capacity_only.status,
                        "solver_objective": capacity_only.objective,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )

    mandatory_count = int(np.floor(len(probabilities) * float(opt["mandatory_fraction"])))
    mandatory_rng = np.random.default_rng(int(opt["mandatory_seed"]))
    mandatory = mandatory_rng.choice(len(probabilities), size=mandatory_count, replace=False)
    structural = solve_milp(
        values,
        probabilities,
        float(opt["penalty"]),
        float(opt["capacity"]),
        mandatory_indices=mandatory,
        groups=test_frame["carrier"].astype(str),
        group_min_fraction=float(opt["carrier_min_fraction"]),
        time_limit_seconds=int(opt["solver_time_limit_seconds"]),
    )
    selections["structural_milp"] = structural.selected
    strategies = pd.concat(
        [
            strategies,
            pd.DataFrame(
                [
                    {
                        **decision_metrics(
                            "structural_milp",
                            structural.selected,
                            values,
                            probabilities,
                            float(opt["penalty"]),
                        ),
                        "solver_status": structural.status,
                        "solver_objective": structural.objective,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )

    scenarios = scenario_analysis(
        values,
        probabilities,
        opt["scenario_penalties"],
        opt["scenario_capacities"],
    )
    strategies.to_csv(run_dir / f"{prefix}_optimization_strategies.csv", index=False)
    scenarios.to_csv(run_dir / f"{prefix}_scenarios.csv", index=False)
    decisions = pd.DataFrame(
        {
            "commercial_value": values,
            "probability": probabilities,
            "target": targets.to_numpy(),
            "carrier": test_frame["carrier"].to_numpy(),
            "airport": test_frame["airport"].to_numpy(),
            **{f"selected_{name}": selected.astype(int) for name, selected in selections.items()},
        }
    )
    decisions.to_csv(run_dir / f"{prefix}_optimization_decisions.csv.gz", index=False)
    if prefix == "random_split":
        plot_strategy_costs(strategies, run_dir)
        plot_scenarios(scenarios, run_dir)
    return strategies


def run_experiment(config: dict[str, Any], input_path: str | Path | None = None) -> Path:
    seed = int(config["run"]["seed"])
    run_dir = create_run_directory(config["run"]["output_root"], seed)
    configure_logging(run_dir)
    record_run_context(run_dir, config)
    started = utc_now()
    LOGGER.info("Run directory: %s", run_dir.resolve())

    try:
        if config["data"].get("synthetic"):
            raw = make_synthetic_bts(int(config["data"]["synthetic_rows"]), seed)
            LOGGER.info("Using synthetic smoke data with %d rows", len(raw))
        else:
            if input_path is None:
                raise ValueError("A BTS data path is required for a formal run.")
            raw = load_raw_data(input_path)
            LOGGER.info("Loaded %d raw BTS rows", len(raw))

        prepared, summary = prepare_dataset(raw, config["data"])
        report = validation_report(summary, config["data"])
        _save_json(run_dir / "data_validation.json", report)
        LOGGER.info(
            "Prepared %d rows; positive rate %.4f; period %s to %s",
            summary.valid_rows,
            summary.positive_rate,
            summary.start_period,
            summary.end_period,
        )

        x_train, y_train, x_test, y_test, test_indices = random_split_and_balance(
            prepared, float(config["data"]["test_size"]), seed
        )
        split_summary = {
            "balanced_train_rows": len(x_train),
            "test_rows": len(x_test),
            "balanced_train_class_counts": class_counts(y_train),
            "test_class_counts": class_counts(y_test),
        }
        _save_json(run_dir / "random_split_summary.json", split_summary)
        LOGGER.info("Fitting %d candidate models", len(config["models"]["names"]))
        models = build_models(config["models"], seed)
        metrics, probabilities, fitted = fit_and_evaluate(
            models,
            x_train,
            y_train,
            x_test,
            y_test,
            float(config["evaluation"]["threshold"]),
        )
        metrics.to_csv(run_dir / "random_split_model_metrics.csv", index=False)
        plot_model_metrics(metrics, run_dir)

        selected_name = str(config["models"]["selected"])
        selected_probability = probabilities[selected_name]
        selected_model = fitted[selected_name]
        prediction_frame = x_test.copy()
        prediction_frame["target"] = y_test.to_numpy()
        for name, probability in probabilities.items():
            prediction_frame[f"probability_{name}"] = probability
        prediction_frame.to_csv(run_dir / "random_split_predictions.csv.gz", index=False)
        joblib.dump(selected_model, run_dir / f"model_{selected_name}.joblib")
        if config["run"].get("save_all_models"):
            for name, model in fitted.items():
                if name != selected_name:
                    joblib.dump(model, run_dir / f"model_{name}.joblib")

        plot_calibration(
            y_test.to_numpy(),
            selected_probability,
            int(config["evaluation"]["calibration_bins"]),
            run_dir,
        )
        random_test_frame = prepared.iloc[test_indices].reset_index(drop=True)
        _run_optimization(
            run_dir,
            random_test_frame,
            y_test,
            selected_probability,
            config,
        )

        if config["evaluation"].get("run_temporal_arm"):
            LOGGER.info("Running temporal validation arm")
            tx_train, ty_train, tx_test, ty_test, temporal_indices = temporal_split_and_balance(
                prepared,
                int(config["data"]["temporal_train_end_year"]),
                int(config["data"]["temporal_test_start_year"]),
                seed,
            )
            temporal_model = build_selected_model(config["models"], seed)
            temporal_model.fit(tx_train, ty_train)
            temporal_probability = positive_probability(temporal_model, tx_test)
            intervals = bootstrap_metric_intervals(
                ty_test,
                temporal_probability,
                float(config["evaluation"]["threshold"]),
                int(config["evaluation"]["bootstrap_samples"]),
                seed,
            )
            intervals.to_csv(run_dir / "temporal_metrics_with_bootstrap_ci.csv", index=False)
            temporal_predictions = tx_test.copy()
            temporal_predictions["target"] = ty_test.to_numpy()
            temporal_predictions["probability_stacking"] = temporal_probability
            temporal_predictions.to_csv(run_dir / "temporal_predictions.csv.gz", index=False)
            joblib.dump(temporal_model, run_dir / "model_stacking_temporal.joblib")
            temporal_test_frame = prepared.loc[temporal_indices].reset_index(drop=True)
            _run_optimization(
                run_dir,
                temporal_test_frame,
                ty_test,
                temporal_probability,
                config,
                prefix="temporal_split",
            )

        completed = utc_now()
        marker_name = "SMOKE_DONE.json" if config["run"]["kind"] == "smoke" else "DONE.json"
        _save_json(
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
        LOGGER.info("Completed successfully in %.2f seconds", (completed - started).total_seconds())
        return run_dir
    except Exception:
        failed = utc_now()
        _save_json(
            run_dir / "FAILED.json",
            {
                "status": "failed",
                "kind": config["run"]["kind"],
                "started_at_utc": started.isoformat(),
                "failed_at_utc": failed.isoformat(),
                "elapsed_seconds": (failed - started).total_seconds(),
            },
        )
        LOGGER.exception("Run failed; evidence preserved in %s", run_dir.resolve())
        raise


def validate_input_data(config: dict[str, Any], input_path: str | Path) -> dict[str, object]:
    raw = load_raw_data(input_path)
    _, summary = prepare_dataset(raw, config["data"])
    return validation_report(summary, config["data"])
