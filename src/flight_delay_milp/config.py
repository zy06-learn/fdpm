from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a mapping: {config_path}")
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    for section in ("run", "data", "models", "evaluation", "optimization"):
        if section not in config or not isinstance(config[section], dict):
            raise ValueError(f"Missing configuration section: {section}")

    if config["data"].get("target_operator") != "gt":
        raise ValueError("This reproduction freezes target_operator to 'gt'.")
    threshold = float(config["data"].get("target_threshold", -1))
    if not 0 < threshold < 1:
        raise ValueError("target_threshold must be between zero and one.")
    if config["data"].get("missing_arr_del15_policy") not in {"drop", "zero"}:
        raise ValueError("missing_arr_del15_policy must be 'drop' or 'zero'.")
    capacity = float(config["optimization"].get("capacity", -1))
    if not 0 < capacity <= 1:
        raise ValueError("optimization.capacity must be in (0, 1].")

    if str(config["run"].get("kind", "")).startswith("tuning"):
        tuning = config.get("tuning")
        if not isinstance(tuning, dict):
            raise ValueError("Missing configuration section: tuning")
        for key in ("outer_folds", "inner_folds", "stack_oof_folds"):
            if int(tuning.get(key, 0)) < 2:
                raise ValueError(f"tuning.{key} must be at least two.")
        if int(tuning.get("n_jobs", 0)) < 1 or int(tuning.get("model_threads", 0)) < 1:
            raise ValueError("Tuning parallelism must use positive worker counts.")
        budgets = tuning.get("budgets")
        if not isinstance(budgets, dict):
            raise ValueError("tuning.budgets must be a mapping.")
        missing_budgets = sorted(set(config["models"]["names"]) - set(budgets))
        if missing_budgets:
            raise ValueError(f"Missing tuning budgets: {missing_budgets}")
