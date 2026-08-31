from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from math import ceil, floor

import numpy as np
import pandas as pd
import pulp


@dataclass(frozen=True)
class OptimizationResult:
    status: str
    selected: np.ndarray
    objective: float


def commercial_values(size: int, low: int, high: int, seed: int) -> np.ndarray:
    if low > high:
        raise ValueError("Commercial value lower bound exceeds upper bound.")
    return np.random.default_rng(seed).integers(low, high + 1, size=size).astype(float)


def capacity_count(size: int, capacity: float) -> int:
    if not 0 < capacity <= 1:
        raise ValueError("Capacity must be in (0, 1].")
    return floor(size * capacity)


def rank_selection(score: np.ndarray, count: int, *, ascending: bool = False) -> np.ndarray:
    if not 0 <= count <= len(score):
        raise ValueError("Selection count is outside the candidate set.")
    order = np.argsort(score, kind="stable")
    if not ascending:
        order = order[::-1]
    selected = np.zeros(len(score), dtype=bool)
    selected[order[:count]] = True
    return selected


def solve_milp(
    values: np.ndarray,
    probabilities: np.ndarray,
    penalty: float,
    capacity: float,
    mandatory_indices: Iterable[int] = (),
    groups: Iterable[str] | None = None,
    group_min_fraction: float | None = None,
    time_limit_seconds: int | None = None,
) -> OptimizationResult:
    values = np.asarray(values, dtype=float)
    probabilities = np.asarray(probabilities, dtype=float)
    if values.shape != probabilities.shape:
        raise ValueError("Values and probabilities must have the same shape.")
    if np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("Probabilities must be in [0, 1].")

    size = len(values)
    cap = capacity_count(size, capacity)
    mandatory = sorted(set(int(index) for index in mandatory_indices))
    if mandatory and (mandatory[0] < 0 or mandatory[-1] >= size):
        raise ValueError("Mandatory index is outside the candidate set.")
    if len(mandatory) > cap:
        raise ValueError("Mandatory units exceed capacity.")

    problem = pulp.LpProblem("flight_selection", pulp.LpMinimize)
    decisions = [pulp.LpVariable(f"x_{index}", cat="Binary") for index in range(size)]
    problem += pulp.lpSum(
        (1 - decisions[index]) * float(values[index])
        + float(penalty) * float(probabilities[index]) * decisions[index]
        for index in range(size)
    )
    problem += pulp.lpSum(decisions) <= cap, "capacity"
    for index in mandatory:
        problem += decisions[index] == 1, f"mandatory_{index}"

    if groups is not None and group_min_fraction is not None:
        group_array = np.asarray(list(groups), dtype=str)
        if len(group_array) != size:
            raise ValueError("Group labels must align with candidate units.")
        if not 0 <= group_min_fraction <= 1:
            raise ValueError("Group minimum fraction must be in [0, 1].")
        for group in sorted(np.unique(group_array)):
            indices = np.flatnonzero(group_array == group)
            lower = ceil(len(indices) * group_min_fraction)
            problem += (
                pulp.lpSum(decisions[index] for index in indices) >= lower,
                f"group_min_{group}",
            )

    solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit_seconds)
    problem.solve(solver)
    status = pulp.LpStatus[problem.status]
    if status != "Optimal":
        raise RuntimeError(f"CBC did not return an optimal solution: {status}")
    selected = np.asarray([pulp.value(decision) > 0.5 for decision in decisions], dtype=bool)
    return OptimizationResult(
        status=status, selected=selected, objective=float(pulp.value(problem.objective))
    )


def decision_metrics(
    name: str,
    selected: np.ndarray,
    values: np.ndarray,
    probabilities: np.ndarray,
    penalty: float,
) -> dict[str, object]:
    selected = np.asarray(selected, dtype=bool)
    values = np.asarray(values, dtype=float)
    probabilities = np.asarray(probabilities, dtype=float)
    commercial_loss = float(values[~selected].sum())
    expected_penalty = float(penalty * probabilities[selected].sum())
    operated = int(selected.sum())
    return {
        "strategy": name,
        "total_cost": commercial_loss + expected_penalty,
        "commercial_loss": commercial_loss,
        "expected_penalty": expected_penalty,
        "risk_sum": float(probabilities[selected].sum()),
        "average_probability": float(probabilities[selected].mean()) if operated else np.nan,
        "operated": operated,
    }


def compare_strategies(
    values: np.ndarray,
    probabilities: np.ndarray,
    targets: np.ndarray,
    penalty: float,
    capacity: float,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    count = capacity_count(len(values), capacity)
    rng = np.random.default_rng(seed)
    random_indices = rng.choice(len(values), size=count, replace=False)
    random_selected = np.zeros(len(values), dtype=bool)
    random_selected[random_indices] = True
    selections = {
        "random": random_selected,
        "value_only": rank_selection(values, count),
        "risk_only": rank_selection(probabilities, count, ascending=True),
        "net_score": rank_selection(values - penalty * probabilities, count),
        "binary_label_policy": rank_selection(values - penalty * np.asarray(targets), count),
    }
    rows = [
        decision_metrics(name, selected, values, probabilities, penalty)
        for name, selected in selections.items()
    ]
    return pd.DataFrame(rows), selections


def scenario_analysis(
    values: np.ndarray,
    probabilities: np.ndarray,
    penalties: Iterable[float],
    capacities: Iterable[float],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for penalty in penalties:
        for capacity in capacities:
            count = capacity_count(len(values), float(capacity))
            selection = rank_selection(values - float(penalty) * probabilities, count)
            rows.append(
                {
                    "penalty": float(penalty),
                    "capacity": float(capacity),
                    **decision_metrics(
                        "net_score", selection, values, probabilities, float(penalty)
                    ),
                }
            )
    return pd.DataFrame(rows)
