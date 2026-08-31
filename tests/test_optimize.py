from __future__ import annotations

import numpy as np

from flight_delay_milp.optimize import (
    capacity_count,
    compare_strategies,
    solve_milp,
)


def test_capacity_only_milp_matches_net_score_ranking() -> None:
    values = np.array([5000, 4500, 4000, 3500, 3000, 2500], dtype=float)
    probabilities = np.array([0.8, 0.1, 0.6, 0.2, 0.5, 0.1], dtype=float)
    targets = (probabilities >= 0.5).astype(int)
    strategies, selections = compare_strategies(
        values, probabilities, targets, penalty=2000, capacity=0.5, seed=42
    )
    result = solve_milp(values, probabilities, penalty=2000, capacity=0.5)
    assert result.status == "Optimal"
    assert result.selected.sum() == capacity_count(len(values), 0.5)
    assert np.array_equal(result.selected, selections["net_score"])
    assert set(strategies["strategy"]) == {
        "random",
        "value_only",
        "risk_only",
        "net_score",
        "binary_label_policy",
    }


def test_structural_milp_obeys_mandatory_and_group_minimums() -> None:
    values = np.full(9, 3000.0)
    probabilities = np.linspace(0.05, 0.95, 9)
    groups = np.array(["A"] * 4 + ["B"] * 5)
    result = solve_milp(
        values,
        probabilities,
        penalty=2000,
        capacity=0.6,
        mandatory_indices=[8],
        groups=groups,
        group_min_fraction=0.4,
    )
    assert result.selected[8]
    assert result.selected[groups == "A"].sum() >= 2
    assert result.selected[groups == "B"].sum() >= 2
    assert result.selected.sum() <= 5
