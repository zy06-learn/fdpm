from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve


def _save(fig: plt.Figure, base_path: Path) -> None:
    fig.savefig(base_path.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(base_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_model_metrics(metrics: pd.DataFrame, output_dir: Path) -> None:
    ordered = metrics.sort_values("roc_auc", ascending=True)
    fig, axes = plt.subplots(1, 3, figsize=(13, max(4.5, len(ordered) * 0.42)))
    colors = ["#35618f", "#c06c3e", "#637c48"]
    for axis, metric, title, color in zip(
        axes,
        ["roc_auc", "pr_auc", "brier"],
        ["ROC-AUC", "PR-AUC", "Brier score (lower is better)"],
        colors,
        strict=True,
    ):
        axis.barh(ordered["model"], ordered[metric], color=color, alpha=0.9)
        axis.set_title(title)
        axis.grid(axis="x", alpha=0.2)
    fig.suptitle("Predictive model benchmark", fontsize=14)
    fig.tight_layout()
    _save(fig, output_dir / "model_benchmark")


def plot_calibration(
    target: np.ndarray, probability: np.ndarray, bins: int, output_dir: Path
) -> None:
    observed, predicted = calibration_curve(target, probability, n_bins=bins, strategy="quantile")
    fig, axis = plt.subplots(figsize=(5.4, 5.0))
    axis.plot([0, 1], [0, 1], linestyle="--", color="#777777", label="ideal")
    axis.plot(predicted, observed, marker="o", color="#35618f", label="stacking")
    axis.set(xlabel="Mean predicted probability", ylabel="Observed positive rate")
    axis.set_title("Probability calibration")
    axis.legend(frameon=False)
    axis.grid(alpha=0.2)
    fig.tight_layout()
    _save(fig, output_dir / "stacking_calibration")


def plot_strategy_costs(strategies: pd.DataFrame, output_dir: Path) -> None:
    frame = strategies.sort_values("total_cost", ascending=False)
    fig, axis = plt.subplots(figsize=(7.2, 4.8))
    axis.barh(frame["strategy"], frame["total_cost"] / 1_000_000, color="#35618f")
    axis.set_xlabel("Expected total cost (USD millions)")
    axis.set_title("Scheduling strategy comparison")
    axis.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    _save(fig, output_dir / "strategy_costs")


def plot_scenarios(scenarios: pd.DataFrame, output_dir: Path) -> None:
    cost = scenarios.pivot(index="penalty", columns="capacity", values="total_cost") / 1_000_000
    risk = scenarios.pivot(index="penalty", columns="capacity", values="average_probability")
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))
    sns.heatmap(cost, annot=True, fmt=".2f", cmap="Blues", ax=axes[0])
    axes[0].set_title("Total expected cost (USD millions)")
    sns.heatmap(risk, annot=True, fmt=".3f", cmap="YlOrRd", ax=axes[1])
    axes[1].set_title("Average operated delay probability")
    fig.tight_layout()
    _save(fig, output_dir / "scenario_analysis")
