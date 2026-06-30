"""Plotting utilities for introductory evaluation metrics."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


def plot_success_curve_by_token_budget(
    curves_by_label: Mapping[str, Mapping[int, float]],
    output_path: Path,
    *,
    title: str = "Toy eval success by token budget",
) -> Path:
    """Plot token-budget success curves and return the output path."""
    return plot_success_curve(
        curves_by_label,
        output_path,
        xlabel="Token budget",
        title=title,
    )


def plot_success_curve(
    curves_by_label: Mapping[str, Mapping[int, float] | Mapping[float, float]],
    output_path: Path,
    *,
    xlabel: str,
    title: str,
) -> Path:
    """Plot success curves and return the output path."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(7, 4.5))
    for label, curve in curves_by_label.items():
        points = sorted(curve.items())
        if not points:
            continue
        x_values = [budget for budget, _ in points]
        y_values = [success_fraction for _, success_fraction in points]
        axis.plot(x_values, y_values, marker="o", label=label)

    axis.set_title(title)
    axis.set_xlabel(xlabel)
    axis.set_ylabel("Fraction solved")
    axis.set_ylim(0.0, 1.05)
    axis.grid(True, alpha=0.3)
    if curves_by_label:
        axis.legend()
    plt.tight_layout()
    figure.savefig(output_path)
    plt.close(figure)
    return output_path


__all__ = ["plot_success_curve", "plot_success_curve_by_token_budget"]
