"""Plot the five JAX-MD regimes together using consistent presentation colors."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pandas as pd

from condensate_ml.constants import N_C, N_R


REGIME_COLORS = {
    "control": "#526579",
    "early": "#188C8D",
    "nucleation": "#7351AB",
    "weak_coarsening": "#C98A2F",
    "coarsening": "#B64F52",
}

REGIME_LABELS = {
    "control": "Control",
    "early": "Early",
    "nucleation": "Nucleation",
    "weak_coarsening": "Weak coarsening",
    "coarsening": "Coarsening",
}


def read_regime_analysis(csv_path: str | Path) -> pd.DataFrame:
    """Read one post-simulation analysis CSV and validate its plot columns."""
    data = pd.read_csv(csv_path)
    required_columns = {"time", "clusters", "fraction"}
    missing_columns = required_columns.difference(data.columns)
    if missing_columns:
        raise ValueError(
            f"{csv_path} is missing required columns: {sorted(missing_columns)}"
        )
    return data


def load_regime_analysis(
    csv_paths_by_regime: Mapping[str, str | Path],
) -> dict[str, pd.DataFrame]:
    """Load analysis data for each named regime in a deterministic order."""
    unknown_regimes = set(csv_paths_by_regime).difference(REGIME_COLORS)
    if unknown_regimes:
        raise ValueError(f"Unknown regime names: {sorted(unknown_regimes)}")
    return {
        regime: read_regime_analysis(csv_paths_by_regime[regime])
        for regime in REGIME_COLORS
        if regime in csv_paths_by_regime
    }


def plot_regime_metric(
    data_by_regime: Mapping[str, pd.DataFrame],
    metric: str,
    ylabel: str,
    output_path: str | Path,
) -> Path:
    """Plot one metric for all regimes and save it as a JPEG."""
    if metric not in {"clusters", "fraction"}:
        raise ValueError(f"Unsupported regime metric: {metric}")

    figure, axis = plt.subplots(figsize=(12, 7))
    for regime, data in data_by_regime.items():
        axis.plot(
            data["time"],
            data[metric],
            color=REGIME_COLORS[regime],
            linewidth=2.5,
            label=REGIME_LABELS[regime],
        )

    axis.set_xlabel("Time")
    axis.set_ylabel(ylabel)
    axis.grid(alpha=0.2)
    axis.legend(frameon=False, ncol=len(data_by_regime), loc="upper center")
    figure.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    return output_path


def write_regime_summary_plots(
    csv_paths_by_regime: Mapping[str, str | Path],
    plots_dir: str | Path,
) -> tuple[Path, Path]:
    """Write the largest-cluster and connected-component regime plots."""
    data_by_regime = load_regime_analysis(csv_paths_by_regime)
    plots_dir = Path(plots_dir)
    largest_fraction_path = plot_regime_metric(
        data_by_regime=data_by_regime,
        metric="fraction",
        ylabel=f"Largest-cluster fraction: f_max = N_max / {N_R + N_C}",
        output_path=plots_dir / "largest_cluster_fraction.jpeg",
    )
    cluster_count_path = plot_regime_metric(
        data_by_regime=data_by_regime,
        metric="clusters",
        ylabel="Number of connected components",
        output_path=plots_dir / "number_of_clusters.jpeg",
    )
    return largest_fraction_path, cluster_count_path


def write_regime_comparison_plots(
    reference_plots_dir: str | Path,
    jaxmd_plots_dir: str | Path,
    output_dir: str | Path,
) -> tuple[Path, ...]:
    """Write one HOOMD-versus-JAX-MD figure per regime, with each metric's plots stacked."""
    reference_plots_dir = Path(reference_plots_dir)
    jaxmd_plots_dir = Path(jaxmd_plots_dir)
    output_dir = Path(output_dir)
    metrics = ("bonds", "clusters", "fraction", "largest")
    output_paths = []

    for regime in REGIME_COLORS:
        reference_images = {
            metric: mpimg.imread(reference_plots_dir / f"{regime}_{metric}.jpeg")
            for metric in metrics
        }
        jaxmd_images = {
            metric: mpimg.imread(jaxmd_plots_dir / f"{regime}_{metric}.jpeg")
            for metric in metrics
        }

        figure = plt.figure(figsize=(16, 8))
        grid = figure.add_gridspec(
            2,
            len(metrics),
            left=0.08,
            right=0.99,
            bottom=0.06,
            top=0.88,
            wspace=0.15,
            hspace=0.12,
        )
        for column, metric in enumerate(metrics):
            reference_axis = figure.add_subplot(grid[0, column])
            jaxmd_axis = figure.add_subplot(grid[1, column])
            reference_axis.imshow(reference_images[metric])
            jaxmd_axis.imshow(jaxmd_images[metric])
            reference_axis.axis("off")
            jaxmd_axis.axis("off")
            reference_axis.set_title(metric.title(), fontsize=13, fontweight="bold")

        figure.text(0.025, 0.67, "HOOMD", ha="center", va="center", rotation=90,
                    fontsize=13, fontweight="bold")
        figure.text(0.025, 0.30, "JAX-MD", ha="center", va="center", rotation=90,
                    fontsize=13, fontweight="bold")
        figure.suptitle(REGIME_LABELS[regime], fontsize=16, fontweight="bold")
        figure.tight_layout(rect=(0.04, 0.0, 1.0, 0.9))

        output_path = output_dir / f"{regime}_comparison.jpeg"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=180)
        plt.close(figure)
        output_paths.append(output_path)

    return tuple(output_paths)
