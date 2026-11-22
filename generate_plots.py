"""Run all five regimes, then write the cross-regime summary (and HOOMD comparison) plots."""

from __future__ import annotations

import os
from pathlib import Path

from condensate_ml.helper.regime_plots import (
    write_regime_comparison_plots,
    write_regime_summary_plots,
)
from condensate_ml.helper.regimes import REGIMES, artifact_directory, run_regime
from generate import get_output_dir

COMPARISON_METRICS = ("bonds", "clusters", "fraction", "largest")


def reference_plots_complete(reference_dir: Path) -> bool:
    """True if every HOOMD reference plot needed for the comparison figures exists."""
    return all(
        (reference_dir / f"{regime.name}_{metric}.jpeg").exists()
        for regime in REGIMES
        for metric in COMPARISON_METRICS
    )


def run_all_regimes(output_dir: Path):
    """Run every regime and write the summary plots; add HOOMD comparisons when references exist."""
    elapsed_by_regime = {}
    for regime in REGIMES:
        _trajectory, elapsed_by_regime[regime.name] = run_regime(regime, output_dir)

    csv_paths = {
        regime.name: artifact_directory(output_dir, "data") / f"{regime.name}.csv"
        for regime in REGIMES
    }
    plots_dir = artifact_directory(output_dir, "plots")
    largest_fraction_path, cluster_count_path = write_regime_summary_plots(csv_paths, plots_dir)
    print(f"Largest-cluster plot written to: {largest_fraction_path}")
    print(f"Cluster-count plot written to: {cluster_count_path}")

    # Optional: HOOMD reference plots, located through the plots_patchy_gpu environment variable.
    reference_dir = os.getenv("plots_patchy_gpu")
    if reference_dir and reference_plots_complete(Path(reference_dir)):
        for path in write_regime_comparison_plots(Path(reference_dir), plots_dir, plots_dir):
            print(f"Comparison plot written to: {path}")
    return elapsed_by_regime


if __name__ == "__main__":
    run_all_regimes(get_output_dir())
