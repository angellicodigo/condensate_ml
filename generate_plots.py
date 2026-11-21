"""Run all JAX-MD regimes and write their analysis plots."""

from __future__ import annotations

import os
import time
from pathlib import Path

import jax
import jax.numpy as jnp
from dotenv import load_dotenv

from condensate_ml import RubiscoEpyc1Simulation
from condensate_ml.helper.regime_plots import (
    write_regime_comparison_plots,
    write_regime_summary_plots,
)


dotenv_path = Path(__file__).resolve().parent.parent.parent / ".env"
load_dotenv(dotenv_path=dotenv_path)

output_dir = os.getenv("OUTPUT_DIR")

if output_dir is None:
    raise RuntimeError("OUTPUT_DIR was not found in the environment or .env file")


params_list = [
    {
        "name": "control",
        "BOX": 40,
        "EPS": 0.5,
        "N_STEPS": 100000,
        "WRITE_EVERY": 500,
        "SEED": 101,
    },
    {
        "name": "early",
        "BOX": 36,
        "EPS": 0.8,
        "N_STEPS": 100000,
        "WRITE_EVERY": 500,
        "SEED": 102,
    },
    {
        "name": "nucleation",
        "BOX": 33,
        "EPS": 1.2,
        "N_STEPS": 100000,
        "WRITE_EVERY": 500,
        "SEED": 103,
    },
    {
        "name": "weak_coarsening",
        "BOX": 31,
        "EPS": 1.8,
        "N_STEPS": 100000,
        "WRITE_EVERY": 500,
        "SEED": 104,
    },
    {
        "name": "coarsening",
        "BOX": 29,
        "EPS": 2.5,
        "N_STEPS": 100000,
        "WRITE_EVERY": 500,
        "SEED": 105,
    },
]


def jaxmd_artifact_directory(output_dir: str | Path, kind: str) -> Path:
    """Return the device-specific artifact directory used by the simulation."""
    device_label = "gpu" if any(device.platform == "gpu" for device in jax.devices()) else "cpu"
    return Path(output_dir) / f"{kind}_jaxmd_{device_label}"


def run_regime(params: dict[str, object], output_dir: str | Path):
    """Run one regime and return its trajectory and elapsed wall time."""
    name = params["name"]

    print(
        f"Starting {name}: "
        f"BOX={params['BOX']}, "
        f"EPS={params['EPS']}, "
        f"SEED={params['SEED']}"
    )

    simulation = RubiscoEpyc1Simulation(
        n_steps=params["N_STEPS"],
        box=params["BOX"],
        write_every=params["WRITE_EVERY"],
        seed=params["SEED"],
        checkpoint_blocks=True,
        output_dir=output_dir,
        output_name=name,
    )

    # EPS is the positive sticker-patch binding strength from the physics model.
    epsilon = jnp.asarray(params["EPS"], dtype=jnp.float32)

    start = time.perf_counter()

    trajectory = simulation({
        "epsilon": epsilon,
    })

    # Ensure asynchronous accelerator work has completed.
    trajectory.block_until_ready()

    elapsed = time.perf_counter() - start

    expected_frames = params["N_STEPS"] // params["WRITE_EVERY"] + 1

    print(
        f"Finished {name} in {elapsed:.2f} seconds. "
        f"Shape={trajectory.shape}, "
        f"expected_frames={expected_frames}"
    )

    print(
        f"GSD written to: "
        f"{jaxmd_artifact_directory(output_dir, 'traj') / f'{name}.gsd'}"
    )

    return trajectory, elapsed


def run_all_regimes(output_dir: str | Path):
    """Run all five regimes and write the two cross-regime summary plots."""
    trajectories = {}
    elapsed_by_scenario = {}
    for params in params_list:
        name = params["name"]
        trajectories[name], elapsed_by_scenario[name] = run_regime(params, output_dir)

    csv_paths = {
        params["name"]: jaxmd_artifact_directory(output_dir, "data") / f"{params['name']}.csv"
        for params in params_list
    }
    plots_dir = jaxmd_artifact_directory(output_dir, "plots")
    summary_paths = write_regime_summary_plots(csv_paths, plots_dir)
    print(f"Largest-cluster plot written to: {summary_paths[0]}")
    print(f"Cluster-count plot written to: {summary_paths[1]}")
    reference_plots_dir = os.getenv("plots_patchy_gpu")
    reference_files = (
        Path(reference_plots_dir) / f"{regime}_{metric}.jpeg"
        for regime in (params["name"] for params in params_list)
        for metric in ("bonds", "clusters", "fraction", "largest")
    ) if reference_plots_dir is not None else ()
    if (
        reference_plots_dir is not None
        and Path(reference_plots_dir).exists()
        and all(path.exists() for path in reference_files)
    ):
        comparison_paths = write_regime_comparison_plots(
            reference_plots_dir=reference_plots_dir,
            jaxmd_plots_dir=plots_dir,
            output_dir=plots_dir,
        )
        for comparison_path in comparison_paths:
            print(f"Comparison plot written to: {comparison_path}")
    return trajectories, elapsed_by_scenario


if __name__ == "__main__":
    run_all_regimes(output_dir)
