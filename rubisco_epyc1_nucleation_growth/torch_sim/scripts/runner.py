"""Run complete simulation workflows (rollout + XYZ/CSV/summary/plots) and the CLI.

Nothing here executes at import time. Use ``python -m condensate_ml.runner params.json`` or
the thin wrapper ``torch_kf_bd_kf_sweep_reformat.py``.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch

from . import output_io
from .constants import CHECKPOINT_EVERY, CONTACT_CUTOFF, DT
from .stepper import RubiscoEpyc1TimeStepper


def make_output_dirs(output_base_dir: str, name: str) -> dict[str, str]:
    """Create and return output directory paths (traj, data, plots)."""
    traj_dir = os.path.join(output_base_dir, "traj_bd")
    data_dir = os.path.join(output_base_dir, "data_bd")
    plots_dir = os.path.join(output_base_dir, "plots_bd")

    for folder in (traj_dir, data_dir, plots_dir):
        os.makedirs(folder, exist_ok=True)

    return {
        "traj": traj_dir,
        "data": data_dir,
        "plots": plots_dir,
        "xyz": os.path.join(traj_dir, f"{name}.xyz"),
        "csv": os.path.join(data_dir, f"{name}.csv"),
        "summary": os.path.join(data_dir, f"{name}_summary.json"),
    }


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _get_param(run_params: Mapping[str, Any], keys: Sequence[str], default: Any) -> Any:
    """Return the value of the first key in ``keys`` present in run_params, else default."""
    for key in keys:
        if key in run_params:
            return run_params[key]
    return default


def run_single_simulation(
    run_params: Mapping[str, Any],
    device: torch.device,
    output_base_dir: str = ".",
) -> dict[str, Any]:
    """Run one parameter configuration (no gradients) and write all output artifacts."""
    name = str(_get_param(run_params, ("name", "NAME"), "sim"))
    box = float(_get_param(run_params, ("BOX", "box"), 40.0))
    kon = float(_get_param(run_params, ("KON", "EPS", "kon_or_epsilon", "kon", "eps"), 1.0))
    koff = float(_get_param(run_params, ("KOFF", "koff"), 0.0))
    seed = int(_get_param(run_params, ("SEED", "seed"), 42))
    n_steps = int(_get_param(run_params, ("N_STEPS", "n_steps"), 10000))
    write_every = int(_get_param(run_params, ("WRITE_EVERY", "write_every"), 100))

    requested_dt = _get_param(run_params, ("DT", "dt"), None)
    if requested_dt is not None and float(requested_dt) != DT:
        print(f"WARNING [{name}]: ignoring dt={requested_dt}; all runs use DT={DT}.")

    epsilon_used = -abs(kon)
    dirs = make_output_dirs(output_base_dir, name)

    print(
        f"\nStarting {name}: BOX={box:.1f}, KON={kon:.3f}, epsilon_used={epsilon_used:.3f}, "
        f"KOFF={koff:.3f}, N_STEPS={n_steps}, WRITE_EVERY={write_every}, DT={DT:.5f}"
    )
    t0 = time.time()

    torch.manual_seed(seed)
    np.random.seed(seed)

    stepper = RubiscoEpyc1TimeStepper(
        n_steps=n_steps,
        dt=DT,
        box=box,
        write_every=write_every,
        seed=seed,
        grad_mode="checkpoint",
        checkpoint_every=CHECKPOINT_EVERY,
        field="epsilon",
        device=device,
    )

    eps_tensor = torch.tensor(epsilon_used, device=device, dtype=torch.float32)
    with torch.no_grad():
        trajectory = stepper({"epsilon": eps_tensor})

    elapsed = time.time() - t0
    print(f"Simulation completed in {elapsed:.2f} s. Exporting results...")

    # Frame 0 is step 0; frame i is step i * write_every.
    expected_frames = n_steps // write_every
    if trajectory.shape[0] not in (expected_frames, expected_frames + 1):
        print(
            f"WARNING [{name}]: got {trajectory.shape[0]} frames, expected "
            f"{expected_frames} or {expected_frames + 1}; check the step/frame mapping."
        )

    output_io.export_ovito_xyz(dirs["xyz"], trajectory, stepper.particle_labels, DT, write_every)
    df = output_io.export_csv_and_summary(
        dirs["csv"],
        dirs["summary"],
        name,
        box,
        kon,
        epsilon_used,
        koff,
        DT,
        n_steps,
        write_every,
        trajectory,
        stepper.sticker_indices,
        stepper.patch_indices,
        stepper.sticker_chain_ids,
        stepper.patch_rubisco_ids,
        stepper.n_rubisco,
        stepper.n_chains,
        contact_cutoff=CONTACT_CUTOFF,
    )
    output_io.generate_all_diagnostic_plots(df, dirs["plots"], name)

    print(f"Wrote XYZ: {dirs['xyz']}")
    print(f"Wrote CSV: {dirs['csv']}")
    print(f"Wrote Summary: {dirs['summary']}")
    print(f"Wrote Plots in {dirs['plots']}")

    return {
        "name": name,
        "elapsed_seconds": elapsed,
        "final_contacts": int(df["contacts"].iloc[-1]) if len(df) > 0 else 0,
        "final_clusters": (
            int(df["clusters"].iloc[-1]) if len(df) > 0 else stepper.n_rubisco + stepper.n_chains
        ),
    }


def run_simulation(
    params_list: Sequence[Mapping[str, Any]],
    device: torch.device,
    output_base_dir: str = ".",
) -> list[dict[str, Any]]:
    """Run every configuration in params_list in order; one result dict per run."""
    names = [str(_get_param(p, ("name", "NAME"), "sim")) for p in params_list]
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        raise ValueError(f"Run names must be unique (outputs would be overwritten): {duplicates}")
    return [run_single_simulation(p, device, output_base_dir) for p in params_list]


def load_param_list(source: str) -> list[dict[str, Any]]:
    """Load runs from a JSON file path, a /dev/fd path (process substitution) or inline JSON."""
    text = source.strip()
    params = json.loads(text) if text[:1] in ("{", "[") else json.load(open(source, "r"))

    if isinstance(params, dict):
        param_list = [params]
    elif isinstance(params, list):
        param_list = params
    else:
        raise ValueError(
            "Parameter JSON must contain an object or a list of objects; "
            f"received {type(params).__name__}."
        )
    if not param_list:
        raise ValueError("Parameter JSON contains no simulation runs.")
    if not all(isinstance(p, dict) for p in param_list):
        raise ValueError("Every entry in the parameter list must be a JSON object.")
    return param_list


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Rubisco-EPYC1 differentiable BD sweep")
    parser.add_argument("params", help="params.json path, or an inline JSON object/list")
    parser.add_argument("--output-dir", default=".", help="base dir for traj_bd/data_bd/plots_bd")
    args = parser.parse_args(argv)

    device = get_device()
    param_list = load_param_list(args.params)
    print(f"Rubisco-EPYC1 Differentiable Brownian Dynamics. Device: {device}")
    print(f"Found {len(param_list)} simulation run(s) to execute.")
    run_simulation(param_list, device, args.output_dir)
    print("\nAll simulations completed successfully.")


if __name__ == "__main__":
    main()
