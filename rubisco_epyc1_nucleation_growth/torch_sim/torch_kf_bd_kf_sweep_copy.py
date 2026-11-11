"""Differentiable Brownian Dynamics Simulation for Rubisco-EPYC1 Condensates.

Reorganized to match the section layout of simulate_hoomd_patchy_full.py
while preserving the differentiable PyTorch functionality of
torch_kf_bd_kf_sweep.py.

Pure helper functions live in the scripts/ package alongside this file:

- scripts/geometry.py   : coordinate wrapping, minimum image displacement
- scripts/initial_snapshot.py : initial grid layout and initial coordinate snapshot
- scripts/topology.py           : particle labels, type IDs, mobilities, bond topologies,
                                  steric parameter matrices
- scripts/forces.py             : steric repulsion, sticker/patch attraction, bond forces,
                                  Brownian integration step
- scripts/post_simulation_analysis.py : soft/hard contacts, cluster graph metrics
- scripts/io.py                 : XYZ export, per-frame metrics, CSV writing, summary JSON,
                                  diagnostic plots

Layout (top to bottom, no main()):
  1. Input params (module-level)
  2. Model parameters
  3. Output folders
  4. Device
  5. Patch geometry
  6. Build initial snapshot
  7. Brownian dynamics
  8. RubiscoEpyc1TimeStepper (TimeStepper subclass)
  9. GSD output
  10. Analysis helpers
  11. Postprocess: CSV, plots
  12. Run: single simulation
  13. Module-level execution loop
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
import torch
from nefi.operators.timestepping import TimeStepper

from scripts import forces
from scripts import io
from scripts import initial_snapshot
from scripts import post_simulation_analysis
from scripts import topology


# ============================================================
# INPUT PARAMS
# ============================================================


if len(sys.argv) < 2:
    raise SystemExit("Usage: python torch_kf_bd_kf_sweep.py params.json")

with open(sys.argv[1], "r") as params_file:
    params = json.load(params_file)

# Accept either one parameter object or a list of objects for a parameter sweep.
if isinstance(params, dict):
    PARAM_LIST = [params]
elif isinstance(params, list):
    PARAM_LIST = params
else:
    raise ValueError(
        "Parameter JSON must contain an object or a list of objects; "
        f"received {type(params).__name__}."
    )

if not PARAM_LIST:
    raise ValueError("Parameter JSON contains no simulation runs.")
if not all(isinstance(run_params, dict) for run_params in PARAM_LIST):
    raise ValueError("Every entry in the parameter list must be a JSON object.")

# Initialize the named input variables from the first run. These are updated
# for each sweep entry in run_single_simulation() below.
params = PARAM_LIST[0]
NAME = str(params.get("name", params.get("NAME", "sim")))
BOX = float(params.get("BOX", params.get("box", 40.0)))
KON = float(
    params.get(
        "KON",
        params.get(
            "EPS",
            params.get(
                "kon_or_epsilon",
                params.get("kon", params.get("eps", 1.0)),
            ),
        ),
    )
)
KOFF = float(params.get("KOFF", params.get("koff", 0.0)))

SEED = int(params.get("SEED", params.get("seed", 42)))
N_STEPS = int(params.get("N_STEPS", params.get("n_steps", 10000)))
WRITE_EVERY = int(params.get("WRITE_EVERY", params.get("write_every", 100)))
DT = float(params.get("DT", params.get("dt", 0.0005)))


# ============================================================
# MODEL PARAMETERS
# ============================================================


N_R = 24
N_C = 48
N_BEADS = 12
STICKER_INDEX = [1, 3, 5, 7, 9, 11]

PATCHES = 8
PATCH_RADIUS = 2.5

CHAIN_K = 10.0
CHAIN_R0 = 1.2

PATCH_TETHER_K = 50.0

PATCH_EPSILON = -abs(KON)
PATCH_SIGMA = 1.0
PATCH_RCUT = 3.0

CONTACT_CUTOFF = 3.0


# ============================================================
# OUTPUT FOLDERS
# ============================================================


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


# ============================================================
# DEVICE
# ============================================================


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ============================================================
# PATCH GEOMETRY
# ============================================================


patch_dirs = np.array(
    [
        [1, 1, 1],
        [1, 1, -1],
        [1, -1, 1],
        [1, -1, -1],
        [-1, 1, 1],
        [-1, 1, -1],
        [-1, -1, 1],
        [-1, -1, -1],
    ],
    dtype=float,
)
patch_dirs /= np.linalg.norm(patch_dirs, axis=1, keepdims=True)


# ============================================================
# BUILD INITIAL SNAPSHOT
# ============================================================

# wrap_centered_coordinates, apply_minimum_image_displacement,
# make_grid_positions, generate_initial_coordinates live in
# scripts/geometry.py and scripts/initial_snapshot.py


# ============================================================
# BROWNIAN DYNAMICS
# ============================================================

# compute_gaussian_steric_forces, compute_gaussian_attraction_forces,
# compute_harmonic_bond_forces, step_brownian_dynamics live in
# scripts/forces.py


# ============================================================
# RUBISCO EPYC1 TIMESTEPPER (NEFI TimeStepper subclass)
# ============================================================


class RubiscoEpyc1TimeStepper(TimeStepper):
    """Differentiable Brownian Dynamics TimeStepper for Rubisco-EPYC1 condensates.

    Subclasses NEFI TimeStepper with gradient checkpointing and continuous
    Gaussian forces. Primary observation: particle trajectory coordinates
    of shape (T, 792, 3).
    """

    traceable: bool = False

    def __init__(
        self,
        n_steps: int,
        dt: float = 0.0005,
        box: float = 40.0,
        write_every: int = 100,
        seed: int = 42,
        grad_mode: str = "checkpoint",
        checkpoint_every: int = 32,
        field: str = "epsilon",
        fields: Sequence[str] | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.box = float(box)
        self.write_every = int(write_every)
        self.seed = int(seed)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype

        self.n_rubisco = N_R
        self.n_patches_per_rubisco = PATCHES
        self.n_chains = N_C
        self.n_beads_per_chain = N_BEADS
        self.sticker_local_indices = STICKER_INDEX
        self.n_particles = N_R + N_R * PATCHES + N_C * N_BEADS

        self.particle_labels = topology.build_particle_labels(
            self.n_rubisco,
            self.n_patches_per_rubisco,
            self.n_chains,
            self.n_beads_per_chain,
            self.sticker_local_indices,
        )
        self.particle_type_ids = topology.build_particle_type_ids(
            self.n_rubisco,
            self.n_patches_per_rubisco,
            self.n_chains,
            self.n_beads_per_chain,
            self.sticker_local_indices,
            device=self.device,
        )
        self.mobilities_by_particle = topology.build_mobilities_by_particle(
            self.n_particles, device=self.device, dtype=self.dtype
        )

        self.bond_pairs_by_type = topology.build_bond_topologies(
            self.n_rubisco,
            self.n_patches_per_rubisco,
            self.n_chains,
            self.n_beads_per_chain,
            device=self.device,
        )
        self.k_by_type = {"chain": CHAIN_K, "patch_tether": PATCH_TETHER_K}
        self.r0_by_type = {"chain": CHAIN_R0, "patch_tether": PATCH_RADIUS}

        eps_mat, sig_mat, rcut_mat = topology.build_steric_parameter_matrices(
            self.particle_type_ids, device=self.device, dtype=self.dtype
        )
        self.epsilon_steric_matrix = eps_mat
        self.sigma_steric_matrix = sig_mat
        self.r_cut_steric_matrix = rcut_mat

        patch_start = self.n_rubisco
        chain_start = self.n_rubisco + self.n_rubisco * self.n_patches_per_rubisco
        patch_idx_list = list(range(patch_start, chain_start))
        self.patch_indices = torch.tensor(patch_idx_list, dtype=torch.long, device=self.device)

        sticker_idx_list = []
        sticker_chain_list = []
        for c in range(self.n_chains):
            base = chain_start + c * self.n_beads_per_chain
            for b in self.sticker_local_indices:
                sticker_idx_list.append(base + b)
                sticker_chain_list.append(c)

        self.sticker_indices = torch.tensor(sticker_idx_list, dtype=torch.long, device=self.device)
        self.sticker_chain_ids = torch.tensor(sticker_chain_list, dtype=torch.long, device=self.device)

        patch_rubisco_list = []
        for p_local in range(len(patch_idx_list)):
            patch_rubisco_list.append(p_local // self.n_patches_per_rubisco)
        self.patch_rubisco_ids = torch.tensor(patch_rubisco_list, dtype=torch.long, device=self.device)

        # Preserve the public attribute names used by torch_kf_bd_kf_sweep.py.
        self.particleLabels = self.particle_labels
        self.particleTypeIds = self.particle_type_ids
        self.mobilitiesByParticle = self.mobilities_by_particle
        self.bondPairsByType = self.bond_pairs_by_type
        self.kByType = self.k_by_type
        self.r0ByType = self.r0_by_type
        self.epsilonStericMatrix = self.epsilon_steric_matrix
        self.sigmaStericMatrix = self.sigma_steric_matrix
        self.rCutStericMatrix = self.r_cut_steric_matrix
        self.patchIndices = self.patch_indices
        self.stickerIndices = self.sticker_indices
        self.stickerChainIds = self.sticker_chain_ids
        self.patchRubiscoIds = self.patch_rubisco_ids

        super().__init__(
            step=self.step,
            init=self.init,
            n_steps=n_steps,
            dt=dt,
            observe=self.observe,
            field=field,
            fields=fields if fields is not None else (field,),
            grad_mode=grad_mode,
            checkpoint_every=checkpoint_every,
        )

    def init(self, fields: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Initializes coordinates of shape (792, 3)."""
        if "positions_0" in fields:
            return fields["positions_0"].to(device=self.device, dtype=self.dtype)
        return initial_snapshot.generate_initial_coordinates(
            self.box,
            self.seed,
            self.n_rubisco,
            self.n_patches_per_rubisco,
            self.n_chains,
            self.n_beads_per_chain,
            patch_radius=PATCH_RADIUS,
            chain_r0=CHAIN_R0,
            device=self.device,
            dtype=self.dtype,
        )

    def step(
        self,
        state: torch.Tensor,
        params: Mapping[str, Any],
        dt: float,
        n: int,
    ) -> torch.Tensor:
        """Executes one Brownian dynamics step with explicit leaf forces and noise injection."""
        epsilon_att = params[self.primary]
        if not torch.is_tensor(epsilon_att):
            epsilon_att = torch.tensor(epsilon_att, device=state.device, dtype=state.dtype)

        f_bonds = forces.compute_harmonic_bond_forces(
            state, self.bond_pairs_by_type, self.k_by_type, self.r0_by_type, self.box
        )
        f_sterics = forces.compute_gaussian_steric_forces(
            state,
            self.epsilon_steric_matrix,
            self.sigma_steric_matrix,
            self.r_cut_steric_matrix,
            self.box,
        )
        f_attract = forces.compute_gaussian_attraction_forces(
            state,
            self.sticker_indices,
            self.patch_indices,
            epsilon_att,
            PATCH_SIGMA,
            PATCH_RCUT,
            self.box,
        )
        f_total = f_bonds + f_sterics + f_attract

        noise_std = torch.sqrt(2.0 * self.mobilities_by_particle * dt)
        noise_displacement = noise_std * torch.randn_like(state)

        return forces.step_brownian_dynamics(
            state, f_total, self.mobilities_by_particle, dt, noise_displacement, self.box
        )

    def observe(self, state: torch.Tensor, n: int) -> torch.Tensor | None:
        """Observes coordinates of shape (792, 3) every write_every steps."""
        if n % self.write_every == 0:
            return state
        return None


# ============================================================
# GSD OUTPUT
# ============================================================


def export_ovito_xyz(
    filepath: str,
    trajectory: torch.Tensor,
    particle_labels: Sequence[str],
    dt: float,
    write_every: int,
) -> None:
    """Exports multi-frame trajectory to standard OVITO-compatible XYZ format."""
    io.export_ovito_xyz(filepath, trajectory, particle_labels, dt, write_every)


# ============================================================
# ANALYSIS HELPERS
# ============================================================


def compute_soft_contacts(
    positions: torch.Tensor,
    sticker_indices: torch.Tensor,
    patch_indices: torch.Tensor,
    box: float,
    cutoff: float = 3.0,
    tau: float = 0.1,
) -> torch.Tensor:
    """Computes smooth, differentiable sigmoid contact indicators between stickers and patches."""
    return post_simulation_analysis.compute_soft_contacts(
        positions, sticker_indices, patch_indices, box, cutoff, tau
    )


def compute_hard_contacts(
    positions: torch.Tensor,
    sticker_indices: torch.Tensor,
    patch_indices: torch.Tensor,
    box: float,
    cutoff: float = 3.0,
) -> torch.Tensor:
    """Computes exact binary indicator matrix for sticker-patch contacts within cutoff."""
    return post_simulation_analysis.compute_hard_contacts(
        positions, sticker_indices, patch_indices, box, cutoff
    )


def compute_cluster_graph_metrics(
    contacts_binary: torch.Tensor,
    sticker_chain_ids: torch.Tensor,
    patch_rubisco_ids: torch.Tensor,
    n_rubisco: int = N_R,
    n_chains: int = N_C,
) -> tuple[int, int, float]:
    """Computes bipartite connected components over Rubisco complexes and EPYC1 chains."""
    return post_simulation_analysis.compute_cluster_graph_metrics(
        contacts_binary, sticker_chain_ids, patch_rubisco_ids, n_rubisco, n_chains
    )


# ============================================================
# POSTPROCESS: CSV, PLOTS
# ============================================================


def _draw_line_plot_pillow(
    x: Sequence[float],
    y: Sequence[float],
    xlabel: str,
    ylabel: str,
    title: str,
    save_path: str,
    width: int = 1200,
    height: int = 800,
) -> None:
    """Renders a 2D line plot to JPEG using Pillow with zero C-backend dependencies."""
    io._draw_line_plot_pillow(x, y, xlabel, ylabel, title, save_path, width, height)


def export_csv_and_summary(
    csv_filepath: str,
    summary_filepath: str,
    name: str,
    box: float,
    kon: float,
    epsilon_used: float,
    koff: float,
    dt: float,
    n_steps: int,
    write_every: int,
    trajectory: torch.Tensor,
    sticker_indices: torch.Tensor,
    patch_indices: torch.Tensor,
    sticker_chain_ids: torch.Tensor,
    patch_rubisco_ids: torch.Tensor,
    n_rubisco: int = N_R,
    n_chains: int = N_C,
) -> pd.DataFrame:
    """Computes contact and graph metrics, writing HOOMD-format CSV and summary JSON."""
    csv_dir = os.path.dirname(os.path.abspath(csv_filepath))
    if csv_dir:
        os.makedirs(csv_dir, exist_ok=True)
    summary_dir = os.path.dirname(os.path.abspath(summary_filepath))
    if summary_dir:
        os.makedirs(summary_dir, exist_ok=True)

    records = []
    n_frames = trajectory.shape[0]

    for frame_idx in range(n_frames):
        step = frame_idx * write_every
        sim_time = step * dt
        pos_frame = trajectory[frame_idx]

        contacts_binary = compute_hard_contacts(
            pos_frame, sticker_indices, patch_indices, box, cutoff=CONTACT_CUTOFF
        )
        n_contacts = int(contacts_binary.sum().item())
        n_clusters, largest_size, fraction = compute_cluster_graph_metrics(
            contacts_binary, sticker_chain_ids, patch_rubisco_ids, n_rubisco, n_chains
        )

        records.append(
            {
                "frame": frame_idx,
                "step": step,
                "time": sim_time,
                "box": box,
                "kon_or_epsilon": kon,
                "epsilon_used": epsilon_used,
                "koff_recorded": koff,
                "contacts": n_contacts,
                "bonds": n_contacts,
                "clusters": n_clusters,
                "largest": largest_size,
                "fraction": fraction,
            }
        )

    df = pd.DataFrame(records)
    df.to_csv(csv_filepath, index=False)

    final_contacts = int(df["contacts"].iloc[-1]) if len(df) > 0 else 0
    final_clusters = int(df["clusters"].iloc[-1]) if len(df) > 0 else (n_rubisco + n_chains)
    final_largest = int(df["largest"].iloc[-1]) if len(df) > 0 else 1
    final_fraction = float(df["fraction"].iloc[-1]) if len(df) > 0 else 1.0 / (n_rubisco + n_chains)

    summary = {
        "name": name,
        "box": box,
        "kon": kon,
        "epsilon_used": epsilon_used,
        "dt": dt,
        "n_steps": n_steps,
        "frames_written": len(records),
        "final_contacts": final_contacts,
        "final_clusters": final_clusters,
        "final_largest_cluster": final_largest,
        "final_fraction": final_fraction,
    }

    with open(summary_filepath, "w") as fh:
        json.dump(summary, fh, indent=2)

    return df


def export_diagnostic_plots(df: pd.DataFrame, plots_dir: str, name: str) -> None:
    """Generates 4 time-series JPEG plots matching HOOMD reference outputs."""
    io.generate_all_diagnostic_plots(df, plots_dir, name)


# ============================================================
# RUN
# ============================================================


def run_single_simulation(
    params_by_field: Mapping[str, Any],
    device: torch.device,
    output_base_dir: str = ".",
) -> dict[str, Any]:
    """Runs a complete simulation workflow for a single parameter configuration."""
    name = str(params_by_field.get("name", params_by_field.get("NAME", "sim")))
    box = float(params_by_field.get("BOX", params_by_field.get("box", 40.0)))
    kon = float(
        params_by_field.get(
            "KON",
            params_by_field.get(
                "EPS",
                params_by_field.get(
                    "kon_or_epsilon",
                    params_by_field.get("kon", params_by_field.get("eps", 1.0)),
                ),
            ),
        )
    )
    koff = float(params_by_field.get("KOFF", params_by_field.get("koff", 0.0)))
    seed = int(params_by_field.get("SEED", params_by_field.get("seed", 42)))
    n_steps = int(params_by_field.get("N_STEPS", params_by_field.get("n_steps", 10000)))
    write_every = int(params_by_field.get("WRITE_EVERY", params_by_field.get("write_every", 100)))
    dt = float(params_by_field.get("DT", params_by_field.get("dt", 0.0005)))

    global NAME, BOX, KON, KOFF, SEED, N_STEPS, WRITE_EVERY, DT, PATCH_EPSILON
    NAME = name
    BOX = box
    KON = kon
    KOFF = koff
    SEED = seed
    N_STEPS = n_steps
    WRITE_EVERY = write_every
    DT = dt
    PATCH_EPSILON = -abs(KON)
    epsilon_used = PATCH_EPSILON

    dirs = make_output_dirs(output_base_dir, name)
    xyz_path = dirs["xyz"]
    csv_path = dirs["csv"]
    summary_path = dirs["summary"]
    plots_dir = dirs["plots"]

    print(
        f"\nStarting {name}: BOX={box:.1f}, KON={kon:.3f}, epsilon_used={epsilon_used:.3f}, "
        f"KOFF={koff:.3f}, N_STEPS={n_steps}, WRITE_EVERY={write_every}, DT={dt:.5f}"
    )
    t0 = time.time()

    torch.manual_seed(seed)
    np.random.seed(seed)

    stepper = RubiscoEpyc1TimeStepper(
        n_steps=n_steps,
        dt=dt,
        box=box,
        write_every=write_every,
        seed=seed,
        grad_mode="checkpoint",
        checkpoint_every=32,
        field="epsilon",
        device=device,
    )

    eps_tensor = torch.tensor(epsilon_used, device=device, dtype=torch.float32)
    with torch.no_grad():
        trajectory = stepper({"epsilon": eps_tensor})

    elapsed = time.time() - t0
    print(f"Simulation completed in {elapsed:.2f} s. Exporting results...")

    export_ovito_xyz(xyz_path, trajectory, stepper.particle_labels, dt, write_every)
    df = export_csv_and_summary(
        csv_path,
        summary_path,
        name,
        box,
        kon,
        epsilon_used,
        koff,
        dt,
        n_steps,
        write_every,
        trajectory,
        stepper.sticker_indices,
        stepper.patch_indices,
        stepper.sticker_chain_ids,
        stepper.patch_rubisco_ids,
        stepper.n_rubisco,
        stepper.n_chains,
    )
    export_diagnostic_plots(df, plots_dir, name)

    print(f"Wrote XYZ: {xyz_path}")
    print(f"Wrote CSV: {csv_path}")
    print(f"Wrote Summary: {summary_path}")
    print(f"Wrote Plots in {plots_dir}")

    return {
        "name": name,
        "elapsed_seconds": elapsed,
        "final_contacts": int(df["contacts"].iloc[-1]) if len(df) > 0 else 0,
        "final_clusters": int(df["clusters"].iloc[-1]) if len(df) > 0 else 72,
    }


# Execute the supplied configuration(s) from top to bottom. A single JSON
# object runs once; a JSON list runs a sweep in list order.
print(f"Rubisco-EPYC1 Differentiable Brownian Dynamics. Device: {device}")
print(f"Found {len(PARAM_LIST)} simulation run(s) to execute.")
for run_params in PARAM_LIST:
    run_single_simulation(run_params, device)
print("\nAll simulations completed successfully.")
