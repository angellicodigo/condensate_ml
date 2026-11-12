"""Differentiable Brownian Dynamics Simulation for Rubisco-EPYC1 Condensates.

Pure helper functions live in the scripts/ package alongside this file:

- scripts/geometry.py                 : coordinate wrapping, minimum image, safe norm
- scripts/initial_snapshot.py         : initial grid layout and coordinate snapshot
- scripts/topology.py                 : labels, type IDs, mobilities, bond topologies,
                                        steric parameter matrices, non-bonded pair mask
- scripts/forces.py                   : steric repulsion, sticker/patch attraction,
                                        bond forces, Brownian integration step
- scripts/post_simulation_analysis.py : soft/hard contacts, cluster graph metrics
- scripts/output_io.py                : XYZ export, per-frame metrics, CSV, summary JSON, plots

Layout (top to bottom, no main()):
  1. Input params (module-level)
  2. Model parameters
  3. Output folders
  4. Device
  5. Patch geometry
  6. Build initial snapshot
  7. Brownian dynamics
  8. RubiscoEpyc1TimeStepper (TimeStepper subclass)
  9. Run: simulations
  10. Module-level execution
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
from nefi.operators.timestepping import TimeStepper

from scripts import forces
from scripts import output_io
from scripts import initial_snapshot
from scripts import topology


# ============================================================
# INPUT PARAMS
# python torch_kf_bd_kf_sweep.py params.json
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

PATCH_SIGMA = 1.0
PATCH_RCUT = 3.0

CONTACT_CUTOFF = 3.0

# Time step shared by every run. Not read from the params JSON.
DT = 0.0005
CHECKPOINT_EVERY = 32


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
# GPU DEVICE
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

    Primary observation: particle trajectory coordinates of shape (T, n_particles, 3).
    """

    traceable: bool = False

    def __init__(
        self,
        n_steps: int,
        dt: float = DT,
        box: float = 40.0,
        write_every: int = 100,
        seed: int = 42,
        grad_mode: str = "checkpoint",
        checkpoint_every: int = CHECKPOINT_EVERY,
        field: str = "epsilon",
        fields: Sequence[str] | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.box = float(box)
        self.patch_dirs = patch_dirs
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
            self.n_rubisco,
            self.n_patches_per_rubisco,
            self.n_chains,
            self.n_beads_per_chain,
            device=self.device,
            dtype=self.dtype,
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

        # True where a pair interacts sterically: bonded pairs and self-pairs are
        # False, matching HOOMD's default bond exclusions in the neighbor list.
        self.nonbonded_pair_mask = topology.build_nonbonded_pair_mask(
            self.bond_pairs_by_type, self.n_particles, device=self.device
        )

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

        patch_rubisco_list = [
            p_local // self.n_patches_per_rubisco for p_local in range(len(patch_idx_list))
        ]
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
        """Initializes coordinates of shape (n_particles, 3)."""
        if "positions_0" in fields:
            return fields["positions_0"].to(device=self.device, dtype=self.dtype)
        return initial_snapshot.generate_initial_coordinates(
            self.box,
            self.patch_dirs,
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
            self.nonbonded_pair_mask,
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
        """Observes coordinates every write_every steps."""
        if n % self.write_every == 0:
            return state
        return None


# ============================================================
# RUN: SIMULATIONS
# ============================================================
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
    """Runs a complete simulation workflow for one parameter configuration.

    Uses the module constant DT; no module-level state is modified.
    """
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

    # The CSV assumes frame 0 is step 0 and frame i is step i * write_every.
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
        CONTACT_CUTOFF,
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
    """Runs every configuration in params_list in order and returns one result dict per run."""
    names = [str(_get_param(p, ("name", "NAME"), "sim")) for p in params_list]
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        raise ValueError(f"Run names must be unique (outputs would be overwritten): {duplicates}")

    results = []
    for run_params in params_list:
        results.append(run_single_simulation(run_params, device, output_base_dir))
    return results


# ============================================================
# RUN
# ============================================================
print(f"Rubisco-EPYC1 Differentiable Brownian Dynamics. Device: {device}")
print(f"Found {len(PARAM_LIST)} simulation run(s) to execute.")
run_simulation(PARAM_LIST, device)
print("\nAll simulations completed successfully.")