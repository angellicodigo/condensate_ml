"""JAX-MD Differentiable Brownian Dynamics Simulation for Rubisco-EPYC1 Condensates.

This module implements the exact continuous, force-based patchy particle model of
rubisco_epyc1_nucleation_growth/molecular_sim/simulate_hoomd_patchy_full.py using
JAX and JAX-MD primitives.

Key Capabilities:
- Full physical parity with HOOMD: 792 particles (24 Rubisco centers, 192 patches,
  48 EPYC1 chains with 12 beads alternating S and T).
- Exact harmonic bonds for polymer backbones and Rubisco patch tethers.
- Bounded Gaussian soft sterics across all steric pairs.
- Continuous Gaussian sticker-patch attraction (depth = -|KON|).
- Vectorized Euler-Maruyama Brownian dynamics with particle-specific mobilities
  (mu_R = 0.2, mu_P = 0.2, mu_S = 1.0, mu_T = 1.0) and kBT = 1.0.
- End-to-end differentiability via JAX autograd and jax.jit XLA kernel compilation.
- Takes CLI input identically to HOOMD (params.json or <(echo "$row")).
- Emits OVITO-compatible XYZ, 12-column CSV, summary JSON, and diagnostic plots.

Requirements:
    pip install jax jaxlib jax-md numpy pandas matplotlib
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping, NamedTuple, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import jax
    import jax.numpy as jnp
    from jax import jit, grad, vmap, random
    from jax_md import space, energy, partition, util
    HAS_JAX = True
except ImportError:
    HAS_JAX = False


# ==============================================================================
# Model Architecture Constants
# ==============================================================================

PARTICLE_TYPE_NAMES: Tuple[str, ...] = ("R", "P", "S", "T")
TYPE_ID_RUBISCO_CENTER: int = 0
TYPE_ID_RUBISCO_PATCH: int = 1
TYPE_ID_CHAIN_SPACER: int = 2
TYPE_ID_CHAIN_STICKER: int = 3

N_RUBISCO: int = 24
N_PATCHES_PER_RUBISCO: int = 8
PATCH_RADIUS: float = 2.5
PATCH_TETHER_K: float = 50.0

N_CHAINS: int = 48
N_BEADS_PER_CHAIN: int = 12
CHAIN_BOND_K: float = 10.0
CHAIN_BOND_R0: float = 1.2
STICKER_LOCAL_INDICES: Tuple[int, ...] = (1, 3, 5, 7, 9, 11)

N_TOTAL_PARTICLES: int = (
    N_RUBISCO + (N_RUBISCO * N_PATCHES_PER_RUBISCO) + (N_CHAINS * N_BEADS_PER_CHAIN)
)  # 792

CONTACT_CUTOFF: float = 3.0


# ==============================================================================
# Configuration & Invariant Data Types
# ==============================================================================

@dataclasses.dataclass(frozen=True)
class SimulationConfig:
    """Immutable configuration container for simulation runs."""
    name: str
    box: float
    kon_or_eps: float
    koff_recorded: float = 0.0
    seed: int = 42
    n_steps: int = 100000
    write_every: int = 500
    dt: float = 0.0005
    kt: float = 1.0
    output_dir: Path = dataclasses.field(default_factory=lambda: Path("."))


@dataclasses.dataclass(frozen=True)
class SystemTopology:
    """Topology definition storing particle assignments and bond graphs."""
    type_ids: np.ndarray
    initial_positions: np.ndarray
    bond_pairs: np.ndarray
    bond_k: np.ndarray
    bond_r0: np.ndarray
    patch_indices: np.ndarray
    sticker_indices: np.ndarray
    chain_by_sticker: np.ndarray


class ForceFieldTables(NamedTuple):
    """Pair potential lookup matrices on device."""
    epsilon_matrix: Any
    sigma_matrix: Any
    r_cut_matrix: Any
    mobility_vector: Any


# ==============================================================================
# Topology & Initial Snapshot Construction
# ==============================================================================

def compute_normalized_cube_vertices() -> np.ndarray:
    """Return 8 unit vectors pointing to the corners of a cube."""
    vertices = np.array(
        [
            [1.0, 1.0, 1.0],
            [1.0, 1.0, -1.0],
            [1.0, -1.0, 1.0],
            [1.0, -1.0, -1.0],
            [-1.0, 1.0, 1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
            [-1.0, -1.0, -1.0],
        ],
        dtype=np.float64,
    )
    norms = np.linalg.norm(vertices, axis=1, keepdims=True)
    return vertices / norms


def wrap_centered(coords: np.ndarray, box: float) -> np.ndarray:
    """Wrap coordinates to centered periodic box [-box/2, box/2)."""
    return ((coords + box / 2.0) % box) - box / 2.0


def generate_grid_positions(n: int, box: float, margin: float) -> np.ndarray:
    """Generate n initial positions on a regular 3D lattice."""
    n_side = int(np.ceil(n ** (1.0 / 3.0)))
    vals = np.linspace(-box / 2.0 + margin, box / 2.0 - margin, n_side)
    grid = np.array(np.meshgrid(vals, vals, vals)).T.reshape(-1, 3)
    return grid[:n].copy()


def build_system_topology(box: float, seed: int) -> SystemTopology:
    """Construct initial coordinates, types, and bonds matching HOOMD reference."""
    rng = np.random.default_rng(seed)
    patch_dirs = compute_normalized_cube_vertices()

    positions = np.zeros((N_TOTAL_PARTICLES, 3), dtype=np.float32)
    type_ids = np.zeros(N_TOTAL_PARTICLES, dtype=np.int32)

    # 1. Rubisco Centers (Type R = 0)
    rubisco_centers = generate_grid_positions(N_RUBISCO, box, margin=7.0)
    rubisco_centers += rng.normal(scale=0.20, size=rubisco_centers.shape)
    rubisco_centers = wrap_centered(rubisco_centers, box)

    idx = 0
    for r in range(N_RUBISCO):
        positions[idx] = rubisco_centers[r]
        type_ids[idx] = TYPE_ID_RUBISCO_CENTER
        idx += 1

    # 2. Tethered Rubisco Patches (Type P = 1)
    patch_start = idx
    patch_tethers = []
    for r in range(N_RUBISCO):
        center = rubisco_centers[r]
        for p in range(N_PATCHES_PER_RUBISCO):
            positions[idx] = center + PATCH_RADIUS * patch_dirs[p]
            type_ids[idx] = TYPE_ID_RUBISCO_PATCH
            patch_tethers.append([r, idx])
            idx += 1
    patch_end = idx

    # 3. EPYC1 Polymer Chains (Type S = 2, Type T = 3)
    chain_start = idx
    chain_centers = generate_grid_positions(N_CHAINS, box, margin=5.0)
    chain_centers += rng.normal(scale=0.25, size=chain_centers.shape)
    chain_centers = wrap_centered(chain_centers, box)

    chain_bonds = []
    sticker_indices_list = []
    chain_by_sticker_list = []

    for c in range(N_CHAINS):
        center = chain_centers[c]
        direction = rng.normal(size=3)
        direction /= np.linalg.norm(direction)
        chain_base = idx

        for b in range(N_BEADS_PER_CHAIN):
            offset = (b - (N_BEADS_PER_CHAIN - 1) / 2.0) * CHAIN_BOND_R0 * direction
            bead_pos = wrap_centered(center + offset, box)
            positions[idx] = bead_pos

            if b in STICKER_LOCAL_INDICES:
                type_ids[idx] = TYPE_ID_CHAIN_STICKER
                sticker_indices_list.append(idx)
                chain_by_sticker_list.append(c)
            else:
                type_ids[idx] = TYPE_ID_CHAIN_SPACER

            if b > 0:
                chain_bonds.append([idx - 1, idx])
            idx += 1

    all_bonds = chain_bonds + patch_tethers
    bond_pairs = np.array(all_bonds, dtype=np.int32)

    bond_k = np.empty(len(all_bonds), dtype=np.float32)
    bond_k[:len(chain_bonds)] = CHAIN_BOND_K
    bond_k[len(chain_bonds):] = PATCH_TETHER_K

    bond_r0 = np.empty(len(all_bonds), dtype=np.float32)
    bond_r0[:len(chain_bonds)] = CHAIN_BOND_R0
    bond_r0[len(chain_bonds):] = PATCH_RADIUS

    return SystemTopology(
        type_ids=type_ids,
        initial_positions=positions,
        bond_pairs=bond_pairs,
        bond_k=bond_k,
        bond_r0=bond_r0,
        patch_indices=np.arange(patch_start, patch_end, dtype=np.int32),
        sticker_indices=np.array(sticker_indices_list, dtype=np.int32),
        chain_by_sticker=np.array(chain_by_sticker_list, dtype=np.int32),
    )


# ==============================================================================
# Force Field Construction
# ==============================================================================

def build_force_field_tables(kon_or_eps: float) -> ForceFieldTables:
    """Assemble pair parameter matrices for 4 particle types (R, P, S, T)."""
    eps = np.zeros((4, 4), dtype=np.float32)
    sig = np.ones((4, 4), dtype=np.float32)
    rcut = np.zeros((4, 4), dtype=np.float32)

    def set_pair(type_a: int, type_b: int, e: float, s: float, r: float) -> None:
        eps[type_a, type_b] = eps[type_b, type_a] = e
        sig[type_a, type_b] = sig[type_b, type_a] = s
        rcut[type_a, type_b] = rcut[type_b, type_a] = r

    # 1. Bounded Gaussian Steric Repulsions
    set_pair(TYPE_ID_RUBISCO_CENTER, TYPE_ID_RUBISCO_CENTER, 2.0, 1.5, 3.0)
    set_pair(TYPE_ID_CHAIN_SPACER, TYPE_ID_CHAIN_SPACER, 1.0, 1.0, 2.5)
    set_pair(TYPE_ID_CHAIN_STICKER, TYPE_ID_CHAIN_STICKER, 1.0, 1.0, 2.5)
    set_pair(TYPE_ID_CHAIN_SPACER, TYPE_ID_CHAIN_STICKER, 1.0, 1.0, 2.5)
    set_pair(TYPE_ID_RUBISCO_CENTER, TYPE_ID_CHAIN_SPACER, 1.5, 1.5, 3.0)
    set_pair(TYPE_ID_RUBISCO_CENTER, TYPE_ID_CHAIN_STICKER, 1.5, 1.5, 3.0)
    set_pair(TYPE_ID_RUBISCO_PATCH, TYPE_ID_RUBISCO_PATCH, 0.2, 0.8, 2.0)
    set_pair(TYPE_ID_RUBISCO_PATCH, TYPE_ID_CHAIN_SPACER, 0.2, 0.8, 2.0)
    set_pair(TYPE_ID_RUBISCO_PATCH, TYPE_ID_RUBISCO_CENTER, 0.2, 0.8, 2.0)

    # 2. Continuous Gaussian Attraction (T - P pairs)
    att_epsilon = -abs(float(kon_or_eps))
    att_sigma = 1.0
    att_rcut = 3.0
    set_pair(TYPE_ID_CHAIN_STICKER, TYPE_ID_RUBISCO_PATCH, att_epsilon, att_sigma, att_rcut)

    # 3. Particle Mobilities (mu = 1 / gamma)
    mobilities = np.array([0.2, 0.2, 1.0, 1.0], dtype=np.float32)

    if HAS_JAX:
        return ForceFieldTables(
            epsilon_matrix=jnp.array(eps),
            sigma_matrix=jnp.array(sig),
            r_cut_matrix=jnp.array(rcut),
            mobility_vector=jnp.array(mobilities),
        )
    return ForceFieldTables(
        epsilon_matrix=eps,
        sigma_matrix=sig,
        r_cut_matrix=rcut,
        mobility_vector=mobilities,
    )


# ==============================================================================
# JAX Differentiable Physics Kernels
# ==============================================================================

if HAS_JAX:
    def create_jax_simulation_kernels(box: float, force_tables: ForceFieldTables):
        """Construct JIT-compiled displacement, force, and integration functions."""
        displacement_fn, shift_fn = space.periodic(box)

        @jit
        def compute_harmonic_bond_forces(positions, bond_pairs, bond_k, bond_r0):
            """Compute vectorized harmonic spring forces for all 720 bonds."""
            idx_i = bond_pairs[:, 0]
            idx_j = bond_pairs[:, 1]
            pos_i = positions[idx_i]
            pos_j = positions[idx_j]

            # Displacement vector pointing from j to i
            delta = vmap(displacement_fn)(pos_i, pos_j)
            dist = jnp.linalg.norm(delta, axis=-1, keepdims=True) + 1e-8
            direction = delta / dist

            force_mag = -bond_k[:, None] * (dist - bond_r0[:, None])
            force_ij = force_mag * direction

            # Accumulate onto particles
            forces = jnp.zeros_like(positions)
            forces = forces.at[idx_i].add(force_ij)
            forces = forces.at[idx_j].add(-force_ij)
            return forces

        @jit
        def compute_gaussian_pair_forces(positions, type_ids):
            """Compute full pairwise Gaussian forces with bounded sterics and attraction."""
            n = positions.shape[0]
            # Pairwise displacement tensor (N, N, 3)
            delta = vmap(lambda p1: vmap(lambda p2: displacement_fn(p1, p2))(positions))(positions)
            dist = jnp.linalg.norm(delta, axis=-1) + 1e-8

            t_i = type_ids[:, None]
            t_j = type_ids[None, :]
            eps_pair = force_tables.epsilon_matrix[t_i, t_j]
            sig_pair = force_tables.sigma_matrix[t_i, t_j]
            rcut_pair = force_tables.r_cut_matrix[t_i, t_j]

            # Gaussian gradient: -d/dr [eps * exp(-0.5*(r/sigma)^2)] = eps/(sigma^2) * exp(...) * r
            within_cut = (dist < rcut_pair) & (dist > 1e-6)
            gauss_factor = (eps_pair / (sig_pair ** 2)) * jnp.exp(-0.5 * ((dist / sig_pair) ** 2))
            pref = jnp.where(within_cut, gauss_factor, 0.0)

            # Force on i from j: pref * delta_ij
            forces = jnp.sum(pref[:, :, None] * delta, axis=1)
            return forces

        @jit
        def brownian_step(positions, rng_key, dt, kt, bond_pairs, bond_k, bond_r0, type_ids):
            """Single Euler-Maruyama Brownian dynamics step."""
            f_bond = compute_harmonic_bond_forces(positions, bond_pairs, bond_k, bond_r0)
            f_pair = compute_gaussian_pair_forces(positions, type_ids)
            f_total = f_bond + f_pair

            particle_mobilities = force_tables.mobility_vector[type_ids][:, None]
            deterministic_disp = particle_mobilities * f_total * dt

            noise_std = jnp.sqrt(2.0 * kt * particle_mobilities * dt)
            noise = random.normal(rng_key, shape=positions.shape)
            stochastic_disp = noise_std * noise

            new_positions = vmap(shift_fn)(positions, deterministic_disp + stochastic_disp)
            return new_positions

        return shift_fn, brownian_step


# ==============================================================================
# Graph & Cluster Analysis (Pure CPU Functions)
# ==============================================================================

def union_find(n_nodes: int, edges: Sequence[Tuple[int, int]]) -> Tuple[int, int, float]:
    """Calculate connected components on bipartite Rubisco-EPYC1 contact graph."""
    parent = list(range(n_nodes))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for u, v in edges:
        union(u, v)

    groups: dict[int, list[int]] = {}
    for i in range(n_nodes):
        root = find(i)
        groups.setdefault(root, []).append(i)

    sizes = sorted((len(v) for v in groups.values()), reverse=True)
    largest = sizes[0] if sizes else 0
    fraction = largest / float(n_nodes) if n_nodes > 0 else 0.0
    return len(sizes), largest, fraction


def compute_contacts_and_clusters(
    positions_np: np.ndarray,
    patch_indices: np.ndarray,
    sticker_indices: np.ndarray,
    chain_by_sticker: np.ndarray,
    box: float,
) -> Tuple[int, int, int, float]:
    """Detect contacts by distance threshold and construct the Rubisco-EPYC1 cluster graph."""
    patch_pos = positions_np[patch_indices]
    sticker_pos = positions_np[sticker_indices]

    delta = sticker_pos[:, None, :] - patch_pos[None, :, :]
    delta_pbc = delta - box * np.round(delta / box)
    dist = np.linalg.norm(delta_pbc, axis=-1)

    contacts = np.argwhere(dist < CONTACT_CUTOFF)
    n_contacts = len(contacts)

    graph_edges: list[Tuple[int, int]] = []
    for s_local, p_local in contacts:
        chain_id = int(chain_by_sticker[s_local])
        rubisco_id = int(p_local // N_PATCHES_PER_RUBISCO)
        graph_edges.append((N_RUBISCO + chain_id, rubisco_id))

    n_clusters, largest, fraction = union_find(N_RUBISCO + N_CHAINS, graph_edges)
    return n_contacts, n_clusters, largest, fraction


# ==============================================================================
# Exporters & Plotting
# ==============================================================================

def write_xyz_frame(file_handle, step: int, positions_np: np.ndarray, type_ids_np: np.ndarray) -> None:
    """Write one frame of particle coordinates formatted for OVITO visualization."""
    file_handle.write(f"{len(positions_np)}\n")
    file_handle.write(f"step {step}\n")
    for pos, tid in zip(positions_np, type_ids_np):
        label = PARTICLE_TYPE_NAMES[tid]
        file_handle.write(f"{label} {pos[0]:.5f} {pos[1]:.5f} {pos[2]:.5f}\n")


def generate_diagnostic_plots(df: pd.DataFrame, plots_dir: Path, name: str) -> None:
    """Render publication-quality JPEG plots of physical observables vs time."""
    plots_dir.mkdir(parents=True, exist_ok=True)
    time_series = df["time"]

    metrics = [
        ("bonds", "contacts / bonds", "bonds"),
        ("clusters", "number of clusters", "clusters"),
        ("largest", "largest cluster size", "largest"),
        ("fraction", "largest cluster fraction", "fraction"),
    ]

    for key, ylabel, filename_suffix in metrics:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(time_series, df[key], lw=1.5, color="#1f77b4")
        ax.set_xlabel("time")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{name}: {ylabel} vs time")
        fig.tight_layout()
        fig.savefig(plots_dir / f"{name}_{filename_suffix}.jpeg", dpi=180)
        plt.close(fig)


# ==============================================================================
# Simulation Runner
# ==============================================================================

def run_simulation(config: SimulationConfig) -> None:
    """High-level orchestrator executing the JAX-MD Brownian simulation."""
    if not HAS_JAX:
        raise RuntimeError(
            "JAX and JAX-MD are required to run simulate_jax_md_patchy.py. "
            "Install with: pip install jax jaxlib jax-md"
        )

    traj_dir = config.output_dir / "traj_jax"
    data_dir = config.output_dir / "data_jax"
    plots_dir = config.output_dir / "plots_jax"

    for d in (traj_dir, data_dir, plots_dir):
        d.mkdir(parents=True, exist_ok=True)

    xyz_path = traj_dir / f"{config.name}.xyz"
    csv_path = data_dir / f"{config.name}.csv"
    summary_path = data_dir / f"{config.name}_summary.json"

    print(f"\n===========================================================")
    print(f"JAX-MD Simulation: {config.name}")
    print(f"BOX={config.box:.2f}, KON/EPS={config.kon_or_eps:.3f}, STEPS={config.n_steps}")
    print(f"===========================================================")

    topology = build_system_topology(config.box, config.seed)
    force_tables = build_force_field_tables(config.kon_or_eps)

    shift_fn, brownian_step = create_jax_simulation_kernels(config.box, force_tables)

    # Move data to JAX device arrays
    positions = jnp.array(topology.initial_positions)
    type_ids = jnp.array(topology.type_ids)
    bond_pairs = jnp.array(topology.bond_pairs)
    bond_k = jnp.array(topology.bond_k)
    bond_r0 = jnp.array(topology.bond_r0)

    rng_key = random.PRNGKey(config.seed)
    records: list[dict[str, Any]] = []

    t0 = time.time()
    with open(xyz_path, "w") as xyz_file:
        for step in range(config.n_steps + 1):
            if step % config.write_every == 0:
                pos_cpu = np.asarray(positions)
                n_contacts, n_clusters, largest, fraction = compute_contacts_and_clusters(
                    pos_cpu,
                    topology.patch_indices,
                    topology.sticker_indices,
                    topology.chain_by_sticker,
                    config.box,
                )

                record = {
                    "frame": len(records),
                    "step": step,
                    "time": step * config.dt,
                    "box": config.box,
                    "kon_or_epsilon": config.kon_or_eps,
                    "epsilon_used": -abs(config.kon_or_eps),
                    "koff_recorded": config.koff_recorded,
                    "contacts": n_contacts,
                    "bonds": n_contacts,
                    "clusters": n_clusters,
                    "largest": largest,
                    "fraction": fraction,
                }
                records.append(record)
                write_xyz_frame(xyz_file, step, pos_cpu, topology.type_ids)

                if step % (config.write_every * 10) == 0 or step == config.n_steps:
                    print(
                        f"step={step:6d} time={step*config.dt:.4f} "
                        f"bonds={n_contacts:4d} clusters={n_clusters:3d} "
                        f"largest={largest:3d} frac={fraction:.4f}"
                    )

            if step < config.n_steps:
                rng_key, subkey = random.split(rng_key)
                positions = brownian_step(
                    positions,
                    subkey,
                    config.dt,
                    config.kt,
                    bond_pairs,
                    bond_k,
                    bond_r0,
                    type_ids,
                )

    elapsed = time.time() - t0
    df = pd.DataFrame(records)
    df.to_csv(csv_path, index=False)
    generate_diagnostic_plots(df, plots_dir, config.name)

    summary = {
        "name": config.name,
        "box": config.box,
        "kon_or_epsilon": config.kon_or_eps,
        "steps": config.n_steps,
        "final_bonds": int(df["bonds"].iloc[-1]),
        "final_clusters": int(df["clusters"].iloc[-1]),
        "final_largest_cluster": int(df["largest"].iloc[-1]),
        "final_fraction": float(df["fraction"].iloc[-1]),
        "elapsed_seconds": elapsed,
    }
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)

    print(f"Finished {config.name} in {elapsed:.2f} s")
    print(f"Wrote XYZ: {xyz_path}")
    print(f"Wrote CSV: {csv_path}")
    print(f"Wrote plots: {plots_dir}")


# ==============================================================================
# CLI Entrypoint
# ==============================================================================

def parse_simulation_inputs(argv: Sequence[str]) -> list[SimulationConfig]:
    """Parse JSON configuration from file, process substitution, or stdin."""
    if len(argv) < 2:
        default_params = Path("params.json")
        if default_params.exists():
            raw_data = json.loads(default_params.read_text(encoding="utf-8"))
        else:
            raise ValueError("Usage: python simulate_jax_md_patchy.py <params.json>")
    else:
        arg = argv[1].strip()
        if arg.startswith("{") or arg.startswith("["):
            raw_data = json.loads(arg)
        else:
            raw_data = json.loads(Path(arg).read_text(encoding="utf-8"))

    raw_items = [raw_data] if isinstance(raw_data, dict) else list(raw_data)
    configs: list[SimulationConfig] = []

    for item in raw_items:
        cfg = SimulationConfig(
            name=str(item.get("name", "unnamed")),
            box=float(item.get("BOX", item.get("box", 40.0))),
            kon_or_eps=float(item.get("KON", item.get("EPS", item.get("eps", 1.0)))),
            koff_recorded=float(item.get("KOFF", item.get("koff", 0.0))),
            seed=int(item.get("SEED", item.get("seed", 42))),
            n_steps=int(item.get("N_STEPS", item.get("n_steps", 10000))),
            write_every=int(item.get("WRITE_EVERY", item.get("write_every", 100))),
            dt=float(item.get("DT", item.get("dt", 0.0005))),
            output_dir=Path(os.environ.get("OUTPUT_DIR", ".")),
        )
        configs.append(cfg)

    return configs


def main() -> None:
    configs = parse_simulation_inputs(sys.argv)
    for cfg in configs:
        run_simulation(cfg)


if __name__ == "__main__":
    main()
