"""Differentiable Brownian Dynamics Simulation for Rubisco-EPYC1 Condensates.

Subclasses nefi.operators.timestepping.TimeStepper to provide exact physical parity
with HOOMD-blue reference simulations (simulate_hoomd_patchy_full.py) alongside
PyTorch gradient checkpointing and continuous autograd differentiability.
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


# =============================================================================
# Honest Leaf Functions: Core Physics & Numerical Mechanics
# =============================================================================

def wrap_centered_coordinates(positions: torch.Tensor, box: float) -> torch.Tensor:
    """Wraps particle coordinates into the origin-centered periodic box [-L/2, +L/2)."""
    return ((positions + box * 0.5) % box) - box * 0.5


def apply_minimum_image_displacement(delta: torch.Tensor, box: float) -> torch.Tensor:
    """Applies the minimum image convention to pairwise displacement vectors."""
    return delta - box * torch.round(delta / box)


def compute_harmonic_bond_forces(
    positions: torch.Tensor,
    bond_pairs_by_type: Mapping[str, torch.Tensor],
    k_by_type: Mapping[str, float],
    r0_by_type: Mapping[str, float],
    box: float,
) -> torch.Tensor:
    """Computes differentiable harmonic bond forces across all bonded topology groups."""
    total_bond_forces = torch.zeros_like(positions)
    for bond_type, pairs in bond_pairs_by_type.items():
        k = k_by_type[bond_type]
        r0 = r0_by_type[bond_type]
        i_indices = pairs[:, 0]
        j_indices = pairs[:, 1]
        delta = positions[i_indices] - positions[j_indices]
        delta_min = apply_minimum_image_displacement(delta, box)
        distances = torch.linalg.vector_norm(delta_min, dim=-1, keepdim=True)
        safe_distances = torch.clamp(distances, min=1e-8)
        force_magnitudes = -k * (distances - r0)
        forces_on_i = force_magnitudes * (delta_min / safe_distances)
        total_bond_forces = total_bond_forces.scatter_add(
            0, i_indices.unsqueeze(-1).expand(-1, 3), forces_on_i
        )
        total_bond_forces = total_bond_forces.scatter_add(
            0, j_indices.unsqueeze(-1).expand(-1, 3), -forces_on_i
        )
    return total_bond_forces


def compute_gaussian_steric_forces(
    positions: torch.Tensor,
    epsilon_matrix: torch.Tensor,
    sigma_matrix: torch.Tensor,
    r_cut_matrix: torch.Tensor,
    box: float,
) -> torch.Tensor:
    """Computes vectorized bounded Gaussian repulsive forces between all non-bonded pairs."""
    delta = positions[:, None, :] - positions[None, :, :]
    delta_min = apply_minimum_image_displacement(delta, box)
    dist_sq = torch.sum(delta_min * delta_min, dim=-1)
    within_cutoff_mask = (dist_sq < (r_cut_matrix * r_cut_matrix)) & (dist_sq > 1e-12)
    sig_sq = sigma_matrix * sigma_matrix
    safe_sig_sq = torch.clamp(sig_sq, min=1e-8)
    prefactors = (epsilon_matrix / safe_sig_sq) * torch.exp(-0.5 * dist_sq / safe_sig_sq)
    masked_prefactors = torch.where(within_cutoff_mask, prefactors, torch.zeros_like(prefactors))
    return torch.sum(masked_prefactors.unsqueeze(-1) * delta_min, dim=1)


def compute_gaussian_attraction_forces(
    positions: torch.Tensor,
    sticker_indices: torch.Tensor,
    patch_indices: torch.Tensor,
    epsilon_att: torch.Tensor,
    sigma_att: float,
    r_cut_att: float,
    box: float,
) -> torch.Tensor:
    """Computes continuous Gaussian attractive forces between EPYC1 stickers and Rubisco patches."""
    pos_stickers = positions[sticker_indices]
    pos_patches = positions[patch_indices]
    delta = pos_stickers[:, None, :] - pos_patches[None, :, :]
    delta_min = apply_minimum_image_displacement(delta, box)
    dist_sq = torch.sum(delta_min * delta_min, dim=-1)
    within_cutoff_mask = dist_sq < (r_cut_att * r_cut_att)
    sig_sq = sigma_att * sigma_att
    prefactors = (epsilon_att / sig_sq) * torch.exp(-0.5 * dist_sq / sig_sq)
    masked_prefactors = torch.where(within_cutoff_mask, prefactors, torch.zeros_like(prefactors))
    forces_on_stickers = torch.sum(masked_prefactors.unsqueeze(-1) * delta_min, dim=1)
    forces_on_patches = -torch.sum(masked_prefactors.unsqueeze(-1) * delta_min, dim=0)
    total_attraction_forces = torch.zeros_like(positions)
    total_attraction_forces = total_attraction_forces.scatter_add(
        0, sticker_indices.unsqueeze(-1).expand(-1, 3), forces_on_stickers
    )
    total_attraction_forces = total_attraction_forces.scatter_add(
        0, patch_indices.unsqueeze(-1).expand(-1, 3), forces_on_patches
    )
    return total_attraction_forces


def step_brownian_dynamics(
    positions: torch.Tensor,
    total_force: torch.Tensor,
    mobilities: torch.Tensor,
    dt: float,
    noise_displacement: torch.Tensor,
    box: float,
) -> torch.Tensor:
    """Executes a single overdamped Langevin (Euler-Maruyama) integration step."""
    deterministic_displacement = mobilities * total_force * dt
    updated_positions = positions + deterministic_displacement + noise_displacement
    return wrap_centered_coordinates(updated_positions, box)


# =============================================================================
# Observables & Graph Analysis Helpers
# =============================================================================

def compute_soft_contacts(
    positions: torch.Tensor,
    sticker_indices: torch.Tensor,
    patch_indices: torch.Tensor,
    box: float,
    cutoff: float = 3.0,
    tau: float = 0.1,
) -> torch.Tensor:
    """Computes smooth, differentiable sigmoid contact indicators between stickers and patches."""
    pos_stickers = positions[..., sticker_indices, :]
    pos_patches = positions[..., patch_indices, :]
    delta = pos_stickers[..., :, None, :] - pos_patches[..., None, :, :]
    delta_min = apply_minimum_image_displacement(delta, box)
    distances = torch.linalg.vector_norm(delta_min, dim=-1)
    return torch.sigmoid((cutoff - distances) / tau)


def compute_hard_contacts(
    positions: torch.Tensor,
    sticker_indices: torch.Tensor,
    patch_indices: torch.Tensor,
    box: float,
    cutoff: float = 3.0,
) -> torch.Tensor:
    """Computes exact binary indicator matrix for sticker-patch contacts within cutoff."""
    pos_stickers = positions[..., sticker_indices, :]
    pos_patches = positions[..., patch_indices, :]
    delta = pos_stickers[..., :, None, :] - pos_patches[..., None, :, :]
    delta_min = apply_minimum_image_displacement(delta, box)
    distances = torch.linalg.vector_norm(delta_min, dim=-1)
    return distances < cutoff


def compute_cluster_graph_metrics(
    contacts_binary: torch.Tensor,
    sticker_chain_ids: torch.Tensor,
    patch_rubisco_ids: torch.Tensor,
    n_rubisco: int = 24,
    n_chains: int = 48,
) -> tuple[int, int, float]:
    """Computes bipartite connected components over Rubisco complexes and EPYC1 chains."""
    total_nodes = n_rubisco + n_chains
    parent = list(range(total_nodes))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra = find(a)
        rb = find(b)
        if ra != rb:
            parent[rb] = ra

    active_contacts = torch.nonzero(contacts_binary, as_tuple=False)
    for row in range(active_contacts.shape[0]):
        s_idx = active_contacts[row, 0].item()
        p_idx = active_contacts[row, 1].item()
        c = sticker_chain_ids[s_idx].item()
        r = patch_rubisco_ids[p_idx].item()
        union(n_rubisco + c, r)

    groups: dict[int, list[int]] = {}
    for node in range(total_nodes):
        root = find(node)
        groups.setdefault(root, []).append(node)

    cluster_sizes = sorted([len(members) for members in groups.values()], reverse=True)
    cluster_count = len(cluster_sizes)
    largest_cluster_size = cluster_sizes[0] if cluster_sizes else 0
    cluster_fraction = float(largest_cluster_size) / float(total_nodes)
    return cluster_count, largest_cluster_size, cluster_fraction


# =============================================================================
# Topology Setup & Geometry Construction
# =============================================================================

def make_grid_positions(n: int, box: float, margin: float = 6.0) -> np.ndarray:
    """Generates initial grid positions spaced symmetrically inside a cubic box."""
    n_side = int(np.ceil(n ** (1 / 3)))
    vals = np.linspace(-box / 2.0 + margin, box / 2.0 - margin, n_side)
    pts = np.array(np.meshgrid(vals, vals, vals)).T.reshape(-1, 3)
    return pts[:n].copy()


def build_particle_labels(
    n_rubisco: int = 24,
    n_patches_per_rubisco: int = 8,
    n_chains: int = 48,
    n_beads_per_chain: int = 12,
    sticker_local_indices: Sequence[int] = (1, 3, 5, 7, 9, 11),
) -> list[str]:
    """Generates sequential particle type labels: R (0..23), P (24..215), S/T (216..791)."""
    labels = ["R"] * n_rubisco
    labels.extend(["P"] * (n_rubisco * n_patches_per_rubisco))
    sticker_set = set(sticker_local_indices)
    for _ in range(n_chains):
        for b in range(n_beads_per_chain):
            labels.append("T" if b in sticker_set else "S")
    return labels


def build_particle_type_ids(
    n_rubisco: int = 24,
    n_patches_per_rubisco: int = 8,
    n_chains: int = 48,
    n_beads_per_chain: int = 12,
    sticker_local_indices: Sequence[int] = (1, 3, 5, 7, 9, 11),
    device: torch.device | None = None,
) -> torch.Tensor:
    """Builds integer type tensor: R:0, P:1, S:2, T:3."""
    type_list = [0] * n_rubisco
    type_list.extend([1] * (n_rubisco * n_patches_per_rubisco))
    sticker_set = set(sticker_local_indices)
    for _ in range(n_chains):
        for b in range(n_beads_per_chain):
            type_list.append(3 if b in sticker_set else 2)
    return torch.tensor(type_list, dtype=torch.long, device=device)


def build_mobilities_by_particle(
    n_particles: int = 792,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Builds mobility factors mu = 1 / gamma (gamma_R=5, gamma_P=5, gamma_S=1, gamma_T=1)."""
    mobilities = torch.ones(n_particles, 1, device=device, dtype=dtype)
    mobilities[0:216] = 1.0 / 5.0
    return mobilities


def build_bond_topologies(
    n_rubisco: int = 24,
    n_patches_per_rubisco: int = 8,
    n_chains: int = 48,
    n_beads_per_chain: int = 12,
    device: torch.device | None = None,
) -> dict[str, torch.Tensor]:
    """Pre-computes bond index matrices: 528 chain bonds and 192 patch tethers."""
    patch_start = n_rubisco
    chain_start = n_rubisco + n_rubisco * n_patches_per_rubisco

    chain_bonds = []
    for c in range(n_chains):
        base = chain_start + c * n_beads_per_chain
        for b in range(n_beads_per_chain - 1):
            chain_bonds.append([base + b, base + b + 1])

    patch_tethers = []
    for r in range(n_rubisco):
        for p in range(n_patches_per_rubisco):
            patch_tethers.append([r, patch_start + r * n_patches_per_rubisco + p])

    return {
        "chain": torch.tensor(chain_bonds, dtype=torch.long, device=device),
        "patch_tether": torch.tensor(patch_tethers, dtype=torch.long, device=device),
    }


def build_steric_parameter_matrices(
    type_ids: torch.Tensor,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Builds (792, 792) pairwise steric parameter matrices (epsilon, sigma, r_cut)."""
    eps_table = torch.zeros((4, 4), device=device, dtype=dtype)
    sig_table = torch.ones((4, 4), device=device, dtype=dtype)
    rcut_table = torch.zeros((4, 4), device=device, dtype=dtype)

    def set_pair(t1: int, t2: int, eps: float, sig: float, rcut: float) -> None:
        eps_table[t1, t2] = eps
        eps_table[t2, t1] = eps
        sig_table[t1, t2] = sig
        sig_table[t2, t1] = sig
        rcut_table[t1, t2] = rcut
        rcut_table[t2, t1] = rcut

    set_pair(0, 0, 2.0, 1.5, 3.0)
    set_pair(2, 2, 1.0, 1.0, 2.5)
    set_pair(3, 3, 1.0, 1.0, 2.5)
    set_pair(2, 3, 1.0, 1.0, 2.5)
    set_pair(0, 2, 1.5, 1.5, 3.0)
    set_pair(0, 3, 1.5, 1.5, 3.0)
    set_pair(1, 1, 0.2, 0.8, 2.0)
    set_pair(1, 2, 0.2, 0.8, 2.0)
    set_pair(1, 0, 0.2, 0.8, 2.0)

    pair_type_a = type_ids[:, None]
    pair_type_b = type_ids[None, :]

    epsilon_matrix = eps_table[pair_type_a, pair_type_b]
    sigma_matrix = sig_table[pair_type_a, pair_type_b]
    r_cut_matrix = rcut_table[pair_type_a, pair_type_b]

    epsilon_matrix.fill_diagonal_(0.0)
    r_cut_matrix.fill_diagonal_(0.0)

    return epsilon_matrix, sigma_matrix, r_cut_matrix


def generate_initial_coordinates(
    box: float,
    seed: int = 42,
    n_rubisco: int = 24,
    n_patches_per_rubisco: int = 8,
    n_chains: int = 48,
    n_beads_per_chain: int = 12,
    patch_radius: float = 2.5,
    chain_r0: float = 1.2,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Generates initial 792 coordinates matching simulate_hoomd_patchy_full.py."""
    rng = np.random.default_rng(seed)
    n_patch = n_rubisco * n_patches_per_rubisco
    n_chain = n_chains * n_beads_per_chain
    n_total = n_rubisco + n_patch + n_chain

    pos = np.zeros((n_total, 3), dtype=np.float64)

    rubisco_centers = make_grid_positions(n_rubisco, box, margin=7.0)
    rubisco_centers += rng.normal(scale=0.20, size=rubisco_centers.shape)
    rubisco_centers = ((rubisco_centers + box / 2.0) % box) - box / 2.0

    idx = 0
    for r in range(n_rubisco):
        pos[idx] = rubisco_centers[r]
        idx += 1

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
        dtype=np.float64,
    )
    patch_dirs /= np.linalg.norm(patch_dirs, axis=1, keepdims=True)

    for r in range(n_rubisco):
        center = rubisco_centers[r]
        for p in range(n_patches_per_rubisco):
            pos[idx] = center + patch_radius * patch_dirs[p]
            idx += 1

    chain_centers = make_grid_positions(n_chains, box, margin=5.0)
    chain_centers += rng.normal(scale=0.25, size=chain_centers.shape)
    chain_centers = ((chain_centers + box / 2.0) % box) - box / 2.0

    for c in range(n_chains):
        center = chain_centers[c]
        direction = rng.normal(size=3)
        direction /= np.linalg.norm(direction)

        for b in range(n_beads_per_chain):
            bead_pos = center + (b - (n_beads_per_chain - 1) / 2.0) * chain_r0 * direction
            bead_pos = ((bead_pos + box / 2.0) % box) - box / 2.0
            pos[idx] = bead_pos
            idx += 1

    return torch.tensor(pos, device=device, dtype=dtype)


# =============================================================================
# RubiscoEpyc1TimeStepper: NEFI Forward Operator Subclass
# =============================================================================

class RubiscoEpyc1TimeStepper(TimeStepper):
    """Differentiable Brownian Dynamics TimeStepper for Rubisco-EPYC1 condensates.

    Subclasses NEFI TimeStepper with gradient checkpointing and continuous Gaussian forces.
    Primary observation: particle trajectory coordinates of shape (T, 792, 3).
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

        self.n_rubisco = 24
        self.n_patches_per_rubisco = 8
        self.n_chains = 48
        self.n_beads_per_chain = 12
        self.sticker_local_indices = (1, 3, 5, 7, 9, 11)
        self.n_particles = 792

        self.particleLabels = build_particle_labels(
            self.n_rubisco,
            self.n_patches_per_rubisco,
            self.n_chains,
            self.n_beads_per_chain,
            self.sticker_local_indices,
        )
        self.particleTypeIds = build_particle_type_ids(
            self.n_rubisco,
            self.n_patches_per_rubisco,
            self.n_chains,
            self.n_beads_per_chain,
            self.sticker_local_indices,
            device=self.device,
        )
        self.mobilitiesByParticle = build_mobilities_by_particle(
            self.n_particles, device=self.device, dtype=self.dtype
        )

        self.bondPairsByType = build_bond_topologies(
            self.n_rubisco,
            self.n_patches_per_rubisco,
            self.n_chains,
            self.n_beads_per_chain,
            device=self.device,
        )
        self.kByType = {"chain": 10.0, "patch_tether": 50.0}
        self.r0ByType = {"chain": 1.2, "patch_tether": 2.5}

        eps_mat, sig_mat, rcut_mat = build_steric_parameter_matrices(
            self.particleTypeIds, device=self.device, dtype=self.dtype
        )
        self.epsilonStericMatrix = eps_mat
        self.sigmaStericMatrix = sig_mat
        self.rCutStericMatrix = rcut_mat

        patch_start = self.n_rubisco
        chain_start = self.n_rubisco + self.n_rubisco * self.n_patches_per_rubisco
        patch_idx_list = list(range(patch_start, chain_start))
        self.patchIndices = torch.tensor(patch_idx_list, dtype=torch.long, device=self.device)

        sticker_idx_list = []
        sticker_chain_list = []
        for c in range(self.n_chains):
            base = chain_start + c * self.n_beads_per_chain
            for b in self.sticker_local_indices:
                sticker_idx_list.append(base + b)
                sticker_chain_list.append(c)

        self.stickerIndices = torch.tensor(sticker_idx_list, dtype=torch.long, device=self.device)
        self.stickerChainIds = torch.tensor(sticker_chain_list, dtype=torch.long, device=self.device)

        patch_rubisco_list = []
        for p_local in range(len(patch_idx_list)):
            patch_rubisco_list.append(p_local // self.n_patches_per_rubisco)
        self.patchRubiscoIds = torch.tensor(patch_rubisco_list, dtype=torch.long, device=self.device)

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
        return generate_initial_coordinates(
            self.box,
            self.seed,
            self.n_rubisco,
            self.n_patches_per_rubisco,
            self.n_chains,
            self.n_beads_per_chain,
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

        f_bonds = compute_harmonic_bond_forces(
            state, self.bondPairsByType, self.kByType, self.r0ByType, self.box
        )
        f_sterics = compute_gaussian_steric_forces(
            state, self.epsilonStericMatrix, self.sigmaStericMatrix, self.rCutStericMatrix, self.box
        )
        f_attract = compute_gaussian_attraction_forces(
            state, self.stickerIndices, self.patchIndices, epsilon_att, 1.0, 3.0, self.box
        )
        f_total = f_bonds + f_sterics + f_attract

        noise_std = torch.sqrt(2.0 * self.mobilitiesByParticle * dt)
        noise_displacement = noise_std * torch.randn_like(state)

        return step_brownian_dynamics(
            state, f_total, self.mobilitiesByParticle, dt, noise_displacement, self.box
        )

    def observe(self, state: torch.Tensor, n: int) -> torch.Tensor | None:
        """Observes coordinates of shape (792, 3) every write_every steps."""
        if n % self.write_every == 0:
            return state
        return None


# =============================================================================
# Output Exporters & Serialization
# =============================================================================

def export_ovito_xyz(
    filepath: str,
    trajectory: torch.Tensor,
    particle_labels: Sequence[str],
    dt: float,
    write_every: int,
) -> None:
    """Exports multi-frame trajectory to standard OVITO-compatible XYZ format."""
    parent_dir = os.path.dirname(os.path.abspath(filepath))
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    traj_cpu = trajectory.detach().cpu().numpy()
    n_frames, n_particles, _ = traj_cpu.shape
    with open(filepath, "w") as fh:
        for frame_idx in range(n_frames):
            step = frame_idx * write_every
            fh.write(f"{n_particles}\n")
            fh.write(f"step {step}\n")
            coords = traj_cpu[frame_idx]
            for label, pos in zip(particle_labels, coords):
                fh.write(f"{label} {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}\n")


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
    n_rubisco: int = 24,
    n_chains: int = 48,
) -> pd.DataFrame:
    """Computes contact and graph metrics, writing HOOMD-format 12-column CSV and summary JSON."""
    csv_dir = os.path.dirname(os.path.abspath(csv_filepath))
    if csv_dir:
        os.makedirs(csv_dir, exist_ok=True)
    summary_dir = os.path.dirname(os.path.abspath(summary_filepath))
    if summary_dir:
        os.makedirs(summary_dir, exist_ok=True)

    recordsByFrame = []
    n_frames = trajectory.shape[0]

    for frame_idx in range(n_frames):
        step = frame_idx * write_every
        sim_time = step * dt
        pos_frame = trajectory[frame_idx]

        contacts_binary = compute_hard_contacts(
            pos_frame, sticker_indices, patch_indices, box, cutoff=3.0
        )
        n_contacts = int(contacts_binary.sum().item())
        n_clusters, largest_size, fraction = compute_cluster_graph_metrics(
            contacts_binary, sticker_chain_ids, patch_rubisco_ids, n_rubisco, n_chains
        )

        recordsByFrame.append(
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

    df = pd.DataFrame(recordsByFrame)
    df.to_csv(csv_filepath, index=False)

    final_contacts = int(df["contacts"].iloc[-1]) if len(df) > 0 else 0
    final_clusters = int(df["clusters"].iloc[-1]) if len(df) > 0 else (n_rubisco + n_chains)
    final_largest = int(df["largest"].iloc[-1]) if len(df) > 0 else 1
    final_fraction = float(df["fraction"].iloc[-1]) if len(df) > 0 else 1.0 / (n_rubisco + n_chains)

    summaryByField = {
        "name": name,
        "box": box,
        "kon": kon,
        "epsilon_used": epsilon_used,
        "dt": dt,
        "n_steps": n_steps,
        "frames_written": len(recordsByFrame),
        "final_contacts": final_contacts,
        "final_clusters": final_clusters,
        "final_largest_cluster": final_largest,
        "final_fraction": final_fraction,
    }

    with open(summary_filepath, "w") as fh:
        json.dump(summaryByField, fh, indent=2)

    return df


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
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    margin_left = 120
    margin_right = 60
    margin_top = 80
    margin_bottom = 100

    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom

    draw.line(
        [(margin_left, margin_top), (margin_left, height - margin_bottom)],
        fill=(0, 0, 0),
        width=2,
    )
    draw.line(
        [(margin_left, height - margin_bottom), (width - margin_right, height - margin_bottom)],
        fill=(0, 0, 0),
        width=2,
    )

    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)

    if len(x_arr) > 0:
        x_min, x_max = float(x_arr.min()), float(x_arr.max())
        y_min, y_max = float(y_arr.min()), float(y_arr.max())
        if x_max == x_min:
            x_max += 1.0
        if y_max == y_min:
            y_max += 1.0

        pts = []
        for xi, yi in zip(x_arr, y_arr):
            px = margin_left + ((xi - x_min) / (x_max - x_min)) * plot_w
            py = (height - margin_bottom) - ((yi - y_min) / (y_max - y_min)) * plot_h
            pts.append((px, py))

        if len(pts) > 1:
            draw.line(pts, fill=(31, 119, 180), width=3)
        elif len(pts) == 1:
            draw.ellipse(
                [pts[0][0] - 3, pts[0][1] - 3, pts[0][0] + 3, pts[0][1] + 3],
                fill=(31, 119, 180),
            )

    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    draw.text((width // 2 - 60, margin_top // 2), title, fill=(0, 0, 0), font=font)
    draw.text((width // 2 - 20, height - margin_bottom // 2), xlabel, fill=(0, 0, 0), font=font)
    draw.text((margin_left // 4, height // 2), ylabel, fill=(0, 0, 0), font=font)

    img.save(save_path, "JPEG", quality=95)


def export_diagnostic_plots(df: pd.DataFrame, plots_dir: str, name: str) -> None:
    """Generates 4 time-series JPEG plots matching HOOMD reference outputs."""
    if plots_dir:
        os.makedirs(plots_dir, exist_ok=True)
    if len(df) == 0:
        return

    plotsByMetric = [
        ("bonds", "contacts / bonds"),
        ("clusters", "number of clusters"),
        ("largest", "largest cluster size"),
        ("fraction", "largest cluster fraction"),
    ]

    for key, ylabel in plotsByMetric:
        save_path = os.path.join(plots_dir, f"{name}_{key}.jpeg")
        _draw_line_plot_pillow(
            df["time"], df[key], "time", ylabel, f"{name}: {ylabel} vs time", save_path
        )


# =============================================================================
# CLI Parser & Orchestration
# =============================================================================

def parse_simulation_inputs(cli_args: Sequence[str]) -> list[dict[str, Any]]:
    """Robustly parses simulation parameter inputs from arguments, STDIN, or params.json."""
    if len(cli_args) > 1:
        arg = cli_args[1].strip()
        if (arg.startswith("{") and arg.endswith("}")) or (arg.startswith("[") and arg.endswith("]")):
            raw_data = json.loads(arg)
        else:
            try:
                is_file = os.path.exists(arg)
            except OSError:
                is_file = False
            if is_file:
                with open(arg, "r") as fh:
                    raw_data = json.load(fh)
            else:
                raw_data = json.loads(arg)
    elif not sys.stdin.isatty():
        raw_data = json.load(sys.stdin)
    elif os.path.exists("params.json"):
        with open("params.json", "r") as fh:
            raw_data = json.load(fh)
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        script_params = os.path.join(script_dir, "params.json")
        if os.path.exists(script_params):
            with open(script_params, "r") as fh:
                raw_data = json.load(fh)
        else:
            raw_data = [
                {
                    "name": "control",
                    "BOX": 40.0,
                    "KON": 1.0,
                    "SEED": 42,
                    "N_STEPS": 1000,
                    "WRITE_EVERY": 100,
                    "DT": 0.0005,
                }
            ]

    if isinstance(raw_data, list):
        return raw_data
    if isinstance(raw_data, dict):
        return [raw_data]
    raise ValueError(f"Unrecognized parameter JSON structure: {type(raw_data)}")


def run_single_simulation(
    paramsByField: Mapping[str, Any],
    device: torch.device,
    output_base_dir: str = ".",
) -> dict[str, Any]:
    """Runs a complete simulation workflow for a single parameter configuration."""
    name = str(paramsByField.get("name", paramsByField.get("NAME", "sim")))
    box = float(paramsByField.get("BOX", paramsByField.get("box", 40.0)))
    kon = float(
        paramsByField.get(
            "KON",
            paramsByField.get(
                "EPS",
                paramsByField.get(
                    "kon_or_epsilon",
                    paramsByField.get("kon", paramsByField.get("eps", 1.0)),
                ),
            ),
        )
    )
    koff = float(paramsByField.get("KOFF", paramsByField.get("koff", 0.0)))
    seed = int(paramsByField.get("SEED", paramsByField.get("seed", 42)))
    n_steps = int(paramsByField.get("N_STEPS", paramsByField.get("n_steps", 10000)))
    write_every = int(paramsByField.get("WRITE_EVERY", paramsByField.get("write_every", 100)))
    dt = float(paramsByField.get("DT", paramsByField.get("dt", 0.0005)))

    epsilon_used = -abs(kon)

    traj_dir = os.path.join(output_base_dir, "traj_bd")
    data_dir = os.path.join(output_base_dir, "data_bd")
    plots_dir = os.path.join(output_base_dir, "plots_bd")

    xyz_path = os.path.join(traj_dir, f"{name}.xyz")
    csv_path = os.path.join(data_dir, f"{name}.csv")
    summary_path = os.path.join(data_dir, f"{name}_summary.json")

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

    export_ovito_xyz(xyz_path, trajectory, stepper.particleLabels, dt, write_every)
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
        stepper.stickerIndices,
        stepper.patchIndices,
        stepper.stickerChainIds,
        stepper.patchRubiscoIds,
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


def main() -> None:
    """Main CLI entrypoint."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Rubisco-EPYC1 Differentiable Brownian Dynamics. Device: {device}")
    param_list = parse_simulation_inputs(sys.argv)
    print(f"Found {len(param_list)} simulation run(s) to execute.")
    for params in param_list:
        run_single_simulation(params, device)
    print("\nAll simulations completed successfully.")


if __name__ == "__main__":
    main()
