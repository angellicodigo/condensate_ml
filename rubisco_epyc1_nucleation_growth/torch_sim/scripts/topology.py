"""Topology setup for the Rubisco-EPYC1 simulation.

Builds particle labels, integer type IDs, per-particle mobilities, bond
topology index matrices, and pairwise steric parameter matrices. All functions
are pure and return fresh tensors or lists.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch


def build_particle_labels(
    n_rubisco: int = 24,
    n_patches_per_rubisco: int = 8,
    n_chains: int = 48,
    n_beads_per_chain: int = 12,
    sticker_local_indices: Sequence[int] = (1, 3, 5, 7, 9, 11),
) -> list[str]:
    """Generate sequential particle type labels: R, P, S, T.

    - ``R`` : Rubisco center particles
    - ``P`` : Rubisco patch particles
    - ``S`` : EPYC1 non-sticker (solvent-like) beads
    - ``T`` : EPYC1 sticker beads (the ones listed in sticker_local_indices)
    """
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
    """Build an integer type tensor: R:0, P:1, S:2, T:3.

    Used to index into the pairwise steric parameter tables.
    """
    type_list = [0] * n_rubisco
    type_list.extend([1] * (n_rubisco * n_patches_per_rubisco))
    sticker_set = set(sticker_local_indices)
    for _ in range(n_chains):
        for b in range(n_beads_per_chain):
            type_list.append(3 if b in sticker_set else 2)
    return torch.tensor(type_list, dtype=torch.long, device=device)


def build_mobilities_by_particle(
    n_particles: int = 24 + 24 * 8 + 48 * 12,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build mobility factors mu = 1 / gamma per particle.

    Rubisco centers and patches have gamma=5 (slow), chain beads have gamma=1
    (fast). Returns a tensor of shape (n_particles, 1).
    """
    mobilities = torch.ones(n_particles, 1, device=device, dtype=dtype)
    n_rubisco_and_patches = 24 + 24 * 8
    mobilities[0:n_rubisco_and_patches] = 1.0 / 5.0
    return mobilities


def build_bond_topologies(
    n_rubisco: int = 24,
    n_patches_per_rubisco: int = 8,
    n_chains: int = 48,
    n_beads_per_chain: int = 12,
    device: torch.device | None = None,
) -> dict[str, torch.Tensor]:
    """Pre-compute bond index matrices: chain bonds and patch tethers.

    Returns a dict with keys:
      - ``"chain"``      : consecutive bead pairs within each EPYC1 chain
      - ``"patch_tether"`` : each Rubisco center to each of its patch particles
    """
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
    """Build (N, N) pairwise steric parameter matrices (epsilon, sigma, r_cut).

    Uses a 4x4 type table (R:0, P:1, S:2, T:3) and lookups by particle type.
    The diagonal is zeroed so a particle does not repel itself.
    """
    n_types = 4
    eps_table = torch.zeros((n_types, n_types), device=device, dtype=dtype)
    sig_table = torch.ones((n_types, n_types), device=device, dtype=dtype)
    rcut_table = torch.zeros((n_types, n_types), device=device, dtype=dtype)

    def set_pair(t1: int, t2: int, eps: float, sig: float, rcut: float) -> None:
        eps_table[t1, t2] = eps
        eps_table[t2, t1] = eps
        sig_table[t1, t2] = sig
        sig_table[t2, t1] = sig
        rcut_table[t1, t2] = rcut
        rcut_table[t2, t1] = rcut

    set_pair(0, 0, 2.0, 1.5, 3.0)  # R-R
    set_pair(2, 2, 1.0, 1.0, 2.5)  # S-S
    set_pair(3, 3, 1.0, 1.0, 2.5)  # T-T
    set_pair(2, 3, 1.0, 1.0, 2.5)  # S-T
    set_pair(0, 2, 1.5, 1.5, 3.0)  # R-S
    set_pair(0, 3, 1.5, 1.5, 3.0)  # R-T
    set_pair(1, 1, 0.2, 0.8, 2.0)  # P-P
    set_pair(1, 2, 0.2, 0.8, 2.0)  # P-S
    set_pair(1, 0, 0.2, 0.8, 2.0)  # P-R

    pair_type_a = type_ids[:, None]
    pair_type_b = type_ids[None, :]

    epsilon_matrix = eps_table[pair_type_a, pair_type_b]
    sigma_matrix = sig_table[pair_type_a, pair_type_b]
    r_cut_matrix = rcut_table[pair_type_a, pair_type_b]

    epsilon_matrix.fill_diagonal_(0.0)
    r_cut_matrix.fill_diagonal_(0.0)

    return epsilon_matrix, sigma_matrix, r_cut_matrix
