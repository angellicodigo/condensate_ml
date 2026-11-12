"""Particle labels, type ids, mobilities, bond topology and steric parameter matrices.

Particle ordering (identical to the HOOMD snapshot):
  [0, n_rubisco)                         Rubisco cores  (R)
  [n_rubisco, n_rubisco*(1+patches))     Rubisco patches (P), grouped by Rubisco
  [..., end)                             EPYC1 chains, n_beads per chain (S / T alternating)
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch

from .constants import STERIC_PAIR_PARAMS, TYPE_INDEX, TYPE_NAMES


def build_particle_labels(
    n_rubisco: int,
    n_patches_per_rubisco: int,
    n_chains: int,
    n_beads_per_chain: int,
    sticker_local_indices: Sequence[int],
) -> list[str]:
    """Return one type label ('R', 'P', 'S' or 'T') per particle."""
    stickers = set(sticker_local_indices)
    labels = ["R"] * n_rubisco + ["P"] * (n_rubisco * n_patches_per_rubisco)
    chain_labels = ["T" if b in stickers else "S" for b in range(n_beads_per_chain)]
    labels += chain_labels * n_chains
    return labels


def build_particle_type_ids(
    n_rubisco: int,
    n_patches_per_rubisco: int,
    n_chains: int,
    n_beads_per_chain: int,
    sticker_local_indices: Sequence[int],
    device: torch.device | None = None,
) -> torch.Tensor:
    """Integer type id per particle (R=0, P=1, S=2, T=3), shape (n_particles,)."""
    labels = build_particle_labels(
        n_rubisco, n_patches_per_rubisco, n_chains, n_beads_per_chain, sticker_local_indices
    )
    return torch.tensor([TYPE_INDEX[x] for x in labels], dtype=torch.long, device=device)


def build_mobilities_by_particle(
    n_rubisco: int,
    n_patches_per_rubisco: int,
    n_chains: int,
    n_beads_per_chain: int,
    gamma_rubisco: float = 5.0,
    gamma_chain: float = 1.0,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build mobility factors mu = 1 / gamma per particle, shape (n_particles, 1).

    Rubisco centers and patches (the first n_rubisco * (1 + patches) particles)
    use gamma_rubisco; chain beads use gamma_chain.
    """
    n_slow = n_rubisco * (1 + n_patches_per_rubisco)
    n_particles = n_slow + n_chains * n_beads_per_chain
    mobilities = torch.full((n_particles, 1), 1.0 / gamma_chain, device=device, dtype=dtype)
    mobilities[:n_slow] = 1.0 / gamma_rubisco
    return mobilities


def build_bond_topologies(
    n_rubisco: int,
    n_patches_per_rubisco: int,
    n_chains: int,
    n_beads_per_chain: int,
    device: torch.device | None = None,
) -> dict[str, torch.Tensor]:
    """Bond index pairs per bond type: 'chain' (528 pairs) and 'patch_tether' (192 pairs)."""
    patch_start = n_rubisco
    chain_start = n_rubisco + n_rubisco * n_patches_per_rubisco

    chain_pairs = []
    for c in range(n_chains):
        base = chain_start + c * n_beads_per_chain
        for b in range(n_beads_per_chain - 1):
            chain_pairs.append([base + b, base + b + 1])

    tether_pairs = []
    for r in range(n_rubisco):
        for p in range(n_patches_per_rubisco):
            tether_pairs.append([r, patch_start + r * n_patches_per_rubisco + p])

    return {
        "chain": torch.tensor(chain_pairs, dtype=torch.long, device=device),
        "patch_tether": torch.tensor(tether_pairs, dtype=torch.long, device=device),
    }


def build_steric_parameter_matrices(
    particle_type_ids: torch.Tensor,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-pair (epsilon, sigma, r_cut) matrices of shape (N, N) for the Gaussian sterics.

    Pairs absent from STERIC_PAIR_PARAMS get epsilon=0, sigma=1, r_cut=0 (no interaction),
    matching the defaults set in the HOOMD script.
    """
    n_types = len(TYPE_NAMES)
    eps_table = torch.zeros((n_types, n_types), dtype=dtype)
    sig_table = torch.ones((n_types, n_types), dtype=dtype)
    rcut_table = torch.zeros((n_types, n_types), dtype=dtype)
    for (a, b), (eps, sig, rcut) in STERIC_PAIR_PARAMS.items():
        for i, j in ((TYPE_INDEX[a], TYPE_INDEX[b]), (TYPE_INDEX[b], TYPE_INDEX[a])):
            eps_table[i, j] = eps
            sig_table[i, j] = sig
            rcut_table[i, j] = rcut

    ids = particle_type_ids.to("cpu")
    pair_i, pair_j = ids[:, None], ids[None, :]
    return (
        eps_table[pair_i, pair_j].to(device),
        sig_table[pair_i, pair_j].to(device),
        rcut_table[pair_i, pair_j].to(device),
    )


def build_nonbonded_pair_mask(
    bond_pairs_by_type: Mapping[str, torch.Tensor],
    n_particles: int,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Boolean (N, N) matrix: True where a pair interacts via pair forces.

    Directly bonded pairs and self-pairs are False, matching HOOMD's default
    ``exclusions=('bond',)`` for the neighbor list.
    """
    mask = torch.ones((n_particles, n_particles), dtype=torch.bool, device=device)
    for pairs in bond_pairs_by_type.values():
        mask[pairs[:, 0], pairs[:, 1]] = False
        mask[pairs[:, 1], pairs[:, 0]] = False
    mask.fill_diagonal_(False)
    return mask
