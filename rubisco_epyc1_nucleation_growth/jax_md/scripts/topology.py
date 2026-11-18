"""Construction of the fixed Rubisco-EPYC1 particle topology.

This module contains no simulation-time state.  The functions here build the
labels, type tables, mobilities, bonds, and exclusions that are shared by every
rollout.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np

from .constants import (
    CHAIN_K,
    CHAIN_R0,
    GAMMA_CHAIN,
    GAMMA_RUBISCO,
    N_BEADS,
    N_C,
    N_R,
    PATCHES,
    PATCH_RADIUS,
    PATCH_TETHER_K,
    STERIC_PAIR_PARAMS,
    STICKER_INDEX,
    TYPE_INDEX,
)


def build_particle_labels() -> list[str]:
    """Return the HOOMD-compatible label for every particle."""
    labels = ["R"] * N_R + ["P"] * (N_R * PATCHES)
    labels += ["T" if bead in STICKER_INDEX else "S" for bead in range(N_BEADS)] * N_C
    return labels


def build_particle_type_ids(dtype: Any = jnp.int32) -> jnp.ndarray:
    """Convert the fixed particle labels to integer type IDs."""
    return jnp.asarray([TYPE_INDEX[label] for label in build_particle_labels()], dtype=dtype)


def build_particle_mobilities(n_particles: int, dtype: Any) -> jnp.ndarray:
    """Return one broadcastable inverse-friction value per particle."""
    n_slow = N_R * (1 + PATCHES)
    values = np.full(n_particles, 1.0 / GAMMA_CHAIN, dtype=np.float32)
    values[:n_slow] = 1.0 / GAMMA_RUBISCO
    return jnp.asarray(values[:, None], dtype=dtype)


def build_bond_data(dtype: Any) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Return bond endpoints, spring constants, and rest lengths."""
    chain_start = N_R + N_R * PATCHES
    chain = [
        [chain_start + chain_id * N_BEADS + bead,
         chain_start + chain_id * N_BEADS + bead + 1]
        for chain_id in range(N_C)
        for bead in range(N_BEADS - 1)
    ]
    patch_tethers = [
        [rubisco_id, N_R + rubisco_id * PATCHES + patch_id]
        for rubisco_id in range(N_R)
        for patch_id in range(PATCHES)
    ]
    pairs = jnp.asarray(chain + patch_tethers, dtype=jnp.int32)
    spring_constants = jnp.asarray(
        [CHAIN_K] * len(chain) + [PATCH_TETHER_K] * len(patch_tethers), dtype=dtype
    )
    rest_lengths = jnp.asarray(
        [CHAIN_R0] * len(chain) + [PATCH_RADIUS] * len(patch_tethers), dtype=dtype
    )
    return pairs, spring_constants, rest_lengths


def build_bond_exclusion_mask(n_particles: int, bond_pairs: jnp.ndarray) -> jnp.ndarray:
    """Exclude self-pairs and directly bonded pairs from steric forces."""
    mask = np.ones((n_particles, n_particles), dtype=bool)
    np.fill_diagonal(mask, False)
    pairs = np.asarray(bond_pairs)
    mask[pairs[:, 0], pairs[:, 1]] = False
    mask[pairs[:, 1], pairs[:, 0]] = False
    return jnp.asarray(mask)


def build_steric_parameter_tables(dtype: Any) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Build epsilon, sigma, and cutoff tables indexed by particle type."""
    epsilon = np.zeros((4, 4), dtype=np.float32)
    sigma = np.ones((4, 4), dtype=np.float32)
    cutoff = np.zeros((4, 4), dtype=np.float32)
    for (first, second), values in STERIC_PAIR_PARAMS.items():
        first_id, second_id = TYPE_INDEX[first], TYPE_INDEX[second]
        for left, right in ((first_id, second_id), (second_id, first_id)):
            epsilon[left, right], sigma[left, right], cutoff[left, right] = values
    return tuple(jnp.asarray(table, dtype=dtype) for table in (epsilon, sigma, cutoff))


def build_particle_indices() -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Return patch/sticker indices and their Rubisco/chain ownership IDs."""
    patch_start = N_R
    chain_start = N_R + N_R * PATCHES
    patch_indices = jnp.arange(patch_start, chain_start, dtype=jnp.int32)
    sticker_indices = [
        chain_start + chain_id * N_BEADS + bead
        for chain_id in range(N_C)
        for bead in STICKER_INDEX
    ]
    return (
        patch_indices,
        jnp.asarray(sticker_indices, dtype=jnp.int32),
        jnp.repeat(jnp.arange(N_R, dtype=jnp.int32), PATCHES),
        jnp.repeat(jnp.arange(N_C, dtype=jnp.int32), len(STICKER_INDEX)),
    )
