"""Topology construction functions for Rubisco and EPYC1 complexes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import jax.numpy as jnp
import numpy as np


def build_particle_labels(
    n_rubisco: int,
    n_patches_per_rubisco: int,
    n_chains: int,
    n_beads_per_chain: int,
    sticker_indices: Sequence[int],
) -> list[str]:
    """Constructs the sequence of string type labels for all particles in the system.

    Inputs:
        n_rubisco: Number of Rubisco centers.
        n_patches_per_rubisco: Number of surface patches per Rubisco.
        n_chains: Number of EPYC1 chains.
        n_beads_per_chain: Number of beads per EPYC1 chain.
        sticker_indices: Local bead indices within a chain acting as stickers.
    Outputs:
        List of particle type labels ('R', 'P', 'S', 'T').
    """
    labels = ["R"] * n_rubisco + ["P"] * (n_rubisco * n_patches_per_rubisco)
    labels += [
        "T" if bead in sticker_indices else "S"
        for bead in range(n_beads_per_chain)
    ] * n_chains
    return labels


def build_particle_type_ids(
    particle_labels: Sequence[str],
    indexByTypeName: Mapping[str, int],
    dtype: Any = jnp.int32,
) -> jnp.ndarray:
    """Converts a sequence of string particle labels to integer type identifiers.

    Inputs:
        particle_labels: Sequence of string particle type labels.
        indexByTypeName: Dictionary mapping type label strings to integer type IDs.
        dtype: Integer data type for output array.
    Outputs:
        Array of integer particle type IDs shaped (n_particles,).
    """
    return jnp.asarray([indexByTypeName[label] for label in particle_labels], dtype=dtype)


def build_particle_mobilities(
    n_particles: int,
    n_rubisco: int,
    n_patches_per_rubisco: int,
    gamma_rubisco: float,
    gamma_chain: float,
    dtype: Any,
) -> jnp.ndarray:
    """Computes hydrodynamic inverse-friction mobility values for all particles.

    Inputs:
        n_particles: Total number of particles in the system.
        n_rubisco: Number of Rubisco centers.
        n_patches_per_rubisco: Number of surface patches per Rubisco.
        gamma_rubisco: Friction coefficient for Rubisco core and patch particles.
        gamma_chain: Friction coefficient for EPYC1 chain particles.
        dtype: Target floating-point data type.
    Outputs:
        Mobility array shaped (n_particles, 1).
    """
    n_slow = n_rubisco * (1 + n_patches_per_rubisco)
    values = np.full(n_particles, 1.0 / gamma_chain, dtype=np.float32)
    values[:n_slow] = 1.0 / gamma_rubisco
    return jnp.asarray(values[:, None], dtype=dtype)


def build_bond_data(
    n_rubisco: int,
    n_patches_per_rubisco: int,
    n_chains: int,
    n_beads_per_chain: int,
    chain_k: float,
    chain_r0: float,
    patch_tether_k: float,
    patch_radius: float,
    dtype: Any,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Constructs bonded particle pairs, spring constants, rest lengths, and bond type IDs.

    Inputs:
        n_rubisco: Number of Rubisco centers.
        n_patches_per_rubisco: Number of surface patches per Rubisco.
        n_chains: Number of EPYC1 chains.
        n_beads_per_chain: Number of beads per EPYC1 chain.
        chain_k: Spring constant for EPYC1 backbone harmonic bonds.
        chain_r0: Rest length for EPYC1 backbone harmonic bonds.
        patch_tether_k: Spring constant for Rubisco core-patch harmonic tethers.
        patch_radius: Rest length for Rubisco core-patch harmonic tethers.
        dtype: Target floating-point data type.
    Outputs:
        Tuple of (bond_pairs, spring_constants, rest_lengths, bond_type_ids).
    """
    chain_start = n_rubisco + n_rubisco * n_patches_per_rubisco
    chain = [
        [chain_start + chain_id * n_beads_per_chain + bead,
         chain_start + chain_id * n_beads_per_chain + bead + 1]
        for chain_id in range(n_chains)
        for bead in range(n_beads_per_chain - 1)
    ]
    patch_tethers = [
        [rubisco_id, n_rubisco + rubisco_id * n_patches_per_rubisco + patch_id]
        for rubisco_id in range(n_rubisco)
        for patch_id in range(n_patches_per_rubisco)
    ]
    pairs = jnp.asarray(chain + patch_tethers, dtype=jnp.int32)
    spring_constants = jnp.asarray(
        [chain_k] * len(chain) + [patch_tether_k] * len(patch_tethers), dtype=dtype
    )
    rest_lengths = jnp.asarray(
        [chain_r0] * len(chain) + [patch_radius] * len(patch_tethers), dtype=dtype
    )
    bond_type_ids = jnp.asarray(
        [0] * len(chain) + [1] * len(patch_tethers), dtype=jnp.int32
    )
    return pairs, spring_constants, rest_lengths, bond_type_ids


def build_bond_exclusion_mask(n_particles: int, bond_pairs: jnp.ndarray) -> jnp.ndarray:
    """Constructs a boolean exclusion matrix excluding self-pairs and directly bonded pairs from nonbonded forces.

    Inputs:
        n_particles: Total number of particles in the system.
        bond_pairs: Array of bonded particle index pairs shaped (n_bonds, 2).
    Outputs:
        Boolean exclusion mask matrix shaped (n_particles, n_particles).
    """
    mask = np.ones((n_particles, n_particles), dtype=bool)
    np.fill_diagonal(mask, False)
    pairs = np.asarray(bond_pairs)
    mask[pairs[:, 0], pairs[:, 1]] = False
    mask[pairs[:, 1], pairs[:, 0]] = False
    return jnp.asarray(mask)


def build_steric_parameter_tables(
    steric_pair_params: Mapping[tuple[str, str], tuple[float, float, float]],
    indexByTypeName: Mapping[str, int],
    dtype: Any,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Constructs pairwise parameter matrices for energy depths, diameters, and interaction cutoffs.

    Inputs:
        steric_pair_params: Mapping of type pairs to (epsilon, sigma, cutoff) parameter tuples.
        indexByTypeName: Mapping of particle type names to integer IDs.
        dtype: Target floating-point data type.
    Outputs:
        Tuple of (epsilon_table, sigma_table, cutoff_table) matrices shaped (n_types, n_types).
    """
    n_types = len(indexByTypeName)
    epsilon = np.zeros((n_types, n_types), dtype=np.float32)
    sigma = np.ones((n_types, n_types), dtype=np.float32)
    cutoff = np.zeros((n_types, n_types), dtype=np.float32)
    for (first, second), values in steric_pair_params.items():
        first_id, second_id = indexByTypeName[first], indexByTypeName[second]
        for left, right in ((first_id, second_id), (second_id, first_id)):
            epsilon[left, right], sigma[left, right], cutoff[left, right] = values
    return tuple(jnp.asarray(table, dtype=dtype) for table in (epsilon, sigma, cutoff))
