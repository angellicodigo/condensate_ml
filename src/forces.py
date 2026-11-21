"""Differentiable pair and bond forces for Brownian dynamics."""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp


def map_pair_displacements(displacement: Callable, left: jax.Array, right: jax.Array) -> jax.Array:
    """Applies periodic minimum-image displacement across matching pairs of coordinates.

    Inputs:
        displacement: Periodic displacement function for two 3D positions.
        left: Starting positions array.
        right: Ending positions array with matching leading dimensions.
    Outputs:
        Displacement vectors array corresponding to each pair.
    """
    return jax.vmap(displacement)(left, right)


def map_neighbor_displacements(
    displacement: Callable,
    positions: jax.Array,
    neighbor_positions: jax.Array,
) -> jax.Array:
    """Applies periodic displacement between all particles and their dense neighbor configurations.

    Inputs:
        displacement: Periodic displacement function.
        positions: Particle coordinates shaped (n_particles, 3).
        neighbor_positions: Dense neighbor coordinates shaped (n_particles, n_neighbors, 3).
    Outputs:
        Displacement vectors array shaped (n_particles, n_neighbors, 3).
    """
    pairwise = jax.vmap(displacement, in_axes=(None, 0))
    return jax.vmap(pairwise, in_axes=(0, 0))(positions, neighbor_positions)


def calculate_bond_forces(
    positions: jax.Array,
    pairs: jax.Array,
    spring_constants: jax.Array,
    rest_lengths: jax.Array,
    displacement: Callable,
) -> jax.Array:
    """Computes harmonic Hookean spring restoring forces for a list of bonded particle pairs.

    Inputs:
        positions: Particle coordinates shaped (n_particles, 3).
        pairs: Bonded particle index pairs shaped (n_bonds, 2).
        spring_constants: Spring constant array shaped (n_bonds,).
        rest_lengths: Preferred equilibrium length array shaped (n_bonds,).
        displacement: Periodic displacement function.
    Outputs:
        Net bonded force array shaped (n_particles, 3).
    """
    i, j = pairs[:, 0], pairs[:, 1]
    delta = map_pair_displacements(displacement, positions[i], positions[j])
    distance_sq = jnp.sum(delta * delta, axis=-1, keepdims=True)
    distance = jnp.sqrt(jnp.maximum(distance_sq, 1e-12))
    force = (-spring_constants[:, None] * (distance - rest_lengths[:, None])) * delta / distance
    result = jnp.zeros_like(positions)
    result = result.at[i].add(force)
    return result.at[j].add(-force)


def calculate_neighbor_pair_forces(
    positions: jax.Array,
    neighbor_indices: jax.Array,
    type_ids: jax.Array,
    epsilon_table: jax.Array,
    sigma_table: jax.Array,
    cutoff_table: jax.Array,
    nonbonded_mask: jax.Array,
    displacement: Callable,
) -> jax.Array:
    """Evaluates nonbonded Gaussian pair forces over a dense cell-list neighbor list.

    Inputs:
        positions: Particle coordinates shaped (n_particles, 3).
        neighbor_indices: Dense neighbor index table shaped (n_particles, n_neighbors).
        type_ids: Integer particle type IDs shaped (n_particles,).
        epsilon_table: Interaction energy scale matrix shaped (n_types, n_types).
        sigma_table: Interaction characteristic distance matrix shaped (n_types, n_types).
        cutoff_table: Interaction cutoff distance matrix shaped (n_types, n_types).
        nonbonded_mask: Boolean exclusion mask matrix shaped (n_particles, n_particles).
        displacement: Periodic displacement function.
    Outputs:
        Summed nonbonded force array shaped (n_particles, 3).
    """
    n_particles = positions.shape[0]
    row = jnp.arange(n_particles)[:, None]
    valid = (neighbor_indices >= 0) & (neighbor_indices < n_particles)
    safe_indices = jnp.clip(neighbor_indices, 0, n_particles - 1)
    i_types = type_ids[:, None]
    j_types = type_ids[safe_indices]
    delta = map_neighbor_displacements(displacement, positions, positions[safe_indices])
    dist_sq = jnp.sum(delta * delta, axis=-1)

    epsilon = epsilon_table[i_types, j_types]
    sigma_sq = jnp.maximum(sigma_table[i_types, j_types] ** 2, 1e-8)
    cutoff_sq = cutoff_table[i_types, j_types] ** 2
    active = valid & nonbonded_mask[row, safe_indices] & (dist_sq < cutoff_sq)
    prefactor = epsilon / sigma_sq * jnp.exp(-0.5 * dist_sq / sigma_sq)
    return jnp.sum(jnp.where(active[..., None], prefactor[..., None] * delta, 0.0), axis=1)


def calculate_force_components(
    positions: jax.Array,
    neighbor_indices: jax.Array,
    type_ids: jax.Array,
    epsilon_table: jax.Array,
    sigma_table: jax.Array,
    cutoff_table: jax.Array,
    nonbonded_mask: jax.Array,
    bond_pairs: jax.Array,
    bond_k: jax.Array,
    bond_r0: jax.Array,
    attraction_epsilon: jax.Array,
    attraction_sigma: float,
    displacement: Callable,
    attraction_cutoff: float | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Computes steric, sticker-patch attraction, and bonded force contributions separately.

    Inputs:
        positions: Particle coordinates shaped (n_particles, 3).
        neighbor_indices: Dense neighbor index table shaped (n_particles, n_neighbors).
        type_ids: Integer particle type IDs shaped (n_particles,).
        epsilon_table: Steric energy scale matrix shaped (n_types, n_types).
        sigma_table: Steric characteristic distance matrix shaped (n_types, n_types).
        cutoff_table: Interaction cutoff distance matrix shaped (n_types, n_types).
        nonbonded_mask: Boolean exclusion mask matrix shaped (n_particles, n_particles).
        bond_pairs: Bonded particle index pairs shaped (n_bonds, 2).
        bond_k: Bond spring constants array shaped (n_bonds,).
        bond_r0: Bond rest lengths array shaped (n_bonds,).
        attraction_epsilon: Negative attraction energy depth scalar or array.
        attraction_sigma: Gaussian attraction potential width.
        displacement: Periodic displacement function.
        attraction_cutoff: Optional explicit cutoff distance override.
    Outputs:
        Tuple of (steric, attraction, bonded) force arrays each shaped (n_particles, 3).
    """
    steric = calculate_neighbor_pair_forces(
        positions,
        neighbor_indices,
        type_ids,
        epsilon_table,
        sigma_table,
        cutoff_table,
        nonbonded_mask,
        displacement,
    )
    attraction_table = jnp.zeros_like(epsilon_table).at[3, 1].set(attraction_epsilon)
    attraction_table = attraction_table.at[1, 3].set(attraction_epsilon)
    attraction_sigma_table = jnp.zeros_like(sigma_table).at[3, 1].set(attraction_sigma)
    attraction_sigma_table = attraction_sigma_table.at[1, 3].set(attraction_sigma)
    if attraction_cutoff is not None:
        attraction_cutoff_table = jnp.zeros_like(cutoff_table).at[3, 1].set(attraction_cutoff)
        attraction_cutoff_table = attraction_cutoff_table.at[1, 3].set(attraction_cutoff)
    else:
        attraction_cutoff_table = cutoff_table
    attraction = calculate_neighbor_pair_forces(
        positions,
        neighbor_indices,
        type_ids,
        attraction_table,
        attraction_sigma_table,
        attraction_cutoff_table,
        nonbonded_mask=jnp.ones_like(nonbonded_mask),
        displacement=displacement,
    )
    bonded = calculate_bond_forces(
        positions, bond_pairs, bond_k, bond_r0, displacement
    )
    return steric, attraction, bonded


def calculate_total_force(
    positions: jax.Array,
    neighbor_indices: jax.Array,
    type_ids: jax.Array,
    epsilon_table: jax.Array,
    sigma_table: jax.Array,
    cutoff_table: jax.Array,
    nonbonded_mask: jax.Array,
    bond_pairs: jax.Array,
    bond_k: jax.Array,
    bond_r0: jax.Array,
    attraction_epsilon: jax.Array,
    attraction_sigma: float,
    displacement: Callable,
    attraction_cutoff: float | None = None,
) -> jax.Array:
    """Computes the total net force acting on each particle by summing all interaction components.

    Inputs:
        positions: Particle coordinates shaped (n_particles, 3).
        neighbor_indices: Dense neighbor index table shaped (n_particles, n_neighbors).
        type_ids: Integer particle type IDs shaped (n_particles,).
        epsilon_table: Steric energy scale matrix shaped (n_types, n_types).
        sigma_table: Steric characteristic distance matrix shaped (n_types, n_types).
        cutoff_table: Interaction cutoff distance matrix shaped (n_types, n_types).
        nonbonded_mask: Boolean exclusion mask matrix shaped (n_particles, n_particles).
        bond_pairs: Bonded particle index pairs shaped (n_bonds, 2).
        bond_k: Bond spring constants array shaped (n_bonds,).
        bond_r0: Bond rest lengths array shaped (n_bonds,).
        attraction_epsilon: Negative attraction energy depth scalar or array.
        attraction_sigma: Gaussian attraction potential width.
        displacement: Periodic displacement function.
        attraction_cutoff: Optional explicit cutoff distance override.
    Outputs:
        Total force array shaped (n_particles, 3).
    """
    return sum(
        calculate_force_components(
            positions, neighbor_indices, type_ids, epsilon_table, sigma_table,
            cutoff_table, nonbonded_mask, bond_pairs, bond_k, bond_r0,
            attraction_epsilon, attraction_sigma, displacement, attraction_cutoff,
        )
    )
