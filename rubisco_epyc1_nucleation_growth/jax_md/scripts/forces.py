"""Differentiable pair and bond forces for the JAX MD backend."""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp


def _mapped_displacement(displacement: Callable, left: jax.Array, right: jax.Array) -> jax.Array:
    """Apply a vector-to-vector displacement function over matching leading axes.

    Input:
        displacement: A JAX MD displacement function for two 3D positions.
        left, right: Arrays with matching leading dimensions, usually one
            position per bond pair.
    Output:
        A displacement array with one minimum-image vector for each matching
        pair in ``left`` and ``right``.
    """
    return jax.vmap(displacement)(left, right)


def _neighbor_displacement(
    displacement: Callable,
    positions: jax.Array,
    neighbor_positions: jax.Array,
) -> jax.Array:
    """Apply displacement to every particle/neighbor combination.

    Input:
        displacement: A periodic displacement function.
        positions: Particle positions with shape ``(n_particles, 3)``.
        neighbor_positions: Dense neighbor positions with shape
            ``(n_particles, n_neighbors, 3)``.
    Output:
        Minimum-image displacement vectors with shape
        ``(n_particles, n_neighbors, 3)``.
    """
    pairwise = jax.vmap(displacement, in_axes=(None, 0))
    return jax.vmap(pairwise, in_axes=(0, 0))(positions, neighbor_positions)


def _bond_forces(
    positions: jax.Array,
    pairs: jax.Array,
    spring_constants: jax.Array,
    rest_lengths: jax.Array,
    displacement: Callable,
) -> jax.Array:
    """Compute Hookean spring forces for a list of bonded pairs.

    Input:
        positions: Particle coordinates, shaped ``(n_particles, 3)``.
        pairs: Integer pair indices shaped ``(n_bonds, 2)``.
        spring_constants: Spring constant for each bond.
        rest_lengths: Preferred length for each bond.
        displacement: Periodic displacement function.
    Output:
        Force array shaped ``(n_particles, 3)``. Each bond contributes equal
        and opposite forces to its two endpoint particles.
    """
    i, j = pairs[:, 0], pairs[:, 1]
    delta = _mapped_displacement(displacement, positions[i], positions[j])
    distance_sq = jnp.sum(delta * delta, axis=-1, keepdims=True)
    distance = jnp.sqrt(jnp.maximum(distance_sq, 1e-12))
    force = (-spring_constants[:, None] * (distance - rest_lengths[:, None])) * delta / distance
    result = jnp.zeros_like(positions)
    result = result.at[i].add(force)
    return result.at[j].add(-force)


def _neighbor_pair_force(
    positions: jax.Array,
    neighbor_indices: jax.Array,
    type_ids: jax.Array,
    epsilon_table: jax.Array,
    sigma_table: jax.Array,
    cutoff_table: jax.Array,
    nonbonded_mask: jax.Array,
    displacement: Callable,
) -> jax.Array:
    """Evaluate one symmetric pair potential on a dense neighbor list.

    Input:
        positions: Particle coordinates shaped ``(n_particles, 3)``.
        neighbor_indices: Dense neighbor-list indices; invalid entries are
            marked with negative values.
        type_ids: Integer particle type for each particle.
        epsilon_table, sigma_table, cutoff_table: Pair-parameter tables
            indexed by the two particle types.
        nonbonded_mask: Boolean matrix excluding self and bonded pairs.
        displacement: Periodic displacement function.
    Output:
        Force array shaped ``(n_particles, 3)`` containing the summed force
        from all active neighbors.
    """
    n_particles = positions.shape[0]
    row = jnp.arange(n_particles)[:, None]
    valid = (neighbor_indices >= 0) & (neighbor_indices < n_particles)
    safe_indices = jnp.clip(neighbor_indices, 0, n_particles - 1)
    i_types = type_ids[:, None]
    j_types = type_ids[safe_indices]
    delta = _neighbor_displacement(displacement, positions, positions[safe_indices])
    dist_sq = jnp.sum(delta * delta, axis=-1)

    epsilon = epsilon_table[i_types, j_types]
    sigma_sq = jnp.maximum(sigma_table[i_types, j_types] ** 2, 1e-8)
    cutoff_sq = cutoff_table[i_types, j_types] ** 2
    active = valid & nonbonded_mask[row, safe_indices] & (dist_sq < cutoff_sq)
    prefactor = epsilon / sigma_sq * jnp.exp(-0.5 * dist_sq / sigma_sq)
    return jnp.sum(jnp.where(active[..., None], prefactor[..., None] * delta, 0.0), axis=1)


def force_components(
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
    attraction_cutoff: float,
    displacement: Callable,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Return the three physical force components separately.

    Input:
        positions, neighbor_indices, type_ids: Current configuration and
            dense neighbor-list metadata.
        epsilon_table, sigma_table, cutoff_table: Steric pair parameters.
        nonbonded_mask: Pair-exclusion mask for the steric interaction.
        bond_pairs, bond_k, bond_r0: Bond topology and Hookean parameters.
        attraction_epsilon, attraction_sigma, attraction_cutoff: Parameters
            for the sticker-patch Gaussian interaction.
        displacement: Periodic displacement function.
    Output:
        Tuple ``(steric, attraction, bonded)``. Each entry is an
        ``(n_particles, 3)`` force array.
    """
    steric = _neighbor_pair_force(
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
    attraction_cutoff_table = jnp.zeros_like(cutoff_table).at[3, 1].set(attraction_cutoff)
    attraction_cutoff_table = attraction_cutoff_table.at[1, 3].set(attraction_cutoff)
    attraction_table = attraction_table.at[1, 3].set(attraction_epsilon)
    attraction_sigma_table = jnp.zeros_like(sigma_table).at[3, 1].set(attraction_sigma)
    attraction_sigma_table = attraction_sigma_table.at[1, 3].set(attraction_sigma)
    attraction = _neighbor_pair_force(
        positions,
        neighbor_indices,
        type_ids,
        attraction_table,
        attraction_sigma_table,
        attraction_cutoff_table,
        nonbonded_mask=jnp.ones_like(nonbonded_mask),
        displacement=displacement,
    )
    bonded = _bond_forces(
        positions, bond_pairs, bond_k, bond_r0, displacement
    )
    return steric, attraction, bonded


def total_forces(
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
    attraction_cutoff: float,
    displacement: Callable,
) -> jax.Array:
    """Return the total force as the sum of all physical force components.

    Input:
        The same configuration, interaction tables, bond data, attraction
        parameters, and displacement function accepted by ``force_components``.
    Output:
        One ``(n_particles, 3)`` array containing steric, attraction, and
        bonded forces added together.
    """
    return sum(
        force_components(
            positions, neighbor_indices, type_ids, epsilon_table, sigma_table,
            cutoff_table, nonbonded_mask, bond_pairs, bond_k, bond_r0,
            attraction_epsilon, attraction_sigma, attraction_cutoff, displacement,
        )
    )
