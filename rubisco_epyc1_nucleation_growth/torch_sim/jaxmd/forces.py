"""Differentiable pair and bond forces for the JAX MD backend."""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp


def _bond_forces(
    positions: jax.Array,
    pairs: jax.Array,
    spring_constants: jax.Array,
    rest_lengths: jax.Array,
    displacement: Callable,
) -> jax.Array:
    i, j = pairs[:, 0], pairs[:, 1]
    delta = displacement(positions[i], positions[j])
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
    """Evaluate one symmetric pair potential on a dense cell-list neighbor list."""
    n_particles = positions.shape[0]
    row = jnp.arange(n_particles)[:, None]
    valid = (neighbor_indices >= 0) & (neighbor_indices < n_particles)
    safe_indices = jnp.clip(neighbor_indices, 0, n_particles - 1)
    i_types = type_ids[:, None]
    j_types = type_ids[safe_indices]
    delta = displacement(positions[:, None, :], positions[safe_indices])
    dist_sq = jnp.sum(delta * delta, axis=-1)

    epsilon = epsilon_table[i_types, j_types]
    sigma_sq = jnp.maximum(sigma_table[i_types, j_types] ** 2, 1e-8)
    cutoff_sq = cutoff_table[i_types, j_types] ** 2
    active = valid & nonbonded_mask[row, safe_indices] & (dist_sq < cutoff_sq)
    prefactor = epsilon / sigma_sq * jnp.exp(-0.5 * dist_sq / sigma_sq)
    return jnp.sum(jnp.where(active[..., None], prefactor[..., None] * delta, 0.0), axis=1)


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
    return steric + attraction + _bond_forces(
        positions, bond_pairs, bond_k, bond_r0, displacement
    )
