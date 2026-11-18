"""Small, named helpers for JAX MD cell-list neighbor lists."""

from __future__ import annotations

import jax.numpy as jnp


def shift_to_neighbor_list_coordinates(positions: jnp.ndarray, box: float) -> jnp.ndarray:
    """Shift centered coordinates into the ``[0, box)`` cell-list convention."""
    return positions + 0.5 * box


def allocate_neighbors(neighbor_fn, positions: jnp.ndarray, box: float, extra_capacity: int):
    """Allocate dense neighbor-list storage for one centered configuration."""
    shifted = shift_to_neighbor_list_coordinates(positions, box)
    return neighbor_fn.allocate(shifted, extra_capacity=extra_capacity)


def update_neighbors(neighbors, positions: jnp.ndarray, box: float):
    """Update an existing neighbor list from centered coordinates."""
    shifted = shift_to_neighbor_list_coordinates(positions, box)
    return neighbors.update(shifted)
