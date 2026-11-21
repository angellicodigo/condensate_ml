"""Coordinate transformations and storage allocation for JAX-MD neighbor lists."""

from __future__ import annotations

import jax.numpy as jnp


def shift_to_neighbor_list_coordinates(positions: jnp.ndarray, box: float) -> jnp.ndarray:
    """Shifts centered coordinates into the [0, box) convention required by JAX-MD cell lists.

    Inputs:
        positions: Particle coordinates in origin-centered periodic space.
        box: Periodic box length.
    Outputs:
        Shifted particle coordinates in [0, box).
    """
    return positions + 0.5 * box


def allocate_neighbors(neighbor_fn, positions: jnp.ndarray, box: float, extra_capacity: int):
    """Allocates dense neighbor-list buffer storage for one centered configuration.

    Inputs:
        neighbor_fn: JAX-MD neighbor list allocation function.
        positions: Particle coordinates in origin-centered periodic space.
        box: Periodic box length.
        extra_capacity: Additional neighbor slots reserved per particle.
    Outputs:
        Pre-allocated JAX-MD dense neighbor list.
    """
    shifted = shift_to_neighbor_list_coordinates(positions, box)
    return neighbor_fn.allocate(shifted, extra_capacity=extra_capacity)
