"""JAX-independent construction of the Rubisco-EPYC1 initial coordinates."""

from __future__ import annotations

import numpy as np


def make_grid_positions(n: int, box: float, margin: float) -> np.ndarray:
    n_side = int(np.ceil(n ** (1.0 / 3.0)))
    values = np.linspace(-box / 2.0 + margin, box / 2.0 - margin, n_side)
    points = np.array(np.meshgrid(values, values, values)).T.reshape(-1, 3)
    return points[:n].copy()


def generate_initial_coordinates(
    box: float,
    patch_dirs: np.ndarray,
    seed: int,
    n_rubisco: int,
    n_patches_per_rubisco: int,
    n_chains: int,
    n_beads_per_chain: int,
    patch_radius: float,
    chain_r0: float,
) -> np.ndarray:
    """Return the same deterministic layout as the PyTorch implementation."""
    rng = np.random.default_rng(seed)
    n_patch = n_rubisco * n_patches_per_rubisco
    n_chain = n_chains * n_beads_per_chain
    positions = np.zeros((n_rubisco + n_patch + n_chain, 3), dtype=np.float32)

    centers = make_grid_positions(n_rubisco, box, margin=7.0)
    centers += rng.normal(scale=0.20, size=centers.shape)
    centers = ((centers + box / 2.0) % box) - box / 2.0
    positions[:n_rubisco] = centers

    cursor = n_rubisco
    for center in centers:
        for direction in patch_dirs:
            positions[cursor] = center + patch_radius * direction
            cursor += 1

    chain_centers = make_grid_positions(n_chains, box, margin=5.0)
    chain_centers += rng.normal(scale=0.25, size=chain_centers.shape)
    chain_centers = ((chain_centers + box / 2.0) % box) - box / 2.0
    for center in chain_centers:
        direction = rng.normal(size=3)
        direction /= np.linalg.norm(direction)
        for bead in range(n_beads_per_chain):
            value = center + (bead - (n_beads_per_chain - 1) / 2.0) * chain_r0 * direction
            positions[cursor] = ((value + box / 2.0) % box) - box / 2.0
            cursor += 1
    return positions
