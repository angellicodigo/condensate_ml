"""Initial coordinate generation for the Rubisco-EPYC1 model."""

from __future__ import annotations

import numpy as np


def make_grid_positions(n: int, box: float, margin: float) -> np.ndarray:
    """Generates the first n coordinates of a cubic lattice within a bordered box volume.

    Inputs:
        n: Number of 3D grid points to generate.
        box: Periodic box length.
        margin: Boundary margin distance between points and box faces.
    Outputs:
        Array of 3D grid coordinates shaped (n, 3).
    """
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
    """Generates deterministic initial 3D coordinates for Rubisco centers, surface patches, and EPYC1 beads.

    Inputs:
        box: Periodic box length.
        patch_dirs: Unit vectors defining patch orientations on each Rubisco core.
        seed: Random seed for coordinate perturbations and chain orientations.
        n_rubisco: Number of Rubisco centers.
        n_patches_per_rubisco: Number of surface patches per Rubisco.
        n_chains: Number of EPYC1 chains.
        n_beads_per_chain: Number of beads per EPYC1 chain.
        patch_radius: Radial distance from Rubisco center to its patches.
        chain_r0: Equilibrium spacing between consecutive chain beads.
    Outputs:
        Initial particle coordinates array shaped (n_particles, 3).
    """
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
