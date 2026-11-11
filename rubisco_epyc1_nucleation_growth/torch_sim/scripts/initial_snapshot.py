"""Initial snapshot builders for the Rubisco-EPYC1 simulation.

Pure functions that generate the initial coordinate layout: grid positions and
the full (Rubisco + patches + EPYC1 chains) starting configuration.
"""

from __future__ import annotations

import numpy as np
import torch


def make_grid_positions(n: int, box: float, margin: float = 6.0) -> np.ndarray:
    """Generate initial grid positions spaced symmetrically inside a cubic box.

    Places ``n`` particles on a 3-D cubic lattice centered in the box, leaving
    a ``margin`` gap from each box face. Returns the first ``n`` lattice points.
    """
    n_side = int(np.ceil(n ** (1 / 3)))
    vals = np.linspace(-box / 2.0 + margin, box / 2.0 - margin, n_side)
    pts = np.array(np.meshgrid(vals, vals, vals)).T.reshape(-1, 3)
    return pts[:n].copy()


def generate_initial_coordinates(
    box: float,
    seed: int = 42,
    n_rubisco: int = 24,
    n_patches_per_rubisco: int = 8,
    n_chains: int = 48,
    n_beads_per_chain: int = 12,
    patch_radius: float = 2.5,
    chain_r0: float = 1.2,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Generate the full initial coordinate snapshot matching the HOOMD reference.

    Layout (792 particles total):
      1. Rubisco centers (24) — placed on a grid with small random jitter.
      2. Rubisco patch particles (192) — one per patch direction at patch_radius.
      3. EPYC1 chain beads (576) — 48 chains of 12 beads each, arranged in lines.

    Returns a tensor of shape (n_total, 3).
    """
    rng = np.random.default_rng(seed)

    n_patch = n_rubisco * n_patches_per_rubisco
    n_chain = n_chains * n_beads_per_chain
    n_total = n_rubisco + n_patch + n_chain

    pos = np.zeros((n_total, 3), dtype=np.float64)

    # ------------------------------------------------------------------
    # 1. Rubisco centers on a jittered grid
    # ------------------------------------------------------------------
    patch_dirs = np.array(
        [
            [1, 1, 1],
            [1, 1, -1],
            [1, -1, 1],
            [1, -1, -1],
            [-1, 1, 1],
            [-1, 1, -1],
            [-1, -1, 1],
            [-1, -1, -1],
        ],
        dtype=float,
    )
    patch_dirs /= np.linalg.norm(patch_dirs, axis=1, keepdims=True)

    rubisco_centers = make_grid_positions(n_rubisco, box, margin=7.0)
    rubisco_centers += rng.normal(scale=0.20, size=rubisco_centers.shape)
    rubisco_centers = ((rubisco_centers + box / 2.0) % box) - box / 2.0

    idx = 0
    for r in range(n_rubisco):
        pos[idx] = rubisco_centers[r]
        idx += 1

    # ------------------------------------------------------------------
    # 2. Rubisco patch particles along the 8 diagonal directions
    # ------------------------------------------------------------------
    for r in range(n_rubisco):
        center = rubisco_centers[r]
        for p in range(n_patches_per_rubisco):
            pos[idx] = center + patch_radius * patch_dirs[p]
            idx += 1

    # ------------------------------------------------------------------
    # 3. EPYC1 chains as short linear polymers
    # ------------------------------------------------------------------
    chain_centers = make_grid_positions(n_chains, box, margin=5.0)
    chain_centers += rng.normal(scale=0.25, size=chain_centers.shape)
    chain_centers = ((chain_centers + box / 2.0) % box) - box / 2.0

    for c in range(n_chains):
        center = chain_centers[c]
        direction = rng.normal(size=3)
        direction /= np.linalg.norm(direction)

        for b in range(n_beads_per_chain):
            bead_pos = center + (b - (n_beads_per_chain - 1) / 2.0) * chain_r0 * direction
            bead_pos = ((bead_pos + box / 2.0) % box) - box / 2.0
            pos[idx] = bead_pos
            idx += 1

    return torch.tensor(pos, device=device, dtype=dtype)
