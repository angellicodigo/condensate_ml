"""Geometry and coordinate helpers for the Rubisco-EPYC1 simulation.

Pure functions that handle periodic box wrapping and minimum-image displacements.
None of these functions mutate their inputs; they all return new tensors.
"""

from __future__ import annotations

import torch


def wrap_centered_coordinates(positions: torch.Tensor, box: float) -> torch.Tensor:
    """Wrap particle coordinates into the origin-centered periodic box [-L/2, +L/2).

    Any coordinate that drifts outside the box is folded back in using modular
    arithmetic. This keeps all particles inside the simulation volume.
    """
    return ((positions + box * 0.5) % box) - box * 0.5


def apply_minimum_image_displacement(delta: torch.Tensor, box: float) -> torch.Tensor:
    """Apply the minimum image convention to pairwise displacement vectors.

    For each component of ``delta``, if the absolute value exceeds half the box
    length, subtract or add the full box length so that the shortest periodic
    image is used.
    """
    return delta - box * torch.round(delta / box)
