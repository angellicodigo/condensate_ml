"""Geometry and coordinate helpers for the Rubisco-EPYC1 simulation.

Pure functions that handle periodic box wrapping, minimum-image displacements
and a gradient-safe vector norm. None of them mutate their inputs.
"""

from __future__ import annotations

import torch


def wrap_centered_coordinates(positions: torch.Tensor, box: float) -> torch.Tensor:
    """Wrap particle coordinates into the origin-centered periodic box [-L/2, +L/2)."""
    return ((positions + box * 0.5) % box) - box * 0.5


def apply_minimum_image_displacement(delta: torch.Tensor, box: float) -> torch.Tensor:
    """Apply the minimum image convention to pairwise displacement vectors."""
    return delta - box * torch.round(delta / box)


def safe_vector_norm(
    vec: torch.Tensor, dim: int = -1, keepdim: bool = False, eps: float = 1e-12
) -> torch.Tensor:
    """Euclidean norm whose gradient stays finite at zero length.

    The squared norm is clamped before the square root, so the backward pass
    never divides by zero. Matches torch.linalg.vector_norm for lengths > 1e-6.
    """
    sq = torch.sum(vec * vec, dim=dim, keepdim=keepdim)
    return torch.sqrt(torch.clamp(sq, min=eps))