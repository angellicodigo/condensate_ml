"""Small numerical-comparison helpers for validating against HOOMD."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np


def force_error(reference: Any, candidate: Any, floor: float = 1e-12) -> dict[str, float]:
    """Return absolute and relative L2 force errors for two ``(N, 3)`` arrays."""
    reference = np.asarray(reference)
    candidate = np.asarray(candidate)
    difference = candidate - reference
    reference_norm = float(np.linalg.norm(reference))
    return {
        "absolute_l2": float(np.linalg.norm(difference)),
        "relative_l2": float(np.linalg.norm(difference) / max(reference_norm, floor)),
        "max_absolute": float(np.max(np.abs(difference))),
    }


def component_force_errors(
    reference: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> dict[str, dict[str, float]]:
    """Compare matching ``steric``, ``sticker_patch``, ``bonded``, and ``total`` arrays."""
    names = sorted(set(reference).intersection(candidate))
    return {name: force_error(reference[name], candidate[name]) for name in names}
