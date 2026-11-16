"""Compare shared HOOMD and JAX MD initial snapshots."""

from __future__ import annotations

import numpy as np
import pytest

from condensate_ml.helper import (
    BOX,
    EXPECTED_PARTICLES,
    build_hoomd_initial_positions,
    build_jax_initial_positions,
)

INITIAL_SEEDS = tuple(range(101, 126))
POSITION_ATOL = 1e-6


@pytest.mark.parametrize("seed", INITIAL_SEEDS)
def test_hoomd_and_jax_md_initial_positions_match(seed: int) -> None:
    """Every JAX particle must match HOOMD for 25 independent seeds."""
    hoomd_positions = build_hoomd_initial_positions(BOX, seed)
    jax_positions = build_jax_initial_positions(BOX, seed)
    assert hoomd_positions.shape == (EXPECTED_PARTICLES, 3)
    assert jax_positions.shape == (EXPECTED_PARTICLES, 3)

    difference = jax_positions - hoomd_positions
    print(f"\nseed={seed} max error={np.max(np.abs(difference)):.6e}")
    print(f"seed={seed} RMS error={np.sqrt(np.mean(difference**2)):.6e}")
    np.testing.assert_allclose(jax_positions, hoomd_positions, atol=POSITION_ATOL, rtol=0.0)
