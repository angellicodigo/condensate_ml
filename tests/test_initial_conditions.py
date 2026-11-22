"""Compare the HOOMD and JAX MD initial layouts particle by particle (needs a HOOMD GPU).

Positions are float32 in both codes. The half-spacing of float32 numbers near |x|=16-32 is 9.5e-7,
which is the largest difference expected, so POSITION_ATOL=2e-6 leaves a small safety margin.
"""

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
POSITION_ATOL = 2e-6


@pytest.mark.parametrize("seed", INITIAL_SEEDS)
def test_hoomd_and_jax_md_initial_positions_match(seed: int) -> None:
    """Every JAX particle must match HOOMD (25 independent seeds)."""
    hoomd_positions = build_hoomd_initial_positions(BOX, seed)
    jax_positions = build_jax_initial_positions(BOX, seed)
    assert hoomd_positions.shape == (EXPECTED_PARTICLES, 3)
    assert jax_positions.shape == (EXPECTED_PARTICLES, 3)

    difference = jax_positions - hoomd_positions
    print(f"\nseed={seed} max error={np.max(np.abs(difference)):.6e}")
    print(f"seed={seed} RMS error={np.sqrt(np.mean(difference**2)):.6e}")
    np.testing.assert_allclose(jax_positions, hoomd_positions, atol=POSITION_ATOL, rtol=0.0)
