"""Compare one zero-noise Brownian-dynamics step against HOOMD (needs a HOOMD GPU).

At dt=0.0005 one step moves particles by <~6e-4, only a few hundred times the float32 rounding
level, so this test mainly catches gross errors; ``test_forces.py`` is the sensitive check.
"""

from __future__ import annotations

import numpy as np
import pytest

from condensate_ml.helper import (
    BOX,
    COMPONENTS,
    DT,
    EXPECTED_PARTICLES,
    build_hoomd_initial_state,
    configure_hoomd_forces,
    make_simulation,
)

ONE_STEP_SEEDS = tuple(range(101, 121))
ONE_STEP_EPSILONS = (-0.1, -0.2, -0.5, -0.8, -1.2, -2.0)
POSITION_ATOL = 1e-4
POSITION_RTOL = 1e-5


@pytest.mark.parametrize("seed", ONE_STEP_SEEDS)
@pytest.mark.parametrize("epsilon", ONE_STEP_EPSILONS)
def test_one_deterministic_step_matches_hoomd(seed: int, epsilon: float) -> None:
    """Compare corresponding positions for every seed/epsilon combination."""
    pytest.importorskip("jax_md")
    hoomd_simulation, initial_positions = build_hoomd_initial_state(BOX, seed)
    jax_simulation = make_simulation(BOX, seed, dt=DT)

    jax_positions = np.asarray(
        jax_simulation.brownian_step(
            initial_positions,
            epsilon,
            noise=np.zeros_like(initial_positions),
        ),
        dtype=np.float64,
    )
    assert jax_positions.shape == (EXPECTED_PARTICLES, 3)
    assert np.isfinite(jax_positions).all()

    configure_hoomd_forces(hoomd_simulation, epsilon, DT, set(COMPONENTS))
    hoomd_simulation.run(1)
    hoomd_positions = np.asarray(
        hoomd_simulation.state.get_snapshot().particles.position,
        dtype=np.float64,
    )

    difference = jax_positions - hoomd_positions
    print(
        f"\nseed={seed} epsilon={epsilon} "
        f"max error={np.max(np.abs(difference)):.6e} "
        f"RMS error={np.sqrt(np.mean(difference**2)):.6e}"
    )
    np.testing.assert_allclose(
        jax_positions,
        hoomd_positions,
        atol=POSITION_ATOL,
        rtol=POSITION_RTOL,
    )
