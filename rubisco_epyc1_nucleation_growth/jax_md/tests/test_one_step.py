"""Compare one deterministic Brownian-dynamics step against HOOMD."""

from __future__ import annotations

import numpy as np
import pytest

from condensate_ml.helper import (
    BOX,
    EXPECTED_PARTICLES,
    build_hoomd_initial_state,
    configure_hoomd_forces,
)

ONE_STEP_SEEDS = tuple(range(101, 121))
EPSILON_VALUES = (-0.1, -0.2, -0.5, -0.8, -1.2, -2.0)
DT = 0.0005


@pytest.mark.parametrize("seed", ONE_STEP_SEEDS)
@pytest.mark.parametrize("epsilon", EPSILON_VALUES)
def test_one_deterministic_step_matches_hoomd(seed: int, epsilon: float) -> None:
    """Compare corresponding positions for every seed/epsilon combination."""
    pytest.importorskip("jax")
    pytest.importorskip("jax_md")
    from condensate_ml import RubiscoEpyc1Simulation

    hoomd_simulation, initial_positions = build_hoomd_initial_state(BOX, seed)
    jax_simulation = RubiscoEpyc1Simulation(
        n_steps=1,
        dt=DT,
        box=BOX,
        write_every=1,
        seed=seed,
    )

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

    configure_hoomd_forces(
        hoomd_simulation,
        epsilon,
        DT,
        {"steric", "sticker_patch", "bonded"},
    )
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
        atol=1e-4,
        rtol=1e-5,
    )
