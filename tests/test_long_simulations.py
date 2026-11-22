"""Full-length (100,000-step) JAX MD rollouts for each regime: smoke tests.

They check that a long stochastic rollout stays finite and inside the box, returns the expected
number of frames, and never loses neighbor-list pairs. They do not compare positions with HOOMD
because the two engines use different random-number generators.
"""

from __future__ import annotations

import numpy as np
import pytest

from condensate_ml.helper.regimes import REGIMES, Regime

EXPECTED_PARTICLES = 792


@pytest.mark.parametrize("regime", REGIMES, ids=[regime.name for regime in REGIMES])
def test_full_rollout_is_finite_and_in_box(regime: Regime) -> None:
    pytest.importorskip("jax_md")
    from condensate_ml import RubiscoEpyc1Simulation

    simulation = RubiscoEpyc1Simulation(
        n_steps=regime.n_steps,
        box=regime.box,
        write_every=regime.write_every,
        seed=regime.seed,
        checkpoint_blocks=True,
    )
    trajectory = simulation({"epsilon": np.asarray(regime.epsilon, dtype=np.float32)})
    trajectory.block_until_ready()
    trajectory = np.asarray(trajectory)

    half_box = regime.box / 2.0
    assert trajectory.shape == (regime.n_frames, EXPECTED_PARTICLES, 3)
    assert np.isfinite(trajectory).all()
    assert np.all(trajectory >= -half_box)
    assert np.all(trajectory < half_box)
    assert not simulation.last_neighbor_overflow, "neighbor list overflowed during the rollout"
