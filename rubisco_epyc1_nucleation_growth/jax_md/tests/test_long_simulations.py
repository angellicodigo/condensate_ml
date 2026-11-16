"""Long JAX MD rollout smoke tests across simulation parameters.

These tests intentionally use 100,000 Brownian dynamics steps for every case.
They validate that long stochastic rollouts remain finite, stay in the box,
return the expected sampled-frame shape, and do not report a lost-neighbor
overflow. They do not compare stochastic positions to HOOMD because the two
engines use different random-number generators.
"""

from __future__ import annotations

import numpy as np
import pytest


N_STEPS = 100_000
WRITE_EVERY = 500
CASES = (
    pytest.param(40.0, -0.5, 101, id="box40_eps-0.5_seed101_control"),
    pytest.param(36.0, -0.8, 102, id="box36_eps-0.8_seed102_early"),
    pytest.param(33.0, -1.2, 103, id="box33_eps-1.2_seed103_nucleation"),
    pytest.param(31.0, -1.8, 104, id="box31_eps-1.8_seed104_weak_coarsening"),
    pytest.param(29.0, -2.5, 105, id="box29_eps-2.5_seed105_coarsening"),
)


@pytest.mark.parametrize("box,epsilon,seed", CASES)
def test_100000_step_rollout_is_finite_and_in_box(
    box: float,
    epsilon: float,
    seed: int,
) -> None:
    """Run and validate one full 100,000-step JAX MD simulation."""
    pytest.importorskip("jax")
    pytest.importorskip("jax_md")
    from condensate_ml import RubiscoEpyc1Simulation

    simulation = RubiscoEpyc1Simulation(
        n_steps=N_STEPS,
        box=box,
        write_every=WRITE_EVERY,
        seed=seed,
        checkpoint_blocks=True,
    )
    trajectory = simulation({"epsilon": np.asarray(epsilon, dtype=np.float32)})
    trajectory.block_until_ready()
    trajectory = np.asarray(trajectory)

    expected_frames = N_STEPS // WRITE_EVERY + 1
    assert trajectory.shape == (expected_frames, 792, 3)
    assert np.isfinite(trajectory).all()
    assert np.all(trajectory >= -box / 2.0)
    assert np.all(trajectory < box / 2.0)

    overflow = getattr(simulation, "last_neighbor_overflow", False)
    assert not overflow, "JAX MD neighbor list overflowed during the rollout"

    print(
        f"box={box} epsilon={epsilon} seed={seed} "
        f"frames={trajectory.shape[0]} max_abs={np.max(np.abs(trajectory)):.6f}"
    )
