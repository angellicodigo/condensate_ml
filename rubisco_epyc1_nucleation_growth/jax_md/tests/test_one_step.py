"""Compare one deterministic Brownian-dynamics step against HOOMD.

The reference HOOMD simulation uses ``kT=1`` and therefore draws random noise.
Because HOOMD and JAX use different random-number generators, their stochastic
positions cannot be expected to match exactly. This test instead sets HOOMD's
``kT`` to zero and supplies zero noise to JAX, so both engines perform the same
deterministic force-driven Brownian update.

Run with::

    pytest rubisco_epyc1_nucleation_growth/jax_md/tests/test_one_step.py -v -s
"""

from __future__ import annotations

import numpy as np
import pytest

from test_initial_conditions import (
    BOX,
    EXPECTED_PARTICLES,
    build_hoomd_initial_state,
)

# The Cartesian product below gives 20 seeds x 6 interaction strengths = 120
# deterministic one-step comparisons.
ONE_STEP_SEEDS = tuple(range(101, 121))
EPSILON_VALUES = (-0.1, -0.2, -0.5, -0.8, -1.2, -2.0)


def configure_deterministic_hoomd(simulation, epsilon: float, dt: float) -> None:
    """Configure the reference pair, bond, and zero-noise Brownian forces."""
    hoomd = pytest.importorskip("hoomd")
    md = hoomd.md
    particle_types = ["R", "P", "S", "T"]
    neighbor_list = md.nlist.Cell(buffer=0.4)

    steric = md.pair.Gaussian(nlist=neighbor_list)
    for first in particle_types:
        for second in particle_types:
            steric.params[(first, second)] = dict(epsilon=0.0, sigma=1.0)
            steric.r_cut[(first, second)] = 0.0
    steric.params[("R", "R")] = dict(epsilon=2.0, sigma=1.5)
    steric.r_cut[("R", "R")] = 3.0
    for pair in [("S", "S"), ("T", "T"), ("S", "T")]:
        steric.params[pair] = dict(epsilon=1.0, sigma=1.0)
        steric.r_cut[pair] = 2.5
    for pair in [("R", "S"), ("R", "T")]:
        steric.params[pair] = dict(epsilon=1.5, sigma=1.5)
        steric.r_cut[pair] = 3.0
    for pair in [("P", "P"), ("P", "S"), ("P", "R")]:
        steric.params[pair] = dict(epsilon=0.2, sigma=0.8)
        steric.r_cut[pair] = 2.0

    attraction = md.pair.Gaussian(nlist=neighbor_list)
    for first in particle_types:
        for second in particle_types:
            attraction.params[(first, second)] = dict(epsilon=0.0, sigma=1.0)
            attraction.r_cut[(first, second)] = 0.0
    attraction.params[("T", "P")] = dict(epsilon=epsilon, sigma=1.0)
    attraction.r_cut[("T", "P")] = 3.0

    harmonic = md.bond.Harmonic()
    harmonic.params["chain"] = dict(k=10.0, r0=1.2)
    harmonic.params["patch_tether"] = dict(k=50.0, r0=2.5)

    integrator = md.Integrator(dt=dt)
    brownian = md.methods.Brownian(filter=hoomd.filter.All(), kT=0.0)
    for particle_type, gamma in {"R": 5.0, "P": 5.0, "S": 1.0, "T": 1.0}.items():
        brownian.gamma[particle_type] = gamma
    integrator.methods.append(brownian)
    integrator.forces.extend([steric, attraction, harmonic])
    simulation.operations.integrator = integrator


@pytest.mark.parametrize("seed", ONE_STEP_SEEDS)
@pytest.mark.parametrize("epsilon", EPSILON_VALUES)
def test_one_deterministic_step_matches_hoomd(seed: int, epsilon: float) -> None:
    """Compare one deterministic step for every seed/epsilon combination."""
    pytest.importorskip("jax")
    pytest.importorskip("jax_md")
    from condensate_ml import RubiscoEpyc1Simulation

    dt = 0.0005
    print(f"\nComparing one step with seed={seed}, epsilon={epsilon}")
    hoomd_simulation, initial_positions = build_hoomd_initial_state(BOX, seed)
    jax_simulation = RubiscoEpyc1Simulation(
        n_steps=1,
        dt=dt,
        box=BOX,
        write_every=1,
        seed=seed,
    )

    jax_forces = np.asarray(
        jax_simulation.compute_forces(initial_positions, epsilon),
        dtype=np.float64,
    )
    assert jax_forces.shape == (EXPECTED_PARTICLES, 3)
    assert np.isfinite(jax_forces).all()

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
    assert np.all(jax_positions >= -BOX / 2.0)
    assert np.all(jax_positions < BOX / 2.0)

    configure_deterministic_hoomd(hoomd_simulation, epsilon, dt)
    hoomd_simulation.run(1)
    hoomd_positions = np.asarray(
        hoomd_simulation.state.get_snapshot().particles.position,
        dtype=np.float64,
    )

    difference = jax_positions - hoomd_positions
    max_error = float(np.max(np.abs(difference)))
    rms_error = float(np.sqrt(np.mean(difference**2)))
    print(f"one-step deterministic max error: {max_error:.6e}")
    print(f"one-step deterministic RMS error: {rms_error:.6e}")

    assert hoomd_positions.shape == (EXPECTED_PARTICLES, 3)
    assert np.isfinite(hoomd_positions).all()
    np.testing.assert_allclose(
        jax_positions,
        hoomd_positions,
        atol=1e-4,
        rtol=1e-5,
        err_msg=(
            "One-step deterministic positions differ. Check force formulas, "
            "bond exclusions, neighbor-list cutoffs, and Brownian mobilities."
        ),
    )
