"""Compare JAX MD forces and short deterministic trajectories against HOOMD (needs a HOOMD GPU).

HOOMD and JAX use different random-number generators, so everything here runs HOOMD with
``kT=0`` (no random kick). JAX-only checks live in ``test_internal_functions.py`` and
``test_differentiable.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

from condensate_ml.helper import (
    BOX,
    COMPONENTS,
    DT,
    EPSILON_VALUES,
    EXPECTED_PARTICLES,
    FORCE_SEEDS,
    PERTURBATIONS,
    build_hoomd_initial_state,
    configure_hoomd_forces,
    force_report,
    jax_force_components,
    make_simulation,
    minimum_image,
    perturbed_state,
)

# Float32 JAX vs float32 HOOMD. On the unperturbed layout the bonded force is ~0, so the
# relative error of the total is dominated by rounding and gets a looser bound.
REL_L2_TOL = 1e-4
REL_MAX_TOL = 1e-3
REL_L2_TOL_INITIAL_LAYOUT = 3e-4
FIVE_STEP_ATOL = 5e-5


@pytest.mark.parametrize("sigma", PERTURBATIONS)
@pytest.mark.parametrize("epsilon", EPSILON_VALUES)
@pytest.mark.parametrize("seed", FORCE_SEEDS)
def test_force_components_match_hoomd(seed: int, epsilon: float, sigma: float) -> None:
    """Compare each HOOMD force term against JAX (5 seeds x 4 epsilons x 3 perturbations = 60)."""
    pytest.importorskip("jax_md")
    hoomd_simulation, positions = perturbed_state(seed, sigma)
    hoomd_forces = configure_hoomd_forces(
        hoomd_simulation,
        epsilon,
        DT,
        set(COMPONENTS),
    )
    hoomd_simulation.run(0)
    jax_components = jax_force_components(positions, epsilon)
    print(f"\nseed={seed} epsilon={epsilon} perturbation={sigma}")

    total_hoomd = np.zeros_like(positions)
    for name in COMPONENTS:
        reference = np.asarray(hoomd_forces[name].forces, dtype=np.float64)
        candidate = jax_components[name]
        assert reference.shape == (EXPECTED_PARTICLES, 3)
        assert np.isfinite(reference).all()
        total_hoomd += reference
        relative_l2, relative_max = force_report(name, candidate, reference)
        if np.linalg.norm(reference) > 1e-8:
            assert relative_l2 < REL_L2_TOL
            assert relative_max < REL_MAX_TOL
        else:
            np.testing.assert_allclose(candidate, 0.0, atol=2e-4, rtol=0.0)

    total_l2, _ = force_report("total", jax_components["total"], total_hoomd)
    assert total_l2 < (REL_L2_TOL_INITIAL_LAYOUT if sigma == 0.0 else REL_L2_TOL)


def test_five_deterministic_steps_match_hoomd() -> None:
    """Five zero-noise steps: positions must still agree with HOOMD (errors must not grow).

    Five steps move particles by up to ~2.7e-3, so the 5e-5 tolerance is ~50x below the motion.
    """
    pytest.importorskip("jax_md")
    seed, epsilon, n_steps = 107, -0.8, 5
    hoomd_simulation, initial_positions = build_hoomd_initial_state(BOX, seed)
    configure_hoomd_forces(hoomd_simulation, epsilon, DT, set(COMPONENTS))
    jax_simulation = make_simulation(BOX, seed, dt=DT)

    jax_positions = initial_positions.copy()
    for _ in range(n_steps):
        jax_positions = np.asarray(
            jax_simulation.brownian_step(jax_positions, epsilon, noise=np.zeros_like(jax_positions))
        )

    hoomd_simulation.run(n_steps)
    hoomd_positions = np.asarray(
        hoomd_simulation.state.get_snapshot().particles.position,
        dtype=np.float64,
    )
    difference = minimum_image(jax_positions - hoomd_positions, BOX)
    print(f"five-step max error: {np.max(np.abs(difference)):.6e}")
    print(f"five-step RMS error: {np.sqrt(np.mean(difference**2)):.6e}")
    np.testing.assert_allclose(jax_positions, hoomd_positions, atol=FIVE_STEP_ATOL, rtol=0.0)
