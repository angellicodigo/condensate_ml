"""Force, gradient, and short deterministic rollout validation."""

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
    minimum_image,
    perturbed_state,
)

REL_L2_TOL = 1e-4
REL_MAX_TOL = 1e-3


@pytest.mark.parametrize("sigma", PERTURBATIONS)
@pytest.mark.parametrize("epsilon", EPSILON_VALUES)
@pytest.mark.parametrize("seed", FORCE_SEEDS)
def test_force_components_match_hoomd(seed: int, epsilon: float, sigma: float) -> None:
    """Compare HOOMD force objects directly for 60 configurations."""
    pytest.importorskip("jax")
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
    assert total_l2 < (3e-4 if sigma == 0.0 else REL_L2_TOL)


def test_jax_force_satisfies_newtons_third_law() -> None:
    """Each force component should have approximately zero net force."""
    _, positions = build_hoomd_initial_state(BOX, 107)
    components = jax_force_components(positions, -0.8)
    for name in (*COMPONENTS, "total"):
        tolerance = 1e-5 * max(1.0, float(np.max(np.abs(components[name]))))
        np.testing.assert_allclose(components[name].sum(axis=0), 0.0, atol=tolerance)


def test_periodic_force_is_finite() -> None:
    """Minimum-image forces remain finite near a periodic boundary."""
    pytest.importorskip("jax")
    pytest.importorskip("jax_md")
    from condensate_ml import RubiscoEpyc1Simulation

    simulation = RubiscoEpyc1Simulation(n_steps=1, box=BOX, write_every=1)
    positions = np.asarray(simulation.init(), dtype=np.float32).copy()
    positions[0] = [BOX / 2.0 - 0.1, 0.0, 0.0]
    positions[1] = [-BOX / 2.0 + 0.1, 0.0, 0.0]
    forces = np.asarray(simulation.compute_forces(positions, -0.8))
    assert np.isfinite(forces).all()


def test_rollout_gradient_matches_finite_difference() -> None:
    """Autodiff through a short rollout agrees with finite differences."""
    pytest.importorskip("jax")
    pytest.importorskip("jax_md")
    import jax
    import jax.numpy as jnp
    from condensate_ml import RubiscoEpyc1Simulation

    simulation = RubiscoEpyc1Simulation(n_steps=40, box=29.0, write_every=10, seed=105)

    def loss(epsilon):
        last = simulation({"epsilon": epsilon})[-1]
        return jnp.mean(last**2) + jnp.mean(jnp.var(last, axis=0))

    epsilon = jnp.asarray(-2.0, dtype=jnp.float32)
    h = jnp.asarray(0.2, dtype=jnp.float32)
    automatic = jax.grad(loss)(epsilon)
    finite_difference = (loss(epsilon + h) - loss(epsilon - h)) / (2.0 * h)
    np.testing.assert_allclose(automatic, finite_difference, rtol=5e-2, atol=0.0)


def test_five_deterministic_steps_match_hoomd() -> None:
    """Check that deterministic position errors do not rapidly grow."""
    pytest.importorskip("jax")
    pytest.importorskip("jax_md")
    from condensate_ml import RubiscoEpyc1Simulation

    seed, epsilon, n_steps = 107, -0.8, 5
    hoomd_simulation, initial_positions = build_hoomd_initial_state(BOX, seed)
    configure_hoomd_forces(hoomd_simulation, epsilon, DT, set(COMPONENTS))
    jax_simulation = RubiscoEpyc1Simulation(n_steps=1, dt=DT, box=BOX, write_every=1, seed=seed)

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
    np.testing.assert_allclose(jax_positions, hoomd_positions, atol=5e-5, rtol=0.0)
