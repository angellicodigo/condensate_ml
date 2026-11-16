"""Force and short-rollout validation for the JAX MD Rubisco model.

HOOMD and JAX use different random-number generators, so deterministic force
checks use ``kT=0``. HOOMD force objects are read directly after ``run(0)``;
test_one_step.py separately checks the deterministic position update.
"""

from __future__ import annotations

import numpy as np
import pytest

from test_initial_conditions import (
    BOX,
    EXPECTED_PARTICLES,
    build_hoomd_initial_state,
)


DT = 0.0005
EPSILON_VALUES = (-0.2, -0.8, -1.5, -2.0)
FORCE_SEEDS = (101, 107, 113, 119, 125)
PERTURBATIONS = (0.0, 0.2, 0.5)
REL_L2_TOL = 1e-4
REL_MAX_TOL = 1e-3
COMPONENTS = ("steric", "sticker_patch", "bonded")


def minimum_image(delta: np.ndarray, box: float) -> np.ndarray:
    """Return the origin-centered minimum-image displacement."""
    return delta - box * np.round(delta / box)


def configure_hoomd(simulation, epsilon: float, dt: float, components: set[str]):
    """Attach selected reference force components and return force objects."""
    hoomd = pytest.importorskip("hoomd")
    md = hoomd.md
    types = ["R", "P", "S", "T"]
    nlist = md.nlist.Cell(buffer=0.4)
    force_objects = []
    force_by_name = {}

    if "steric" in components:
        steric = md.pair.Gaussian(nlist=nlist)
        for first in types:
            for second in types:
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
        force_objects.append(steric)
        force_by_name["steric"] = steric

    if "sticker_patch" in components:
        attraction = md.pair.Gaussian(nlist=nlist)
        for first in types:
            for second in types:
                attraction.params[(first, second)] = dict(epsilon=0.0, sigma=1.0)
                attraction.r_cut[(first, second)] = 0.0
        attraction.params[("T", "P")] = dict(epsilon=epsilon, sigma=1.0)
        attraction.r_cut[("T", "P")] = 3.0
        force_objects.append(attraction)
        force_by_name["sticker_patch"] = attraction

    if "bonded" in components:
        harmonic = md.bond.Harmonic()
        harmonic.params["chain"] = dict(k=10.0, r0=1.2)
        harmonic.params["patch_tether"] = dict(k=50.0, r0=2.5)
        force_objects.append(harmonic)
        force_by_name["bonded"] = harmonic

    integrator = md.Integrator(dt=dt)
    brownian = md.methods.Brownian(filter=hoomd.filter.All(), kT=0.0)
    for particle_type, gamma in {"R": 5.0, "P": 5.0, "S": 1.0, "T": 1.0}.items():
        brownian.gamma[particle_type] = gamma
    integrator.methods.append(brownian)
    integrator.forces.extend(force_objects)
    simulation.operations.integrator = integrator
    return force_by_name


def perturbed_state(seed: int, sigma: float):
    """Return a GPU HOOMD state and identical host-side positions."""
    simulation, positions = build_hoomd_initial_state(BOX, seed)
    if sigma > 0.0:
        rng = np.random.default_rng(1000 + seed)
        positions = minimum_image(
            positions + rng.normal(0.0, sigma, positions.shape),
            BOX,
        )
        snapshot = simulation.state.get_snapshot()
        snapshot.particles.position[:] = positions
        simulation.state.set_snapshot(snapshot)
    return simulation, positions


def jax_force_components(positions: np.ndarray, epsilon: float) -> dict[str, np.ndarray]:
    """Evaluate JAX MD force components at a fixed configuration."""
    pytest.importorskip("jax")
    pytest.importorskip("jax_md")
    from condensate_ml import RubiscoEpyc1Simulation

    simulation = RubiscoEpyc1Simulation(
        n_steps=1,
        dt=DT,
        box=BOX,
        write_every=1,
        seed=101,
    )
    return {
        name: np.asarray(value, dtype=np.float64)
        for name, value in simulation.compute_force_components(positions, epsilon).items()
    }


def force_report(name: str, candidate: np.ndarray, reference: np.ndarray) -> tuple[float, float]:
    """Print and return relative L2 and relative maximum force errors."""
    difference = candidate - reference
    reference_norm = np.linalg.norm(reference)
    reference_max = np.max(np.abs(reference))
    if reference_norm <= 1e-8:
        # Relative error is undefined for a zero reference. Report NaN and let
        # the caller apply an absolute roundoff tolerance instead.
        rel_l2 = float("nan")
        rel_max = float("nan")
    else:
        rel_l2 = float(np.linalg.norm(difference) / reference_norm)
        rel_max = float(np.max(np.abs(difference)) / max(reference_max, 1e-12))
    print(
        f"  {name:14s} |F|={reference_norm:10.4f} "
        f"rel L2={rel_l2:.2e} max|dF|/max|F|={rel_max:.2e}"
    )
    return rel_l2, rel_max


@pytest.mark.parametrize("sigma", PERTURBATIONS)
@pytest.mark.parametrize("epsilon", EPSILON_VALUES)
@pytest.mark.parametrize("seed", FORCE_SEEDS)
def test_force_components_match_hoomd(seed: int, epsilon: float, sigma: float) -> None:
    """Compare HOOMD force objects directly for 60 configurations."""
    pytest.importorskip("jax")
    pytest.importorskip("jax_md")
    hoomd_simulation, positions = perturbed_state(seed, sigma)
    hoomd_forces = configure_hoomd(
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
        rel_l2, rel_max = force_report(name, candidate, reference)
        if np.linalg.norm(reference) > 1e-8:
            assert rel_l2 < REL_L2_TOL, f"{name} relative L2 error={rel_l2:.2e}"
            assert rel_max < REL_MAX_TOL, f"{name} relative max error={rel_max:.2e}"
        else:
            # The reference force is zero, so relative error is undefined.
            # The ~1e-4 JAX residual is float32 rounding from ideal bond lengths.
            np.testing.assert_allclose(candidate, 0.0, atol=2e-4, rtol=0.0)

    total_l2, _ = force_report("total", jax_components["total"], total_hoomd)
    total_tolerance = 3e-4 if sigma == 0.0 else REL_L2_TOL
    assert total_l2 < total_tolerance


def test_jax_force_satisfies_newtons_third_law() -> None:
    """The net force of every pair/bond force component should be zero."""
    _, positions = build_hoomd_initial_state(BOX, 107)
    components = jax_force_components(positions, -0.8)
    for name in ("steric", "sticker_patch", "bonded", "total"):
        # float32 sums of ~800 forces: scale the tolerance with the largest force.
        tolerance = 1e-5 * max(1.0, float(np.max(np.abs(components[name]))))
        np.testing.assert_allclose(
            components[name].sum(axis=0),
            np.zeros(3),
            atol=tolerance,
            rtol=0.0,
            err_msg=f"Net force is nonzero for {name}",
        )


def _rollout_simulation():
    from condensate_ml import RubiscoEpyc1Simulation

    return RubiscoEpyc1Simulation(n_steps=40, box=29.0, write_every=10, seed=105)


def test_rollout_gradient_with_respect_to_epsilon_matches_finite_difference() -> None:
    """Autodiff through a 40-step rollout agrees with central finite differences.

    The force is linear in epsilon, so differentiating a single force call is a
    trivial check. A rollout also differentiates through the neighbor list,
    periodic wrapping, ``lax.scan`` and ``jax.checkpoint``.
    """
    pytest.importorskip("jax")
    pytest.importorskip("jax_md")
    import jax
    import jax.numpy as jnp

    simulation = _rollout_simulation()
    sticker = np.asarray(simulation.type_ids) == 3
    patch = np.asarray(simulation.type_ids) == 1
    box = simulation.box

    def mean_square_loss(epsilon):
        last = simulation({"epsilon": epsilon})[-1]
        return jnp.mean(last**2) + jnp.mean(jnp.var(last, axis=0))

    def contact_loss(epsilon):
        last = simulation({"epsilon": epsilon})[-1]
        delta = last[sticker][:, None, :] - last[patch][None, :, :]
        delta = delta - box * jnp.round(delta / box)
        return jnp.mean(jnp.exp(-0.5 * jnp.sum(delta**2, axis=-1)))

    epsilon = jnp.asarray(-2.0, dtype=jnp.float32)
    h = jnp.asarray(0.2, dtype=jnp.float32)
    for loss in (mean_square_loss, contact_loss):
        automatic = jax.grad(loss)(epsilon)
        finite_difference = (loss(epsilon + h) - loss(epsilon - h)) / (2.0 * h)
        print(f"{loss.__name__}: autodiff={float(automatic):.6e} finite={float(finite_difference):.6e}")
        assert np.isfinite(float(automatic))
        assert float(automatic) != 0.0
        # float32 + a finite step: agreement of a few percent is the expected level.
        np.testing.assert_allclose(automatic, finite_difference, rtol=5e-2, atol=0.0)


def test_five_deterministic_steps_match_hoomd() -> None:
    """Check that small deterministic position errors do not rapidly grow."""
    pytest.importorskip("jax")
    pytest.importorskip("jax_md")
    from condensate_ml import RubiscoEpyc1Simulation

    seed = 107
    epsilon = -0.8
    n_steps = 5
    hoomd_simulation, initial_positions = build_hoomd_initial_state(BOX, seed)
    configure_hoomd(hoomd_simulation, epsilon, DT, {"steric", "sticker_patch", "bonded"})

    jax_simulation = RubiscoEpyc1Simulation(
        n_steps=1,
        dt=DT,
        box=BOX,
        write_every=1,
        seed=seed,
    )
    jax_positions = initial_positions.copy()
    for _ in range(n_steps):
        jax_positions = np.asarray(
            jax_simulation.brownian_step(
                jax_positions,
                epsilon,
                noise=np.zeros_like(jax_positions),
            )
        )

    hoomd_simulation.run(n_steps)
    hoomd_positions = np.asarray(
        hoomd_simulation.state.get_snapshot().particles.position,
        dtype=np.float64,
    )
    difference = minimum_image(jax_positions - hoomd_positions, BOX)
    print(f"five-step max error: {np.max(np.abs(difference)):.6e}")
    print(f"five-step RMS error: {np.sqrt(np.mean(difference**2)):.6e}")
    np.testing.assert_allclose(
        jax_positions,
        hoomd_positions,
        # Five steps move particles by up to ~2.7e-3, so the tolerance must be far
        # smaller than that. Float32 rounding accumulates to ~2e-5 at most.
        atol=5e-5,
        rtol=0.0,
        err_msg="Deterministic five-step trajectories diverged from HOOMD.",
    )
