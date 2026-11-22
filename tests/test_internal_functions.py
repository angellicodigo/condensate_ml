"""JAX-only consistency tests (no HOOMD needed; run on CPU or GPU).

* forces agree with an independent float64 all-pairs implementation
* the neighbor list contains every interacting pair (guards the periodic-box bug)
* net force is zero (Newton's third law) and forces ignore a periodic shift of the whole system
* one step equals x + mobility * force * dt + noise, with mobility = 1/gamma
* the reference constants used against HOOMD agree with the package constants
"""

from __future__ import annotations

import numpy as np
import pytest

from condensate_ml.helper import (
    BOX,
    brute_force_components,
    make_simulation,
    minimum_image,
    perturbed_positions,
    wrap_centered,
)

pytest.importorskip("jax")
pytest.importorskip("jax_md")

CONDITIONS = ((40.0, -0.5, 101), (33.0, -1.2, 103), (29.0, -2.5, 105))


@pytest.mark.parametrize("sigma", (0.0, 0.2, 0.5))
@pytest.mark.parametrize("box,epsilon,seed", CONDITIONS)
def test_forces_match_brute_force(box, epsilon, seed, sigma):
    simulation = make_simulation(box, seed)
    positions = perturbed_positions(simulation, sigma, seed)
    computed = {
        name: np.asarray(value, dtype=np.float64)
        for name, value in simulation.compute_force_components(positions, epsilon).items()
    }
    reference = brute_force_components(simulation, positions, epsilon)
    for name in ("steric", "sticker_patch", "bonded", "total"):
        scale = np.linalg.norm(reference[name])
        error = np.linalg.norm(computed[name] - reference[name])
        assert error <= 1e-4 * scale + 5e-3, f"{name} differs from brute force"


@pytest.mark.parametrize("box,epsilon,seed", CONDITIONS)
def test_neighbor_list_contains_all_interacting_pairs(box, epsilon, seed):
    import jax.numpy as jnp

    simulation = make_simulation(box, seed)
    positions = perturbed_positions(simulation, 0.3, seed)
    neighbors = simulation.neighbor_fn.allocate(
        jnp.asarray(positions + box / 2, jnp.float32),
        extra_capacity=32,
    )
    assert not bool(neighbors.did_buffer_overflow)
    listed = {
        (i, int(j))
        for i, row in enumerate(np.asarray(neighbors.idx))
        for j in row
        if j < len(positions)
    }
    delta = minimum_image(positions[:, None] - positions[None, :], box)
    distances = np.linalg.norm(delta, axis=-1)
    needed = {
        (i, j)
        for i, j in zip(*np.where((distances < 3.0) & ~np.eye(len(positions), dtype=bool)))
    }
    assert not needed - listed


@pytest.mark.parametrize("box,epsilon,seed", CONDITIONS)
def test_total_force_sums_to_zero(box, epsilon, seed):
    simulation = make_simulation(box, seed)
    positions = perturbed_positions(simulation, 0.3, seed)
    for force in simulation.compute_force_components(positions, epsilon).values():
        force = np.asarray(force, dtype=np.float64)
        assert np.abs(force.sum(axis=0)).max() < 1e-3 * max(np.abs(force).max(), 1.0)


@pytest.mark.parametrize("box,epsilon,seed", CONDITIONS)
def test_forces_invariant_under_periodic_translation(box, epsilon, seed):
    simulation = make_simulation(box, seed)
    positions = perturbed_positions(simulation, 0.3, seed)
    shift = np.array([0.37 * box, -0.21 * box, 0.49 * box])
    moved = wrap_centered(positions + shift, box).astype(np.float32).astype(np.float64)
    first = np.asarray(simulation.compute_forces(positions, epsilon), dtype=np.float64)
    second = np.asarray(simulation.compute_forces(moved, epsilon), dtype=np.float64)
    assert np.linalg.norm(first - second) / np.linalg.norm(first) < 1e-3


def test_step_is_position_plus_mobility_force_dt_plus_noise():
    simulation = make_simulation(40.0, 101)
    positions = perturbed_positions(simulation, 0.2, 7)
    noise = np.random.default_rng(3).normal(0, 1e-3, positions.shape)
    force = np.asarray(simulation.compute_forces(positions, -1.0), dtype=np.float64)
    mobility = np.asarray(simulation.mobilities, dtype=np.float64).reshape(len(positions), -1)
    computed = np.asarray(simulation.brownian_step(positions, -1.0, noise=noise), dtype=np.float64)
    expected = wrap_centered(positions + mobility * force * simulation.dt + noise, 40.0)
    assert np.abs(computed - expected).max() < 5e-6


def test_mobilities_follow_hoomd_gammas():
    simulation = make_simulation(40.0, 101)
    mobility = np.asarray(simulation.mobilities, dtype=np.float64).reshape(len(simulation.type_ids), -1)[:, 0]
    types = np.asarray(simulation.type_ids)
    expected = np.where((types == 0) | (types == 1), 0.2, 1.0)
    np.testing.assert_allclose(mobility, expected, rtol=1e-6)


def test_short_stochastic_rollout_is_sane():
    simulation = make_simulation(29.0, 105, n_steps=200, write_every=100)
    frames = np.asarray(simulation({"epsilon": -2.5}))
    assert np.isfinite(frames).all()
    assert frames.max() < 14.5 + 1e-4 and frames.min() >= -14.5 - 1e-4
    assert not bool(simulation.last_neighbor_overflow)


def test_pair_across_periodic_boundary_matches_brute_force():
    simulation = make_simulation(BOX, 101)
    positions = np.asarray(simulation.init(), dtype=np.float64).copy()
    positions[0] = [BOX / 2 - 0.1, 0.0, 0.0]
    positions[1] = [-BOX / 2 + 0.1, 0.0, 0.0]
    positions = positions.astype(np.float32).astype(np.float64)
    computed = np.asarray(simulation.compute_force_components(positions, -0.8)["steric"])
    reference = brute_force_components(simulation, positions, -0.8)["steric"]
    assert np.isfinite(computed).all()
    # The two cores are 0.2 apart through the periodic face, so they must repel along x.
    assert computed[0, 0] < 0.0 < computed[1, 0]
    np.testing.assert_allclose(computed[:2], reference[:2], rtol=1e-4, atol=1e-4)


def test_reference_constants_match_package():
    """The HOOMD-side reference tables must not drift away from condensate_ml.constants."""
    from condensate_ml import constants as c
    from condensate_ml.helper import validate_forces as reference
    from condensate_ml.helper import validate_snapshot as snapshot

    assert (snapshot.N_R, snapshot.N_C, snapshot.N_BEADS, snapshot.PATCHES) == (
        c.N_R, c.N_C, c.N_BEADS, c.PATCHES)
    assert tuple(snapshot.STICKER_INDEX) == tuple(c.STICKER_INDEX)
    assert (snapshot.PATCH_RADIUS, snapshot.CHAIN_R0) == (c.PATCH_RADIUS, c.CHAIN_R0)
    assert reference.DT == c.DT
    assert reference.HOOMD_TYPES == c.TYPE_NAMES

    # Every steric pair except sticker-patch (handled by the attraction term) must agree.
    package_pairs = {pair: values for pair, values in c.STERIC_PAIR_PARAMS.items() if pair != ("P", "T")}
    assert reference.HOOMD_STERIC_PARAMS == package_pairs
    assert (reference.HOOMD_ATTRACTION_SIGMA, reference.HOOMD_ATTRACTION_RCUT) == (c.PATCH_SIGMA, c.PATCH_RCUT)

    assert reference.HOOMD_BOND_PARAMS["chain"] == dict(k=c.CHAIN_K, r0=c.CHAIN_R0)
    assert reference.HOOMD_BOND_PARAMS["patch_tether"] == dict(k=c.PATCH_TETHER_K, r0=c.PATCH_RADIUS)
    assert reference.HOOMD_GAMMA == {
        "R": c.GAMMA_RUBISCO, "P": c.GAMMA_RUBISCO, "S": c.GAMMA_CHAIN, "T": c.GAMMA_CHAIN}
