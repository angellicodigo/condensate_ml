"""Shared force-validation helpers for HOOMD/JAX MD tests."""

from __future__ import annotations

import numpy as np
import pytest

from .validate_snapshot import BOX, build_hoomd_initial_state, minimum_image

DT = 0.0005
EPSILON_VALUES = (-0.2, -0.8, -1.5, -2.0)
FORCE_SEEDS = (101, 107, 113, 119, 125)
PERTURBATIONS = (0.0, 0.2, 0.5)
COMPONENTS = ("steric", "sticker_patch", "bonded")


def make_simulation(box: float, seed: int, **kwargs):
    """Construct a one-step JAX simulation for helper/test calculations."""
    from condensate_ml import RubiscoEpyc1Simulation

    n_steps = kwargs.pop("n_steps", 1)
    write_every = kwargs.pop("write_every", 1)
    return RubiscoEpyc1Simulation(
        n_steps=n_steps,
        box=box,
        write_every=write_every,
        seed=seed,
        **kwargs,
    )


def perturbed_positions(simulation, sigma: float, seed: int) -> np.ndarray:
    """Return a wrapped float64 perturbation of a simulation's initial state."""
    positions = np.asarray(simulation.init(), dtype=np.float64)
    rng = np.random.default_rng(seed)
    return minimum_image(
        positions + rng.normal(0.0, sigma, positions.shape), simulation.box
    ).astype(np.float32).astype(np.float64)


def brute_force_components(simulation, positions: np.ndarray, epsilon: float) -> dict[str, np.ndarray]:
    """Independent float64 all-pairs reference implementation."""
    positions = np.asarray(positions, dtype=np.float64)
    types = np.asarray(simulation.type_ids)
    delta = minimum_image(positions[:, None, :] - positions[None, :, :], simulation.box)
    distance_sq = np.sum(delta * delta, axis=-1)

    epsilon_pair = np.asarray(simulation.epsilon_table, dtype=np.float64)[types[:, None], types[None, :]]
    sigma_sq = np.asarray(simulation.sigma_table, dtype=np.float64)[types[:, None], types[None, :]] ** 2
    cutoff = np.asarray(simulation.cutoff_table, dtype=np.float64)[types[:, None], types[None, :]]
    active = np.asarray(simulation.nonbonded_pair_mask) & (distance_sq < cutoff**2)
    steric = (
        np.where(active, epsilon_pair / sigma_sq * np.exp(-0.5 * distance_sq / sigma_sq), 0.0)[..., None]
        * delta
    ).sum(axis=1)

    is_sticker, is_patch = types == 3, types == 1
    active = ((is_sticker[:, None] & is_patch[None, :]) | (is_patch[:, None] & is_sticker[None, :]))
    active &= distance_sq < 3.0**2
    attraction = (
        np.where(active, epsilon * np.exp(-0.5 * distance_sq), 0.0)[..., None] * delta
    ).sum(axis=1)

    bonded = np.zeros_like(positions)
    for (i, j), spring, rest in zip(
        np.asarray(simulation.bond_pairs),
        np.asarray(simulation.bond_k, dtype=np.float64),
        np.asarray(simulation.bond_r0, dtype=np.float64),
    ):
        bond_delta = minimum_image(positions[i] - positions[j], simulation.box)
        distance = np.linalg.norm(bond_delta)
        force = -spring * (distance - rest) * bond_delta / distance
        bonded[i] += force
        bonded[j] -= force
    return {"steric": steric, "sticker_patch": attraction, "bonded": bonded, "total": steric + attraction + bonded}


def configure_hoomd_forces(simulation, epsilon: float, dt: float, components: set[str]):
    """Attach selected reference force components and return them by name."""
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
    """Return a GPU HOOMD state and identical host-side perturbed positions."""
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


def force_report(
    name: str,
    candidate: np.ndarray,
    reference: np.ndarray,
) -> tuple[float, float]:
    """Print relative L2 and relative maximum force errors."""
    difference = candidate - reference
    reference_norm = np.linalg.norm(reference)
    reference_max = np.max(np.abs(reference))
    if reference_norm <= 1e-8:
        relative_l2 = float("nan")
        relative_max = float("nan")
    else:
        relative_l2 = float(np.linalg.norm(difference) / reference_norm)
        relative_max = float(np.max(np.abs(difference)) / max(reference_max, 1e-12))
    print(
        f"  {name:14s} |F|={reference_norm:10.4f} "
        f"rel L2={relative_l2:.2e} max|dF|/max|F|={relative_max:.2e}"
    )
    return relative_l2, relative_max
