"""Shared force-validation helpers for HOOMD/JAX MD tests.

The ``HOOMD_*`` tables below mirror the force-field section of
``molecular_sim/simulate_hoomd_patchy_full.py``. They are written out here (and
not imported from ``condensate_ml.constants``) so that the reference stays
independent of the code under test; a test checks that both agree.
"""

from __future__ import annotations

import numpy as np
import pytest

from .validate_snapshot import BOX, build_hoomd_initial_state, minimum_image, wrap_centered

DT = 0.0005
EPSILON_VALUES = (-0.2, -0.8, -1.5, -2.0)
FORCE_SEEDS = (101, 107, 113, 119, 125)
PERTURBATIONS = (0.0, 0.2, 0.5)
COMPONENTS = ("steric", "sticker_patch", "bonded")

HOOMD_TYPES = ("R", "P", "S", "T")
# (epsilon, sigma, r_cut) of the Gaussian steric repulsion; every other pair is switched off.
HOOMD_STERIC_PARAMS = {
    ("R", "R"): (2.0, 1.5, 3.0),
    ("S", "S"): (1.0, 1.0, 2.5),
    ("T", "T"): (1.0, 1.0, 2.5),
    ("S", "T"): (1.0, 1.0, 2.5),
    ("R", "S"): (1.5, 1.5, 3.0),
    ("R", "T"): (1.5, 1.5, 3.0),
    ("P", "P"): (0.2, 0.8, 2.0),
    ("P", "S"): (0.2, 0.8, 2.0),
    ("P", "R"): (0.2, 0.8, 2.0),
}
HOOMD_ATTRACTION_PAIR = ("T", "P")  # sticker-patch; its epsilon is the value under test
HOOMD_ATTRACTION_SIGMA = 1.0
HOOMD_ATTRACTION_RCUT = 3.0
HOOMD_BOND_PARAMS = {
    "chain": dict(k=10.0, r0=1.2),
    "patch_tether": dict(k=50.0, r0=2.5),
}
HOOMD_GAMMA = {"R": 5.0, "P": 5.0, "S": 1.0, "T": 1.0}


# ---------------------------------------------------------------------------
# JAX side
# ---------------------------------------------------------------------------
def make_simulation(box: float, seed: int, **kwargs):
    """Construct a short JAX simulation (default: one step) for helper/test calculations."""
    from condensate_ml import RubiscoEpyc1Simulation

    kwargs.setdefault("n_steps", 1)
    kwargs.setdefault("write_every", 1)
    return RubiscoEpyc1Simulation(box=box, seed=seed, **kwargs)


def jax_force_components(positions: np.ndarray, epsilon: float) -> dict[str, np.ndarray]:
    """Evaluate JAX MD force components at a fixed configuration (box = BOX)."""
    pytest.importorskip("jax")
    pytest.importorskip("jax_md")
    simulation = make_simulation(BOX, seed=101, dt=DT)
    return {
        name: np.asarray(value, dtype=np.float64)
        for name, value in simulation.compute_force_components(positions, epsilon).items()
    }


# ---------------------------------------------------------------------------
# Configurations
# ---------------------------------------------------------------------------
def perturb_positions(positions: np.ndarray, sigma: float, box: float, seed: int) -> np.ndarray:
    """Add Gaussian noise, wrap into the box, and round through float32.

    Rounding through float32 guarantees that HOOMD (float32 storage) and JAX
    (float32 dtype) are handed exactly the same numbers.
    """
    if sigma <= 0.0:
        return np.asarray(positions, dtype=np.float64)
    rng = np.random.default_rng(seed)
    moved = np.asarray(positions, dtype=np.float64) + rng.normal(0.0, sigma, np.shape(positions))
    return wrap_centered(moved, box).astype(np.float32).astype(np.float64)


def perturbed_positions(simulation, sigma: float, seed: int) -> np.ndarray:
    """Perturbed copy of a JAX simulation's initial layout."""
    initial = np.asarray(simulation.init(), dtype=np.float64)
    return perturb_positions(initial, sigma, simulation.box, seed)


def perturbed_state(seed: int, sigma: float):
    """Return a GPU HOOMD state and the identical host-side perturbed positions."""
    simulation, positions = build_hoomd_initial_state(BOX, seed)
    if sigma > 0.0:
        positions = perturb_positions(positions, sigma, BOX, 1000 + seed)
        snapshot = simulation.state.get_snapshot()
        snapshot.particles.position[:] = positions
        simulation.state.set_snapshot(snapshot)
    return simulation, positions


# ---------------------------------------------------------------------------
# Independent references
# ---------------------------------------------------------------------------
def brute_force_components(simulation, positions: np.ndarray, epsilon: float) -> dict[str, np.ndarray]:
    """Independent float64 all-pairs reference for the three force terms.

    ``epsilon`` is the signed sticker-patch well depth (negative = attractive).
    """
    positions = np.asarray(positions, dtype=np.float64)
    types = np.asarray(simulation.type_ids)
    delta = minimum_image(positions[:, None, :] - positions[None, :, :], simulation.box)
    distance_sq = np.sum(delta * delta, axis=-1)

    pair = (types[:, None], types[None, :])
    epsilon_pair = np.asarray(simulation.epsilon_table, dtype=np.float64)[pair]
    sigma_sq = np.asarray(simulation.sigma_table, dtype=np.float64)[pair] ** 2
    cutoff = np.asarray(simulation.cutoff_table, dtype=np.float64)[pair]
    active = np.asarray(simulation.nonbonded_pair_mask) & (distance_sq < cutoff**2)
    steric = (
        np.where(active, epsilon_pair / sigma_sq * np.exp(-0.5 * distance_sq / sigma_sq), 0.0)[..., None]
        * delta
    ).sum(axis=1)

    sticker, patch = types == 3, types == 1
    active = (sticker[:, None] & patch[None, :]) | (patch[:, None] & sticker[None, :])
    active &= distance_sq < HOOMD_ATTRACTION_RCUT**2
    sigma_attr_sq = HOOMD_ATTRACTION_SIGMA**2
    attraction = (
        np.where(active, epsilon / sigma_attr_sq * np.exp(-0.5 * distance_sq / sigma_attr_sq), 0.0)[..., None]
        * delta
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
    return {
        "steric": steric,
        "sticker_patch": attraction,
        "bonded": bonded,
        "total": steric + attraction + bonded,
    }


def force_report(name: str, candidate: np.ndarray, reference: np.ndarray) -> tuple[float, float]:
    """Print and return the relative L2 and relative maximum force error (NaN if |reference| ~ 0)."""
    difference = candidate - reference
    reference_norm = np.linalg.norm(reference)
    if reference_norm <= 1e-8:
        relative_l2 = relative_max = float("nan")
    else:
        relative_l2 = float(np.linalg.norm(difference) / reference_norm)
        relative_max = float(np.max(np.abs(difference)) / max(np.max(np.abs(reference)), 1e-12))
    print(
        f"  {name:14s} |F|={reference_norm:10.4f} "
        f"rel L2={relative_l2:.2e} max|dF|/max|F|={relative_max:.2e}"
    )
    return relative_l2, relative_max


# ---------------------------------------------------------------------------
# HOOMD side
# ---------------------------------------------------------------------------
def configure_hoomd_forces(simulation, epsilon: float, dt: float, components: set[str]):
    """Attach the selected zero-noise reference forces to a HOOMD simulation.

    Returns the force objects by name so their ``.forces`` can be read after ``run(0)``.
    """
    hoomd = pytest.importorskip("hoomd")
    md = hoomd.md
    nlist = md.nlist.Cell(buffer=0.4)
    force_by_name = {}

    def switched_off_gaussian() -> object:
        force = md.pair.Gaussian(nlist=nlist)
        for first in HOOMD_TYPES:
            for second in HOOMD_TYPES:
                force.params[(first, second)] = dict(epsilon=0.0, sigma=1.0)
                force.r_cut[(first, second)] = 0.0
        return force

    if "steric" in components:
        steric = switched_off_gaussian()
        for pair, (pair_epsilon, sigma, r_cut) in HOOMD_STERIC_PARAMS.items():
            steric.params[pair] = dict(epsilon=pair_epsilon, sigma=sigma)
            steric.r_cut[pair] = r_cut
        force_by_name["steric"] = steric

    if "sticker_patch" in components:
        attraction = switched_off_gaussian()
        attraction.params[HOOMD_ATTRACTION_PAIR] = dict(epsilon=epsilon, sigma=HOOMD_ATTRACTION_SIGMA)
        attraction.r_cut[HOOMD_ATTRACTION_PAIR] = HOOMD_ATTRACTION_RCUT
        force_by_name["sticker_patch"] = attraction

    if "bonded" in components:
        harmonic = md.bond.Harmonic()
        for bond_type, params in HOOMD_BOND_PARAMS.items():
            harmonic.params[bond_type] = dict(params)
        force_by_name["bonded"] = harmonic

    # kT = 0 removes the random kick, leaving the deterministic drift x += F * dt / gamma.
    integrator = md.Integrator(dt=dt)
    brownian = md.methods.Brownian(filter=hoomd.filter.All(), kT=0.0)
    for particle_type, gamma in HOOMD_GAMMA.items():
        brownian.gamma[particle_type] = gamma
    integrator.methods.append(brownian)
    integrator.forces.extend(force_by_name.values())
    simulation.operations.integrator = integrator
    return force_by_name
