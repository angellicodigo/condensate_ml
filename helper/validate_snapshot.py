"""HOOMD reference initial-condition builder and the JAX counterpart.

The model constants below are deliberately written out again instead of being
imported from ``condensate_ml.constants``. They mirror the MODEL PARAMETERS in
``molecular_sim/simulate_hoomd_patchy_full.py`` so that the comparison against
HOOMD stays independent of the package under test.
``tests/test_internal_functions.py::test_reference_constants_match_package``
fails if the two ever drift apart.
"""

from __future__ import annotations

import numpy as np
import pytest

BOX = 40.0
N_R = 24
N_C = 48
N_BEADS = 12
STICKER_INDEX = (1, 3, 5, 7, 9, 11)
PATCHES = 8
PATCH_RADIUS = 2.5
CHAIN_R0 = 1.2
EXPECTED_PARTICLES = N_R + N_R * PATCHES + N_C * N_BEADS  # 792


def minimum_image(delta: np.ndarray, box: float) -> np.ndarray:
    """Map displacement vectors to their nearest periodic image."""
    return delta - box * np.round(delta / box)


def wrap_centered(positions: np.ndarray, box: float) -> np.ndarray:
    """Wrap coordinates into ``[-box/2, box/2)`` (same rule as the JAX rollout)."""
    return ((positions + box / 2.0) % box) - box / 2.0


def make_grid_positions(n: int, box: float, margin: float) -> np.ndarray:
    """Create the cubic lattice used by the HOOMD reference initialization."""
    n_side = int(np.ceil(n ** (1.0 / 3.0)))
    values = np.linspace(-box / 2.0 + margin, box / 2.0 - margin, n_side)
    points = np.array(np.meshgrid(values, values, values)).T.reshape(-1, 3)
    return points[:n].copy()


def _patch_directions() -> np.ndarray:
    """Eight unit vectors pointing to the corners of a cube."""
    directions = np.asarray(
        [
            [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
            [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1],
        ],
        dtype=float,
    )
    return directions / np.linalg.norm(directions, axis=1, keepdims=True)


def build_hoomd_initial_state(box: float, seed: int):
    """Build the reference initial state on an assigned HOOMD GPU."""
    hoomd = pytest.importorskip("hoomd")
    try:
        simulation = hoomd.Simulation(device=hoomd.device.GPU(), seed=seed)
    except Exception as error:
        pytest.skip(f"HOOMD GPU is unavailable: {error}")

    snapshot = hoomd.Snapshot()
    rng = np.random.default_rng(seed)
    n_patch = N_R * PATCHES
    n_total = N_R + n_patch + N_C * N_BEADS
    snapshot.configuration.box = [box, box, box, 0, 0, 0]
    snapshot.particles.N = n_total
    snapshot.particles.types = ["R", "P", "S", "T"]

    positions = np.zeros((n_total, 3), dtype=float)
    type_ids = np.zeros(n_total, dtype=np.uint32)
    patch_dirs = _patch_directions()

    rubisco_centers = make_grid_positions(N_R, box, margin=7.0)
    rubisco_centers = wrap_centered(
        rubisco_centers + rng.normal(scale=0.20, size=rubisco_centers.shape),
        box,
    )

    index = 0
    for center in rubisco_centers:
        positions[index] = center
        type_ids[index] = 0
        index += 1

    patch_start = index
    for center in rubisco_centers:
        for direction in patch_dirs:
            positions[index] = center + PATCH_RADIUS * direction
            type_ids[index] = 1
            index += 1

    chain_centers = make_grid_positions(N_C, box, margin=5.0)
    chain_centers = wrap_centered(
        chain_centers + rng.normal(scale=0.25, size=chain_centers.shape),
        box,
    )
    chain_start = N_R + n_patch
    for center in chain_centers:
        direction = rng.normal(size=3)
        direction /= np.linalg.norm(direction)
        for bead in range(N_BEADS):
            value = center + (bead - (N_BEADS - 1) / 2.0) * CHAIN_R0 * direction
            positions[index] = wrap_centered(value, box)
            type_ids[index] = 3 if bead in STICKER_INDEX else 2
            index += 1

    snapshot.particles.position[:] = positions
    snapshot.particles.typeid[:] = type_ids

    chain_bonds = [
        [chain_start + c * N_BEADS + bead, chain_start + c * N_BEADS + bead + 1]
        for c in range(N_C)
        for bead in range(N_BEADS - 1)
    ]
    patch_tethers = [
        [rubisco, patch_start + rubisco * PATCHES + patch]
        for rubisco in range(N_R)
        for patch in range(PATCHES)
    ]
    bonds = chain_bonds + patch_tethers
    snapshot.bonds.N = len(bonds)
    snapshot.bonds.types = ["chain", "patch_tether"]
    bond_type_ids = np.zeros(len(bonds), dtype=np.uint32)
    bond_type_ids[len(chain_bonds):] = 1
    snapshot.bonds.typeid[:] = bond_type_ids
    snapshot.bonds.group[:] = np.asarray(bonds, dtype=np.uint32)

    simulation.create_state_from_snapshot(snapshot)
    state_snapshot = simulation.state.get_snapshot()
    return simulation, np.asarray(state_snapshot.particles.position, dtype=np.float64)


def build_hoomd_initial_positions(box: float, seed: int) -> np.ndarray:
    """Return only the GPU-backed HOOMD initial positions."""
    _simulation, positions = build_hoomd_initial_state(box, seed)
    return positions


def build_jax_initial_positions(box: float, seed: int) -> np.ndarray:
    """Return initial positions from ``RubiscoEpyc1Simulation``."""
    pytest.importorskip("jax")
    pytest.importorskip("jax_md")
    from condensate_ml import RubiscoEpyc1Simulation

    simulation = RubiscoEpyc1Simulation(
        n_steps=1,
        box=box,
        write_every=1,
        seed=seed,
    )
    return np.asarray(simulation.init(), dtype=np.float64)
