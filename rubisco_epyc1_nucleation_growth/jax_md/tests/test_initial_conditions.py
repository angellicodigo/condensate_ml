"""Compare HOOMD and JAX MD initial particle positions on a GPU.

The HOOMD builder below follows the initial-snapshot code in
``molecular_sim/simulate_hoomd_patchy_full.py``. The full HOOMD script is not
imported or executed because it parses command-line arguments, writes files,
configures an integrator, and runs a simulation.

Run from the project root with::

    pytest rubisco_epyc1_nucleation_growth/jax_md/tests/test_initial_conditions.py -v -s
"""

from __future__ import annotations

import numpy as np
import pytest


BOX = 40.0
SEED = 101
# A wider seed sweep checks that the two implementations agree for different
# Rubisco jitters, chain-center jitters, and random chain orientations.
INITIAL_SEEDS = tuple(range(101, 126))
EXPECTED_PARTICLES = 792
POSITION_ATOL = 1e-6
POSITION_RTOL = 0.0

# These values mirror the MODEL PARAMETERS section of the HOOMD reference.
N_R = 24
N_C = 48
N_BEADS = 12
STICKER_INDEX = [1, 3, 5, 7, 9, 11]
PATCHES = 8
PATCH_RADIUS = 2.5
CHAIN_R0 = 1.2


def wrap_centered(positions: np.ndarray, box: float) -> np.ndarray:
    """Wrap coordinates into the origin-centered periodic box."""
    return ((positions + box / 2.0) % box) - box / 2.0


def make_grid_positions(n: int, box: float, margin: float = 6.0) -> np.ndarray:
    """Copy the cubic-grid helper from the HOOMD reference script."""
    n_side = int(np.ceil(n ** (1.0 / 3.0)))
    values = np.linspace(-box / 2.0 + margin, box / 2.0 - margin, n_side)
    points = np.array(np.meshgrid(values, values, values)).T.reshape(-1, 3)
    return points[:n].copy()


def build_hoomd_initial_state(box: float, seed: int):
    """Create the initial HOOMD state on the assigned GPU."""
    hoomd = pytest.importorskip("hoomd")

    try:
        device = hoomd.device.GPU()
        simulation = hoomd.Simulation(device=device, seed=seed)
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
    typeid = np.zeros(n_total, dtype=np.uint32)

    # This is the PATCH GEOMETRY block from the HOOMD script.
    patch_dirs = np.array(
        [
            [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
            [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1],
        ],
        dtype=float,
    )
    patch_dirs /= np.linalg.norm(patch_dirs, axis=1, keepdims=True)

    # Rubisco centers: margin=7.0 and jitter scale=0.20.
    rubisco_centers = make_grid_positions(N_R, box, margin=7.0)
    rubisco_centers += rng.normal(scale=0.20, size=rubisco_centers.shape)
    rubisco_centers = wrap_centered(rubisco_centers, box)

    index = 0
    for center in rubisco_centers:
        positions[index] = center
        typeid[index] = 0
        index += 1

    # Do not wrap patches independently: they must remain local to their core.
    patch_start = index
    for center in rubisco_centers:
        for direction in patch_dirs:
            positions[index] = center + PATCH_RADIUS * direction
            typeid[index] = 1
            index += 1

    # EPYC1 chain centers and randomly oriented linear chains.
    chain_centers = make_grid_positions(N_C, box, margin=5.0)
    chain_centers += rng.normal(scale=0.25, size=chain_centers.shape)
    chain_centers = wrap_centered(chain_centers, box)

    chain_start = N_R + n_patch
    for center in chain_centers:
        direction = rng.normal(size=3)
        direction /= np.linalg.norm(direction)
        for bead in range(N_BEADS):
            bead_position = center + (bead - (N_BEADS - 1) / 2.0) * CHAIN_R0 * direction
            positions[index] = wrap_centered(bead_position, box)
            typeid[index] = 3 if bead in STICKER_INDEX else 2
            index += 1

    snapshot.particles.position[:] = positions
    snapshot.particles.typeid[:] = typeid

    # Include the same bonds as the reference snapshot.
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
    all_bonds = chain_bonds + patch_tethers
    snapshot.bonds.N = len(all_bonds)
    snapshot.bonds.types = ["chain", "patch_tether"]
    bond_typeid = np.zeros(len(all_bonds), dtype=np.uint32)
    bond_typeid[len(chain_bonds):] = 1
    snapshot.bonds.typeid[:] = bond_typeid
    snapshot.bonds.group[:] = np.asarray(all_bonds, dtype=np.uint32)

    simulation.create_state_from_snapshot(snapshot)
    state_snapshot = simulation.state.get_snapshot()
    return simulation, np.asarray(state_snapshot.particles.position, dtype=np.float64)


def build_hoomd_initial_positions(box: float, seed: int) -> np.ndarray:
    """Return only positions from the GPU-backed HOOMD initial state."""
    _simulation, positions = build_hoomd_initial_state(box, seed)
    return positions


def build_jax_initial_positions(box: float, seed: int) -> np.ndarray:
    """Create the initial state from the JAX MD simulation class."""
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


def assert_initial_positions_match(seed: int) -> None:
    """Compare all corresponding particles for one random initial condition."""
    hoomd_positions = build_hoomd_initial_positions(BOX, seed)
    jax_positions = build_jax_initial_positions(BOX, seed)

    assert hoomd_positions.shape == (EXPECTED_PARTICLES, 3)
    assert jax_positions.shape == (EXPECTED_PARTICLES, 3)

    difference = jax_positions - hoomd_positions
    max_error = float(np.max(np.abs(difference)))
    rms_error = float(np.sqrt(np.mean(difference**2)))

    print(f"initial-position max error: {max_error:.6e}")
    print(f"initial-position RMS error: {rms_error:.6e}")

    np.testing.assert_allclose(
        jax_positions,
        hoomd_positions,
        atol=POSITION_ATOL,
        rtol=POSITION_RTOL,
        err_msg=(
            "JAX MD and HOOMD initial positions differ. Check particle ordering, "
            "seed handling, patch wrapping, and chain initialization."
        ),
    )


@pytest.mark.parametrize("seed", INITIAL_SEEDS)
def test_hoomd_and_jax_md_initial_positions_match(seed: int) -> None:
    """Every JAX particle must match HOOMD for 25 independent seeds."""
    print(f"\nComparing initial condition with seed={seed}")
    assert_initial_positions_match(seed)
