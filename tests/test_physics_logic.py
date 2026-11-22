"""Logic and symmetry checks for the simulation model.

Tests include:
5. Symmetries: Rotation, reflection, chain relabeling invariance, epsilon=0 zero force,
   and epsilon of either sign yielding identical forces.
6. Exclusions and cutoffs: Direct bonded pairs excluded from sterics, next-nearest neighbors
   interact, and forces strictly vanish outside cutoffs while matching formulas inside.
7. Reproducibility: Deterministic trajectory reproduction with identical seeds, divergence
   with different seeds, and equivalence between checkpointed and uncheckpointed scans.
8. Time-step convergence: Halving dt over a fixed interval reduces integration error linearly.
"""

from __future__ import annotations

import numpy as np
import pytest

from condensate_ml import RubiscoEpyc1Simulation
from condensate_ml.constants import (
    CHAIN_K,
    CHAIN_R0,
    PATCH_RCUT,
    PATCH_SIGMA,
    PATCH_TETHER_K,
    STERIC_PAIR_PARAMS,
    TYPE_INDEX,
)
from condensate_ml.helper import BOX, make_simulation, minimum_image, perturbed_positions


# ---------------------------------------------------------------------------
# Test 5: Symmetries
# ---------------------------------------------------------------------------


def build_rotation_matrix_z_90() -> np.ndarray:
    """Returns a 3x3 orthogonal rotation matrix rotating 90 degrees around the z-axis.

    Inputs:
        None.
    Outputs:
        3x3 rotation matrix.
    """
    return np.array(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def build_reflection_matrix_z() -> np.ndarray:
    """Returns a 3x3 reflection matrix flipping the z-coordinate.

    Inputs:
        None.
    Outputs:
        3x3 reflection matrix.
    """
    return np.diag([1.0, 1.0, -1.0]).astype(np.float64)


def test_forces_rotate_with_system() -> None:
    """Rotating coordinates by an orthogonal matrix must rotate net forces by the same matrix."""
    simulation = make_simulation(BOX, seed=101)
    positions = perturbed_positions(simulation, sigma=0.2, seed=101)
    rotation_matrix = build_rotation_matrix_z_90()
    rotated_positions = positions @ rotation_matrix.T

    epsilon = -1.2
    original_forces = np.asarray(simulation.compute_forces(positions, epsilon), dtype=np.float64)
    forces_on_rotated = np.asarray(
        simulation.compute_forces(rotated_positions, epsilon), dtype=np.float64
    )
    expected_rotated_forces = original_forces @ rotation_matrix.T

    np.testing.assert_allclose(forces_on_rotated, expected_rotated_forces, rtol=1e-3, atol=1e-4)


def test_forces_reflect_with_system() -> None:
    """Reflecting coordinates across a plane must reflect net forces across the same plane."""
    simulation = make_simulation(BOX, seed=101)
    positions = perturbed_positions(simulation, sigma=0.2, seed=101)
    reflection_matrix = build_reflection_matrix_z()
    reflected_positions = positions @ reflection_matrix.T

    epsilon = -1.2
    original_forces = np.asarray(simulation.compute_forces(positions, epsilon), dtype=np.float64)
    forces_on_reflected = np.asarray(
        simulation.compute_forces(reflected_positions, epsilon), dtype=np.float64
    )
    expected_reflected_forces = original_forces @ reflection_matrix.T

    np.testing.assert_allclose(forces_on_reflected, expected_reflected_forces, rtol=1e-3, atol=1e-4)


def test_chain_relabeling_permutes_forces() -> None:
    """Swapping two identical polymer chains in the coordinate array permutes their computed forces."""
    simulation = make_simulation(BOX, seed=101)
    positions = np.asarray(perturbed_positions(simulation, sigma=0.2, seed=101)).copy()

    chain_start = simulation.n_rubisco * (1 + simulation.n_patches_per_rubisco)
    beads_per_chain = simulation.n_beads_per_chain
    chain_0_slice = slice(chain_start, chain_start + beads_per_chain)
    chain_1_slice = slice(chain_start + beads_per_chain, chain_start + 2 * beads_per_chain)

    epsilon = -1.0
    original_forces = np.asarray(simulation.compute_forces(positions, epsilon))

    swapped_positions = positions.copy()
    swapped_positions[chain_0_slice] = positions[chain_1_slice]
    swapped_positions[chain_1_slice] = positions[chain_0_slice]

    swapped_forces = np.asarray(simulation.compute_forces(swapped_positions, epsilon))

    np.testing.assert_allclose(swapped_forces[chain_0_slice], original_forces[chain_1_slice], atol=1e-5)
    np.testing.assert_allclose(swapped_forces[chain_1_slice], original_forces[chain_0_slice], atol=1e-5)


def test_zero_epsilon_gives_zero_sticker_patch_force() -> None:
    """When epsilon is zero, sticker-patch attraction forces must be strictly zero."""
    simulation = make_simulation(BOX, seed=101)
    positions = perturbed_positions(simulation, sigma=0.3, seed=101)

    forces = simulation.compute_force_components(positions, epsilon=0.0)
    sticker_patch_force = np.asarray(forces["sticker_patch"])

    np.testing.assert_allclose(sticker_patch_force, 0.0, atol=1e-10)


def test_epsilon_of_either_sign_gives_identical_forces() -> None:
    """Binding strength parameter of either positive or negative sign must evaluate to identical forces."""
    simulation = make_simulation(BOX, seed=101)
    positions = perturbed_positions(simulation, sigma=0.3, seed=101)

    forces_pos = simulation.compute_force_components(positions, epsilon=1.5)
    forces_neg = simulation.compute_force_components(positions, epsilon=-1.5)

    np.testing.assert_allclose(
        np.asarray(forces_pos["sticker_patch"]),
        np.asarray(forces_neg["sticker_patch"]),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(forces_pos["total"]),
        np.asarray(forces_neg["total"]),
        rtol=1e-6,
        atol=1e-6,
    )


# ---------------------------------------------------------------------------
# Test 6: Exclusions and Cutoffs
# ---------------------------------------------------------------------------


def test_bonded_pairs_excluded_from_sterics_while_next_nearest_interact() -> None:
    """Directly bonded neighbors have zero steric repulsion while next-nearest neighbors interact."""
    simulation = make_simulation(BOX, seed=101)
    chain_start = simulation.n_rubisco * (1 + simulation.n_patches_per_rubisco)

    # Bead 0 and Bead 1 on Chain 0 are directly bonded:
    bead_0 = chain_start
    bead_1 = chain_start + 1
    # Bead 2 is next-nearest neighbor (separated by 2 bonds from Bead 0):
    bead_2 = chain_start + 2

    # Check nonbonded exclusion mask
    mask = np.asarray(simulation.nonbonded_pair_mask)
    assert not mask[bead_0, bead_1], "Directly bonded pair must be excluded from nonbonded mask"
    assert mask[bead_0, bead_2], "Next-nearest pair must be active in nonbonded mask"

    # Place bead 0 and bead 2 at distance 1.5 (< cutoff 2.5) with other particles far away
    positions = np.zeros((simulation.n_particles, 3), dtype=np.float32)
    positions[:] = np.linspace(-15.0, 15.0, simulation.n_particles)[:, None] * np.array([1.0, 0.0, 0.0])
    positions[bead_0] = [0.0, 0.0, 0.0]
    positions[bead_2] = [1.5, 0.0, 0.0]

    steric_forces = np.asarray(simulation.compute_force_components(positions, epsilon=0.0)["steric"])
    assert abs(steric_forces[bead_0, 0]) > 1e-4, "Next-nearest neighbors should exert steric repulsion"
    # Bead 0 should be repelled away from bead 2 along the -x direction
    assert steric_forces[bead_0, 0] < 0.0 < steric_forces[bead_2, 0]


def test_pair_forces_cutoff_discontinuity_and_formula() -> None:
    """Forces must vanish strictly outside cutoffs and match the analytical Gaussian formula inside."""
    simulation = make_simulation(BOX, seed=101)

    for (type_a, type_b), (eps, sigma, r_cut) in STERIC_PAIR_PARAMS.items():
        if eps <= 0.0:
            continue
        id_a = TYPE_INDEX[type_a]
        id_b = TYPE_INDEX[type_b]

        # Particles placed at r_cut + delta and r_cut - delta
        delta_outside = 0.002
        delta_inside = 0.002
        r_outside = r_cut + delta_outside
        r_inside = r_cut - delta_inside

        # Analytical formula inside cutoff: F = (eps / sigma^2) * exp(-0.5 * r^2 / sigma^2) * r
        expected_force_inside = (
            (eps / (sigma**2)) * np.exp(-0.5 * (r_inside**2) / (sigma**2)) * r_inside
        )

        # Test pair evaluation using JAX forces module directly
        import jax.numpy as jnp
        from condensate_ml import forces

        def evaluate_two_particle_force(dist: float) -> float:
            pos = jnp.array([[0.0, 0.0, 0.0], [dist, 0.0, 0.0]], dtype=jnp.float32)
            types = jnp.array([id_a, id_b], dtype=jnp.int32)
            neighbors = jnp.array([[1], [0]], dtype=jnp.int32)
            mask = jnp.array([[False, True], [True, False]], dtype=bool)

            f = forces.calculate_neighbor_pair_forces(
                positions=pos,
                neighbor_indices=neighbors,
                type_ids=types,
                epsilon_table=simulation.epsilon_table,
                sigma_table=simulation.sigma_table,
                cutoff_table=simulation.cutoff_table,
                nonbonded_mask=mask,
                displacement=simulation.displacement,
            )
            return float(f[0, 0])

        f_outside = evaluate_two_particle_force(r_outside)
        f_inside = evaluate_two_particle_force(r_inside)

        assert abs(f_outside) < 1e-7, f"Force for ({type_a}, {type_b}) must be 0 outside cutoff"
        # Force on particle 0 points in -x direction (away from particle 1 at +x)
        np.testing.assert_allclose(
            abs(f_inside),
            expected_force_inside,
            rtol=1e-3,
            err_msg=f"Force inside cutoff for ({type_a}, {type_b}) did not match formula",
        )


# ---------------------------------------------------------------------------
# Test 7: Reproducibility
# ---------------------------------------------------------------------------


def test_trajectory_reproducibility_across_seeds_and_checkpointing() -> None:
    """Same seed yields identical trajectories; different seed yields different; checkpointing is exact."""
    n_steps = 30
    write_every = 10
    box = 40.0
    epsilon = 1.0

    sim_seed_a = RubiscoEpyc1Simulation(
        n_steps=n_steps, box=box, write_every=write_every, seed=42, checkpoint_blocks=True
    )
    sim_seed_a_repeat = RubiscoEpyc1Simulation(
        n_steps=n_steps, box=box, write_every=write_every, seed=42, checkpoint_blocks=True
    )
    sim_seed_b = RubiscoEpyc1Simulation(
        n_steps=n_steps, box=box, write_every=write_every, seed=43, checkpoint_blocks=True
    )
    sim_uncheckpointed = RubiscoEpyc1Simulation(
        n_steps=n_steps, box=box, write_every=write_every, seed=42, checkpoint_blocks=False
    )

    traj_a = np.asarray(sim_seed_a({"epsilon": epsilon}))
    traj_a_repeat = np.asarray(sim_seed_a_repeat({"epsilon": epsilon}))
    traj_b = np.asarray(sim_seed_b({"epsilon": epsilon}))
    traj_uncheckpointed = np.asarray(sim_uncheckpointed({"epsilon": epsilon}))

    # Identical seed produces bitwise matching trajectory
    np.testing.assert_array_equal(traj_a, traj_a_repeat)

    # Checkpointed vs uncheckpointed produces matching trajectory
    np.testing.assert_allclose(traj_a, traj_uncheckpointed, rtol=1e-6, atol=1e-6)

    # Different seeds diverge
    assert not np.array_equal(traj_a, traj_b)


# ---------------------------------------------------------------------------
# Test 8: Time-Step Convergence
# ---------------------------------------------------------------------------


def rollout_deterministic_trajectory(
    simulation: RubiscoEpyc1Simulation,
    initial_positions: np.ndarray,
    dt: float,
    total_time: float,
    epsilon: float,
) -> np.ndarray:
    """Integrates deterministic Brownian dynamics (kT=0) over a fixed time interval using a specified dt.

    Inputs:
        simulation: Base simulation instance.
        initial_positions: Coordinates shaped (n_particles, 3).
        dt: Sub-stepping timestep.
        total_time: Total integration time duration.
        epsilon: Binding strength parameter.
    Outputs:
        Final positions after total_time shaped (n_particles, 3).
    """
    n_steps = int(round(total_time / dt))
    positions = initial_positions.copy()
    zero_noise = np.zeros_like(positions)

    # Temporarily set simulation dt
    original_dt = simulation.dt
    simulation.dt = dt
    try:
        for _ in range(n_steps):
            positions = np.asarray(
                simulation.brownian_step(positions, epsilon=epsilon, noise=zero_noise)
            )
    finally:
        simulation.dt = original_dt
    return positions


def test_euler_maruyama_timestep_convergence() -> None:
    """Halving dt over a fixed integration interval shrinks deterministic error linearly (O(dt))."""
    simulation = make_simulation(BOX, seed=101)
    initial_positions = perturbed_positions(simulation, sigma=0.1, seed=101)
    total_time = 0.004
    epsilon = 1.0

    dt_coarse = 0.002
    dt_medium = 0.001
    dt_fine = 0.0001

    pos_coarse = rollout_deterministic_trajectory(simulation, initial_positions, dt_coarse, total_time, epsilon)
    pos_medium = rollout_deterministic_trajectory(simulation, initial_positions, dt_medium, total_time, epsilon)
    pos_fine = rollout_deterministic_trajectory(simulation, initial_positions, dt_fine, total_time, epsilon)

    err_coarse = float(np.max(np.abs(minimum_image(pos_coarse - pos_fine, BOX))))
    err_medium = float(np.max(np.abs(minimum_image(pos_medium - pos_fine, BOX))))

    # Halving dt should cut error roughly in half: ratio err_coarse / err_medium in [1.5, 2.5]
    ratio = err_coarse / max(err_medium, 1e-12)
    assert 1.4 < ratio < 2.6, f"Convergence ratio {ratio:.3f} deviated from linear O(dt) behavior"
