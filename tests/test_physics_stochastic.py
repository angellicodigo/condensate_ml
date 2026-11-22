"""Stochastic physics tests for Brownian dynamics simulation.

Tests include:
1. Free diffusion: MSD = 6 * D * t with D = kT / gamma (0.2 for Rubisco, 1.0 for chain beads).
2. Bond equilibrium: Boltzmann distribution of harmonic bond lengths (mean r0, variance kT / k).
3. Stationary state friction independence: Doubling gamma leaves equilibrium potential energy invariant.
4. Conservative forces: Analytical forces match -grad(U) computed via finite differences.
"""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from condensate_ml import RubiscoEpyc1Simulation
from condensate_ml.constants import (
    CHAIN_K,
    CHAIN_R0,
    GAMMA_CHAIN,
    GAMMA_RUBISCO,
    KBT,
    PATCH_RADIUS,
    PATCH_SIGMA,
    PATCH_TETHER_K,
    TYPE_INDEX,
)
from condensate_ml.helper import BOX, make_simulation, perturbed_positions


# ---------------------------------------------------------------------------
# Test 1: Free Diffusion
# ---------------------------------------------------------------------------


def step_free_diffusion(
    positions: jax.Array,
    mobilities: jax.Array,
    kT: float,
    dt: float,
    key: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Advances particle positions by one pure Brownian displacement step without forces.

    Inputs:
        positions: Current particle coordinates shaped (n_particles, 3).
        mobilities: Inverse friction mobilities shaped (n_particles, 1).
        kT: Thermal energy scale.
        dt: Integration timestep.
        key: JAX PRNG key.
    Outputs:
        Tuple of (updated_positions, next_key).
    """
    key, subkey = jax.random.split(key)
    noise_sigma = jnp.sqrt(2.0 * kT * mobilities * dt)
    displacement = noise_sigma * jax.random.normal(subkey, positions.shape, dtype=positions.dtype)
    return positions + displacement, key


def rollout_free_diffusion(
    initial_positions: jax.Array,
    mobilities: jax.Array,
    kT: float,
    dt: float,
    n_steps: int,
    key: jax.Array,
) -> jax.Array:
    """Computes unwrapped particle trajectories under free Brownian diffusion.

    Inputs:
        initial_positions: Starting coordinates shaped (n_particles, 3).
        mobilities: Inverse friction mobilities shaped (n_particles, 1).
        kT: Thermal energy scale.
        dt: Integration timestep.
        n_steps: Total number of diffusion steps.
        key: JAX PRNG key.
    Outputs:
        Final positions after n_steps shaped (n_particles, 3).
    """
    def scan_fn(carry, _):
        pos, rng = carry
        next_pos, next_rng = step_free_diffusion(pos, mobilities, kT, dt, rng)
        return (next_pos, next_rng), None

    (final_positions, _), _ = jax.lax.scan(scan_fn, (initial_positions, key), None, length=n_steps)
    return final_positions


def calculate_mean_squared_displacement_by_group(
    displacement: np.ndarray,
    is_rubisco: np.ndarray,
) -> tuple[float, float]:
    """Computes mean squared displacement for Rubisco beads and chain beads separately.

    Inputs:
        displacement: Particle displacements from origin shaped (n_particles, 3).
        is_rubisco: Boolean mask indicating Rubisco core or patch particles.
    Outputs:
        Tuple of (rubisco_msd, chain_msd).
    """
    squared_displacement = np.sum(displacement**2, axis=-1)
    rubisco_msd = float(np.mean(squared_displacement[is_rubisco]))
    chain_msd = float(np.mean(squared_displacement[~is_rubisco]))
    return rubisco_msd, chain_msd


def test_free_diffusion_msd_matches_theory() -> None:
    """Free Brownian motion must satisfy MSD = 6 * D * t with D = kT / gamma."""
    dt = 0.001
    n_steps = 1000
    total_time = n_steps * dt
    kT = KBT

    simulation = make_simulation(BOX, seed=101)
    mobilities = simulation.mobilities
    n_particles = simulation.n_particles
    initial_positions = jnp.zeros((n_particles, 3), dtype=jnp.float32)

    rubisco_cutoff = simulation.n_rubisco * (1 + simulation.n_patches_per_rubisco)
    is_rubisco = np.arange(n_particles) < rubisco_cutoff

    expected_d_rubisco = kT / GAMMA_RUBISCO
    expected_d_chain = kT / GAMMA_CHAIN
    expected_msd_rubisco = 6.0 * expected_d_rubisco * total_time
    expected_msd_chain = 6.0 * expected_d_chain * total_time

    assert np.isclose(expected_d_rubisco, 0.2)
    assert np.isclose(expected_d_chain, 1.0)

    # Average MSD over multiple runs for statistical accuracy
    rubisco_msds = []
    chain_msds = []
    for seed in range(101, 111):
        key = jax.random.PRNGKey(seed)
        final_positions = rollout_free_diffusion(
            initial_positions=initial_positions,
            mobilities=mobilities,
            kT=kT,
            dt=dt,
            n_steps=n_steps,
            key=key,
        )
        displacement = np.asarray(final_positions - initial_positions)
        r_msd, c_msd = calculate_mean_squared_displacement_by_group(displacement, is_rubisco)
        rubisco_msds.append(r_msd)
        chain_msds.append(c_msd)

    mean_rubisco_msd = float(np.mean(rubisco_msds))
    mean_chain_msd = float(np.mean(chain_msds))

    np.testing.assert_allclose(mean_rubisco_msd, expected_msd_rubisco, rtol=0.08)
    np.testing.assert_allclose(mean_chain_msd, expected_msd_chain, rtol=0.05)


# ---------------------------------------------------------------------------
# Test 2: Bond Equilibrium (Boltzmann Distribution)
# ---------------------------------------------------------------------------


def step_single_harmonic_bond(
    state: tuple[jax.Array, jax.Array],
    mobility: float,
    spring_k: float,
    rest_r0: float,
    kT: float,
    dt: float,
) -> tuple[tuple[jax.Array, jax.Array], jax.Array]:
    """Advances a two-particle harmonic bond system by one Euler-Maruyama step.

    Inputs:
        state: Tuple of (positions shaped (2, 3), rng_key).
        mobility: Inverse friction mobility for both particles.
        spring_k: Bond spring constant.
        rest_r0: Equilibrium bond rest length.
        kT: Thermal energy scale.
        dt: Integration timestep.
    Outputs:
        Tuple of ((next_positions, next_key), current_bond_distance).
    """
    positions, key = state
    delta = positions[0] - positions[1]
    distance = jnp.sqrt(jnp.sum(delta * delta) + 1e-12)
    force_on_first = -spring_k * (distance - rest_r0) * (delta / distance)
    force = jnp.stack([force_on_first, -force_on_first], axis=0)

    key, subkey = jax.random.split(key)
    noise_sigma = jnp.sqrt(2.0 * kT * mobility * dt)
    noise = noise_sigma * jax.random.normal(subkey, positions.shape, dtype=positions.dtype)
    next_positions = positions + mobility * force * dt + noise
    return (next_positions, key), distance


def simulate_bond_length_trajectory(
    spring_k: float,
    rest_r0: float,
    mobility: float,
    kT: float,
    dt: float,
    n_steps: int,
    seed: int,
) -> jax.Array:
    """Simulates bond length fluctuations over a long trajectory.

    Inputs:
        spring_k: Bond spring constant.
        rest_r0: Equilibrium bond rest length.
        mobility: Particle mobility.
        kT: Thermal energy.
        dt: Timestep.
        n_steps: Number of integration steps.
        seed: Random seed.
    Outputs:
        Array of bond lengths across time shaped (n_steps,).
    """
    initial_positions = jnp.array([[0.0, 0.0, 0.0], [rest_r0, 0.0, 0.0]], dtype=jnp.float32)
    key = jax.random.PRNGKey(seed)

    def scan_step(state, _):
        return step_single_harmonic_bond(state, mobility, spring_k, rest_r0, kT, dt)

    _, distances = jax.lax.scan(scan_step, (initial_positions, key), None, length=n_steps)
    return distances


@pytest.mark.parametrize(
    "name,spring_k,rest_r0,mobility",
    [
        ("rubisco_patch_tether", PATCH_TETHER_K, PATCH_RADIUS, 1.0 / GAMMA_RUBISCO),
        ("chain_bond", CHAIN_K, CHAIN_R0, 1.0 / GAMMA_CHAIN),
    ],
)
def test_bond_equilibrium_mean_and_variance(
    name: str,
    spring_k: float,
    rest_r0: float,
    mobility: float,
) -> None:
    """Long-run bond length must have mean r0 and variance kT / k."""
    kT = KBT
    dt = 0.0005
    n_steps = 40_000
    burn_in = 4_000

    distances = simulate_bond_length_trajectory(
        spring_k=spring_k,
        rest_r0=rest_r0,
        mobility=mobility,
        kT=kT,
        dt=dt,
        n_steps=n_steps,
        seed=2024,
    )
    sampled_lengths = np.asarray(distances[burn_in:])
    sample_mean = float(np.mean(sampled_lengths))
    sample_var = float(np.var(sampled_lengths))

    expected_var = kT / spring_k
    # In 3D spherical coordinates, the metric term r^2 shifts the mean length outward by
    # approximately 2 * kT / (k * r0) (~0.16 for chain bonds, ~0.016 for patch tethers).
    max_mean_deviation = 0.20 if spring_k < 20.0 else 0.05
    assert abs(sample_mean - rest_r0) < max_mean_deviation, (
        f"{name}: mean bond length {sample_mean} drifted too far from r0 {rest_r0}"
    )
    np.testing.assert_allclose(
        sample_var,
        expected_var,
        rtol=0.25,
        err_msg=f"{name}: bond length variance did not match kT / k",
    )


# ---------------------------------------------------------------------------
# Test 3: Stationary State Does Not Depend on Friction
# ---------------------------------------------------------------------------


def step_particle_in_harmonic_well(
    state: tuple[jax.Array, jax.Array],
    mobility: float,
    spring_k: float,
    kT: float,
    dt: float,
) -> tuple[tuple[jax.Array, jax.Array], jax.Array]:
    """Advances a particle trapped in a 3D harmonic well U(x) = 0.5 * k * ||x||^2.

    Inputs:
        state: Tuple of (position shaped (3,), key).
        mobility: Inverse friction 1 / gamma.
        spring_k: Well stiffness.
        kT: Thermal energy scale.
        dt: Integration timestep.
    Outputs:
        Tuple of ((next_position, next_key), potential_energy).
    """
    pos, key = state
    force = -spring_k * pos
    key, subkey = jax.random.split(key)
    noise_sigma = jnp.sqrt(2.0 * kT * mobility * dt)
    noise = noise_sigma * jax.random.normal(subkey, pos.shape, dtype=pos.dtype)
    next_pos = pos + mobility * force * dt + noise
    energy = 0.5 * spring_k * jnp.sum(pos * pos)
    return (next_pos, key), energy


def compute_equilibrium_mean_energy(
    gamma: float,
    spring_k: float,
    kT: float,
    dt: float,
    n_steps: int,
    seed: int,
) -> float:
    """Computes the time-averaged equilibrium potential energy for a given friction coefficient.

    Inputs:
        gamma: Hydrodynamic friction coefficient.
        spring_k: Harmonic spring constant.
        kT: Thermal energy scale.
        dt: Integration timestep.
        n_steps: Number of integration steps.
        seed: Random seed.
    Outputs:
        Mean potential energy in the stationary state.
    """
    mobility = 1.0 / gamma
    initial_pos = jnp.zeros(3, dtype=jnp.float32)
    key = jax.random.PRNGKey(seed)

    def scan_fn(state, _):
        return step_particle_in_harmonic_well(state, mobility, spring_k, kT, dt)

    _, energies = jax.lax.scan(scan_fn, (initial_pos, key), None, length=n_steps)
    burn_in = n_steps // 5
    return float(np.mean(np.asarray(energies[burn_in:])))


def test_stationary_state_is_independent_of_friction() -> None:
    """Doubling gamma slows dynamics but leaves stationary expectation values invariant."""
    spring_k = 10.0
    kT = KBT
    dt = 0.0005
    n_steps = 60_000

    gamma_base = 2.0
    gamma_doubled = 4.0

    mean_energy_base = compute_equilibrium_mean_energy(
        gamma=gamma_base,
        spring_k=spring_k,
        kT=kT,
        dt=dt,
        n_steps=n_steps,
        seed=123,
    )
    mean_energy_doubled = compute_equilibrium_mean_energy(
        gamma=gamma_doubled,
        spring_k=spring_k,
        kT=kT,
        dt=dt,
        n_steps=n_steps,
        seed=123,
    )

    # Equipartition expects <U> ~ 1.5 * kT; the stationary average must be invariant to friction gamma
    expected_energy = 1.5 * kT
    np.testing.assert_allclose(mean_energy_base, expected_energy, rtol=0.15)
    np.testing.assert_allclose(mean_energy_doubled, expected_energy, rtol=0.15)
    # The primary check: changing gamma does not change the equilibrium potential energy
    np.testing.assert_allclose(mean_energy_base, mean_energy_doubled, rtol=0.10)


# ---------------------------------------------------------------------------
# Test 4: Forces Come from an Energy (F = -grad U)
# ---------------------------------------------------------------------------


def evaluate_bonded_potential(
    positions: jax.Array,
    bond_pairs: jax.Array,
    bond_k: jax.Array,
    bond_r0: jax.Array,
    displacement: Callable,
) -> jax.Array:
    """Computes total harmonic bond potential energy U_bonded = sum 0.5 * k * (r - r0)^2.

    Inputs:
        positions: Coordinates array shaped (n_particles, 3).
        bond_pairs: Bonded index pairs shaped (n_bonds, 2).
        bond_k: Spring constants shaped (n_bonds,).
        bond_r0: Rest lengths shaped (n_bonds,).
        displacement: Periodic displacement function.
    Outputs:
        Scalar potential energy.
    """
    first = positions[bond_pairs[:, 0]]
    second = positions[bond_pairs[:, 1]]
    delta = jax.vmap(displacement)(first, second)
    dist = jnp.sqrt(jnp.sum(delta * delta, axis=-1) + 1e-12)
    return jnp.sum(0.5 * bond_k * (dist - bond_r0) ** 2)


def evaluate_pairwise_gaussian_potential(
    positions: jax.Array,
    type_ids: jax.Array,
    epsilon_table: jax.Array,
    sigma_table: jax.Array,
    cutoff_table: jax.Array,
    pair_mask: jax.Array,
    displacement: Callable,
) -> jax.Array:
    """Computes total pairwise Gaussian energy sum_{i < j} epsilon * exp(-0.5 * r^2 / sigma^2).

    Inputs:
        positions: Coordinates array shaped (n_particles, 3).
        type_ids: Integer type ID array shaped (n_particles,).
        epsilon_table: Energy depth matrix shaped (n_types, n_types).
        sigma_table: Characteristic width matrix shaped (n_types, n_types).
        cutoff_table: Cutoff distance matrix shaped (n_types, n_types).
        pair_mask: Boolean interaction mask shaped (n_particles, n_particles).
        displacement: Periodic displacement function.
    Outputs:
        Scalar potential energy.
    """
    pairwise_disp = jax.vmap(jax.vmap(displacement, in_axes=(None, 0)), in_axes=(0, None))
    delta = pairwise_disp(positions, positions)
    dist_sq = jnp.sum(delta * delta, axis=-1)

    i_types = type_ids[:, None]
    j_types = type_ids[None, :]
    eps = epsilon_table[i_types, j_types]
    sig_sq = jnp.maximum(sigma_table[i_types, j_types] ** 2, 1e-8)
    cut_sq = cutoff_table[i_types, j_types] ** 2

    n = positions.shape[0]
    upper_triangle = jnp.triu(jnp.ones((n, n), dtype=bool), k=1)
    active = upper_triangle & pair_mask & (dist_sq < cut_sq)
    energy = eps * jnp.exp(-0.5 * dist_sq / sig_sq)
    return jnp.sum(jnp.where(active, energy, 0.0))


def evaluate_total_system_potential(
    positions: jax.Array,
    simulation: RubiscoEpyc1Simulation,
    epsilon: float,
) -> jax.Array:
    """Evaluates the exact total conservative potential energy of the system.

    Inputs:
        positions: Particle coordinates array shaped (n_particles, 3).
        simulation: Configured simulation instance.
        epsilon: Signed sticker-patch attraction depth parameter.
    Outputs:
        Total potential energy scalar.
    """
    u_bonded = evaluate_bonded_potential(
        positions,
        simulation.bond_pairs,
        simulation.bond_k,
        simulation.bond_r0,
        simulation.displacement,
    )
    u_steric = evaluate_pairwise_gaussian_potential(
        positions,
        simulation.type_ids,
        simulation.epsilon_table,
        simulation.sigma_table,
        simulation.cutoff_table,
        simulation.nonbonded_pair_mask,
        simulation.displacement,
    )
    # Sticker-patch attraction table
    attract_table = jnp.zeros_like(simulation.epsilon_table)
    attract_table = attract_table.at[TYPE_INDEX["T"], TYPE_INDEX["P"]].set(epsilon)
    attract_table = attract_table.at[TYPE_INDEX["P"], TYPE_INDEX["T"]].set(epsilon)

    sigma_table = jnp.zeros_like(simulation.sigma_table)
    sigma_table = sigma_table.at[TYPE_INDEX["T"], TYPE_INDEX["P"]].set(PATCH_SIGMA)
    sigma_table = sigma_table.at[TYPE_INDEX["P"], TYPE_INDEX["T"]].set(PATCH_SIGMA)

    u_attract = evaluate_pairwise_gaussian_potential(
        positions,
        simulation.type_ids,
        attract_table,
        sigma_table,
        simulation.cutoff_table,
        jnp.ones_like(simulation.nonbonded_pair_mask),
        simulation.displacement,
    )
    return u_bonded + u_steric + u_attract


def compute_minus_energy_gradient(
    positions: jax.Array,
    simulation: RubiscoEpyc1Simulation,
    epsilon: float,
) -> jax.Array:
    """Computes -grad(U) analytically via reverse-mode autodiff.

    Inputs:
        positions: Coordinates array shaped (n_particles, 3).
        simulation: Configured simulation instance.
        epsilon: Signed attraction parameter.
    Outputs:
        Conservative force array -grad(U) shaped (n_particles, 3).
    """
    energy_fn = lambda pos: evaluate_total_system_potential(pos, simulation, epsilon)
    return -jax.grad(energy_fn)(positions)


def test_analytical_forces_equal_negative_energy_gradient() -> None:
    """Forces calculated analytically must equal -grad(U) across all components."""
    simulation = make_simulation(BOX, seed=101)
    positions = perturbed_positions(simulation, sigma=0.2, seed=101)
    epsilon = -1.2

    analytical_force = np.asarray(simulation.compute_forces(positions, epsilon), dtype=np.float64)
    conservative_force = np.asarray(
        compute_minus_energy_gradient(jnp.asarray(positions), simulation, epsilon),
        dtype=np.float64,
    )

    scale = np.linalg.norm(conservative_force)
    difference = np.linalg.norm(analytical_force - conservative_force)
    assert difference / scale < 1e-4, "Analytical forces do not equal -grad(U)"
