"""Differentiable Brownian Dynamics JAX MD simulation for Rubisco-EPYC1 condensates."""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax_md import partition, space

from .constants import (
    CHAIN_K,
    CHAIN_R0,
    DT,
    GAMMA_CHAIN,
    GAMMA_RUBISCO,
    KBT,
    N_BEADS,
    N_C,
    N_R,
    PATCH_DIRS,
    PATCH_RADIUS,
    PATCH_SIGMA,
    PATCH_TETHER_K,
    PATCHES,
    STERIC_PAIR_PARAMS,
    STICKER_INDEX,
    TYPE_INDEX,
)
from . import forces
from .initial_snapshot import generate_initial_coordinates
from .neighbor_list import allocate_neighbors, shift_to_neighbor_list_coordinates
from .output import write_analysis_outputs, write_gsd
from .topology import (
    build_bond_data,
    build_bond_exclusion_mask,
    build_particle_labels,
    build_particle_mobilities,
    build_particle_type_ids,
    build_steric_parameter_tables,
)

__all__ = ["RubiscoEpyc1Simulation"]


class RubiscoEpyc1Simulation:
    """Differentiable Brownian Dynamics JAX MD simulation class for Rubisco-EPYC1 condensates."""

    def __init__(
        self,
        n_steps: int,
        dt: float = DT,
        box: float = 40.0,
        write_every: int = 100,
        seed: int = 42,
        neighbor_buffer: float = 0.4,
        dtype: Any = jnp.float32,
        checkpoint_blocks: bool = True,
        neighbor_extra_capacity: int = 32,
        max_capacity_retries: int = 3,
        output_dir: str | Path | None = None,
        output_name: str = "trajectory",
        kT: float = KBT,
    ) -> None:
        """Initializes simulation parameters, static topology, interaction tables, and compiled rollout code.

        Inputs:
            n_steps: Total number of Brownian dynamics steps to simulate.
            dt: Brownian dynamics integration timestep.
            box: Cubic periodic box length.
            write_every: Timestep interval between saved trajectory frames.
            seed: Random seed for coordinate generation and thermal noise.
            neighbor_buffer: Skin buffer distance for neighbor-list updates.
            dtype: Target floating-point precision type.
            checkpoint_blocks: Whether to checkpoint scan intervals during reverse autodiff.
            neighbor_extra_capacity: Extra neighbor slots allocated per particle.
            max_capacity_retries: Maximum neighbor list allocation retries upon overflow.
            output_dir: Optional path to directory for host-side output artifacts.
            output_name: Base name for generated trajectory and analysis files.
            kT: Thermal energy scale.
        Outputs:
            None.
        """
        if n_steps < 0 or write_every < 1 or n_steps % write_every:
            raise ValueError("n_steps must be nonnegative and divisible by write_every")
        if kT < 0:
            raise ValueError("kT must be nonnegative")
        self.n_steps = int(n_steps)
        self.dt = float(dt)
        self.kT = float(kT)
        self.box = float(box)
        self.write_every = int(write_every)
        self.seed = int(seed)
        self.dtype = dtype
        self._extra_capacity = int(neighbor_extra_capacity)
        self._max_capacity_retries = int(max_capacity_retries)
        self.last_neighbor_overflow = False
        self.output_dir = None if output_dir is None else Path(output_dir)
        self.output_name = str(output_name)
        self.n_rubisco = N_R
        self.n_patches_per_rubisco = PATCHES
        self.n_chains = N_C
        self.n_beads_per_chain = N_BEADS
        self.n_particles = N_R + N_R * PATCHES + N_C * N_BEADS
        self.sticker_local_indices = tuple(STICKER_INDEX)
        self.patch_dirs = np.asarray(PATCH_DIRS)

        particle_labels = build_particle_labels(
            n_rubisco=self.n_rubisco,
            n_patches_per_rubisco=self.n_patches_per_rubisco,
            n_chains=self.n_chains,
            n_beads_per_chain=self.n_beads_per_chain,
            sticker_indices=self.sticker_local_indices,
        )
        self.type_ids = build_particle_type_ids(
            particle_labels,
            indexByTypeName=TYPE_INDEX,
            dtype=jnp.int32,
        )
        self.mobilities = build_particle_mobilities(
            n_particles=self.n_particles,
            n_rubisco=self.n_rubisco,
            n_patches_per_rubisco=self.n_patches_per_rubisco,
            gamma_rubisco=GAMMA_RUBISCO,
            gamma_chain=GAMMA_CHAIN,
            dtype=self.dtype,
        )
        self.bond_pairs, self.bond_k, self.bond_r0, self.bond_type_ids = build_bond_data(
            n_rubisco=self.n_rubisco,
            n_patches_per_rubisco=self.n_patches_per_rubisco,
            n_chains=self.n_chains,
            n_beads_per_chain=self.n_beads_per_chain,
            chain_k=CHAIN_K,
            chain_r0=CHAIN_R0,
            patch_tether_k=PATCH_TETHER_K,
            patch_radius=PATCH_RADIUS,
            dtype=self.dtype,
        )
        self.nonbonded_pair_mask = build_bond_exclusion_mask(self.n_particles, self.bond_pairs)
        self.epsilon_table, self.sigma_table, self.cutoff_table = build_steric_parameter_tables(
            steric_pair_params=STERIC_PAIR_PARAMS,
            indexByTypeName=TYPE_INDEX,
            dtype=dtype,
        )

        self.displacement, _shift = space.periodic(self.box)
        max_cutoff = float(np.max(self.cutoff_table))

        self.neighbor_fn = partition.neighbor_list(
            self.displacement,
            self.box,
            max_cutoff,
            dr_threshold=neighbor_buffer,
            format=partition.NeighborListFormat.Dense,
        )
        self.compiled_simulate_fn = jax.jit(self.simulate(checkpoint_blocks))

    def init(self, fields: Mapping[str, Any] | None = None) -> jax.Array:
        """Returns the initial particle coordinates for a simulation rollout.

        Inputs:
            fields: Optional dictionary optionally containing custom 'positions_0'.
        Outputs:
            Initial coordinate array shaped (n_particles, 3).
        """
        fields = {} if fields is None else fields
        if "positions_0" in fields:
            return jnp.asarray(fields["positions_0"], dtype=self.dtype)
        return jnp.asarray(
            generate_initial_coordinates(
                self.box,
                self.patch_dirs,
                self.seed,
                self.n_rubisco,
                self.n_patches_per_rubisco,
                self.n_chains,
                self.n_beads_per_chain,
                PATCH_RADIUS,
                CHAIN_R0,
            ),
            dtype=self.dtype,
        )

    def simulate(self, checkpoint_blocks: bool):
        """Constructs the pure JAX-compilable scan function advancing the Brownian dynamics trajectory.

        Inputs:
            checkpoint_blocks: Whether to checkpoint scan intervals during reverse autodiff.
        Outputs:
            Rollout callable returning a tuple of (trajectory_frames, did_overflow).
        """
        displacement = self.displacement
        dt = self.dt
        kT = self.kT
        block_size = self.write_every
        n_blocks = self.n_steps // self.write_every
        mobility = self.mobilities
        epsilon_table = self.epsilon_table
        sigma_table = self.sigma_table
        cutoff_table = self.cutoff_table
        type_ids = self.type_ids
        exclusions = self.nonbonded_pair_mask
        bond_pairs, bond_k, bond_r0 = self.bond_pairs, self.bond_k, self.bond_r0
        box = self.box

        def wrap_centered(positions):
            """Wraps coordinates into the origin-centered periodic box interval.

            Inputs:
                positions: Particle coordinates array.
            Outputs:
                Periodic wrapped coordinates mapped to [-box / 2, box / 2).
            """
            return (positions + box * 0.5) % box - box * 0.5

        def one_step(state, _):
            """Advances the simulation state by one overdamped Euler-Maruyama Brownian dynamics step.

            Inputs:
                state: State tuple of (positions, neighbors, key, epsilon).
                _: Unused scan step index.
            Outputs:
                Tuple of (updated_state, None).
            """
            positions, neighbors, key, epsilon = state
            neighbors = neighbors.update(shift_to_neighbor_list_coordinates(positions, box))
            key, noise_key = jax.random.split(key)

            total = forces.calculate_total_force(
                positions, neighbors.idx, type_ids, epsilon_table, sigma_table,
                cutoff_table, exclusions, bond_pairs, bond_k, bond_r0,
                epsilon, PATCH_SIGMA, displacement,
            )
            noise = jnp.sqrt(2.0 * kT * mobility * dt) * jax.random.normal(
                noise_key, positions.shape, dtype=positions.dtype
            )
            next_positions = wrap_centered(positions + mobility * total * dt + noise)
            return (next_positions, neighbors, key, epsilon), None

        def integrate_block(state, _):
            """Advances the simulation through one output block interval.

            Inputs:
                state: State tuple passed through the timestep scan.
                _: Unused outer scan index.
            Outputs:
                Tuple of (updated_state, final_block_positions).
            """
            state, _ = jax.lax.scan(one_step, state, None, length=block_size)
            return state, state[0]

        if checkpoint_blocks:
            integrate_block = jax.checkpoint(integrate_block)

        def rollout(positions, epsilon, key, neighbors):
            """Runs all observation blocks and concatenates the resulting trajectory frames.

            Inputs:
                positions: Starting coordinates array shaped (n_particles, 3).
                epsilon: Signed sticker-patch attraction energy parameter.
                key: JAX PRNG key for thermal noise.
                neighbors: Preallocated dense neighbor list object.
            Outputs:
                Tuple of (trajectory_frames, overflow_flag).
            """
            state = (positions, neighbors, key, epsilon)
            state, block_frames = jax.lax.scan(integrate_block, state, None, length=n_blocks)
            return jnp.concatenate((positions[None], block_frames), axis=0), state[1].did_buffer_overflow

        return rollout

    # FOR DEBUGGING
    def compute_forces(self, positions: Any, epsilon: Any) -> jax.Array:
        """Evaluates the total net force acting on all particles for a given configuration.

        Inputs:
            positions: Particle coordinates shaped (n_particles, 3).
            epsilon: Positive sticker-patch binding strength.
        Outputs:
            Total force array shaped (n_particles, 3).
        """
        positions = jnp.asarray(positions, dtype=self.dtype)
        epsilon = -jnp.abs(jnp.asarray(epsilon, dtype=self.dtype))
        shifted = shift_to_neighbor_list_coordinates(positions, self.box)
        neighbors = allocate_neighbors(
            self.neighbor_fn, positions, self.box, self._extra_capacity
        ).update(shifted)
        return forces.calculate_total_force(
            positions,
            neighbors.idx,
            self.type_ids,
            self.epsilon_table,
            self.sigma_table,
            self.cutoff_table,
            self.nonbonded_pair_mask,
            self.bond_pairs,
            self.bond_k,
            self.bond_r0,
            epsilon,
            PATCH_SIGMA,
            self.displacement,
        )

    # FOR DEBUGGING
    def brownian_step(
        self,
        positions: Any,
        epsilon: Any,
        noise: Any | None = None,
    ) -> jax.Array:
        """Advances positions by one Euler-Maruyama Brownian dynamics step with optional fixed noise.

        Inputs:
            positions: Starting particle coordinates shaped (n_particles, 3).
            epsilon: Positive sticker-patch binding strength.
            noise: Optional precomputed displacement noise array shaped (n_particles, 3).
        Outputs:
            Updated particle coordinates shaped (n_particles, 3).
        """
        positions = jnp.asarray(positions, dtype=self.dtype)
        force = self.compute_forces(positions, epsilon)
        displacement = jnp.zeros_like(positions) if noise is None else jnp.asarray(noise, dtype=self.dtype)
        # Performs the same thing to wrap_centered()
        updated_positions = positions + self.mobilities * force * self.dt + displacement
        return (updated_positions + self.box * 0.5) % self.box - self.box * 0.5

    # FOR DEBUGGING
    def compute_force_components(
        self, positions: Any, epsilon: Any
    ) -> dict[str, jax.Array]:
        """Evaluates steric, sticker-patch attraction, bonded, and total forces individually.

        Inputs:
            positions: Particle coordinates shaped (n_particles, 3).
            epsilon: Positive sticker-patch binding strength.
        Outputs:
            Dictionary containing 'steric', 'sticker_patch', 'bonded', and 'total' force arrays.
        """
        positions = jnp.asarray(positions, dtype=self.dtype)
        epsilon = -jnp.abs(jnp.asarray(epsilon, dtype=self.dtype))
        shifted = shift_to_neighbor_list_coordinates(positions, self.box)
        neighbors = allocate_neighbors(
            self.neighbor_fn, positions, self.box, self._extra_capacity
        ).update(shifted)
        steric, attraction, bonded = forces.calculate_force_components(
            positions,
            neighbors.idx,
            self.type_ids,
            self.epsilon_table,
            self.sigma_table,
            self.cutoff_table,
            self.nonbonded_pair_mask,
            self.bond_pairs,
            self.bond_k,
            self.bond_r0,
            epsilon,
            PATCH_SIGMA,
            self.displacement,
        )
        return {
            "steric": steric,
            "sticker_patch": attraction,
            "bonded": bonded,
            "total": steric + attraction + bonded,
        }

    def __call__(self, fields: Mapping[str, Any]) -> jax.Array:
        """Executes the differentiable simulation rollout and returns sampled trajectory frames.

        Note on gradient sign: The simulation uses -abs(epsilon) internally, so the gradient with
        respect to a positive epsilon has the opposite sign to a negative epsilon. When training
        a neural network or optimizing binding affinity, parameterize epsilon as positive binding
        strength so that d(contacts)/d(epsilon) > 0.

        Inputs:
            fields: Mapping containing required 'epsilon' and optional 'positions_0'.
        Outputs:
            Sampled trajectory array shaped (n_frames, n_particles, 3).
        """
        epsilon = fields.get("epsilon")
        if epsilon is None:
            raise KeyError("JAX MD stepper requires fields['epsilon']")
        positions = self.init(fields)
        binding_strength = jnp.abs(jnp.asarray(epsilon, dtype=self.dtype))
        epsilon = -binding_strength
        key = jax.random.PRNGKey(self.seed)
        for attempt in range(self._max_capacity_retries + 1):
            neighbors = allocate_neighbors(
                self.neighbor_fn, positions, self.box, self._extra_capacity
            )
            frames, overflow = self.compiled_simulate_fn(positions, epsilon, key, neighbors)
            if isinstance(overflow, jax.core.Tracer):
                break
            self.last_neighbor_overflow = bool(overflow)
            if not self.last_neighbor_overflow:
                break
            if attempt < self._max_capacity_retries:
                self._extra_capacity = max(2 * self._extra_capacity, 16)
        else:
            warnings.warn(
                "Neighbor list overflowed even after growing its capacity; some interacting pairs "
                "were dropped. Increase neighbor_extra_capacity.",
                RuntimeWarning,
                stacklevel=2,
            )

        if self.output_dir is not None and not isinstance(frames, jax.core.Tracer):
            self.write_outputs(frames, binding_strength)

        return frames

    def write_outputs(self, trajectory: jax.Array, epsilon: Any) -> None:
        """Writes host-side output files and analysis artifacts for a trajectory.

        Inputs:
            trajectory: Sampled rollout frames array.
            epsilon: Attraction parameter value recorded in analysis output.
        Outputs:
            None.
        """

        def _artifact_directories() -> dict[str, Path]:
            """Creates and returns device-specific output directories for simulation artifacts.

            Inputs:
                None.
            Outputs:
                Dictionary mapping artifact names ('traj', 'xyz', 'data', 'plots') to directory Paths.
            """
            if self.output_dir is None:
                raise ValueError("output_dir is not configured")
            device_label = "gpu" if any(device.platform == "gpu" for device in jax.devices()) else "cpu"
            directories = {
                "traj": self.output_dir / f"traj_jaxmd_{device_label}",
                "xyz": self.output_dir / f"xyz_jaxmd_{device_label}",
                "data": self.output_dir / f"data_jaxmd_{device_label}",
                "plots": self.output_dir / f"plots_jaxmd_{device_label}",
            }
            for directory in directories.values():
                directory.mkdir(parents=True, exist_ok=True)
            return directories

        directories = _artifact_directories()
        gsd_path = write_gsd(
            path=directories["traj"] / f"{self.output_name}.gsd",
            trajectory=trajectory,
            box=self.box,
            particle_type_ids=self.type_ids,
            bond_pairs=self.bond_pairs,
            write_every=self.write_every,
            bond_type_ids=self.bond_type_ids,
        )
        write_analysis_outputs(
            gsd_path=gsd_path,
            xyz_path=directories["xyz"] / f"{self.output_name}.xyz",
            csv_path=directories["data"] / f"{self.output_name}.csv",
            plots_dir=directories["plots"],
            output_name=self.output_name,
            dt=self.dt,
            epsilon=float(np.asarray(epsilon)),
            n_rubisco=self.n_rubisco,
            n_patches_per_rubisco=self.n_patches_per_rubisco,
            n_chains=self.n_chains,
            n_beads_per_chain=self.n_beads_per_chain,
            sticker_indices=self.sticker_local_indices,
            contact_cutoff=float(self.cutoff_table[TYPE_INDEX["P"], TYPE_INDEX["T"]]),
        )
