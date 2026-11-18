"""Differentiable Brownian Dynamics JAX MD prototype for Rubisco-EPYC1 condensates.

Importing this module has no side effects. Typical use from an ML training script::

    import jax
    import jax.numpy as jnp
    from condensate_ml import RubiscoEpyc1Simulation

    stepper = RubiscoEpyc1Simulation(n_steps=2000, box=35.0,
                                     write_every=100, seed=0)
    epsilon = jax.nn.softplus(features)        # positive binding strength
    trajectory = stepper({"epsilon": epsilon}) # (T, 792, 3), differentiable
    loss = loss_fn(trajectory)
    loss, grad = jax.value_and_grad(loss_fn)(epsilon)

The public ``epsilon`` parameter is the positive sticker-patch binding strength.
It is converted internally to the negative Gaussian well depth required for
attraction by the HOOMD Gaussian-potential sign convention.

This module is a separate JAX MD implementation of the PyTorch/NEFI stepper. It uses
JAX MD's cell-list neighbor list and ``jax.lax.scan`` rather than NEFI's generic Python
time loop.
"""

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
    CHAIN_R0,
    DT,
    KBT,
    N_BEADS,
    N_C,
    N_R,
    PATCH_DIRS,
    PATCH_RADIUS,
    PATCH_RCUT,
    PATCH_SIGMA,
    PATCHES,
    STICKER_INDEX,
)
from . import forces
from .post_simulation_analysis import write_analysis_outputs
from .initial_snapshot import generate_initial_coordinates
from .neighbor_list import allocate_neighbors, shift_to_neighbor_list_coordinates
from .output import write_gsd
from .topology import (
    build_bond_data,
    build_bond_exclusion_mask,
    build_particle_indices,
    build_particle_labels,
    build_particle_mobilities,
    build_particle_type_ids,
    build_steric_parameter_tables,
)

__all__ = ["RubiscoEpyc1Simulation"]


class RubiscoEpyc1Simulation:
    """Differentiable Brownian Dynamics JAX MD stepper for Rubisco-EPYC1 condensates.

    The stepper object holds only fixed topology and constants. The learnable / swept
    quantity is passed per call as ``fields["epsilon"]`` (scalar JAX array), so one
    stepper can be reused across optimisation iterations.

    Output: trajectory array (n_frames, n_particles, 3), frames at step 0 and every
    ``write_every`` steps. Particle order is R (24), P (192), then 48 chains of
    12 beads (S/T alternating).

    Stochasticity: every step splits an explicit JAX PRNG key to draw fresh Gaussian
    noise. Reusing the same seed reproduces the same rollout. Gradients are obtained
    with ``jax.grad`` or ``jax.value_and_grad``.

    ``write_every`` must divide ``n_steps`` in this prototype.  The rollout is
    compiled lazily on the first call.  Subsequent calls with the same shapes,
    dtype, device, and timestep settings reuse the compiled executable.

    If ``output_dir`` is supplied, ordinary forward calls write sampled frames
    and analysis artifacts to device-specific subdirectories. GSD writing is a
    host-side side effect and is skipped inside JAX transformations.
    """

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
        """Initialize topology, force tables, neighbor lists, and rollout code.

        Input:
            ``n_steps``: Number of Brownian dynamics steps per rollout.
            ``dt``: Brownian dynamics timestep.
            ``kT``: Thermal energy in reduced units. The reference model uses
                ``kT=1``.
            ``box``: Cubic periodic box length.
            ``write_every``: Steps between returned trajectory frames.
            ``seed``: Seed for initial coordinates and Brownian noise.
            ``neighbor_buffer``: Cell-list skin distance.
            ``dtype``: Floating-point dtype used by the JAX simulation.
            ``checkpoint_blocks``: Whether to rematerialize scan blocks while
                differentiating.
            ``neighbor_extra_capacity``: Spare neighbor slots per particle.
            ``max_capacity_retries``: Maximum neighbor-capacity retries.
            ``output_dir``: Optional directory for host-side output files.
            ``output_name``: Base name for generated artifacts.
        Output:
            None. The instance stores fixed topology and a lazily compiled
            rollout function.
        """
        # Unlike the PyTorch/NEFI implementation, this class is not a
        # TimeStepper subclass. The rollout below is expressed directly with
        # jax.lax.scan so JAX can compile blocks of timesteps together.
        #
        # n_steps       - total Brownian dynamics steps to unroll.
        # dt            - Brownian dynamics timestep.
        # box           - origin-centered periodic simulation box length.
        # write_every   - record a trajectory frame at this interval.
        # seed          - JAX PRNG seed for initial coordinates and noise.
        # neighbor_buffer - cell-list skin distance, analogous to HOOMD's buffer.
        # checkpoint_blocks - rematerialize scan blocks during reverse-mode autodiff.
        # neighbor_extra_capacity - spare neighbor slots per particle beyond what the
        #                   initial layout needs. Condensed clusters have far more
        #                   neighbors than the starting lattice (about 43 vs 22 for
        #                   BOX=29, EPS=2.5); pairs beyond the capacity are dropped silently.
        # max_capacity_retries - a concrete (non-differentiated) call that overflows is
        #                   repeated with doubled capacity up to this many times.
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
        self.last_neighbor_overflow = False   # set by every call that is not being differentiated
        self.output_dir = None if output_dir is None else Path(output_dir)
        self.output_name = str(output_name)
        # Variables defined in constants.py and used by every instance of the
        # Rubisco-EPYC1 JAX MD stepper. Particle order is Rubisco centers,
        # Rubisco patches, then EPYC1 chain beads.
        self.n_rubisco = N_R
        self.n_patches_per_rubisco = PATCHES
        self.n_chains = N_C
        self.n_beads_per_chain = N_BEADS
        self.n_particles = N_R + N_R * PATCHES + N_C * N_BEADS
        self.sticker_local_indices = tuple(STICKER_INDEX)
        self.patch_dirs = np.asarray(PATCH_DIRS)

        # Fixed topology and per-particle properties are created once during
        # construction and reused by every compiled rollout.
        #
        # build_particle_labels() returns one label per particle:
        #     R: Rubisco core center
        #     P: Rubisco surface patch
        #     S: EPYC1 spacer bead
        #     T: EPYC1 sticker bead
        #
        # build_particle_type_ids() converts those labels to integer IDs:
        #     R=0, P=1, S=2, T=3
        self.type_ids = build_particle_type_ids()
        self.particle_type_ids = self.type_ids

        # build_mobilities() returns mobility 1/gamma for every particle as an
        # (n_particles, 1) array. The trailing dimension broadcasts over x/y/z.
        #
        #     GAMMA_RUBISCO = brownian.gamma["R"] = brownian.gamma["P"] = 5.0
        #     GAMMA_CHAIN   = brownian.gamma["S"] = brownian.gamma["T"] = 1.0
        self.mobilities = build_particle_mobilities(self.n_particles, self.dtype)

        # build_bonds() packs two topology groups into one table:
        #     chain:        n_chains * (n_beads_per_chain - 1) = 528 bonds
        #     patch_tether: n_rubisco * n_patches_per_rubisco = 192 bonds
        # This packed representation avoids a separate force/scatter pass for
        # each bond type.
        self.bond_pairs, self.bond_k, self.bond_r0 = build_bond_data(self.dtype)

        # build_steric_parameter_tables() builds the 4-by-4 type-level tables
        # for the Gaussian steric forces. Unlike the PyTorch implementation,
        # these stay at type-level size and are gathered for neighbor pairs:
        #     epsilon_table[type_i, type_j] -> pair repulsion strength
        #     sigma_table[type_i, type_j]   -> pair width
        #     cutoff_table[type_i, type_j]  -> pair cutoff
        self.nonbonded_pair_mask = build_bond_exclusion_mask(self.n_particles, self.bond_pairs)
        self.epsilon_table, self.sigma_table, self.cutoff_table = build_steric_parameter_tables(dtype)

        # Global particle indices used by analysis and by the sticker-patch
        # interaction. There are 192 patches and 6 stickers per chain.
        (
            self.patch_indices,
            self.sticker_indices,
            self.patch_rubisco_ids,
            self.sticker_chain_ids,
        ) = build_particle_indices()
        self.particle_labels = build_particle_labels()

        # JAX MD supplies the periodic minimum-image displacement function.
        # Its displacement function operates on individual vectors; forces.py
        # applies jax.vmap where arrays of vectors are needed.
        self.displacement, _shift = space.periodic(self.box)
        max_cutoff = max(float(np.max(self.cutoff_table)), PATCH_RCUT)

        # A cell-list neighbor list avoids evaluating every N-by-N particle
        # pair.  The buffer (skin) lets JAX MD reuse the list while particles
        # move, rebuilding it only when necessary.
        self.neighbor_fn = partition.neighbor_list(
            self.displacement,
            self.box,
            max_cutoff,
            dr_threshold=neighbor_buffer,
            format=partition.NeighborListFormat.Dense,
        )
        # Compilation is lazy: the first call pays the XLA compilation cost;
        # compatible later calls reuse the compiled executable.
        self._compiled_rollout = jax.jit(self._make_rollout(checkpoint_blocks))

    def _allocate_neighbors(self, positions: jax.Array):
        """Allocate neighbor-list storage for one configuration.

        Input:
            ``positions``: Current coordinates shaped ``(n_particles, 3)``.
        Output:
            A JAX MD dense neighbor-list object with spare capacity.
        """
        return allocate_neighbors(
            self.neighbor_fn, positions, self.box, self._extra_capacity
        )

    def init(self, fields: Mapping[str, Any] | None = None) -> jax.Array:
        """Return initial coordinates for a rollout.

        Input:
            ``fields``: Optional mapping. ``fields['positions_0']`` overrides
                the generated initial snapshot when supplied.
        Output:
            Float array shaped ``(n_particles, 3)``.
        """
        fields = {} if fields is None else fields
        if "positions_0" in fields:
            return jnp.asarray(fields["positions_0"], dtype=self.dtype)
        return jnp.asarray(
            generate_initial_coordinates(
                self.box,
                self.patch_dirs,
                self.seed,
                N_R,
                PATCHES,
                N_C,
                N_BEADS,
                PATCH_RADIUS,
                CHAIN_R0,
            ),
            dtype=self.dtype,
        )

    def _make_rollout(self, checkpoint_blocks: bool):
        """Create the pure function that JAX compiles for the complete rollout.

        The outer scan advances one observation block at a time.  The inner
        scan performs ``write_every`` Brownian steps.  This avoids retaining all
        intermediate positions while still returning frame 0 and each output
        frame.  ``jax.checkpoint`` rematerializes a block during reverse-mode
        differentiation when ``checkpoint_blocks`` is enabled.

        Input:
            ``checkpoint_blocks``: Whether to checkpoint each output interval.
        Output:
            A pure JAX rollout function that returns sampled frames and a
            neighbor-list overflow flag.
        """
        neighbor_fn = self.neighbor_fn
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
            """Wrap coordinates into the origin-centered periodic box."""
            return (positions + box * 0.5) % box - box * 0.5

        def one_step(carry, _):
            """Advance one overdamped Euler-Maruyama Brownian step.

            Input:
                ``carry``: ``(positions, neighbors, key, epsilon)`` state tuple.
                ``_``: Unused scan index.
            Output:
                Updated carry and ``None`` as the per-step scan output.
            """
            positions, neighbors, key, epsilon = carry
            neighbors = neighbors.update(shift_to_neighbor_list_coordinates(positions, box))
            key, noise_key = jax.random.split(key)

            # Compute the three physical force contributions. The Gaussian
            # attraction uses the signed internal epsilon. Public callers pass
            # a positive binding strength, which is converted to a negative
            # Gaussian depth before reaching this compiled function.
            total = forces.calculate_total_force(
                positions, neighbors.idx, type_ids, epsilon_table, sigma_table,
                cutoff_table, exclusions, bond_pairs, bond_k, bond_r0,
                epsilon, PATCH_SIGMA, PATCH_RCUT, displacement,
            )
            # Einstein relation: D_i = kT / gamma_i = kT * mobility_i.
            # Euler-Maruyama for overdamped Langevin dynamics therefore adds
            # sqrt(2 * D_i * dt) times a standard normal vector.
            noise = jnp.sqrt(2.0 * kT * mobility * dt) * jax.random.normal(
                noise_key, positions.shape, dtype=positions.dtype
            )
            # Overdamped Langevin Euler-Maruyama update:
            # x_new = x + (force / gamma) * dt + sqrt(2 D dt) * eta.
            next_positions = wrap_centered(positions + mobility * total * dt + noise)
            return (next_positions, neighbors, key, epsilon), None

        def integrate_block(carry, _):
            """Advance one output interval and return its final coordinates.

            The inner scan replaces the Python timestep loop used by the
            PyTorch/NEFI TimeStepper. The outer scan returns only sampled
            frames, rather than every intermediate state.

            Input:
                ``carry``: State tuple passed through the timestep scan.
                ``_``: Unused outer-scan index.
            Output:
                Updated state and the final coordinates for this block.
            """
            carry, _ = jax.lax.scan(one_step, carry, None, length=block_size)
            return carry, carry[0]

        if checkpoint_blocks:
            integrate_block = jax.checkpoint(integrate_block)

        def rollout(positions, epsilon, key, neighbors):
            """Run all output blocks and return sampled frames.

            Input:
                ``positions``: Initial coordinates.
                ``epsilon``: Positive sticker-patch binding strength.
                ``key``: JAX PRNG key for Brownian noise.
                ``neighbors``: Preallocated dense neighbor list.
            Output:
                ``(frames, overflow)`` with frames shaped
                ``(n_frames, n_particles, 3)``.
            """
            carry = (positions, neighbors, key, epsilon)
            carry, block_frames = jax.lax.scan(integrate_block, carry, None, length=n_blocks)
            # The overflow flag is sticky: True if ANY update in the run lost neighbors.
            return jnp.concatenate((positions[None], block_frames), axis=0), carry[1].did_buffer_overflow

        return rollout

    def compute_forces(self, positions: Any, epsilon: Any) -> jax.Array:
        """Evaluate the total force at one configuration.

        This method is intended for validation against HOOMD. It does not draw
        random noise or advance the positions. The returned array has shape
        ``(n_particles, 3)`` and includes steric, sticker-patch, chain-bond,
        and patch-tether forces.

        The neighbor list is allocated and updated for this configuration, so
        this method is convenient for reference checks but should not be used
        inside a timestep loop. The compiled rollout reuses the list instead.

        Input:
            ``positions``: One configuration shaped ``(n_particles, 3)``.
            ``epsilon``: Positive sticker-patch binding strength.
        Output:
            Total force array shaped ``(n_particles, 3)``.
        """
        positions = jnp.asarray(positions, dtype=self.dtype)
        epsilon = self._signed_attraction_epsilon(epsilon)
        shifted = shift_to_neighbor_list_coordinates(positions, self.box)
        neighbors = self._allocate_neighbors(positions).update(shifted)
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
            PATCH_RCUT,
            self.displacement,
        )

    def brownian_step(
        self,
        positions: Any,
        epsilon: Any,
        noise: Any | None = None,
    ) -> jax.Array:
        """Advance one Euler-Maruyama step from a supplied configuration.

        If ``noise`` is supplied, it must already be a displacement array with
        shape ``(n_particles, 3)``. Supplying the same noise to another
        implementation makes a controlled integrator comparison possible. If
        it is omitted, this method uses a zero-noise deterministic step; the
        normal stochastic rollout generates and carries its own JAX PRNG keys.

        Input:
            ``positions``: Starting coordinates shaped ``(n_particles, 3)``.
            ``epsilon``: Positive sticker-patch binding strength.
            ``noise``: Optional precomputed Brownian displacement with the same
                shape as ``positions``.
        Output:
            Coordinates after one wrapped Brownian dynamics step.
        """
        positions = jnp.asarray(positions, dtype=self.dtype)
        force = self.compute_forces(positions, epsilon)
        displacement = jnp.zeros_like(positions) if noise is None else jnp.asarray(noise, dtype=self.dtype)
        return self._wrap_centered(
            positions + self.mobilities * force * self.dt + displacement
        )

    def _wrap_centered(self, positions: jax.Array) -> jax.Array:
        """Wrap positions into the origin-centered periodic box.

        Input:
            ``positions``: Coordinates in any periodic image.
        Output:
            Coordinates mapped to ``[-box / 2, box / 2)`` with the same shape.
        """
        return (positions + self.box * 0.5) % self.box - self.box * 0.5

    def _signed_attraction_epsilon(self, binding_strength: Any) -> jax.Array:
        """Convert a binding magnitude to the signed attractive well depth."""
        return -jnp.abs(jnp.asarray(binding_strength, dtype=self.dtype))

    def compute_force_components(
        self, positions: Any, epsilon: Any
    ) -> dict[str, jax.Array]:
        """Return steric, sticker-patch, bonded, and total forces separately.

        This is useful when validating the JAX implementation against HOOMD:
        the four returned entries have shape ``(n_particles, 3)`` and can be
        compared component by component before comparing the total.

        Input:
            ``positions``: One configuration shaped ``(n_particles, 3)``.
            ``epsilon``: Positive sticker-patch binding strength.
        Output:
            Dictionary containing ``steric``, ``sticker_patch``, ``bonded``,
            and ``total`` force arrays.
        """
        positions = jnp.asarray(positions, dtype=self.dtype)
        epsilon = self._signed_attraction_epsilon(epsilon)
        shifted = shift_to_neighbor_list_coordinates(positions, self.box)
        neighbors = self._allocate_neighbors(positions).update(shifted)
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
            PATCH_RCUT,
            self.displacement,
        )
        return {
            "steric": steric,
            "sticker_patch": attraction,
            "bonded": bonded,
            "total": steric + attraction + bonded,
        }

    def __call__(self, fields: Mapping[str, Any]) -> jax.Array:
        """Run the differentiable simulation and return sampled trajectory frames.

        ``fields`` normally contains only ``fields["epsilon"]``. Supplying
        ``fields["positions_0"]`` overrides the generated initial snapshot.
        The returned array contains frame 0 followed by frames at every
        ``write_every`` steps.

        Input:
            ``fields``: Mapping containing required ``epsilon`` and optional
                ``positions_0``.
        Output:
            Sampled trajectory shaped ``(n_frames, n_particles, 3)``.
        """
        epsilon = fields.get("epsilon")
        if epsilon is None:
            raise KeyError("JAX MD stepper requires fields['epsilon']")
        positions = self.init(fields)
        binding_strength = jnp.abs(jnp.asarray(epsilon, dtype=self.dtype))
        epsilon = self._signed_attraction_epsilon(binding_strength)
        # JAX uses explicit PRNG keys rather than a mutable global RNG.  The
        # key is carried through lax.scan so every step receives fresh noise.
        key = jax.random.PRNGKey(self.seed)
        for attempt in range(self._max_capacity_retries + 1):
            # Allocate the fixed-shape neighbor-list storage before entering the
            # compiled function.  The list is updated inside each timestep.
            neighbors = self._allocate_neighbors(positions)
            frames, overflow = self._compiled_rollout(positions, epsilon, key, neighbors)
            if isinstance(overflow, jax.core.Tracer):
                # Being differentiated or jitted: the flag cannot be read here. Run one
                # ordinary call first so any needed capacity growth has already happened.
                return self._write_output_if_requested(frames, binding_strength)
            self.last_neighbor_overflow = bool(overflow)
            if not self.last_neighbor_overflow:
                return self._write_output_if_requested(frames, binding_strength)
            if attempt < self._max_capacity_retries:
                self._extra_capacity = max(2 * self._extra_capacity, 16)   # recompiles once
        warnings.warn(
            "Neighbor list overflowed even after growing its capacity; some interacting pairs "
            "were dropped. Increase neighbor_extra_capacity.", RuntimeWarning, stacklevel=2,
        )
        return self._write_output_if_requested(frames, binding_strength)

    def _write_gsd(
        self,
        trajectory: Any,
        path: str | Path | None = None,
    ) -> Path:
        """Write a trajectory to GSD and return its path.

        If ``path`` is omitted, ``output_dir/output_name.gsd`` is used. This
        explicit method is also the recommended way to write a trajectory
        produced inside a differentiated or transformed JAX function.

        Input:
            ``trajectory``: Frames shaped ``(n_frames, n_particles, 3)``.
            ``path``: Optional output path; otherwise the configured directory
                and name are used.
        Output:
            Path to the written GSD file.
        """
        if path is None:
            if self.output_dir is None:
                raise ValueError("Set output_dir or pass an explicit GSD path")
            path = self._artifact_directories()["traj"] / f"{self.output_name}.gsd"
        bond_type_ids = np.concatenate(
            [
                np.zeros(528, dtype=np.uint32),
                np.ones(192, dtype=np.uint32),
            ]
        )
        return write_gsd(
            path=path,
            trajectory=trajectory,
            box=self.box,
            particle_type_ids=self.type_ids,
            bond_pairs=self.bond_pairs,
            write_every=self.write_every,
            bond_type_ids=bond_type_ids,
        )

    def _artifact_directories(self) -> dict[str, Path]:
        """Create and return device-specific artifact directories.

        Input:
            None; ``output_dir`` must be configured on the instance.
        Output:
            Mapping for trajectory, XYZ, CSV, and plot directories.
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

    def _write_output_if_requested(self, trajectory: jax.Array, epsilon: Any) -> jax.Array:
        """Write host-side artifacts for ordinary forward calls.

        Input:
            ``trajectory``: Returned rollout frames.
            ``epsilon``: Attraction parameter recorded in analysis output.
        Output:
            The unchanged trajectory.
        """
        if self.output_dir is not None and not isinstance(trajectory, jax.core.Tracer):
            directories = self._artifact_directories()
            gsd_path = self._write_gsd(trajectory)
            write_analysis_outputs(
                gsd_path=gsd_path,
                xyz_path=directories["xyz"] / f"{self.output_name}.xyz",
                csv_path=directories["data"] / f"{self.output_name}.csv",
                plots_dir=directories["plots"],
                output_name=self.output_name,
                dt=self.dt,
                epsilon=float(np.asarray(epsilon)),
            )
        return trajectory

    def rollout(self, epsilon: Any, positions_0: Any | None = None, seed: int | None = None) -> jax.Array:
        """Run a rollout using a compact convenience API.

        Input:
            ``epsilon``: Positive sticker-patch binding strength.
            ``positions_0``: Optional initial coordinates.
            ``seed``: Optional temporary PRNG seed for this call.
        Output:
            Sampled trajectory shaped ``(n_frames, n_particles, 3)``.
        """
        fields = {"epsilon": epsilon}
        if positions_0 is not None:
            fields["positions_0"] = positions_0
        if seed is not None:
            old_seed = self.seed
            self.seed = int(seed)
            try:
                return self(fields)
            finally:
                self.seed = old_seed
        return self(fields)
