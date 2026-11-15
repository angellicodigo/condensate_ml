"""JAX MD Rubisco-EPYC1 differentiable Brownian dynamics prototype."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax_md import partition, space

from ..scripts.constants import (
    CHAIN_K,
    CHAIN_R0,
    DT,
    GAMMA_CHAIN,
    GAMMA_RUBISCO,
    N_BEADS,
    N_C,
    N_R,
    PATCH_DIRS,
    PATCH_RADIUS,
    PATCH_RCUT,
    PATCH_SIGMA,
    PATCH_TETHER_K,
    PATCHES,
    STERIC_PAIR_PARAMS,
    STICKER_INDEX,
    TYPE_INDEX,
)
from . import forces
from .initial_snapshot import generate_initial_coordinates


def kon_to_epsilon(kon: jax.Array | float) -> jax.Array | float:
    return -kon


def _make_type_tables(dtype: Any) -> tuple[jax.Array, jax.Array, jax.Array]:
    epsilon = np.zeros((4, 4), dtype=np.float32)
    sigma = np.ones((4, 4), dtype=np.float32)
    cutoff = np.zeros((4, 4), dtype=np.float32)
    for (a, b), values in STERIC_PAIR_PARAMS.items():
        i, j = TYPE_INDEX[a], TYPE_INDEX[b]
        for x, y in ((i, j), (j, i)):
            epsilon[x, y], sigma[x, y], cutoff[x, y] = values
    return tuple(jnp.asarray(x, dtype=dtype) for x in (epsilon, sigma, cutoff))


class RubiscoEpyc1JaxMDTimeStepper:
    """Differentiable JAX MD prototype with cell-list neighbors and scan blocks.

    The public operation is ``stepper({"epsilon": epsilon})`` and returns
    frames with shape ``(n_frames, 792, 3)``.  ``epsilon`` may be a JAX scalar;
    gradients are obtained with ``jax.grad`` or ``jax.value_and_grad``.

    ``write_every`` must divide ``n_steps`` in this prototype.  The rollout is
    compiled lazily on the first call.  Subsequent calls with the same shapes,
    dtype, device, and timestep settings reuse the compiled executable.
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
    ) -> None:
        if n_steps < 0 or write_every < 1 or n_steps % write_every:
            raise ValueError("n_steps must be nonnegative and divisible by write_every")
        self.n_steps = int(n_steps)
        self.dt = float(dt)
        self.box = float(box)
        self.write_every = int(write_every)
        self.seed = int(seed)
        self.dtype = dtype
        self.n_rubisco = N_R
        self.n_patches_per_rubisco = PATCHES
        self.n_chains = N_C
        self.n_beads_per_chain = N_BEADS
        self.n_particles = N_R + N_R * PATCHES + N_C * N_BEADS
        self.sticker_local_indices = tuple(STICKER_INDEX)
        self.patch_dirs = np.asarray(PATCH_DIRS)

        self.type_ids = self._build_type_ids()
        self.particle_type_ids = self.type_ids
        self.mobilities = self._build_mobilities()
        self.bond_pairs, self.bond_k, self.bond_r0 = self._build_bonds()
        self.nonbonded_pair_mask = self._build_exclusion_mask()
        self.epsilon_table, self.sigma_table, self.cutoff_table = _make_type_tables(dtype)

        self.patch_indices = jnp.arange(N_R, N_R + N_R * PATCHES, dtype=jnp.int32)
        chain_start = N_R + N_R * PATCHES
        sticker_indices = [chain_start + c * N_BEADS + b for c in range(N_C) for b in STICKER_INDEX]
        self.sticker_indices = jnp.asarray(sticker_indices, dtype=jnp.int32)
        self.patch_rubisco_ids = jnp.repeat(jnp.arange(N_R, dtype=jnp.int32), PATCHES)
        self.sticker_chain_ids = jnp.repeat(jnp.arange(N_C, dtype=jnp.int32), len(STICKER_INDEX))
        self.particle_labels = self._build_labels()

        self.displacement, _shift = space.periodic(self.box)
        max_cutoff = max(float(np.max(self.cutoff_table)), PATCH_RCUT)
        self.neighbor_fn = partition.neighbor_list(
            self.displacement,
            self.box,
            max_cutoff,
            dr_threshold=neighbor_buffer,
            format=partition.NeighborListFormat.Dense,
        )
        self._compiled_rollout = jax.jit(self._make_rollout(checkpoint_blocks))

    def _build_labels(self) -> list[str]:
        labels = ["R"] * N_R + ["P"] * (N_R * PATCHES)
        labels += ["T" if b in STICKER_INDEX else "S" for b in range(N_BEADS)] * N_C
        return labels

    def _build_type_ids(self) -> jax.Array:
        labels = self._build_labels()
        return jnp.asarray([TYPE_INDEX[label] for label in labels], dtype=jnp.int32)

    def _build_mobilities(self) -> jax.Array:
        n_slow = N_R * (1 + PATCHES)
        values = np.full(self.n_particles, 1.0 / GAMMA_CHAIN, dtype=np.float32)
        values[:n_slow] = 1.0 / GAMMA_RUBISCO
        return jnp.asarray(values[:, None], dtype=self.dtype)

    def _build_bonds(self) -> tuple[jax.Array, jax.Array, jax.Array]:
        chain_start = N_R + N_R * PATCHES
        chain = [[chain_start + c * N_BEADS + b, chain_start + c * N_BEADS + b + 1]
                 for c in range(N_C) for b in range(N_BEADS - 1)]
        tether = [[r, N_R + r * PATCHES + p] for r in range(N_R) for p in range(PATCHES)]
        pairs = jnp.asarray(chain + tether, dtype=jnp.int32)
        k = jnp.asarray([CHAIN_K] * len(chain) + [PATCH_TETHER_K] * len(tether), dtype=self.dtype)
        r0 = jnp.asarray([CHAIN_R0] * len(chain) + [PATCH_RADIUS] * len(tether), dtype=self.dtype)
        return pairs, k, r0

    def _build_exclusion_mask(self) -> jax.Array:
        mask = np.ones((self.n_particles, self.n_particles), dtype=bool)
        np.fill_diagonal(mask, False)
        pairs = np.asarray(self.bond_pairs)
        mask[pairs[:, 0], pairs[:, 1]] = False
        mask[pairs[:, 1], pairs[:, 0]] = False
        return jnp.asarray(mask)

    def init(self, fields: Mapping[str, Any] | None = None) -> jax.Array:
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
        neighbor_fn = self.neighbor_fn
        displacement = self.displacement
        dt = self.dt
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
            return (positions + box * 0.5) % box - box * 0.5

        def one_step(carry, _):
            positions, neighbors, key, epsilon = carry
            neighbors = neighbors.update(positions)
            key, noise_key = jax.random.split(key)
            total = forces.total_forces(
                positions, neighbors.idx, type_ids, epsilon_table, sigma_table,
                cutoff_table, exclusions, bond_pairs, bond_k, bond_r0,
                epsilon, PATCH_SIGMA, PATCH_RCUT, displacement,
            )
            noise = jnp.sqrt(2.0 * mobility * dt) * jax.random.normal(
                noise_key, positions.shape, dtype=positions.dtype
            )
            next_positions = wrap_centered(positions + mobility * total * dt + noise)
            return (next_positions, neighbors, key, epsilon), None

        def integrate_block(carry, _):
            carry, _ = jax.lax.scan(one_step, carry, None, length=block_size)
            return carry, carry[0]

        if checkpoint_blocks:
            integrate_block = jax.checkpoint(integrate_block)

        def rollout(positions, epsilon, key, neighbors):
            carry = (positions, neighbors, key, epsilon)
            carry, block_frames = jax.lax.scan(integrate_block, carry, None, length=n_blocks)
            return jnp.concatenate((positions[None], block_frames), axis=0)

        return rollout

    def __call__(self, fields: Mapping[str, Any]) -> jax.Array:
        epsilon = fields.get("epsilon")
        if epsilon is None:
            raise KeyError("JAX MD stepper requires fields['epsilon']")
        positions = self.init(fields)
        neighbors = self.neighbor_fn.allocate(positions)
        key = jax.random.PRNGKey(self.seed)
        return self._compiled_rollout(
            positions,
            jnp.asarray(epsilon, dtype=self.dtype),
            key,
            neighbors,
        )

    def rollout(self, epsilon: Any, positions_0: Any | None = None, seed: int | None = None) -> jax.Array:
        """Convenience API equivalent to ``stepper({'epsilon': epsilon})``."""
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


__all__ = ["RubiscoEpyc1JaxMDTimeStepper", "kon_to_epsilon"]
