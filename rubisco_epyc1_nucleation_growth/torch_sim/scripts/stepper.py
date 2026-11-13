"""Differentiable Brownian Dynamics TimeStepper for Rubisco-EPYC1 condensates.

Importing this module has no side effects. Typical use from an ML training script::

    from condensate_ml import RubiscoEpyc1TimeStepper

    stepper = RubiscoEpyc1TimeStepper(n_steps=2000, box=35.0, write_every=100,
                                      seed=0, device=device)
    epsilon = -torch.nn.functional.softplus(net(features))     # attractive => negative
    trajectory = stepper({"epsilon": epsilon})                  # (T, 792, 3), differentiable
    loss = loss_fn(compute_soft_contacts(trajectory, stepper.sticker_indices,
                                         stepper.patch_indices, box=35.0))
    loss.backward()

Sign convention: ``epsilon`` is the signed Gaussian well depth of the sticker-patch pair
potential. NEGATIVE = attractive (this is -|KON| in the HOOMD script). A positive value
silently turns the sticker-patch interaction repulsive.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from nefi.operators.timestepping import TimeStepper

from . import forces, initial_snapshot, topology
from .constants import (
    CHAIN_K,
    CHAIN_R0,
    CHECKPOINT_EVERY,
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
    STICKER_INDEX,
)


def kon_to_epsilon(kon: torch.Tensor | float) -> torch.Tensor | float:
    """Map a positive binding strength (KON / EPS in params.json) to the signed epsilon."""
    return -kon


class RubiscoEpyc1TimeStepper(TimeStepper):
    """Differentiable Brownian Dynamics TimeStepper for Rubisco-EPYC1 condensates.

    The stepper object holds only fixed topology and constants. The learnable / swept
    quantity is passed per call as ``fields["epsilon"]`` (scalar tensor), so one stepper can
    be reused across optimisation iterations.

    Output: trajectory tensor (n_frames, n_particles, 3), frames at step 0 and every
    ``write_every`` steps (inclusive of the last when divisible). Particle order is
    R (24), P (192), then 48 chains of 12 beads (S/T alternating).

    Stochasticity: each step draws fresh Gaussian noise from torch's global RNG. With
    ``grad_mode="checkpoint"`` torch.utils.checkpoint restores the RNG state when it
    recomputes a block, so backward sees the same noise as forward. Call
    ``torch.manual_seed`` before each rollout for reproducibility.
    """

    traceable: bool = False

    def __init__(
        self,
        n_steps: int,
        dt: float = DT,
        box: float = 40.0,
        write_every: int = 100,
        seed: int = 42,
        grad_mode: str = "checkpoint",
        checkpoint_every: int = CHECKPOINT_EVERY,
        field: str = "epsilon",
        fields: Sequence[str] | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        # nn.Module must be initialised before attributes are assigned.
        super().__init__(
            step=self.step,
            init=self.init,
            n_steps=n_steps,
            dt=dt,
            observe=self.observe,
            field=field,
            fields=fields if fields is not None else (field,),
            grad_mode=grad_mode,
            checkpoint_every=checkpoint_every,
        )

        self.box = float(box)
        self.patch_dirs = PATCH_DIRS
        self.write_every = int(write_every)
        self.seed = int(seed)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype

        self.n_rubisco = N_R
        self.n_patches_per_rubisco = PATCHES
        self.n_chains = N_C
        self.n_beads_per_chain = N_BEADS
        self.sticker_local_indices = STICKER_INDEX
        self.n_particles = N_R + N_R * PATCHES + N_C * N_BEADS

        layout = (self.n_rubisco, self.n_patches_per_rubisco, self.n_chains, self.n_beads_per_chain)

        self.particle_labels = topology.build_particle_labels(*layout, self.sticker_local_indices)
        self.particle_type_ids = topology.build_particle_type_ids(
            *layout, self.sticker_local_indices, device=self.device
        )
        self.mobilities_by_particle = topology.build_mobilities_by_particle(
            *layout,
            gamma_rubisco=GAMMA_RUBISCO,
            gamma_chain=GAMMA_CHAIN,
            device=self.device,
            dtype=self.dtype,
        )
        self.bond_pairs_by_type = topology.build_bond_topologies(*layout, device=self.device)
        self.k_by_type = {"chain": CHAIN_K, "patch_tether": PATCH_TETHER_K}
        self.r0_by_type = {"chain": CHAIN_R0, "patch_tether": PATCH_RADIUS}

        eps_mat, sig_mat, rcut_mat = topology.build_steric_parameter_matrices(
            self.particle_type_ids, device=self.device, dtype=self.dtype
        )
        self.epsilon_steric_matrix = eps_mat
        self.sigma_steric_matrix = sig_mat
        self.r_cut_steric_matrix = rcut_mat

        # True where a pair interacts sterically: bonded pairs and self-pairs are False,
        # matching HOOMD's default bond exclusions in the neighbor list.
        self.nonbonded_pair_mask = topology.build_nonbonded_pair_mask(
            self.bond_pairs_by_type, self.n_particles, device=self.device
        )

        patch_start = self.n_rubisco
        chain_start = self.n_rubisco + self.n_rubisco * self.n_patches_per_rubisco
        patch_idx = list(range(patch_start, chain_start))
        self.patch_indices = torch.tensor(patch_idx, dtype=torch.long, device=self.device)

        sticker_idx, sticker_chain = [], []
        for c in range(self.n_chains):
            base = chain_start + c * self.n_beads_per_chain
            for b in self.sticker_local_indices:
                sticker_idx.append(base + b)
                sticker_chain.append(c)
        self.sticker_indices = torch.tensor(sticker_idx, dtype=torch.long, device=self.device)
        self.sticker_chain_ids = torch.tensor(sticker_chain, dtype=torch.long, device=self.device)

        self.patch_rubisco_ids = torch.tensor(
            [p // self.n_patches_per_rubisco for p in range(len(patch_idx))],
            dtype=torch.long,
            device=self.device,
        )

    # ------------------------------------------------------------------------------
    # TimeStepper callbacks
    # ------------------------------------------------------------------------------
    def init(self, fields: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Initial coordinates (n_particles, 3); override with fields['positions_0']."""
        if "positions_0" in fields:
            return fields["positions_0"].to(device=self.device, dtype=self.dtype)
        return initial_snapshot.generate_initial_coordinates(
            self.box,
            self.patch_dirs,
            self.seed,
            self.n_rubisco,
            self.n_patches_per_rubisco,
            self.n_chains,
            self.n_beads_per_chain,
            patch_radius=PATCH_RADIUS,
            chain_r0=CHAIN_R0,
            device=self.device,
            dtype=self.dtype,
        )

    def step(
        self,
        state: torch.Tensor,
        params: Mapping[str, Any],
        dt: float,
        n: int,
    ) -> torch.Tensor:
        """One Euler-Maruyama Brownian dynamics step."""
        epsilon_att = params[self.primary]
        if not torch.is_tensor(epsilon_att):
            epsilon_att = torch.tensor(epsilon_att, device=state.device, dtype=state.dtype)

        f_bonds = forces.compute_harmonic_bond_forces(
            state, self.bond_pairs_by_type, self.k_by_type, self.r0_by_type, self.box
        )
        f_sterics = forces.compute_gaussian_steric_forces(
            state,
            self.epsilon_steric_matrix,
            self.sigma_steric_matrix,
            self.r_cut_steric_matrix,
            self.box,
            self.nonbonded_pair_mask,
        )
        f_attract = forces.compute_gaussian_attraction_forces(
            state,
            self.sticker_indices,
            self.patch_indices,
            epsilon_att,
            PATCH_SIGMA,
            PATCH_RCUT,
            self.box,
        )
        f_total = f_bonds + f_sterics + f_attract

        # kT = 1 => diffusion coefficient D = mobility; noise std = sqrt(2 D dt)
        noise_std = torch.sqrt(2.0 * self.mobilities_by_particle * dt)
        noise_displacement = noise_std * torch.randn_like(state)

        return forces.step_brownian_dynamics(
            state, f_total, self.mobilities_by_particle, dt, noise_displacement, self.box
        )

    def observe(self, state: torch.Tensor, n: int) -> torch.Tensor | None:
        """Record coordinates at step 0 and every write_every steps."""
        if n % self.write_every == 0:
            return state
        return None


# camelCase aliases kept for backward compatibility with the earlier sweep script and tests.
_CAMEL_ALIASES = {
    "particleLabels": "particle_labels",
    "particleTypeIds": "particle_type_ids",
    "mobilitiesByParticle": "mobilities_by_particle",
    "bondPairsByType": "bond_pairs_by_type",
    "kByType": "k_by_type",
    "r0ByType": "r0_by_type",
    "epsilonStericMatrix": "epsilon_steric_matrix",
    "sigmaStericMatrix": "sigma_steric_matrix",
    "rCutStericMatrix": "r_cut_steric_matrix",
    "patchIndices": "patch_indices",
    "stickerIndices": "sticker_indices",
    "stickerChainIds": "sticker_chain_ids",
    "patchRubiscoIds": "patch_rubisco_ids",
}
for _camel, _snake in _CAMEL_ALIASES.items():
    setattr(
        RubiscoEpyc1TimeStepper,
        _camel,
        property(lambda self, _s=_snake: getattr(self, _s)),
    )
