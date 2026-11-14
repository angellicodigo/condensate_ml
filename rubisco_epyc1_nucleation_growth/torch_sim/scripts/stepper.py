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
        compile_step: bool = False,
    ) -> None:
        # Writing to the TimeStepper Parent Class. This means that a RubiscoEpyc1TimeStepper instance also has methods 
        # from TimeStepper.
        step_fn = self.step
        if compile_step:
            if not hasattr(torch, "compile"):
                raise RuntimeError("compile_step=True requires PyTorch 2.0 or newer")
            step_fn = torch.compile(step_fn, mode="reduce-overhead")

        super().__init__(
            step=step_fn, # What happens in one step. Defined later
            init=self.init, # Initial setup. Defined later
            n_steps=n_steps,
            dt=dt,
            observe=self.observe, # What frames to observed. Defined later
            field=field,
            fields=fields if fields is not None else (field,),
            grad_mode=grad_mode,
            checkpoint_every=checkpoint_every,
        )
        self.box = float(box)
        self.write_every = int(write_every)
        self.seed = int(seed)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype

        # Variables defined in constants.py used by all instances of RubiscoEpyc1TimeStepper
        self.patch_dirs = PATCH_DIRS 
        self.n_rubisco = N_R
        self.n_patches_per_rubisco = PATCHES
        self.n_chains = N_C
        self.n_beads_per_chain = N_BEADS
        self.sticker_local_indices = STICKER_INDEX
        self.n_particles = N_R + N_R * PATCHES + N_C * N_BEADS

        layout = (self.n_rubisco, self.n_patches_per_rubisco, self.n_chains, self.n_beads_per_chain)
        
        """
        topology.build_particle_labels() - returns a tensor of shape (n_particles,) of letters where each letter represents.
            R: Rubisco core center.
            P: Rubisco surface patch.
            S: EPYC1 spacer bead.
            T: EPYC1 sticker bead.
        """
        self.particle_labels = topology.build_particle_labels(*layout, self.sticker_local_indices)
        """
        topology.build_particle_type_ids() - converts those letters to integers (R=0, P=1, S=2, T=3), giving a tensor of shape (n_particles,).
        """
        self.particle_type_ids = topology.build_particle_type_ids(
            *layout, self.sticker_local_indices, device=self.device
        )
        """
        topology.build_mobilities_by_particle() - returns a tensor of shape (n_particles, 1) that represents how fast the particles move in our simulation. This is used to simulate the Brownian motion. 
            GAMMA_RUBISCO = brownian.gamma["R"] = brownian.gamma["P"] = 5.0
            GAMMA_CHAIN = brownian.gamma["S"] = brownian.gamma["T"] = 1.0
        The dimensions has an extra (1) to utilize tensor broadcast mutliplication in step().
        TODO: Question this
        """
        self.mobilities_by_particle = topology.build_mobilities_by_particle(
            *layout,
            gamma_rubisco=GAMMA_RUBISCO,
            gamma_chain=GAMMA_CHAIN,
            device=self.device,
            dtype=self.dtype,
        )
        """
        topology.build_bond_topologies() - returns a dictionary of two tables, "chain" and "patch-tehter". "chain" is a tensor of shape (n_chains * (n_beads_per_chain - 1), 2) and "patch-tehter" is a tensor of shape (n_rubisco * n_patches_per_rubisco, 2). Note that the number of bonds between the beads is equal to n_beads_per_chain - 1. Below is an example.
        {
            "chain": [[216, 217], [217, 218], [218, 219], ... , [790, 791]]
                chain[0]  = [216, 217]    bead 0 -- bead 1
                chain[1]  = [217, 218]    bead 1 -- bead 2
                chain[2]  = [218, 219]    bead 2 -- bead 3
                ...
                chain[10] = [226, 227]    bead 10 -- bead 11   (last bond of chain 0)
                chain[11] = [228, 229]    first bond of chain 1 (chain 1 starts at particle 228)
                ...
            "patch_tether": [[0, 24], [0, 25], ... [0, 31], [1, 32], ... [23, 215]] 
                patch_tether[0]  = [0, 24]       Rubisco center 0 -- Rubisco patch 24
                patch_tether[1]  = [0, 25]       Rubisco center 0 -- Rubisco patch 25
                ...
                patch_tether[7] = [0, 31]        Rubisco center 0 -- Rubisco patch 31
                patch_tether[8] = [1, 32]        Rubisco center 1 -- Rubisco patch 32
                ...
        }

        """
        self.bond_pairs_by_type = topology.build_bond_topologies(*layout, device=self.device)
        # More variables from constraints.py. They are lookup tables, given bond type's name -> return their values. They are used in conjunction with bond_pairs_by_type
        # Defines spring stiffness
        self.k_by_type = {"chain": CHAIN_K, "patch_tether": PATCH_TETHER_K}
        # Defines spring rest / equilibrium length 
        self.r0_by_type = {"chain": CHAIN_R0, "patch_tether": PATCH_RADIUS}
        self.bond_pairs, self.bond_k, self.bond_r0 = self._build_fused_bonds()
        """
        topology.build_steric_parameter_matrices() - returns three tensor matrices of shape (n_particles, n_particles). These three matrices represents the interactions of all particles with each other, depending on what type of particle they are. 
        eps_mat - repulsion strength, epsilon
        sig_mat - repulsion width, sigma
        rcut_mat - distance beyond which they ignore each other. If particle A and B are < rcut_mat[A, B], they can interact with each other.
        """
        eps_mat, sig_mat, rcut_mat = topology.build_steric_parameter_matrices(
            self.particle_type_ids, device=self.device, dtype=self.dtype
        )
        self.epsilon_steric_matrix = eps_mat
        self.sigma_steric_matrix = sig_mat
        self.r_cut_steric_matrix = rcut_mat

        """ 
        topology.build_nonbonded_pair_mask() - boolean tensor matrix of shape (n_particles, n_particles)
        nonbonded_pair_mask[a, b] = 1 -> Particle A and B are allowed to have a repulsive force between them
        """
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

        # This factor is fixed for the lifetime of the stepper.  Computing it
        # once avoids a sqrt and a broadcast setup on every integration step.
        self.noise_std = torch.sqrt(2.0 * self.mobilities_by_particle * self.dt)

    def _build_fused_bonds(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pack the fixed bond tables into tensors consumed by the fused kernel."""
        bond_types = tuple(self.bond_pairs_by_type)
        pairs = torch.cat([self.bond_pairs_by_type[name] for name in bond_types], dim=0)
        k = torch.tensor(
            [self.k_by_type[name] for name in bond_types for _ in range(self.bond_pairs_by_type[name].shape[0])],
            device=self.device,
            dtype=self.dtype,
        )
        r0 = torch.tensor(
            [self.r0_by_type[name] for name in bond_types for _ in range(self.bond_pairs_by_type[name].shape[0])],
            device=self.device,
            dtype=self.dtype,
        )
        return pairs, k, r0

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

        f_bonds = forces.compute_harmonic_bond_forces_fused(
            state, self.bond_pairs, self.bond_k, self.bond_r0, self.box
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
        noise_displacement = self.noise_std * torch.randn_like(state)

        return forces.brownian_dynamics(
            state, f_total, self.mobilities_by_particle, dt, noise_displacement, self.box
        )

    def observe(self, state: torch.Tensor, n: int) -> torch.Tensor | None:
        """Record coordinates at step 0 and every write_every steps."""
        if n % self.write_every == 0:
            return state
        return None
