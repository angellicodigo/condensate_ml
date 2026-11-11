"""Force computations for the Rubisco-EPYC1 Brownian Dynamics simulation.

All functions are differentiable PyTorch operations. They compute forces from
positions and return new force tensors; none mutate their inputs.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch

from .geometry import apply_minimum_image_displacement, wrap_centered_coordinates


def compute_gaussian_steric_forces(
    positions: torch.Tensor,
    epsilon_matrix: torch.Tensor,
    sigma_matrix: torch.Tensor,
    r_cut_matrix: torch.Tensor,
    box: float,
) -> torch.Tensor:
    """Compute bounded Gaussian repulsive forces between all non-bonded pairs.

    The force on particle i is the sum over j of the Gaussian repulsion
    prefactor times the minimum-image displacement. Pairs beyond the cutoff
    contribute zero.
    """
    delta = positions[:, None, :] - positions[None, :, :]
    delta_min = apply_minimum_image_displacement(delta, box)
    dist_sq = torch.sum(delta_min * delta_min, dim=-1)

    within_cutoff_mask = (dist_sq < (r_cut_matrix * r_cut_matrix)) & (dist_sq > 1e-12)
    sig_sq = sigma_matrix * sigma_matrix
    safe_sig_sq = torch.clamp(sig_sq, min=1e-8)
    prefactors = (epsilon_matrix / safe_sig_sq) * torch.exp(-0.5 * dist_sq / safe_sig_sq)
    masked_prefactors = torch.where(
        within_cutoff_mask, prefactors, torch.zeros_like(prefactors)
    )
    return torch.sum(masked_prefactors.unsqueeze(-1) * delta_min, dim=1)


def compute_sticker_attraction_forces(
    positions: torch.Tensor,
    sticker_indices: torch.Tensor,
    patch_indices: torch.Tensor,
    epsilon_att: torch.Tensor,
    sigma_att: float,
    r_cut_att: float,
    box: float,
) -> torch.Tensor:
    """Compute attractive forces on sticker beads from nearby patch particles.

    Returns a tensor of shape (n_stickers, 3) with the force on each sticker.
    """
    pos_stickers = positions[sticker_indices]
    pos_patches = positions[patch_indices]
    delta = pos_stickers[:, None, :] - pos_patches[None, :, :]
    delta_min = apply_minimum_image_displacement(delta, box)
    dist_sq = torch.sum(delta_min * delta_min, dim=-1)

    within_cutoff_mask = dist_sq < (r_cut_att * r_cut_att)
    sig_sq = sigma_att * sigma_att
    prefactors = (epsilon_att / sig_sq) * torch.exp(-0.5 * dist_sq / sig_sq)
    masked_prefactors = torch.where(
        within_cutoff_mask, prefactors, torch.zeros_like(prefactors)
    )
    return torch.sum(masked_prefactors.unsqueeze(-1) * delta_min, dim=1)


def compute_patch_attraction_forces(
    positions: torch.Tensor,
    sticker_indices: torch.Tensor,
    patch_indices: torch.Tensor,
    epsilon_att: torch.Tensor,
    sigma_att: float,
    r_cut_att: float,
    box: float,
) -> torch.Tensor:
    """Compute attractive forces on patch particles from nearby sticker beads.

    By Newton's third law, the force on a patch is the negative of the force
    the corresponding sticker exerts on it.
    """
    pos_stickers = positions[sticker_indices]
    pos_patches = positions[patch_indices]
    delta = pos_stickers[:, None, :] - pos_patches[None, :, :]
    delta_min = apply_minimum_image_displacement(delta, box)
    dist_sq = torch.sum(delta_min * delta_min, dim=-1)

    within_cutoff_mask = dist_sq < (r_cut_att * r_cut_att)
    sig_sq = sigma_att * sigma_att
    prefactors = (epsilon_att / sig_sq) * torch.exp(-0.5 * dist_sq / sig_sq)
    masked_prefactors = torch.where(
        within_cutoff_mask, prefactors, torch.zeros_like(prefactors)
    )
    return -torch.sum(masked_prefactors.unsqueeze(-1) * delta_min, dim=0)


def scatter_forces_onto_particles(
    total_forces: torch.Tensor,
    indices: torch.Tensor,
    per_atom_forces: torch.Tensor,
) -> torch.Tensor:
    """Add per-atom forces into the total force tensor at the given indices.

    Uses scatter_add so forces accumulate correctly when multiple interactions
    target the same particle.
    """
    return total_forces.scatter_add(
        0, indices.unsqueeze(-1).expand(-1, 3), per_atom_forces
    )


def combine_force_tensors(*force_tensors: torch.Tensor) -> torch.Tensor:
    """Sum multiple force tensors element-wise into a single total force tensor."""
    total = torch.zeros_like(force_tensors[0])
    for forces in force_tensors:
        total = total + forces
    return total


def compute_gaussian_attraction_forces(
    positions: torch.Tensor,
    sticker_indices: torch.Tensor,
    patch_indices: torch.Tensor,
    epsilon_att: torch.Tensor,
    sigma_att: float,
    r_cut_att: float,
    box: float,
) -> torch.Tensor:
    """Compute the total Gaussian attraction force on every particle.

    Combines sticker forces and patch forces, then scatters them onto the
    full particle array. Returns a tensor of shape (n_particles, 3).
    """
    forces_on_stickers = compute_sticker_attraction_forces(
        positions, sticker_indices, patch_indices, epsilon_att, sigma_att, r_cut_att, box
    )
    forces_on_patches = compute_patch_attraction_forces(
        positions, sticker_indices, patch_indices, epsilon_att, sigma_att, r_cut_att, box
    )

    total_attraction_forces = torch.zeros_like(positions)
    total_attraction_forces = scatter_forces_onto_particles(
        total_attraction_forces, sticker_indices, forces_on_stickers
    )
    total_attraction_forces = scatter_forces_onto_particles(
        total_attraction_forces, patch_indices, forces_on_patches
    )
    return total_attraction_forces


def compute_single_bond_type_forces(
    positions: torch.Tensor,
    pairs: torch.Tensor,
    k: float,
    r0: float,
    box: float,
) -> torch.Tensor:
    """Compute harmonic bond forces for one bond type (one set of index pairs).

    Returns a tensor of shape (n_particles, 3) with forces scattered onto
    the bonded particles.
    """
    i_indices = pairs[:, 0]
    j_indices = pairs[:, 1]
    delta = positions[i_indices] - positions[j_indices]
    delta_min = apply_minimum_image_displacement(delta, box)
    distances = torch.linalg.vector_norm(delta_min, dim=-1, keepdim=True)
    safe_distances = torch.clamp(distances, min=1e-8)
    force_magnitudes = -k * (distances - r0)
    forces_on_i = force_magnitudes * (delta_min / safe_distances)

    total_forces = torch.zeros_like(positions)
    total_forces = scatter_forces_onto_particles(total_forces, i_indices, forces_on_i)
    total_forces = scatter_forces_onto_particles(total_forces, j_indices, -forces_on_i)
    return total_forces


def accumulate_bond_forces(
    total_forces: torch.Tensor, additional_forces: torch.Tensor
) -> torch.Tensor:
    """Add one bond-type force contribution into the running total."""
    return total_forces + additional_forces


def compute_harmonic_bond_forces(
    positions: torch.Tensor,
    bond_pairs_by_type: Mapping[str, torch.Tensor],
    k_by_type: Mapping[str, float],
    r0_by_type: Mapping[str, float],
    box: float,
) -> torch.Tensor:
    """Compute differentiable harmonic bond forces across all bonded topology groups.

    Loops over bond types (chain bonds, patch tethers), computes the force for
    each type, and accumulates into the total.
    """
    total_bond_forces = torch.zeros_like(positions)
    for bond_type, pairs in bond_pairs_by_type.items():
        k = k_by_type[bond_type]
        r0 = r0_by_type[bond_type]
        type_forces = compute_single_bond_type_forces(
            positions, pairs, k, r0, box
        )
        total_bond_forces = accumulate_bond_forces(total_bond_forces, type_forces)
    return total_bond_forces


def step_brownian_dynamics(
    positions: torch.Tensor,
    total_force: torch.Tensor,
    mobilities: torch.Tensor,
    dt: float,
    noise_displacement: torch.Tensor,
    box: float,
) -> torch.Tensor:
    """Execute one overdamped Langevin (Euler-Maruyama) integration step.

    Updates positions as:
        x_new = x + mobility * force * dt + noise_displacement

    Then wraps the result back into the periodic box.
    """
    deterministic_displacement = mobilities * total_force * dt
    updated_positions = positions + deterministic_displacement + noise_displacement
    return wrap_centered_coordinates(updated_positions, box)
