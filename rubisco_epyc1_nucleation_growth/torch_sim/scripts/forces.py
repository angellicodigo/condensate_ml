"""Force computations for the Rubisco-EPYC1 Brownian Dynamics simulation.

All functions are differentiable PyTorch operations. They compute forces from
positions and return new force tensors; none mutate their inputs.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch

from .geometry import (
    apply_minimum_image_displacement,
    safe_vector_norm,
    wrap_centered_coordinates,
)


def compute_gaussian_steric_forces(
    positions: torch.Tensor,
    epsilon_matrix: torch.Tensor,
    sigma_matrix: torch.Tensor,
    r_cut_matrix: torch.Tensor,
    box: float,
    nonbonded_pair_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute bounded Gaussian repulsive forces between all non-bonded pairs.

    ``nonbonded_pair_mask`` is a boolean (N, N) matrix that is True where a pair
    interacts. Bonded pairs and self-pairs should be False (HOOMD excludes bonded
    pairs from pair forces by default). Pairs beyond the cutoff contribute zero.
    """
    delta = positions[:, None, :] - positions[None, :, :]
    delta_min = apply_minimum_image_displacement(delta, box)
    dist_sq = torch.sum(delta_min * delta_min, dim=-1)

    within_cutoff_mask = (dist_sq < (r_cut_matrix * r_cut_matrix)) & (dist_sq > 1e-12)
    if nonbonded_pair_mask is not None:
        within_cutoff_mask = within_cutoff_mask & nonbonded_pair_mask

    sig_sq = sigma_matrix * sigma_matrix
    safe_sig_sq = torch.clamp(sig_sq, min=1e-8)
    prefactors = (epsilon_matrix / safe_sig_sq) * torch.exp(-0.5 * dist_sq / safe_sig_sq)
    masked_prefactors = torch.where(
        within_cutoff_mask, prefactors, torch.zeros_like(prefactors)
    )
    return torch.sum(masked_prefactors.unsqueeze(-1) * delta_min, dim=1)


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


def compute_sticker_patch_pair_forces(
    positions: torch.Tensor,
    sticker_indices: torch.Tensor,
    patch_indices: torch.Tensor,
    epsilon_att: torch.Tensor,
    sigma_att: float,
    r_cut_att: float,
    box: float,
) -> torch.Tensor:
    """Compute the force on each sticker due to each patch, shape (n_stickers, n_patches, 3).

    By Newton's third law the force on the patch is the negative of this.
    Computed once and reused for both sides.
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
    return masked_prefactors.unsqueeze(-1) * delta_min


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

    Returns a tensor of shape (n_particles, 3).
    """
    pair_forces = compute_sticker_patch_pair_forces(
        positions, sticker_indices, patch_indices, epsilon_att, sigma_att, r_cut_att, box
    )
    forces_on_stickers = torch.sum(pair_forces, dim=1)
    forces_on_patches = -torch.sum(pair_forces, dim=0)

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
    """Compute harmonic bond forces for one bond type (one set of index pairs)."""
    i_indices = pairs[:, 0]
    j_indices = pairs[:, 1]
    delta = positions[i_indices] - positions[j_indices]
    delta_min = apply_minimum_image_displacement(delta, box)
    distances = safe_vector_norm(delta_min, dim=-1, keepdim=True)
    force_magnitudes = -k * (distances - r0)
    forces_on_i = force_magnitudes * (delta_min / distances)

    total_forces = torch.zeros_like(positions)
    total_forces = scatter_forces_onto_particles(total_forces, i_indices, forces_on_i)
    total_forces = scatter_forces_onto_particles(total_forces, j_indices, -forces_on_i)
    return total_forces


def compute_harmonic_bond_forces(
    positions: torch.Tensor,
    bond_pairs_by_type: Mapping[str, torch.Tensor],
    k_by_type: Mapping[str, float],
    r0_by_type: Mapping[str, float],
    box: float,
) -> torch.Tensor:
    """Compute differentiable harmonic bond forces across all bonded topology groups."""
    total_bond_forces = torch.zeros_like(positions)
    for bond_type, pairs in bond_pairs_by_type.items():
        total_bond_forces = total_bond_forces + compute_single_bond_type_forces(
            positions, pairs, k_by_type[bond_type], r0_by_type[bond_type], box
        )
    return total_bond_forces


def compute_harmonic_bond_forces_fused(
    positions: torch.Tensor,
    pairs: torch.Tensor,
    spring_constants: torch.Tensor,
    rest_lengths: torch.Tensor,
    box: float,
) -> torch.Tensor:
    """Compute all harmonic bond forces in one vectorized pass.

    This is equivalent to :func:`compute_harmonic_bond_forces`, but combines all
    bond types before computing forces.  The separate public function remains
    available for validation and compatibility; the fused form avoids one
    displacement/norm/scatter sequence per bond type during a rollout.
    """
    i_indices = pairs[:, 0]
    j_indices = pairs[:, 1]
    delta = positions[i_indices] - positions[j_indices]
    delta_min = apply_minimum_image_displacement(delta, box)
    distances = safe_vector_norm(delta_min, dim=-1, keepdim=True)
    force_magnitudes = -spring_constants[:, None] * (distances - rest_lengths[:, None])
    forces_on_i = force_magnitudes * (delta_min / distances)

    total_forces = torch.zeros_like(positions)
    total_forces = total_forces.scatter_add(
        0, i_indices[:, None].expand(-1, positions.shape[-1]), forces_on_i
    )
    return total_forces.scatter_add(
        0, j_indices[:, None].expand(-1, positions.shape[-1]), -forces_on_i
    )


def brownian_dynamics(
    positions: torch.Tensor,
    total_force: torch.Tensor,
    mobilities: torch.Tensor,
    dt: float,
    noise_displacement: torch.Tensor,
    box: float,
) -> torch.Tensor:
    """Execute one overdamped Langevin (Euler-Maruyama) integration step.

        x_new = x + mobility * force * dt + noise_displacement

    The result is wrapped back into the periodic box.
    """
    deterministic_displacement = mobilities * total_force * dt
    updated_positions = positions + deterministic_displacement + noise_displacement
    return wrap_centered_coordinates(updated_positions, box)
