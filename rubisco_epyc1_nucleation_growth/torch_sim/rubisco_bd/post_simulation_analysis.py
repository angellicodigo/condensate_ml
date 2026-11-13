"""Contact and cluster analysis of trajectory frames.

Hard contacts / clusters reproduce the HOOMD postprocessing exactly but are NOT
differentiable (threshold + union-find). ``compute_soft_contacts`` is the
differentiable surrogate to use inside a training loss.
"""

from __future__ import annotations

import torch

from .geometry import apply_minimum_image_displacement, safe_vector_norm


def _sticker_patch_distances(
    positions: torch.Tensor,
    sticker_indices: torch.Tensor,
    patch_indices: torch.Tensor,
    box: float,
) -> torch.Tensor:
    """Minimum-image sticker-patch distances, shape (..., n_stickers, n_patches).

    ``positions`` may be a single frame (N, 3) or a trajectory (T, N, 3).
    """
    stickers = positions.index_select(-2, sticker_indices)
    patches = positions.index_select(-2, patch_indices)
    delta = stickers.unsqueeze(-2) - patches.unsqueeze(-3)
    delta = apply_minimum_image_displacement(delta, box)
    return safe_vector_norm(delta, dim=-1)


def compute_hard_contacts(
    positions: torch.Tensor,
    sticker_indices: torch.Tensor,
    patch_indices: torch.Tensor,
    box: float,
    cutoff: float = 3.0,
) -> torch.Tensor:
    """Boolean sticker-patch contact matrix (distance < cutoff). Not differentiable."""
    with torch.no_grad():
        return _sticker_patch_distances(positions, sticker_indices, patch_indices, box) < cutoff


def compute_soft_contacts(
    positions: torch.Tensor,
    sticker_indices: torch.Tensor,
    patch_indices: torch.Tensor,
    box: float,
    cutoff: float = 3.0,
    tau: float = 0.1,
) -> torch.Tensor:
    """Differentiable contact matrix: sigmoid((cutoff - distance) / tau) in [0, 1].

    Accepts a frame (N, 3) or trajectory (T, N, 3); returns (..., n_stickers, n_patches).
    As tau -> 0 this approaches compute_hard_contacts.
    """
    dist = _sticker_patch_distances(positions, sticker_indices, patch_indices, box)
    return torch.sigmoid((cutoff - dist) / tau)


def compute_cluster_graph_metrics(
    contacts_binary: torch.Tensor,
    sticker_chain_ids: torch.Tensor,
    patch_rubisco_ids: torch.Tensor,
    n_rubisco: int,
    n_chains: int,
) -> tuple[int, int, float]:
    """(n_clusters, largest_cluster_size, largest_fraction) of the macromolecule graph.

    Nodes are the n_rubisco Rubisco molecules and n_chains EPYC1 chains. A sticker-patch
    contact joins its chain to its Rubisco. Sizes are counted in nodes, so the fraction
    divides by n_rubisco + n_chains (72), as in the HOOMD script.
    """
    n_nodes = n_rubisco + n_chains
    hits = torch.nonzero(contacts_binary > 0, as_tuple=False).cpu()
    chain_ids = sticker_chain_ids.cpu()[hits[:, 0]]
    rubisco_ids = patch_rubisco_ids.cpu()[hits[:, 1]]
    edges = set(zip((n_rubisco + chain_ids).tolist(), rubisco_ids.tolist()))

    parent = list(range(n_nodes))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in edges:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    sizes: dict[int, int] = {}
    for i in range(n_nodes):
        root = find(i)
        sizes[root] = sizes.get(root, 0) + 1

    largest = max(sizes.values()) if sizes else 0
    return len(sizes), largest, largest / n_nodes


def compute_frame_metrics(
    pos_frame: torch.Tensor,
    sticker_indices: torch.Tensor,
    patch_indices: torch.Tensor,
    sticker_chain_ids: torch.Tensor,
    patch_rubisco_ids: torch.Tensor,
    box: float,
    step: int,
    sim_time: float,
    frame_idx: int,
    name: str,
    kon: float,
    epsilon_used: float,
    koff: float,
    n_rubisco: int,
    n_chains: int,
    contact_cutoff: float = 3.0,
) -> dict:
    """Compute contact and cluster metrics for a single trajectory frame."""
    contacts_binary = compute_hard_contacts(
        pos_frame, sticker_indices, patch_indices, box, cutoff=contact_cutoff
    )
    n_contacts = int(contacts_binary.sum().item())
    n_clusters, largest_size, fraction = compute_cluster_graph_metrics(
        contacts_binary, sticker_chain_ids, patch_rubisco_ids, n_rubisco, n_chains
    )

    return {
        "frame": frame_idx,
        "step": step,
        "time": sim_time,
        "box": box,
        "kon_or_epsilon": kon,
        "epsilon_used": epsilon_used,
        "koff_recorded": koff,
        "contacts": n_contacts,
        "bonds": n_contacts,
        "clusters": n_clusters,
        "largest": largest_size,
        "fraction": fraction,
    }
