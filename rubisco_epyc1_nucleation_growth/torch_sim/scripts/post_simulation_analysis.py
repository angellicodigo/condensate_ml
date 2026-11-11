"""Analysis helpers: contact detection and cluster graph metrics.

Functions for computing soft (differentiable) and hard (binary) sticker-patch
contacts, and for computing bipartite connected-component metrics over Rubisco
complexes and EPYC1 chains.
"""

from __future__ import annotations

import torch

from .geometry import apply_minimum_image_displacement


def compute_soft_contacts(
    positions: torch.Tensor,
    sticker_indices: torch.Tensor,
    patch_indices: torch.Tensor,
    box: float,
    cutoff: float = 3.0,
    tau: float = 0.1,
) -> torch.Tensor:
    """Compute smooth, differentiable sigmoid contact indicators between stickers and patches.

    Returns a matrix of shape (n_stickers, n_patches) with values in (0, 1),
    where values near 1 indicate a contact and values near 0 indicate no contact.
    """
    pos_stickers = positions[..., sticker_indices, :]
    pos_patches = positions[..., patch_indices, :]
    delta = pos_stickers[..., :, None, :] - pos_patches[..., None, :, :]
    delta_min = apply_minimum_image_displacement(delta, box)
    distances = torch.linalg.vector_norm(delta_min, dim=-1)
    return torch.sigmoid((cutoff - distances) / tau)


def compute_hard_contacts(
    positions: torch.Tensor,
    sticker_indices: torch.Tensor,
    patch_indices: torch.Tensor,
    box: float,
    cutoff: float = 3.0,
) -> torch.Tensor:
    """Compute exact binary indicator matrix for sticker-patch contacts within cutoff.

    Returns a boolean matrix of shape (n_stickers, n_patches) where True means
    the sticker-patch distance is below the cutoff.
    """
    pos_stickers = positions[..., sticker_indices, :]
    pos_patches = positions[..., patch_indices, :]
    delta = pos_stickers[..., :, None, :] - pos_patches[..., None, :, :]
    delta_min = apply_minimum_image_displacement(delta, box)
    distances = torch.linalg.vector_norm(delta_min, dim=-1)
    return distances < cutoff


def compute_cluster_graph_metrics(
    contacts_binary: torch.Tensor,
    sticker_chain_ids: torch.Tensor,
    patch_rubisco_ids: torch.Tensor,
    n_rubisco: int = 24,
    n_chains: int = 48,
) -> tuple[int, int, float]:
    """Compute bipartite connected-component metrics over Rubisco complexes and EPYC1 chains.

    Builds a union-find graph where nodes are Rubisco complexes (0..n_rubisco-1)
    and EPYC1 chains (n_rubisco..n_rubisco+n_chains-1). An edge exists between
    chain c and Rubisco r if any sticker on chain c contacts any patch on Rubisco r.

    Returns (cluster_count, largest_cluster_size, cluster_fraction).
    """
    total_nodes = n_rubisco + n_chains
    parent = list(range(total_nodes))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        root_a = find(a)
        root_b = find(b)
        if root_a != root_b:
            parent[root_b] = root_a

    active_contacts = torch.nonzero(contacts_binary, as_tuple=False)
    for row in range(active_contacts.shape[0]):
        sticker_idx = active_contacts[row, 0].item()
        patch_idx = active_contacts[row, 1].item()
        chain_id = sticker_chain_ids[sticker_idx].item()
        rubisco_id = patch_rubisco_ids[patch_idx].item()
        union(n_rubisco + chain_id, rubisco_id)

    groups: dict[int, list[int]] = {}
    for node in range(total_nodes):
        root = find(node)
        groups.setdefault(root, []).append(node)

    cluster_sizes = sorted(
        [len(members) for members in groups.values()], reverse=True
    )
    cluster_count = len(cluster_sizes)
    largest_cluster_size = cluster_sizes[0] if cluster_sizes else 0
    cluster_fraction = float(largest_cluster_size) / float(total_nodes)
    return cluster_count, largest_cluster_size, cluster_fraction
