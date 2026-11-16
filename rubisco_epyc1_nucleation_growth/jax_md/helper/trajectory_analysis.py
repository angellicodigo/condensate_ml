"""In-memory GSD analysis and plotting helpers."""

from __future__ import annotations

from pathlib import Path

import gsd.hoomd
import numpy as np
import pandas as pd

from .validate_snapshot import (
    N_BEADS,
    N_C,
    N_R,
    PATCHES,
    STICKER_INDEX,
)

CONTACT_CUTOFF = 3.0
DEFAULT_DT = 0.0005


def _particle_indices():
    """Return patch indices, sticker indices, chain IDs, and Rubisco IDs."""
    patch_start = N_R
    chain_start = N_R + N_R * PATCHES
    patch_indices = np.arange(patch_start, chain_start, dtype=np.int64)
    sticker_indices = []
    sticker_chain_ids = []

    for chain_id in range(N_C):
        base = chain_start + chain_id * N_BEADS
        for local_index in STICKER_INDEX:
            sticker_indices.append(base + local_index)
            sticker_chain_ids.append(chain_id)

    patch_rubisco_ids = np.arange(len(patch_indices), dtype=np.int64) // PATCHES
    return (
        patch_indices,
        np.asarray(sticker_indices, dtype=np.int64),
        np.asarray(sticker_chain_ids, dtype=np.int64),
        patch_rubisco_ids,
    )


def union_find(n_nodes: int, edges: list[tuple[int, int]]) -> tuple[int, int, float]:
    """Return number of clusters, largest cluster size, and largest fraction."""
    parent = list(range(n_nodes))

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for first, second in edges:
        root_first, root_second = find(first), find(second)
        if root_first != root_second:
            parent[root_second] = root_first

    sizes: dict[int, int] = {}
    for node in range(n_nodes):
        root = find(node)
        sizes[root] = sizes.get(root, 0) + 1

    values = sorted(sizes.values(), reverse=True)
    if not values:
        return 0, 0, 0.0
    return len(values), values[0], values[0] / n_nodes


def analyze_frame(
    positions: np.ndarray,
    box: float,
    frame_index: int,
    simulation_step: int,
    dt: float = DEFAULT_DT,
    contact_cutoff: float = CONTACT_CUTOFF,
) -> dict[str, float]:
    """Analyze sticker-patch contacts and condensate graph metrics."""
    patch_indices, sticker_indices, sticker_chain_ids, patch_rubisco_ids = _particle_indices()
    patch_positions = positions[patch_indices]
    sticker_positions = positions[sticker_indices]

    delta = sticker_positions[:, None, :] - patch_positions[None, :, :]
    delta = delta - box * np.round(delta / box)
    distances = np.linalg.norm(delta, axis=-1)
    contacts = np.argwhere(distances < contact_cutoff)

    edges = [
        (N_R + int(sticker_chain_ids[sticker_local]), int(patch_rubisco_ids[patch_local]))
        for sticker_local, patch_local in contacts
    ]
    n_clusters, largest, fraction = union_find(N_R + N_C, edges)
    n_contacts = len(contacts)

    return {
        "frame": frame_index,
        "step": simulation_step,
        "time": simulation_step * dt,
        "contacts": n_contacts,
        "bonds": n_contacts,
        "clusters": n_clusters,
        "largest": largest,
        "fraction": fraction,
        "box": box,
    }


def analyze_gsd(
    data_path: str | Path,
    dt: float = DEFAULT_DT,
    contact_cutoff: float = CONTACT_CUTOFF,
) -> pd.DataFrame:
    """Read a GSD trajectory and return its analysis as a DataFrame."""
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(data_path)

    records = []
    with gsd.hoomd.open(name=str(data_path), mode="r") as trajectory:
        for frame_index, frame in enumerate(trajectory):
            records.append(
                analyze_frame(
                    positions=np.asarray(frame.particles.position),
                    box=float(frame.configuration.box[0]),
                    frame_index=frame_index,
                    simulation_step=int(frame.configuration.step),
                    dt=dt,
                    contact_cutoff=contact_cutoff,
                )
            )
    return pd.DataFrame(records)

