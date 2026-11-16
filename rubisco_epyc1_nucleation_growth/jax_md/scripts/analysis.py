"""GSD analysis and artifact writing for JAX MD simulations."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .constants import N_BEADS, N_C, N_R, PATCHES, STICKER_INDEX

CONTACT_CUTOFF = 3.0


def _minimum_image(delta: np.ndarray, box: float) -> np.ndarray:
    return delta - box * np.round(delta / box)


def _union_find(n_nodes: int, edges: list[tuple[int, int]]) -> tuple[int, int, float]:
    parent = list(range(n_nodes))

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for first, second in edges:
        first_root, second_root = find(first), find(second)
        if first_root != second_root:
            parent[second_root] = first_root

    sizes: dict[int, int] = {}
    for node in range(n_nodes):
        root = find(node)
        sizes[root] = sizes.get(root, 0) + 1
    values = sorted(sizes.values(), reverse=True)
    if not values:
        return 0, 0, 0.0
    return len(values), values[0], values[0] / n_nodes


def _contact_indices() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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


def analyze_gsd(path: str | Path, dt: float, contact_cutoff: float = CONTACT_CUTOFF) -> pd.DataFrame:
    """Analyze GSD frames and return contacts/clusters as a DataFrame."""
    import gsd.hoomd
    import pandas as pd

    patch_indices, sticker_indices, sticker_chain_ids, patch_rubisco_ids = _contact_indices()
    records = []
    with gsd.hoomd.open(name=str(path), mode="r") as trajectory:
        for frame_index, frame in enumerate(trajectory):
            positions = np.asarray(frame.particles.position)
            box = float(frame.configuration.box[0])
            sticker_positions = positions[sticker_indices]
            patch_positions = positions[patch_indices]
            delta = sticker_positions[:, None, :] - patch_positions[None, :, :]
            distances = np.linalg.norm(_minimum_image(delta, box), axis=-1)
            contacts = np.argwhere(distances < contact_cutoff)
            edges = [
                (N_R + int(sticker_chain_ids[s]), int(patch_rubisco_ids[p]))
                for s, p in contacts
            ]
            clusters, largest, fraction = _union_find(N_R + N_C, edges)
            step = int(frame.configuration.step)
            records.append({
                "frame": frame_index,
                "step": step,
                "time": step * dt,
                "contacts": len(contacts),
                "bonds": len(contacts),
                "clusters": clusters,
                "largest": largest,
                "fraction": fraction,
                "box": box,
            })
    return pd.DataFrame(records)


def write_xyz(path: str | Path, gsd_path: str | Path) -> None:
    """Convert GSD frames to the simple XYZ format used by the HOOMD script."""
    import gsd.hoomd

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    particle_types = ("R", "P", "S", "T")
    with gsd.hoomd.open(name=str(gsd_path), mode="r") as trajectory, path.open("w") as output:
        for frame in trajectory:
            positions = np.asarray(frame.particles.position)
            type_ids = np.asarray(frame.particles.typeid)
            output.write(f"{len(positions)}\n")
            output.write(f"step {int(frame.configuration.step)}\n")
            for type_id, position in zip(type_ids, positions):
                output.write(f"{particle_types[int(type_id)]} {position[0]} {position[1]} {position[2]}\n")


def write_analysis_plots(data: pd.DataFrame, plots_dir: str | Path, output_name: str) -> None:
    """Write the four HOOMD-style metric plots."""
    import matplotlib.pyplot as plt

    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    labels = {
        "bonds": "Contacts / bonds",
        "clusters": "Number of clusters",
        "fraction": "Largest cluster fraction",
        "largest": "Largest cluster size",
    }
    for metric, ylabel in labels.items():
        figure, axis = plt.subplots(figsize=(8, 5))
        axis.plot(data["time"], data[metric])
        axis.set_xlabel("Time")
        axis.set_ylabel(ylabel)
        axis.set_title(f"{output_name}: {ylabel} vs time")
        axis.grid(alpha=0.25)
        figure.tight_layout()
        figure.savefig(plots_dir / f"{output_name}_{metric}.jpeg", dpi=180)
        plt.close(figure)


def write_analysis_outputs(
    gsd_path: str | Path,
    xyz_path: str | Path,
    csv_path: str | Path,
    plots_dir: str | Path,
    output_name: str,
    dt: float,
    epsilon: float,
) -> pd.DataFrame:
    """Create XYZ, CSV, and JPEG outputs from one GSD trajectory."""
    data = analyze_gsd(gsd_path, dt=dt)
    data.insert(4, "epsilon_used", epsilon)
    write_xyz(xyz_path, gsd_path)
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(csv_path, index=False)
    write_analysis_plots(data, plots_dir, output_name)
    return data
