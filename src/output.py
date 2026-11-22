"""Trajectory output and post-simulation analysis for JAX MD Rubisco simulations."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from .constants import STERIC_PAIR_PARAMS

if TYPE_CHECKING:
    import pandas as pd


def write_gsd(
    path: str | Path,
    trajectory: Any,
    box: float,
    particle_type_ids: Any,
    bond_pairs: Any,
    write_every: int,
    bond_type_ids: Any,
    particle_types: tuple[str, ...] = ("R", "P", "S", "T"),
    bond_types: tuple[str, ...] = ("chain", "patch_tether"),
) -> Path:
    """Writes trajectory frames and topology data to a HOOMD-compatible GSD file.

    Inputs:
        path: Destination GSD file path.
        trajectory: Trajectory frames shaped (n_frames, n_particles, 3).
        box: Cubic simulation-box length.
        particle_type_ids: Integer type ID for every particle.
        bond_pairs: Bond endpoint indices shaped (n_bonds, 2).
        write_every: Number of simulation steps between saved frames.
        bond_type_ids: Integer bond type ID for every bond.
        particle_types: Names corresponding to particle type IDs.
        bond_types: Names corresponding to bond type IDs.
    Outputs:
        The destination path as a pathlib.Path.
    """
    try:
        import gsd.hoomd
    except ImportError as error:
        raise ImportError("GSD output requires the 'gsd' package") from error

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames = np.asarray(trajectory)
    type_ids = np.asarray(particle_type_ids, dtype=np.uint32)
    bonds = np.asarray(bond_pairs, dtype=np.uint32)
    bond_ids = np.asarray(bond_type_ids, dtype=np.uint32)

    with gsd.hoomd.open(name=str(output_path), mode="w") as stream:
        for frame_index, positions in enumerate(frames):
            frame = gsd.hoomd.Frame()
            frame.configuration.box = [box, box, box, 0, 0, 0]
            frame.configuration.step = int(frame_index * write_every)
            frame.particles.N = positions.shape[0]
            frame.particles.types = list(particle_types)
            frame.particles.position = np.asarray(positions, dtype=np.float32)
            frame.particles.typeid = type_ids
            frame.bonds.N = bonds.shape[0]
            frame.bonds.types = list(bond_types)
            frame.bonds.group = bonds
            frame.bonds.typeid = bond_ids
            stream.append(frame)
    return output_path


def _minimum_image(delta: np.ndarray, box: float) -> np.ndarray:
    """Applies the periodic minimum-image convention to displacement vectors.

    Inputs:
        delta: Displacement vectors with final axis dimension 3.
        box: Periodic box side length.
    Outputs:
        Displacements mapped to the nearest periodic image.
    """
    return delta - box * np.round(delta / box)


def _union_find(n_nodes: int, edges: list[tuple[int, int]]) -> tuple[int, int, float]:
    """Computes connected component metrics in a graph using the union-find algorithm.

    Inputs:
        n_nodes: Total number of graph nodes.
        edges: Pairs of node IDs connected by edges.
    Outputs:
        Tuple of (n_components, largest_component_size, largest_component_fraction).
    """
    parent = list(range(n_nodes))

    def find(node: int) -> int:
        """Returns the representative root for one node with path compression.

        Inputs:
            node: Integer node identifier.
        Outputs:
            Root node identifier.
        """
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


def _contact_indices(
    n_rubisco: int,
    n_patches_per_rubisco: int,
    n_chains: int,
    n_beads_per_chain: int,
    sticker_indices: Sequence[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Builds index arrays linking patches, stickers, chains, and Rubisco complexes.

    Inputs:
        n_rubisco: Number of Rubisco centers.
        n_patches_per_rubisco: Number of patches per Rubisco.
        n_chains: Number of EPYC1 chains.
        n_beads_per_chain: Beads per EPYC1 chain.
        sticker_indices: Local bead indices within a chain acting as stickers.
    Outputs:
        Tuple of (patch_indices, sticker_global_indices, sticker_chain_ids, patch_rubisco_ids).
    """
    patch_start = n_rubisco
    chain_start = n_rubisco + n_rubisco * n_patches_per_rubisco
    patch_indices = np.arange(patch_start, chain_start, dtype=np.int64)
    sticker_global_indices = []
    sticker_chain_ids = []
    for chain_id in range(n_chains):
        base = chain_start + chain_id * n_beads_per_chain
        for local_index in sticker_indices:
            sticker_global_indices.append(base + local_index)
            sticker_chain_ids.append(chain_id)
    patch_rubisco_ids = np.arange(len(patch_indices), dtype=np.int64) // n_patches_per_rubisco
    return (
        patch_indices,
        np.asarray(sticker_global_indices, dtype=np.int64),
        np.asarray(sticker_chain_ids, dtype=np.int64),
        patch_rubisco_ids,
    )


def analyze_gsd(
    path: str | Path,
    dt: float,
    n_rubisco: int,
    n_patches_per_rubisco: int,
    n_chains: int,
    n_beads_per_chain: int,
    sticker_indices: Sequence[int],
    contact_cutoff: float = STERIC_PAIR_PARAMS[("P", "T")][2],
) -> pd.DataFrame:
    """Analyzes GSD trajectory frames and returns contact and cluster statistics as a DataFrame.

    Inputs:
        path: Path to GSD trajectory file.
        dt: Integration timestep used to convert steps to physical time.
        n_rubisco: Number of Rubisco centers.
        n_patches_per_rubisco: Number of patches per Rubisco.
        n_chains: Number of EPYC1 chains.
        n_beads_per_chain: Beads per chain.
        sticker_indices: Local bead indices acting as stickers.
        contact_cutoff: Maximum periodic distance defining a sticker-patch contact.
    Outputs:
        DataFrame containing frame, time, contact, cluster, and box metrics.
    """
    import gsd.hoomd
    import pandas as pd

    (
        patch_indices,
        sticker_global_indices,
        sticker_chain_ids,
        patch_rubisco_ids,
    ) = _contact_indices(
        n_rubisco=n_rubisco,
        n_patches_per_rubisco=n_patches_per_rubisco,
        n_chains=n_chains,
        n_beads_per_chain=n_beads_per_chain,
        sticker_indices=sticker_indices,
    )
    records = []
    with gsd.hoomd.open(name=str(path), mode="r") as trajectory:
        for frame_index, frame in enumerate(trajectory):
            positions = np.asarray(frame.particles.position)
            box = float(frame.configuration.box[0])
            sticker_positions = positions[sticker_global_indices]
            patch_positions = positions[patch_indices]
            delta = sticker_positions[:, None, :] - patch_positions[None, :, :]
            distances = np.linalg.norm(_minimum_image(delta, box), axis=-1)
            contacts = np.argwhere(distances < contact_cutoff)
            edges = [
                (n_rubisco + int(sticker_chain_ids[s]), int(patch_rubisco_ids[p]))
                for s, p in contacts
            ]
            clusters, largest, fraction = _union_find(n_rubisco + n_chains, edges)
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
    """Converts a GSD trajectory into XYZ format with atom type labels.

    Inputs:
        path: Destination XYZ file path.
        gsd_path: Source GSD trajectory file path.
    Outputs:
        None.
    """
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
    """Generates JPEG line plots for simulation contact and cluster time series.

    Inputs:
        data: DataFrame containing trajectory analysis columns.
        plots_dir: Destination directory path for plot images.
        output_name: Base prefix for plot file names and titles.
    Outputs:
        None.
    """
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
    n_rubisco: int,
    n_patches_per_rubisco: int,
    n_chains: int,
    n_beads_per_chain: int,
    sticker_indices: Sequence[int],
    contact_cutoff: float = STERIC_PAIR_PARAMS[("P", "T")][2],
) -> pd.DataFrame:
    """Generates all post-simulation trajectory outputs including XYZ, CSV, and metric plots.

    Inputs:
        gsd_path: Source GSD trajectory file path.
        xyz_path: Destination XYZ file path.
        csv_path: Destination CSV file path.
        plots_dir: Destination directory for metric plots.
        output_name: Prefix used for plot names and titles.
        dt: Simulation integration timestep.
        epsilon: Binding parameter value recorded in the CSV.
        n_rubisco: Number of Rubisco centers.
        n_patches_per_rubisco: Number of patches per Rubisco.
        n_chains: Number of EPYC1 chains.
        n_beads_per_chain: Beads per chain.
        sticker_indices: Local bead indices within a chain acting as stickers.
        contact_cutoff: Maximum periodic distance defining a sticker-patch contact.
    Outputs:
        Pandas DataFrame containing the computed frame metrics.
    """
    data = analyze_gsd(
        path=gsd_path,
        dt=dt,
        n_rubisco=n_rubisco,
        n_patches_per_rubisco=n_patches_per_rubisco,
        n_chains=n_chains,
        n_beads_per_chain=n_beads_per_chain,
        sticker_indices=sticker_indices,
        contact_cutoff=contact_cutoff,
    )
    data.insert(4, "epsilon_used", epsilon)
    write_xyz(xyz_path, gsd_path)
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(csv_path, index=False)
    write_analysis_plots(data, plots_dir, output_name)
    return data
