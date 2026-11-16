"""GSD trajectory output for the JAX MD Rubisco simulation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


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
    """Write sampled positions to a HOOMD-compatible GSD file."""
    try:
        import gsd.hoomd
    except ImportError as error:  # pragma: no cover - environment dependent
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
