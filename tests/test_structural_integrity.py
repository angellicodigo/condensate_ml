"""Structural integrity and analysis logic tests.

Tests include:
9. Chains stay connected and patches stay put: bond lengths stay bounded, patches stay near
   2.5 from Rubisco cores, and 8 patches maintain their cubic symmetry angles.
10. Hand-built frame analysis: verifies cluster counts, largest cluster, and fraction on known
    ground-truth geometries (isolated, multi-contact bridging, and periodic boundary wrapping).
11. GSD round trip: trajectory written via write_gsd matches exactly when read back.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from condensate_ml import RubiscoEpyc1Simulation
from condensate_ml.constants import (
    CHAIN_R0,
    PATCH_DIRS,
    PATCH_RADIUS,
    PATCH_RCUT,
    PATCHES,
    STICKER_INDEX,
    TYPE_INDEX,
)
from condensate_ml.helper import minimum_image
from condensate_ml.output import analyze_gsd, write_gsd


# ---------------------------------------------------------------------------
# Test 9: Chains Stay Connected, Patches Stay Put
# ---------------------------------------------------------------------------


def compute_chain_bond_lengths(
    positions: np.ndarray,
    bond_pairs: np.ndarray,
    box: float,
) -> np.ndarray:
    """Computes minimum-image periodic distances for all harmonic bonds in a configuration.

    Inputs:
        positions: Coordinates array shaped (n_particles, 3).
        bond_pairs: Bond index pairs shaped (n_bonds, 2).
        box: Periodic box length.
    Outputs:
        Bond lengths array shaped (n_bonds,).
    """
    first = positions[bond_pairs[:, 0]]
    second = positions[bond_pairs[:, 1]]
    delta = minimum_image(first - second, box)
    return np.linalg.norm(delta, axis=-1)


def compute_patch_radial_distances_and_angles(
    positions: np.ndarray,
    n_rubisco: int,
    n_patches: int,
    box: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Computes radial distances and pairwise direction dot products for Rubisco surface patches.

    Inputs:
        positions: Coordinates array shaped (n_particles, 3).
        n_rubisco: Number of Rubisco core particles.
        n_patches: Number of patches per Rubisco.
        box: Periodic box length.
    Outputs:
        Tuple of (radial_distances shaped (n_rubisco * n_patches,), pairwise_dot_products).
    """
    patch_start = n_rubisco
    radial_distances = []
    dot_products = []

    for r_id in range(n_rubisco):
        core_pos = positions[r_id]
        patch_indices = np.arange(patch_start + r_id * n_patches, patch_start + (r_id + 1) * n_patches)
        patch_pos = positions[patch_indices]
        delta = minimum_image(patch_pos - core_pos, box)
        distances = np.linalg.norm(delta, axis=-1)
        radial_distances.extend(distances)

        unit_dirs = delta / distances[:, None]
        pair_dots = unit_dirs @ unit_dirs.T
        dot_products.append(pair_dots)

    return np.asarray(radial_distances), np.asarray(dot_products)


def test_chains_connected_and_patches_cubic_in_rollout() -> None:
    """During a simulation rollout, bond lengths stay bounded and patches maintain cubic angles."""
    n_steps = 60
    write_every = 10
    box = 40.0
    epsilon = 1.2

    simulation = RubiscoEpyc1Simulation(
        n_steps=n_steps, box=box, write_every=write_every, seed=101, checkpoint_blocks=True
    )
    trajectory = np.asarray(simulation({"epsilon": epsilon}))

    ideal_patch_dots = PATCH_DIRS @ PATCH_DIRS.T
    bond_types = np.asarray(simulation.bond_type_ids)
    chain_mask = (bond_types == 0)
    tether_mask = (bond_types == 1)

    for frame_idx, frame in enumerate(trajectory):
        # 1. Chain bond lengths check (rest length 1.2, must stay below threshold 2.5)
        bond_lengths = compute_chain_bond_lengths(frame, np.asarray(simulation.bond_pairs), box)
        max_chain_bond = float(np.max(bond_lengths[chain_mask]))
        assert max_chain_bond < 2.5, f"Frame {frame_idx}: max chain bond {max_chain_bond} exceeded 2.5"

        # 2. Patch tether lengths (rest length 2.5, must stay in [2.0, 3.0])
        max_tether = float(np.max(bond_lengths[tether_mask]))
        min_tether = float(np.min(bond_lengths[tether_mask]))
        assert min_tether > 2.0 and max_tether < 3.0, (
            f"Frame {frame_idx}: patch tether outside [2.0, 3.0]: [{min_tether}, {max_tether}]"
        )

        # 2. Rubisco core-patch radial distances (rest length 2.5, must stay in [2.0, 3.0])
        patch_distances, patch_dots = compute_patch_radial_distances_and_angles(
            frame, simulation.n_rubisco, simulation.n_patches_per_rubisco, box
        )
        assert np.all(patch_distances > 2.0), f"Frame {frame_idx}: patch too close to core"
        assert np.all(patch_distances < 3.0), f"Frame {frame_idx}: patch too far from core"

        # 3. Eight patches maintain cubic symmetry angles (dot products close to ideal cubic angles)
        for r_id in range(simulation.n_rubisco):
            dots = patch_dots[r_id]
            max_angle_error = float(np.max(np.abs(dots - ideal_patch_dots)))
            assert max_angle_error < 0.25, (
                f"Frame {frame_idx}, Rubisco {r_id}: cubic patch angle error {max_angle_error} exceeded 0.25"
            )


# ---------------------------------------------------------------------------
# Test 10: Analysis Logic on Hand-Built Frames
# ---------------------------------------------------------------------------


def build_isolated_configuration(
    n_particles: int,
    chain_start: int,
) -> np.ndarray:
    """Creates a configuration where Rubisco and chain particles are widely separated in y by 20.0.

    Inputs:
        n_particles: Total particle count.
        chain_start: Index where chain particles begin.
    Outputs:
        Positions array shaped (n_particles, 3).
    """
    coords = np.zeros((n_particles, 3), dtype=np.float32)
    # Rubisco particles at y = -10.0
    coords[:chain_start, 1] = -10.0
    coords[:chain_start, 0] = np.linspace(-15.0, 15.0, chain_start)

    # Chain particles at y = +10.0 (distance in y = 20.0 >> contact_cutoff = 3.0)
    n_chain_particles = n_particles - chain_start
    coords[chain_start:, 1] = 10.0
    coords[chain_start:, 0] = np.linspace(-15.0, 15.0, n_chain_particles)
    return coords


def create_hand_built_gsd_trajectory(
    filepath: Path,
    frames: list[np.ndarray],
    simulation: RubiscoEpyc1Simulation,
) -> Path:
    """Helper writing a list of hand-built frames to a GSD file for analysis testing.

    Inputs:
        filepath: Destination file path.
        frames: List of coordinate arrays each shaped (n_particles, 3).
        simulation: Configured simulation instance providing topology.
    Outputs:
        Path to the written GSD file.
    """
    trajectory = np.stack(frames, axis=0)
    return write_gsd(
        path=filepath,
        trajectory=trajectory,
        box=simulation.box,
        particle_type_ids=simulation.type_ids,
        bond_pairs=simulation.bond_pairs,
        write_every=1,
        bond_type_ids=simulation.bond_type_ids,
    )


def test_analysis_metrics_on_hand_built_frames() -> None:
    """Validates cluster counts, largest cluster, and fraction on known ground-truth geometries."""
    box = 40.0
    simulation = RubiscoEpyc1Simulation(n_steps=0, box=box, write_every=1, seed=1)
    n_particles = simulation.n_particles
    n_complexes = simulation.n_rubisco + simulation.n_chains  # 24 + 48 = 72

    chain_start = simulation.n_rubisco * (1 + simulation.n_patches_per_rubisco)
    patch_start = simulation.n_rubisco
    beads_per_chain = simulation.n_beads_per_chain

    # -------------------------------------------------------
    # Frame 0: No contacts (all particles far apart)
    # Expected: 0 contacts, 72 clusters, largest = 1, fraction = 1 / 72
    # -------------------------------------------------------
    frame_0 = build_isolated_configuration(n_particles, chain_start)

    # -------------------------------------------------------
    # Frame 1: One chain bridging two Rubisco complexes
    # Chain 0 sticker 0 contacts Rubisco 0 patch 0
    # Chain 0 sticker 1 contacts Rubisco 1 patch 0
    # Expected: 2 contacts, 70 clusters, largest = 3, fraction = 3 / 72
    # -------------------------------------------------------
    frame_1 = build_isolated_configuration(n_particles, chain_start)

    rubisco_0_patch_0 = patch_start
    rubisco_1_patch_0 = patch_start + simulation.n_patches_per_rubisco

    chain_0_sticker_0 = chain_start + STICKER_INDEX[0]
    chain_0_sticker_1 = chain_start + STICKER_INDEX[1]

    frame_1[rubisco_0_patch_0] = np.array([0.0, 0.0, 0.0])
    frame_1[chain_0_sticker_0] = np.array([0.5, 0.0, 0.0])  # distance = 0.5 < 3.0

    frame_1[rubisco_1_patch_0] = np.array([5.0, 0.0, 0.0])
    frame_1[chain_0_sticker_1] = np.array([5.5, 0.0, 0.0])  # distance = 0.5 < 3.0

    # -------------------------------------------------------
    # Frame 2: Contact across the periodic box boundary
    # Rubisco 0 patch 0 at +19.9, Chain 0 sticker 0 at -19.9
    # Periodic minimum-image distance = 0.2 < 3.0
    # Expected: 1 contact, 71 clusters, largest = 2, fraction = 2 / 72
    # -------------------------------------------------------
    frame_2 = build_isolated_configuration(n_particles, chain_start)
    frame_2[rubisco_0_patch_0] = np.array([19.9, 0.0, 0.0])
    frame_2[chain_0_sticker_0] = np.array([-19.9, 0.0, 0.0])

    with tempfile.TemporaryDirectory() as temp_dir:
        gsd_file = Path(temp_dir) / "test_frames.gsd"
        create_hand_built_gsd_trajectory(gsd_file, [frame_0, frame_1, frame_2], simulation)

        df = analyze_gsd(
            path=gsd_file,
            dt=0.001,
            n_rubisco=simulation.n_rubisco,
            n_patches_per_rubisco=simulation.n_patches_per_rubisco,
            n_chains=simulation.n_chains,
            n_beads_per_chain=simulation.n_beads_per_chain,
            sticker_indices=STICKER_INDEX,
            contact_cutoff=PATCH_RCUT,
        )

        assert len(df) == 3

        # Verify Frame 0
        assert df.loc[0, "contacts"] == 0
        assert df.loc[0, "clusters"] == n_complexes
        assert df.loc[0, "largest"] == 1
        assert np.isclose(df.loc[0, "fraction"], 1.0 / n_complexes)

        # Verify Frame 1
        assert df.loc[1, "contacts"] == 2
        assert df.loc[1, "clusters"] == n_complexes - 2  # 70
        assert df.loc[1, "largest"] == 3
        assert np.isclose(df.loc[1, "fraction"], 3.0 / n_complexes)

        # Verify Frame 2
        assert df.loc[2, "contacts"] == 1
        assert df.loc[2, "clusters"] == n_complexes - 1  # 71
        assert df.loc[2, "largest"] == 2
        assert np.isclose(df.loc[2, "fraction"], 2.0 / n_complexes)


# ---------------------------------------------------------------------------
# Test 11: GSD Round Trip
# ---------------------------------------------------------------------------


def test_gsd_round_trip() -> None:
    """Writing a trajectory to GSD and reading it back preserves positions, types, and bonds."""
    import gsd.hoomd

    box = 40.0
    write_every = 5
    simulation = RubiscoEpyc1Simulation(n_steps=10, box=box, write_every=write_every, seed=101)
    trajectory = np.asarray(simulation({"epsilon": 1.0}))
    n_frames, n_particles, _ = trajectory.shape

    with tempfile.TemporaryDirectory() as temp_dir:
        gsd_path = Path(temp_dir) / "round_trip.gsd"
        written_path = write_gsd(
            path=gsd_path,
            trajectory=trajectory,
            box=box,
            particle_type_ids=simulation.type_ids,
            bond_pairs=simulation.bond_pairs,
            write_every=write_every,
            bond_type_ids=simulation.bond_type_ids,
        )
        assert written_path.exists()

        with gsd.hoomd.open(name=str(written_path), mode="r") as stream:
            assert len(stream) == n_frames

            for frame_index, frame in enumerate(stream):
                assert frame.configuration.step == frame_index * write_every
                np.testing.assert_allclose(frame.configuration.box[:3], [box, box, box])

                # Check particles
                assert frame.particles.N == n_particles
                np.testing.assert_allclose(
                    frame.particles.position, trajectory[frame_index], atol=1e-6
                )
                np.testing.assert_array_equal(
                    frame.particles.typeid, np.asarray(simulation.type_ids)
                )

                # Check bonds
                assert frame.bonds.N == len(simulation.bond_pairs)
                np.testing.assert_array_equal(
                    frame.bonds.group, np.asarray(simulation.bond_pairs)
                )
                np.testing.assert_array_equal(
                    frame.bonds.typeid, np.asarray(simulation.bond_type_ids)
                )
