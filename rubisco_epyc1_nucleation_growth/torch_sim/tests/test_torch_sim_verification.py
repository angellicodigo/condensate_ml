"""Automated Verification Suite for Differentiable Rubisco-EPYC1 Simulation.

Satisfies Milestone 3 verification criteria (ORIGINAL_REQUEST.md § 4):
1. Analytical forces vs autograd sanity check across:
   - Harmonic bonds (528 chain, 192 patch tethers)
   - Bounded Gaussian sterics (9 particle pairs)
   - Continuous Gaussian attraction (stickers to patches)
   with relative tolerance < 1e-4.
2. Checkpointed backward pass gradient check:
   Verifies non-zero, finite, non-NaN gradients dLoss/dEpsilon flow through
   RubiscoEpyc1TimeStepper with grad_mode="checkpoint" and checkpoint_every=16.
3. End-to-end simulation execution with params.json:
   Verifies complete multi-run pipeline producing OVITO XYZ trajectories (792 particles),
   12-column CSV datasets, summary JSON, and 4 diagnostic JPEG plots.
"""

from __future__ import annotations

import csv
import json
import math
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, Mapping, Sequence

from PIL import Image
import numpy as np
import pandas as pd
import torch

# tests/ -> torch_sim/ (contains the rubisco_bd package and the sweep CLI)
TORCH_SIM_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TORCH_SIM_DIR))

from rubisco_bd import (
    RubiscoEpyc1TimeStepper,
    compute_harmonic_bond_forces,
    compute_gaussian_steric_forces,
    compute_gaussian_attraction_forces,
    compute_soft_contacts,
    apply_minimum_image_displacement,
    run_single_simulation,
)

PYTHON_EXE = sys.executable
SIM_SCRIPT_PATH = TORCH_SIM_DIR / "torch_kf_bd_kf_sweep_reformat.py"
PARAMS_JSON_PATH = TORCH_SIM_DIR / "params.json"

EXPECTED_HOOMD_COLUMNS: list[str] = [
    "frame",
    "step",
    "time",
    "box",
    "kon_or_epsilon",
    "epsilon_used",
    "koff_recorded",
    "contacts",
    "bonds",
    "clusters",
    "largest",
    "fraction",
]


# =============================================================================
# Potential Energy Functions for Verification (Honest Mathematical Formulations)
# =============================================================================

def compute_harmonic_bond_potential(
    positions: torch.Tensor,
    bondPairsByType: Mapping[str, torch.Tensor],
    kByType: Mapping[str, float],
    r0ByType: Mapping[str, float],
    box: float,
) -> torch.Tensor:
    """Computes total harmonic bond potential energy U_bonds(r) = sum 0.5 * k * (r - r0)^2."""
    total_potential = torch.tensor(0.0, dtype=positions.dtype, device=positions.device)
    for bond_type, pairs in bondPairsByType.items():
        k = kByType[bond_type]
        r0 = r0ByType[bond_type]
        i_indices = pairs[:, 0]
        j_indices = pairs[:, 1]
        delta = positions[i_indices] - positions[j_indices]
        delta_min = apply_minimum_image_displacement(delta, box)
        distances = torch.linalg.vector_norm(delta_min, dim=-1)
        bond_energies = 0.5 * k * torch.sum((distances - r0) ** 2)
        total_potential = total_potential + bond_energies
    return total_potential


def compute_bounded_gaussian_steric_potential(
    positions: torch.Tensor,
    epsilonMatrix: torch.Tensor,
    sigmaMatrix: torch.Tensor,
    rCutMatrix: torch.Tensor,
    box: float,
) -> torch.Tensor:
    """Computes total bounded Gaussian steric potential U_steric(r) across 9 steric pairs."""
    delta = positions[:, None, :] - positions[None, :, :]
    delta_min = apply_minimum_image_displacement(delta, box)
    dist_sq = torch.sum(delta_min * delta_min, dim=-1)
    within_cutoff_mask = (dist_sq < (rCutMatrix * rCutMatrix)) & (dist_sq > 1e-12)
    sig_sq = sigmaMatrix * sigmaMatrix
    safe_sig_sq = torch.clamp(sig_sq, min=1e-8)
    pairwise_potential = epsilonMatrix * torch.exp(-0.5 * dist_sq / safe_sig_sq)
    masked_potential = torch.where(within_cutoff_mask, pairwise_potential, torch.zeros_like(pairwise_potential))
    return 0.5 * torch.sum(masked_potential)


def compute_gaussian_attraction_potential(
    positions: torch.Tensor,
    stickerIndices: torch.Tensor,
    patchIndices: torch.Tensor,
    epsilonAtt: torch.Tensor,
    sigmaAtt: float,
    rCutAtt: float,
    box: float,
) -> torch.Tensor:
    """Computes total continuous Gaussian attractive potential between EPYC1 stickers and patches."""
    pos_stickers = positions[stickerIndices]
    pos_patches = positions[patchIndices]
    delta = pos_stickers[:, None, :] - pos_patches[None, :, :]
    delta_min = apply_minimum_image_displacement(delta, box)
    dist_sq = torch.sum(delta_min * delta_min, dim=-1)
    within_cutoff_mask = dist_sq < (rCutAtt * rCutAtt)
    sig_sq = sigmaAtt * sigmaAtt
    pairwise_potential = epsilonAtt * torch.exp(-0.5 * dist_sq / sig_sq)
    masked_potential = torch.where(within_cutoff_mask, pairwise_potential, torch.zeros_like(pairwise_potential))
    return torch.sum(masked_potential)


def compute_relative_force_error(
    analyticalForces: torch.Tensor,
    autogradForces: torch.Tensor,
) -> float:
    """Computes the relative L-infinity difference between analytical forces and -grad(U)."""
    absolute_difference = torch.abs(analyticalForces - autogradForces)
    max_absolute_error = torch.max(absolute_difference).item()
    max_force_magnitude = torch.max(torch.abs(analyticalForces)).item()
    scale = max(max_force_magnitude, 1e-12)
    return max_absolute_error / scale


# =============================================================================
# Exporter Artifact Verification Helpers
# =============================================================================

def verify_ovito_xyz_format(xyzPath: Path, expectedParticleCount: int = 792) -> None:
    """Validates that an XYZ file complies strictly with OVITO multi-frame specifications."""
    with open(xyzPath, "r", encoding="utf-8") as fh:
        lines = fh.readlines()

    lines_per_frame = expectedParticleCount + 2
    assert len(lines) % lines_per_frame == 0, (
        f"XYZ line count {len(lines)} is not a multiple of {lines_per_frame}!"
    )
    n_frames = len(lines) // lines_per_frame
    assert n_frames >= 2, f"Expected at least 2 frames, found {n_frames}"

    for frame_idx in range(n_frames):
        offset = frame_idx * lines_per_frame
        header_count = int(lines[offset].strip())
        assert header_count == expectedParticleCount, (
            f"Frame {frame_idx}: expected {expectedParticleCount} particles, got {header_count}"
        )
        comment_line = lines[offset + 1].strip()
        assert comment_line.startswith("step "), (
            f"Frame {frame_idx}: invalid comment header '{comment_line}'"
        )

        particle_lines = lines[offset + 2 : offset + lines_per_frame]
        labels = []
        for p_idx, line in enumerate(particle_lines):
            tokens = line.strip().split()
            assert len(tokens) == 4, (
                f"Frame {frame_idx}, particle {p_idx}: expected 4 tokens, got {len(tokens)}"
            )
            label, x_str, y_str, z_str = tokens
            labels.append(label)
            x, y, z = float(x_str), float(y_str), float(z_str)
            assert not (math.isnan(x) or math.isnan(y) or math.isnan(z)), "NaN in XYZ coordinates!"
            assert not (math.isinf(x) or math.isinf(y) or math.isinf(z)), "Inf in XYZ coordinates!"

        assert labels[:24] == ["R"] * 24, "First 24 particles must be Rubisco centers ('R')"
        assert labels[24:216] == ["P"] * 192, "Next 192 particles must be Rubisco patches ('P')"
        expected_chain_labels = ["S", "T", "S", "T", "S", "T", "S", "T", "S", "T", "S", "T"] * 48
        assert labels[216:792] == expected_chain_labels, "Remaining 576 particles must alternate S and T"


def verify_csv_parity_and_consistency(csvPath: Path, expectedColumns: Sequence[str]) -> None:
    """Validates 12-column CSV dataset schema, bounds, and internal mathematical parity."""
    with open(csvPath, mode="r", newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        header = next(reader)
    assert header == list(expectedColumns), (
        f"CSV header mismatch:\nExpected: {list(expectedColumns)}\nActual:   {header}"
    )

    df = pd.read_csv(csvPath)
    assert len(df) >= 2, f"Expected at least 2 rows, found {len(df)}"

    for idx, row in df.iterrows():
        assert row["frame"] == idx, f"Row {idx}: frame index mismatch"
        assert row["contacts"] == row["bonds"], f"Row {idx}: contacts != bonds ({row['contacts']} != {row['bonds']})"
        assert 1 <= row["clusters"] <= 72, f"Row {idx}: clusters out of bounds ({row['clusters']})"
        assert 1 <= row["largest"] <= 72, f"Row {idx}: largest cluster out of bounds ({row['largest']})"
        expected_frac = float(row["largest"]) / 72.0
        assert abs(row["fraction"] - expected_frac) < 1e-5, (
            f"Row {idx}: fraction {row['fraction']} != {expected_frac}"
        )
        assert abs(row["epsilon_used"] - (-abs(row["kon_or_epsilon"]))) < 1e-6, (
            f"Row {idx}: epsilon_used != -|kon_or_epsilon|"
        )


def verify_summary_json(summaryPath: Path, expectedName: str) -> None:
    """Validates summary JSON contains required simulation outcome statistics."""
    with open(summaryPath, "r", encoding="utf-8") as fh:
        summary = json.load(fh)
    assert summary["name"] == expectedName, (
        f"Summary name mismatch: expected '{expectedName}', got '{summary.get('name')}'"
    )
    assert "box" in summary
    assert "kon" in summary
    assert "epsilon_used" in summary
    assert "final_contacts" in summary
    assert "final_clusters" in summary
    assert "final_largest_cluster" in summary
    assert "final_fraction" in summary
    assert summary["final_clusters"] >= 1
    assert summary["final_largest_cluster"] >= 1


def verify_diagnostic_plots(plotsDir: Path, baseName: str, metrics: Sequence[str]) -> None:
    """Validates that all diagnostic plots are non-empty, non-blank JPEG images."""
    for metric in metrics:
        plot_path = plotsDir / f"{baseName}_{metric}.jpeg"
        assert plot_path.exists(), f"Plot image missing: {plot_path}"
        assert plot_path.stat().st_size > 1000, f"Plot image suspiciously small: {plot_path}"

        with Image.open(plot_path) as img:
            assert img.format == "JPEG", f"Expected JPEG format, got {img.format}"
            assert img.mode == "RGB", f"Expected RGB mode, got {img.mode}"
            assert img.width >= 800 and img.height >= 500, (
                f"Dimensions ({img.width}x{img.height}) below minimum 800x500"
            )
            arr = np.array(img)
            pixel_std = float(np.std(arr))
            assert pixel_std > 1.0, f"Plot image appears blank / uniform: std={pixel_std}"


# =============================================================================
# Test Suite 1: Analytical Forces vs Autograd Sanity Check
# =============================================================================

class TestAnalyticalForcesVsAutograd(unittest.TestCase):
    """Verifies analytical leaf forces match -grad(U) to < 10^-4 relative tolerance."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.box = 40.0
        cls.stepper = RubiscoEpyc1TimeStepper(n_steps=10, box=cls.box, seed=123)
        cls.initial_positions = cls.stepper.init({}).clone().detach()

    def test_harmonic_chain_bonds_forces_vs_autograd(self) -> None:
        """Verify 528 EPYC1 harmonic chain bonds match -grad(U_chain) to < 1e-4."""
        pos = self.initial_positions.clone().detach().requires_grad_(True)
        chain_bond_pairs = {"chain": self.stepper.bondPairsByType["chain"]}
        chain_k = {"chain": self.stepper.kByType["chain"]}
        chain_r0 = {"chain": self.stepper.r0ByType["chain"]}

        assert chain_bond_pairs["chain"].shape == (528, 2), "Expected exactly 528 chain bonds"

        u_chain = compute_harmonic_bond_potential(pos, chain_bond_pairs, chain_k, chain_r0, self.box)
        grad_chain = torch.autograd.grad(u_chain, pos, retain_graph=True)[0]
        autograd_force = -grad_chain

        f_chain_analytical = compute_harmonic_bond_forces(
            pos, chain_bond_pairs, chain_k, chain_r0, self.box
        )

        rel_error = compute_relative_force_error(f_chain_analytical, autograd_force)
        self.assertLess(
            rel_error,
            1e-4,
            f"Chain bond force relative error {rel_error:.6e} exceeds 1e-4",
        )

    def test_harmonic_patch_tethers_forces_vs_autograd(self) -> None:
        """Verify 192 Rubisco patch tethers match -grad(U_patch) to < 1e-4."""
        pos = self.initial_positions.clone().detach().requires_grad_(True)
        patch_bond_pairs = {"patch_tether": self.stepper.bondPairsByType["patch_tether"]}
        patch_k = {"patch_tether": self.stepper.kByType["patch_tether"]}
        patch_r0 = {"patch_tether": self.stepper.r0ByType["patch_tether"]}

        assert patch_bond_pairs["patch_tether"].shape == (192, 2), "Expected exactly 192 patch tethers"

        u_patch = compute_harmonic_bond_potential(pos, patch_bond_pairs, patch_k, patch_r0, self.box)
        grad_patch = torch.autograd.grad(u_patch, pos, retain_graph=True)[0]
        autograd_force = -grad_patch

        f_patch_analytical = compute_harmonic_bond_forces(
            pos, patch_bond_pairs, patch_k, patch_r0, self.box
        )

        rel_error = compute_relative_force_error(f_patch_analytical, autograd_force)
        self.assertLess(
            rel_error,
            1e-4,
            f"Patch tether force relative error {rel_error:.6e} exceeds 1e-4",
        )

    def test_harmonic_all_bonds_combined_forces_vs_autograd(self) -> None:
        """Verify all 720 harmonic bonds combined match -grad(U_total_bonds) to < 1e-4."""
        pos = self.initial_positions.clone().detach().requires_grad_(True)
        u_bonds = compute_harmonic_bond_potential(
            pos, self.stepper.bondPairsByType, self.stepper.kByType, self.stepper.r0ByType, self.box
        )
        grad_bonds = torch.autograd.grad(u_bonds, pos, retain_graph=True)[0]
        autograd_force = -grad_bonds

        f_bonds_analytical = compute_harmonic_bond_forces(
            pos, self.stepper.bondPairsByType, self.stepper.kByType, self.stepper.r0ByType, self.box
        )

        rel_error = compute_relative_force_error(f_bonds_analytical, autograd_force)
        self.assertLess(
            rel_error,
            1e-4,
            f"All bonds force relative error {rel_error:.6e} exceeds 1e-4",
        )

    def test_bounded_gaussian_sterics_forces_vs_autograd(self) -> None:
        """Verify bounded Gaussian sterics across 9 pairs match -grad(U_steric) to < 1e-4."""
        pos = self.initial_positions.clone().detach().requires_grad_(True)
        u_steric = compute_bounded_gaussian_steric_potential(
            pos,
            self.stepper.epsilonStericMatrix,
            self.stepper.sigmaStericMatrix,
            self.stepper.rCutStericMatrix,
            self.box,
        )
        grad_steric = torch.autograd.grad(u_steric, pos, retain_graph=True)[0]
        autograd_force = -grad_steric

        f_steric_analytical = compute_gaussian_steric_forces(
            pos,
            self.stepper.epsilonStericMatrix,
            self.stepper.sigmaStericMatrix,
            self.stepper.rCutStericMatrix,
            self.box,
        )

        rel_error = compute_relative_force_error(f_steric_analytical, autograd_force)
        self.assertLess(
            rel_error,
            1e-4,
            f"Sterics force relative error {rel_error:.6e} exceeds 1e-4",
        )

    def test_continuous_gaussian_attraction_forces_vs_autograd(self) -> None:
        """Verify continuous Gaussian attraction matches -grad(U_attract) to < 1e-4."""
        pos = self.initial_positions.clone().detach().requires_grad_(True)
        eps_att = torch.tensor(-1.8, dtype=pos.dtype, requires_grad=True)

        u_attract = compute_gaussian_attraction_potential(
            pos,
            self.stepper.stickerIndices,
            self.stepper.patchIndices,
            eps_att,
            sigmaAtt=1.0,
            rCutAtt=3.0,
            box=self.box,
        )
        grad_attract = torch.autograd.grad(u_attract, pos, retain_graph=True)[0]
        autograd_force = -grad_attract

        f_attract_analytical = compute_gaussian_attraction_forces(
            pos,
            self.stepper.stickerIndices,
            self.stepper.patchIndices,
            eps_att,
            sigma_att=1.0,
            r_cut_att=3.0,
            box=self.box,
        )

        rel_error = compute_relative_force_error(f_attract_analytical, autograd_force)
        self.assertLess(
            rel_error,
            1e-4,
            f"Attraction force relative error {rel_error:.6e} exceeds 1e-4",
        )


# =============================================================================
# Test Suite 2: Checkpointed Backward Pass Gradient Check
# =============================================================================

class TestCheckpointedBackwardPass(unittest.TestCase):
    """Verifies non-zero, finite, non-NaN gradients flow through RubiscoEpyc1TimeStepper."""

    def test_checkpointed_backward_gradient_flow(self) -> None:
        """Verify dLoss/dEpsilon flows through checkpointed simulation with checkpoint_every=16."""
        stepper = RubiscoEpyc1TimeStepper(
            n_steps=32,
            dt=0.0005,
            box=40.0,
            write_every=16,
            seed=42,
            grad_mode="checkpoint",
            checkpoint_every=16,
        )

        eps_param = torch.tensor(-1.5, requires_grad=True)
        trajectory = stepper({"epsilon": eps_param})

        # Expected frames: step 0, step 16, step 32 -> (3, 792, 3)
        self.assertEqual(
            trajectory.shape,
            (3, 792, 3),
            f"Expected trajectory shape (3, 792, 3), got {trajectory.shape}",
        )

        soft_contacts = compute_soft_contacts(
            trajectory,
            stepper.stickerIndices,
            stepper.patchIndices,
            box=40.0,
            cutoff=3.0,
            tau=0.1,
        )
        loss = soft_contacts.sum()
        loss.backward()

        self.assertIsNotNone(eps_param.grad, "Gradient of epsilon is None!")
        grad_val = eps_param.grad.item()
        self.assertFalse(math.isnan(grad_val), "Gradient of epsilon is NaN!")
        self.assertFalse(math.isinf(grad_val), "Gradient of epsilon is Inf!")
        self.assertNotEqual(grad_val, 0.0, "Gradient of epsilon is identically zero!")
        self.assertGreater(abs(grad_val), 1e-8, f"Gradient {grad_val} is negligibly small")

    def test_checkpointed_autograd_matches_finite_differences(self) -> None:
        """Verify checkpointed autograd matches central finite differences to < 1e-4."""
        box = 40.0
        h = 1e-5
        stepper = RubiscoEpyc1TimeStepper(
            n_steps=4,
            dt=0.0005,
            box=box,
            write_every=2,
            seed=42,
            dtype=torch.float64,
            grad_mode="checkpoint",
            checkpoint_every=2,
        )

        orig_randn_like = torch.randn_like
        torch.randn_like = lambda x: torch.zeros_like(x)

        def evaluate_loss(val: float, track_grad: bool) -> tuple[torch.Tensor, torch.Tensor]:
            tensor_val = torch.tensor(val, dtype=torch.float64, requires_grad=track_grad)
            traj = stepper({"epsilon": tensor_val})
            contacts = compute_soft_contacts(
                traj, stepper.stickerIndices, stepper.patchIndices, box=box, cutoff=3.0, tau=0.1
            )
            return contacts.sum(), tensor_val

        try:
            loss_base, eps_tensor = evaluate_loss(-1.5, track_grad=True)
            loss_base.backward()
            grad_autograd = eps_tensor.grad.item()

            loss_plus, _ = evaluate_loss(-1.5 + h, track_grad=False)
            loss_minus, _ = evaluate_loss(-1.5 - h, track_grad=False)
            grad_fd = (loss_plus.item() - loss_minus.item()) / (2.0 * h)
        finally:
            torch.randn_like = orig_randn_like

        rel_error = abs(grad_autograd - grad_fd) / (abs(grad_fd) + 1e-12)
        self.assertGreater(abs(grad_autograd), 1e-6)
        self.assertLess(
            rel_error,
            1e-4,
            f"Autograd gradient {grad_autograd} differs from FD {grad_fd} by {rel_error:.6%}",
        )


# =============================================================================
# Test Suite 3: End-to-End Simulation Execution with params.json
# =============================================================================

class TestEndToEndParamsExecution(unittest.TestCase):
    """Verifies end-to-end simulation execution with params.json creating all required outputs."""

    def test_end_to_end_simulation_with_params_json(self) -> None:
        """Run short simulation for all conditions in params.json and verify all output artifacts."""
        self.assertTrue(PARAMS_JSON_PATH.exists(), f"params.json not found at {PARAMS_JSON_PATH}")

        with open(PARAMS_JSON_PATH, "r", encoding="utf-8") as fh:
            raw_params_list = json.load(fh)

        self.assertEqual(
            len(raw_params_list),
            5,
            f"Expected exactly 5 conditions in params.json, got {len(raw_params_list)}",
        )

        # Scale down step count for swift automated test execution while preserving all physics
        scaled_params_list = []
        for run_cfg in raw_params_list:
            scaled_cfg = dict(run_cfg)
            scaled_cfg["N_STEPS"] = 20
            scaled_cfg["WRITE_EVERY"] = 10
            scaled_params_list.append(scaled_cfg)

        with tempfile.TemporaryDirectory() as tmpdir:
            temp_params_file = Path(tmpdir) / "test_params.json"
            with open(temp_params_file, "w", encoding="utf-8") as fh:
                json.dump(scaled_params_list, fh)

            cmd = [PYTHON_EXE, str(SIM_SCRIPT_PATH), str(temp_params_file)]
            env = dict(os.environ, PYTHONPATH=str(TORCH_SIM_DIR) + os.pathsep + os.environ.get("PYTHONPATH", ""))
            proc = subprocess.run(
                cmd,
                cwd=tmpdir,
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(
                proc.returncode,
                0,
                f"Simulation CLI execution failed with code {proc.returncode}:\n{proc.stderr}",
            )

            expected_metrics = ["bonds", "clusters", "largest", "fraction"]
            for run_cfg in scaled_params_list:
                name = run_cfg["name"]
                xyz_path = Path(tmpdir) / "traj_bd" / f"{name}.xyz"
                csv_path = Path(tmpdir) / "data_bd" / f"{name}.csv"
                summary_path = Path(tmpdir) / "data_bd" / f"{name}_summary.json"
                plots_dir = Path(tmpdir) / "plots_bd"

                # 1. Verify OVITO multi-frame XYZ format
                self.assertTrue(xyz_path.exists(), f"Missing XYZ trajectory: {xyz_path}")
                verify_ovito_xyz_format(xyz_path, expectedParticleCount=792)

                # 2. Verify 12-column CSV dataset schema & physical consistency
                self.assertTrue(csv_path.exists(), f"Missing CSV dataset: {csv_path}")
                verify_csv_parity_and_consistency(csv_path, expectedColumns=EXPECTED_HOOMD_COLUMNS)

                # 3. Verify summary JSON structure
                self.assertTrue(summary_path.exists(), f"Missing summary JSON: {summary_path}")
                verify_summary_json(summary_path, expectedName=name)

                # 4. Verify diagnostic JPEG plots
                verify_diagnostic_plots(plots_dir, baseName=name, metrics=expected_metrics)


if __name__ == "__main__":
    unittest.main()
