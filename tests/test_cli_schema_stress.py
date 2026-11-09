"""Adversarial Stress Test Suite for Milestone 2: CLI, Schema Parity, and Exporters.

Tests:
1. CLI execution with params.json and JSON inputs.
2. Inline JSON parameter parsing with UPPERCASE, LOWERCASE, and edge cases (e.g., 'eps' vs 'EPS').
3. CSV output column parity against simulate_hoomd_patchy_full.py (verbatim 12 columns, types, values).
4. OVITO XYZ format parsing and strict specification conformance.
5. Diagnostic plot image generation (valid JPEG format, dimensions, non-blank content).
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
from typing import Any

from PIL import Image
import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from rubisco_epyc1_nucleation_growth.torch_sim.torch_kf_bd_kf_sweep import (
    RubiscoEpyc1TimeStepper,
    parse_simulation_inputs,
    run_single_simulation,
    export_ovito_xyz,
    export_csv_and_summary,
    export_diagnostic_plots,
)

PYTHON_EXE = sys.executable
SIM_SCRIPT = REPO_ROOT / "rubisco_epyc1_nucleation_growth" / "torch_sim" / "torch_kf_bd_kf_sweep.py"
HOOMD_REF_SCRIPT = REPO_ROOT / "rubisco_epyc1_nucleation_growth" / "molecular_sim" / "simulate_hoomd_patchy_full.py"
PARAMS_JSON_PATH = REPO_ROOT / "rubisco_epyc1_nucleation_growth" / "torch_sim" / "params.json"

EXPECTED_HOOMD_COLUMNS = [
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


class TestCliInputHandling(unittest.TestCase):
    """Adversarial stress-testing of CLI parameter input parsing."""

    def test_params_json_file_parsing(self):
        """Verify that params.json is parsed into exactly 5 runs with all required fields."""
        self.assertTrue(PARAMS_JSON_PATH.exists(), f"Missing params.json at {PARAMS_JSON_PATH}")
        args = ["torch_kf_bd_kf_sweep.py", str(PARAMS_JSON_PATH)]
        runs = parse_simulation_inputs(args)
        self.assertIsInstance(runs, list)
        self.assertEqual(len(runs), 5, f"Expected 5 runs in params.json, got {len(runs)}")

        expected_names = ["control", "early", "nucleation", "weak_coarsening", "coarsening"]
        for idx, (run_params, exp_name) in enumerate(zip(runs, expected_names)):
            self.assertEqual(run_params["name"], exp_name)
            self.assertIn("BOX", run_params)
            self.assertIn("EPS", run_params)
            self.assertIn("N_STEPS", run_params)
            self.assertIn("WRITE_EVERY", run_params)
            self.assertIn("SEED", run_params)
            self.assertGreater(float(run_params["BOX"]), 0.0)
            self.assertGreater(float(run_params["EPS"]), 0.0)

    def test_uppercase_inline_json(self):
        """Verify single inline JSON object with uppercase keys."""
        payload = json.dumps({
            "NAME": "test_upper_inline",
            "BOX": 45.0,
            "KON": 2.25,
            "KOFF": 0.05,
            "SEED": 999,
            "N_STEPS": 20,
            "WRITE_EVERY": 10,
            "DT": 0.001,
        })
        args = ["torch_kf_bd_kf_sweep.py", payload]
        runs = parse_simulation_inputs(args)
        self.assertEqual(len(runs), 1)
        p = runs[0]
        self.assertEqual(p["NAME"], "test_upper_inline")
        self.assertEqual(p["BOX"], 45.0)
        self.assertEqual(p["KON"], 2.25)
        self.assertEqual(p["KOFF"], 0.05)
        self.assertEqual(p["SEED"], 999)
        self.assertEqual(p["N_STEPS"], 20)
        self.assertEqual(p["WRITE_EVERY"], 10)
        self.assertEqual(p["DT"], 0.001)

    def test_lowercase_inline_json(self):
        """Verify single inline JSON object with lowercase keys."""
        payload = json.dumps({
            "name": "test_lower_inline",
            "box": 32.5,
            "kon": 1.75,
            "koff": 0.02,
            "seed": 111,
            "n_steps": 40,
            "write_every": 20,
            "dt": 0.00025,
        })
        args = ["torch_kf_bd_kf_sweep.py", payload]
        runs = parse_simulation_inputs(args)
        self.assertEqual(len(runs), 1)
        p = runs[0]
        self.assertEqual(p["name"], "test_lower_inline")
        self.assertEqual(p["box"], 32.5)
        self.assertEqual(p["kon"], 1.75)
        self.assertEqual(p["koff"], 0.02)
        self.assertEqual(p["seed"], 111)
        self.assertEqual(p["n_steps"], 40)
        self.assertEqual(p["write_every"], 20)
        self.assertEqual(p["dt"], 0.00025)

    def test_mixed_case_and_alias_keys(self):
        """Verify combinations of mixed case and key aliases."""
        # Test EPS alias in uppercase
        p1 = {"name": "test_eps_alias", "BOX": 38.0, "EPS": 1.4}
        args1 = ["torch_kf_bd_kf_sweep.py", json.dumps(p1)]
        runs1 = parse_simulation_inputs(args1)
        self.assertEqual(runs1[0]["EPS"], 1.4)

        # Test lowercase 'kon' key
        p_lower_kon = {"name": "test_lower_kon", "box": 38.0, "kon": 1.9, "n_steps": 10, "write_every": 5}
        with tempfile.TemporaryDirectory() as tmpdir:
            dev = torch.device("cpu")
            res = run_single_simulation(p_lower_kon, dev, output_base_dir=tmpdir)
            summary_path = os.path.join(tmpdir, "data_bd", "test_lower_kon_summary.json")
            with open(summary_path) as fh:
                summary = json.load(fh)
            self.assertEqual(summary["kon"], 1.9)
            self.assertEqual(summary["epsilon_used"], -1.9)

        # Adversarial Bug Check: lowercase 'eps' key
        # In params.json the field is named "EPS". If a user passes lowercase "eps":
        p_lower_eps = {"name": "test_lower_eps", "box": 38.0, "eps": 1.9, "n_steps": 10, "write_every": 5}
        with tempfile.TemporaryDirectory() as tmpdir:
            dev = torch.device("cpu")
            res = run_single_simulation(p_lower_eps, dev, output_base_dir=tmpdir)
            summary_path = os.path.join(tmpdir, "data_bd", "test_lower_eps_summary.json")
            with open(summary_path) as fh:
                summary = json.load(fh)
            # Record whether lowercase 'eps' was honored or silently dropped:
            self.lowercase_eps_honored = (summary["kon"] == 1.9)
            print(f"[ADVERSARIAL CHECK] Lowercase 'eps' honored? {self.lowercase_eps_honored} (parsed kon={summary['kon']})")

    def test_cli_execution_with_params_json_file(self):
        """Test CLI execution directly invoking torch_kf_bd_kf_sweep.py with params.json."""
        # Launch simulation with params.json
        cmd = [PYTHON_EXE, "-u", str(SIM_SCRIPT), str(PARAMS_JSON_PATH)]
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )
        # Read initial output to verify it loads params.json and begins execution
        stdout_lines = []
        try:
            for _ in range(10):
                line = proc.stdout.readline()
                if line:
                    stdout_lines.append(line.strip())
                    if "Starting control" in line:
                        break
        finally:
            proc.terminate()
            proc.wait(timeout=5)


        combined = " ".join(stdout_lines)
        self.assertIn("Found 5 simulation run(s) to execute.", combined)
        self.assertIn("Starting control:", combined)
        self.assertIn("BOX=40.0", combined)
        self.assertIn("KON=0.500", combined)
        self.assertIn("epsilon_used=-0.500", combined)
        self.assertIn("N_STEPS=100000", combined)


    def test_stdin_json_input(self):
        """Verify reading JSON from STDIN when sys.stdin is piped."""
        payload = json.dumps({"name": "stdin_test", "BOX": 40.0, "KON": 1.0, "N_STEPS": 10, "WRITE_EVERY": 5})
        cmd = [
            PYTHON_EXE,
            str(SIM_SCRIPT),
        ]
        # Execute via subprocess with stdin piped
        proc = subprocess.run(cmd, input=payload, text=True, capture_output=True)
        self.assertEqual(proc.returncode, 0, f"STDIN execution failed: {proc.stderr}")
        self.assertIn("Found 1 simulation run(s) to execute.", proc.stdout)
        self.assertIn("Starting stdin_test", proc.stdout)

    def test_multi_run_sequential_execution(self):
        """Verify that CLI executes a multi-run JSON list of 5 conditions to full completion."""
        test_runs = [
            {"name": f"multi_test_{i}", "BOX": 35.0 + i, "EPS": 0.5 + 0.3 * i, "N_STEPS": 10, "WRITE_EVERY": 5, "SEED": 100 + i}
            for i in range(5)
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            json_file = os.path.join(tmpdir, "multi_params.json")
            with open(json_file, "w") as fh:
                json.dump(test_runs, fh)

            cmd = [PYTHON_EXE, str(SIM_SCRIPT), json_file]
            proc = subprocess.run(cmd, cwd=tmpdir, capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, f"Multi-run execution failed: {proc.stderr}")
            self.assertIn("Found 5 simulation run(s) to execute.", proc.stdout)
            self.assertIn("All simulations completed successfully.", proc.stdout)

            # Check that all 5 runs created XYZ, CSV, summary, and 4 plots
            for i in range(5):
                name = f"multi_test_{i}"
                xyz_file = Path(tmpdir) / "traj_bd" / f"{name}.xyz"
                csv_file = Path(tmpdir) / "data_bd" / f"{name}.csv"
                sum_file = Path(tmpdir) / "data_bd" / f"{name}_summary.json"
                self.assertTrue(xyz_file.exists(), f"Missing XYZ: {xyz_file}")
                self.assertTrue(csv_file.exists(), f"Missing CSV: {csv_file}")
                self.assertTrue(sum_file.exists(), f"Missing summary: {sum_file}")
                for key in ["bonds", "clusters", "largest", "fraction"]:
                    plot_file = Path(tmpdir) / "plots_bd" / f"{name}_{key}.jpeg"
                    self.assertTrue(plot_file.exists(), f"Missing plot: {plot_file}")

class TestOutputParityAndSchema(unittest.TestCase):
    """Adversarially verify CSV schema, OVITO XYZ format, and diagnostic plots."""


    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.TemporaryDirectory()
        cls.outdir = cls.tmpdir.name
        cls.device = torch.device("cpu")

        # Run two end-to-end simulations:
        # Run A: Uppercase parameters
        cls.params_upper = {
            "NAME": "sim_upper_schema",
            "BOX": 40.0,
            "KON": 1.6,
            "KOFF": 0.0,
            "SEED": 42,
            "N_STEPS": 20,
            "WRITE_EVERY": 10,
            "DT": 0.0005,
        }
        # Run B: Lowercase parameters
        cls.params_lower = {
            "name": "sim_lower_schema",
            "box": 35.0,
            "kon": 2.1,
            "koff": 0.01,
            "seed": 123,
            "n_steps": 20,
            "write_every": 10,
            "dt": 0.0005,
        }
        run_single_simulation(cls.params_upper, cls.device, output_base_dir=cls.outdir)
        run_single_simulation(cls.params_lower, cls.device, output_base_dir=cls.outdir)

    @classmethod
    def tearDownClass(cls):
        cls.tmpdir.cleanup()

    def test_csv_12_columns_verbatim(self):
        """Verify CSV output has exactly 12 columns matching simulate_hoomd_patchy_full.py verbatim."""
        for name in ["sim_upper_schema", "sim_lower_schema"]:
            csv_path = Path(self.outdir) / "data_bd" / f"{name}.csv"
            self.assertTrue(csv_path.exists(), f"CSV file missing: {csv_path}")

            with open(csv_path, mode="r", newline="") as f:
                reader = csv.reader(f)
                header = next(reader)

            self.assertEqual(
                len(header),
                12,
                f"CSV must have exactly 12 columns, got {len(header)}: {header}",
            )
            self.assertEqual(
                header,
                EXPECTED_HOOMD_COLUMNS,
                f"CSV header columns do not match HOOMD reference verbatim!\nExpected: {EXPECTED_HOOMD_COLUMNS}\nActual:   {header}",
            )

    def test_csv_row_values_and_mathematical_consistency(self):
        """Verify data types, bounds, and mathematical consistency across CSV rows."""
        csv_path = Path(self.outdir) / "data_bd" / "sim_upper_schema.csv"
        df = pd.read_csv(csv_path)

        # 20 steps with write_every=10 means frames: 0, 10, 20 -> 3 frames
        self.assertEqual(len(df), 3, f"Expected 3 frames, got {len(df)}")

        for idx, row in df.iterrows():
            # frame
            self.assertEqual(row["frame"], idx)
            # step
            self.assertEqual(row["step"], idx * 10)
            # time
            self.assertAlmostEqual(row["time"], idx * 10 * 0.0005, places=7)
            # box
            self.assertEqual(row["box"], 40.0)
            # kon_or_epsilon
            self.assertEqual(row["kon_or_epsilon"], 1.6)
            # epsilon_used
            self.assertEqual(row["epsilon_used"], -1.6)
            # koff_recorded
            self.assertEqual(row["koff_recorded"], 0.0)
            # contacts == bonds
            self.assertEqual(row["contacts"], row["bonds"])
            self.assertGreaterEqual(row["contacts"], 0)
            # clusters in [1, 72]
            self.assertGreaterEqual(row["clusters"], 1)
            self.assertLessEqual(row["clusters"], 72)
            # largest in [1, 72]
            self.assertGreaterEqual(row["largest"], 1)
            self.assertLessEqual(row["largest"], 72)
            # fraction == largest / 72
            self.assertAlmostEqual(row["fraction"], row["largest"] / 72.0, places=6)

    def test_ovito_xyz_format_strict(self):
        """Strictly parse the generated OVITO multi-frame XYZ files."""
        for name in ["sim_upper_schema", "sim_lower_schema"]:
            xyz_path = Path(self.outdir) / "traj_bd" / f"{name}.xyz"
            self.assertTrue(xyz_path.exists(), f"XYZ file missing: {xyz_path}")

            with open(xyz_path, "r") as fh:
                lines = fh.readlines()

            # Each frame must consist of:
            # Line 1: Particle count N (792)
            # Line 2: Comment line ('step X')
            # N lines: Type X Y Z
            lines_per_frame = 792 + 2
            self.assertEqual(
                len(lines) % lines_per_frame,
                0,
                f"Total lines {len(lines)} not a multiple of lines_per_frame {lines_per_frame}",
            )
            n_frames = len(lines) // lines_per_frame
            self.assertEqual(n_frames, 3)

            for frame_idx in range(n_frames):
                offset = frame_idx * lines_per_frame
                # Header 1: count
                count_str = lines[offset].strip()
                self.assertEqual(int(count_str), 792)

                # Header 2: comment step
                comment_str = lines[offset + 1].strip()
                self.assertTrue(
                    comment_str.startswith("step "),
                    f"Comment line should start with 'step ', got {comment_str!r}",
                )
                exp_step = frame_idx * 10
                self.assertEqual(comment_str, f"step {exp_step}")

                # Check particle lines
                frame_lines = lines[offset + 2 : offset + lines_per_frame]
                labels = []
                coords = []
                for p_idx, line in enumerate(frame_lines):
                    parts = line.strip().split()
                    self.assertEqual(
                        len(parts),
                        4,
                        f"Line {offset + 2 + p_idx} does not have 4 tokens (type x y z): {line!r}",
                    )
                    label, x_str, y_str, z_str = parts
                    labels.append(label)
                    x, y, z = float(x_str), float(y_str), float(z_str)
                    self.assertFalse(math.isnan(x) or math.isnan(y) or math.isnan(z), "NaN coordinate detected")
                    self.assertFalse(math.isinf(x) or math.isinf(y) or math.isinf(z), "Inf coordinate detected")
                    coords.append([x, y, z])

                # Verify particle type order:
                # 24 R centers
                self.assertEqual(labels[:24], ["R"] * 24)
                # 192 P surface patches
                self.assertEqual(labels[24:216], ["P"] * 192)
                # 48 chains with 12 alternating S and T beads: ["S", "T", "S", "T", "S", "T", "S", "T", "S", "T", "S", "T"]
                expected_chain_beads = ["S", "T", "S", "T", "S", "T", "S", "T", "S", "T", "S", "T"] * 48
                self.assertEqual(labels[216:792], expected_chain_beads)
                self.assertEqual(labels.count("S"), 288)
                self.assertEqual(labels.count("T"), 288)
                self.assertEqual(len(labels), 792)


    def test_diagnostic_plots_generation(self):
        """Verify the 4 diagnostic plots are generated as valid, non-blank JPEGs."""
        metrics = ["bonds", "clusters", "largest", "fraction"]
        for name in ["sim_upper_schema", "sim_lower_schema"]:
            for key in metrics:
                plot_path = Path(self.outdir) / "plots_bd" / f"{name}_{key}.jpeg"
                self.assertTrue(plot_path.exists(), f"Plot missing: {plot_path}")
                self.assertGreater(plot_path.stat().st_size, 1000, f"Plot file unusually small: {plot_path}")

                # Open and verify via PIL
                with Image.open(plot_path) as img:
                    self.assertEqual(img.format, "JPEG")
                    self.assertEqual(img.mode, "RGB")
                    self.assertGreaterEqual(img.width, 800)
                    self.assertGreaterEqual(img.height, 500)

                    # Verify image is not blank (variance of pixel intensities must be non-zero)
                    arr = np.array(img)
                    self.assertGreater(
                        np.std(arr),
                        1.0,
                        f"Plot image appears blank / uniform white: std={np.std(arr)}",
                    )


if __name__ == "__main__":
    unittest.main()
