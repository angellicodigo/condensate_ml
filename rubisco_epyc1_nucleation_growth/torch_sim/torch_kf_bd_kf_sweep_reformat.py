"""Proof-of-concept CLI for the Rubisco-EPYC1 differentiable Brownian Dynamics simulator.

All simulation code lives in the importable ``rubisco_bd`` package next to this file:

    from rubisco_bd import RubiscoEpyc1TimeStepper      # use this from ML code

This file only runs parameter sweeps and writes artifacts (XYZ for OVITO, CSV, summary
JSON, plots), so it is safe to import and nothing happens at import time.

    python torch_kf_bd_kf_sweep_reformat.py params.json [--output-dir OUT]
    python torch_kf_bd_kf_sweep_reformat.py '{"name": "t", "BOX": 35, "EPS": 1.2, "N_STEPS": 2000, "WRITE_EVERY": 100, "SEED": 1}'

Install for use from other directories / scripts:

    pip install -e rubisco_epyc1_nucleation_growth/torch_sim
"""

from rubisco_bd import *  # noqa: F401,F403  (re-export the public API for older imports)
from rubisco_bd.runner import main

if __name__ == "__main__":
    main()
