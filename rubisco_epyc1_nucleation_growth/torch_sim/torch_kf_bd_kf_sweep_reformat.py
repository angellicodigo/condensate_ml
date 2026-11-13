"""Proof-of-concept CLI for the Rubisco-EPYC1 differentiable Brownian Dynamics simulator.

All simulation code lives in the ``scripts/`` folder next to this file, which is installed
and imported under the package name ``condensate_ml``:

    from condensate_ml import RubiscoEpyc1TimeStepper      # use this from ML code

This file only runs parameter sweeps and writes artifacts (XYZ for OVITO, CSV, summary
JSON, plots), so it is safe to import and nothing happens at import time.

    python torch_kf_bd_kf_sweep_reformat.py params.json [--output-dir OUT]
    python torch_kf_bd_kf_sweep_reformat.py '{"name": "t", "BOX": 35, "EPS": 1.2, "N_STEPS": 2000, "WRITE_EVERY": 100, "SEED": 1}'

Install once (required, because the folder name differs from the import name):

    pip install -e rubisco_epyc1_nucleation_growth/torch_sim
"""

from condensate_ml import *  # noqa: F401,F403  (re-export the public API for older imports)
from condensate_ml.runner import main

if __name__ == "__main__":
    main()
