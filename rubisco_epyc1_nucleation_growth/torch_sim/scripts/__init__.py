"""Rubisco-EPYC1 differentiable Brownian Dynamics (PyTorch / NEFI).

Public API (import these from ML code; importing has no side effects):

- RubiscoEpyc1TimeStepper : the differentiable simulator (stepper.py)
- compute_soft_contacts   : differentiable contact observable for losses
- compute_hard_contacts / compute_cluster_graph_metrics : HOOMD-parity analysis (not differentiable)
- run_single_simulation / run_simulation / main : full sweep workflow with artifacts (runner.py)

Modules:
- constants.py   : model constants, steric parameter table, patch geometry
- geometry.py    : coordinate wrapping, minimum image, safe norm
- initial_snapshot.py : initial coordinate layout
- topology.py    : labels, type ids, mobilities, bonds, steric matrices, pair mask
- forces.py      : steric / attraction / bond forces, Brownian step
- post_simulation_analysis.py : soft/hard contacts, cluster graph metrics, per-frame metrics
- output_io.py   : XYZ, CSV, summary JSON, diagnostic plots
- stepper.py     : RubiscoEpyc1TimeStepper
- runner.py      : sweep workflow and CLI
"""

from .forces import (
    compute_gaussian_attraction_forces,
    compute_gaussian_steric_forces,
    compute_harmonic_bond_forces,
)
from .geometry import apply_minimum_image_displacement
from .post_simulation_analysis import (
    compute_cluster_graph_metrics,
    compute_hard_contacts,
    compute_soft_contacts,
)
from .runner import main, run_simulation, run_single_simulation
from .stepper import RubiscoEpyc1TimeStepper, kon_to_epsilon

__all__ = [
    "RubiscoEpyc1TimeStepper",
    "apply_minimum_image_displacement",
    "compute_cluster_graph_metrics",
    "compute_gaussian_attraction_forces",
    "compute_gaussian_steric_forces",
    "compute_hard_contacts",
    "compute_harmonic_bond_forces",
    "compute_soft_contacts",
    "kon_to_epsilon",
    "main",
    "run_simulation",
    "run_single_simulation",
]
