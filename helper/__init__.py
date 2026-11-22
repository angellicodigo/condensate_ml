"""Shared helpers for JAX MD validation and regime runs.

Only the lightweight, test-facing helpers are re-exported here. Plotting
(``helper.regime_plots``) and the regime definitions / runner (``helper.regimes``)
are imported explicitly so that ``import condensate_ml.helper`` does not pull in
matplotlib, pandas, gsd or JAX.
"""

from .validate_snapshot import (
    BOX,
    EXPECTED_PARTICLES,
    build_hoomd_initial_positions,
    build_hoomd_initial_state,
    build_jax_initial_positions,
    minimum_image,
    wrap_centered,
)
from .validate_forces import (
    COMPONENTS,
    DT,
    EPSILON_VALUES,
    FORCE_SEEDS,
    HOOMD_BOND_PARAMS,
    HOOMD_GAMMA,
    HOOMD_STERIC_PARAMS,
    PERTURBATIONS,
    brute_force_components,
    configure_hoomd_forces,
    force_report,
    jax_force_components,
    make_simulation,
    perturb_positions,
    perturbed_positions,
    perturbed_state,
)
from .validate_differentiability import (
    epsilon_value_and_gradient,
    finite_difference_epsilon_gradient,
    make_differentiability_simulation,
    trajectory_loss,
)

__all__ = [
    "BOX",
    "COMPONENTS",
    "DT",
    "EPSILON_VALUES",
    "EXPECTED_PARTICLES",
    "FORCE_SEEDS",
    "HOOMD_BOND_PARAMS",
    "HOOMD_GAMMA",
    "HOOMD_STERIC_PARAMS",
    "PERTURBATIONS",
    "brute_force_components",
    "build_hoomd_initial_positions",
    "build_hoomd_initial_state",
    "build_jax_initial_positions",
    "configure_hoomd_forces",
    "epsilon_value_and_gradient",
    "finite_difference_epsilon_gradient",
    "force_report",
    "jax_force_components",
    "make_differentiability_simulation",
    "make_simulation",
    "minimum_image",
    "perturb_positions",
    "perturbed_positions",
    "perturbed_state",
    "trajectory_loss",
    "wrap_centered",
]
