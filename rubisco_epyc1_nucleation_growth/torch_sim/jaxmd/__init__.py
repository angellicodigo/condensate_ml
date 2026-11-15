"""JAX MD prototype of the Rubisco-EPYC1 Brownian dynamics model.

This backend is deliberately separate from the PyTorch/NEFI implementation.
It uses JAX MD's cell-list neighbor list, JAX ``lax.scan`` for the rollout,
and JAX autodiff for gradients.
"""

from .stepper import RubiscoEpyc1JaxMDTimeStepper, kon_to_epsilon

# Convenient backend-local name for code that selects a timestepper backend.
RubiscoEpyc1TimeStepper = RubiscoEpyc1JaxMDTimeStepper

__all__ = [
    "RubiscoEpyc1JaxMDTimeStepper",
    "RubiscoEpyc1TimeStepper",
    "kon_to_epsilon",
]
