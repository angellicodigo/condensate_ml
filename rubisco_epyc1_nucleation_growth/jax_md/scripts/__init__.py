"""JAX MD implementation of the Rubisco-EPYC1 Brownian dynamics model.

This backend is deliberately separate from the PyTorch/NEFI implementation.
It uses JAX MD's cell-list neighbor list, JAX ``lax.scan`` for the rollout,
and JAX autodiff for gradients.
"""

from .simulation import RubiscoEpyc1Simulation

__all__ = ["RubiscoEpyc1Simulation"]