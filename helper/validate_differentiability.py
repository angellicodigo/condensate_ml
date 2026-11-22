"""Reusable differentiability checks for the JAX MD simulation."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp


def make_differentiability_simulation(
    *,
    n_steps: int = 16,
    write_every: int = 8,
    seed: int = 101,
    checkpoint_blocks: bool = True,
):
    """Construct a small simulation suitable for fast gradient tests."""
    from condensate_ml import RubiscoEpyc1Simulation

    return RubiscoEpyc1Simulation(
        n_steps=n_steps,
        box=40.0,
        write_every=write_every,
        seed=seed,
        checkpoint_blocks=checkpoint_blocks,
        output_dir=None,
    )


def trajectory_loss(simulation, epsilon: Any) -> jax.Array:
    """Use a deterministic differentiable loss on the final trajectory frame."""
    trajectory = simulation({"epsilon": epsilon})
    final_positions = trajectory[-1]
    coordinate_weights = jnp.linspace(
        -1.0, 1.0, final_positions.size, dtype=final_positions.dtype
    ).reshape(final_positions.shape)
    return jnp.mean(final_positions * coordinate_weights) + 1e-3 * jnp.mean(final_positions**2)


def epsilon_value_and_gradient(simulation, epsilon: Any):
    """Return loss and autodiff gradient with respect to epsilon."""
    loss_for_epsilon = lambda value: trajectory_loss(simulation, value)
    return jax.value_and_grad(loss_for_epsilon)(epsilon)


def finite_difference_epsilon_gradient(simulation, epsilon: Any, step: float) -> jax.Array:
    """Return a centered finite-difference gradient with fixed simulation noise.

    ``step`` must be large enough that the two losses differ by far more than float32 rounding
    (~1e-7 for a loss near 1); a step of 0.01 here returns pure rounding noise. Use ~0.1.
    """
    step = jnp.asarray(step, dtype=jnp.asarray(epsilon).dtype)
    return (
        trajectory_loss(simulation, epsilon + step)
        - trajectory_loss(simulation, epsilon - step)
    ) / (2.0 * step)
