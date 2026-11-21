"""Tests that gradients propagate through the JAX MD rollout."""

from __future__ import annotations

import numpy as np
import pytest

from condensate_ml.helper import (
    epsilon_value_and_gradient,
    finite_difference_epsilon_gradient,
    make_differentiability_simulation,
    trajectory_loss,
)


def test_epsilon_gradient_is_finite_and_nonzero() -> None:
    """The rollout loss must receive a usable gradient from epsilon."""
    pytest.importorskip("jax")
    pytest.importorskip("jax_md")
    import jax.numpy as jnp

    simulation = make_differentiability_simulation()
    epsilon = jnp.asarray(-0.8, dtype=jnp.float32)
    loss, gradient = epsilon_value_and_gradient(simulation, epsilon)

    assert np.isfinite(float(loss))
    assert np.isfinite(float(gradient))
    assert abs(float(gradient)) > 1e-10


def test_epsilon_gradient_matches_finite_difference() -> None:
    """Reverse-mode autodiff must agree with a centered finite difference."""
    pytest.importorskip("jax")
    pytest.importorskip("jax_md")
    import jax.numpy as jnp

    simulation = make_differentiability_simulation()
    epsilon = jnp.asarray(-0.8, dtype=jnp.float32)
    _, autodiff_gradient = epsilon_value_and_gradient(simulation, epsilon)
    finite_difference = finite_difference_epsilon_gradient(simulation, epsilon, step=0.01)

    np.testing.assert_allclose(
        autodiff_gradient,
        finite_difference,
        rtol=0.1,
        atol=1e-7,
    )


def test_checkpointing_preserves_the_gradient() -> None:
    """Checkpointed and non-checkpointed rollouts produce matching gradients."""
    pytest.importorskip("jax")
    pytest.importorskip("jax_md")
    import jax.numpy as jnp

    checkpointed = make_differentiability_simulation(checkpoint_blocks=True)
    uncheckpointed = make_differentiability_simulation(checkpoint_blocks=False)
    epsilon = jnp.asarray(-0.8, dtype=jnp.float32)

    checkpointed_loss, checkpointed_gradient = epsilon_value_and_gradient(checkpointed, epsilon)
    uncheckpointed_loss, uncheckpointed_gradient = epsilon_value_and_gradient(uncheckpointed, epsilon)

    np.testing.assert_allclose(checkpointed_loss, uncheckpointed_loss, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(checkpointed_gradient, uncheckpointed_gradient, rtol=1e-3, atol=1e-6)


def test_different_epsilon_values_change_the_loss() -> None:
    """Changing epsilon should affect the simulated trajectory and loss."""
    pytest.importorskip("jax")
    pytest.importorskip("jax_md")
    import jax.numpy as jnp

    simulation = make_differentiability_simulation()
    weak_attraction = trajectory_loss(simulation, jnp.asarray(-0.2, dtype=jnp.float32))
    strong_attraction = trajectory_loss(simulation, jnp.asarray(-2.0, dtype=jnp.float32))

    assert np.isfinite(float(weak_attraction))
    assert np.isfinite(float(strong_attraction))
    assert abs(float(weak_attraction - strong_attraction)) > 1e-6
