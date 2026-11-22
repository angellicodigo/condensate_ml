"""Tests that gradients propagate through the JAX MD rollout."""

from __future__ import annotations

import numpy as np
import pytest

from condensate_ml.helper import (
    epsilon_value_and_gradient,
    finite_difference_epsilon_gradient,
    make_differentiability_simulation,
    make_simulation,
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
    # step=0.1: a smaller step makes the two losses differ by about one float32 rounding unit,
    # so the finite difference measures rounding noise and not the gradient.
    finite_difference = finite_difference_epsilon_gradient(simulation, epsilon, step=0.1)

    np.testing.assert_allclose(autodiff_gradient, finite_difference, rtol=0.05, atol=0.0)


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


def test_gradient_through_denser_rollout_matches_finite_difference() -> None:
    """Same check on the dense BOX=29 system (many neighbor-list pairs), 40 steps."""
    pytest.importorskip("jax")
    pytest.importorskip("jax_md")
    import jax
    import jax.numpy as jnp

    simulation = make_simulation(29.0, 105, n_steps=40, write_every=10)

    def loss(epsilon):
        last = simulation({"epsilon": epsilon})[-1]
        return jnp.mean(last**2) + jnp.mean(jnp.var(last, axis=0))

    epsilon = jnp.asarray(-2.0, dtype=jnp.float32)
    step = jnp.asarray(0.2, dtype=jnp.float32)
    automatic = jax.grad(loss)(epsilon)
    finite_difference = (loss(epsilon + step) - loss(epsilon - step)) / (2.0 * step)
    np.testing.assert_allclose(automatic, finite_difference, rtol=5e-2, atol=0.0)


def test_contact_loss_gradient_sign_and_initial_positions_gradient() -> None:
    """Stronger binding increases contacts (dLoss/dEps < 0) and dLoss/dPositions_0 is finite."""
    pytest.importorskip("jax")
    pytest.importorskip("jax_md")
    import jax
    import jax.numpy as jnp

    simulation = make_differentiability_simulation(n_steps=16, write_every=8, seed=101)
    type_ids = np.asarray(simulation.type_ids)
    patch_indices = jnp.asarray(np.where(type_ids == 1)[0], dtype=jnp.int32)
    sticker_indices = jnp.asarray(np.where(type_ids == 3)[0], dtype=jnp.int32)
    displacement = simulation.displacement

    def soft_contacts_loss(positions):
        patches = positions[patch_indices]
        stickers = positions[sticker_indices]
        pair_disp = jax.vmap(jax.vmap(displacement, in_axes=(None, 0)), in_axes=(0, None))
        delta = pair_disp(stickers, patches)
        dist_sq = jnp.sum(delta * delta, axis=-1)
        # Soft contact proxy: Gaussian proximity within attraction radius
        return jnp.sum(jnp.exp(-0.5 * dist_sq / (1.5**2)))

    # 1. Gradient with respect to positive binding strength epsilon
    # For positive epsilon, increasing epsilon deepens attraction, increasing contacts.
    # Therefore, a loss defined as -contacts must have a negative gradient (dLoss/dEps < 0).
    def loss_wrt_epsilon(eps):
        trajectory = simulation({"epsilon": eps})
        return -soft_contacts_loss(trajectory[-1])

    binding_epsilon = jnp.asarray(1.2, dtype=jnp.float32)
    loss_val, d_loss_d_eps = jax.value_and_grad(loss_wrt_epsilon)(binding_epsilon)

    assert np.isfinite(float(loss_val))
    assert np.isfinite(float(d_loss_d_eps))
    # Increasing binding strength must increase contacts (decrease negative contacts loss)
    assert float(d_loss_d_eps) < 0.0, "Expected d(-contacts)/d(epsilon) < 0 for positive binding strength"

    # 2. Gradient with respect to initial positions
    initial_positions = simulation.init()

    def loss_wrt_positions_0(pos_0):
        trajectory = simulation({"epsilon": binding_epsilon, "positions_0": pos_0})
        return soft_contacts_loss(trajectory[-1])

    d_loss_d_pos_0 = jax.grad(loss_wrt_positions_0)(initial_positions)

    assert d_loss_d_pos_0.shape == initial_positions.shape
    assert bool(jnp.isfinite(d_loss_d_pos_0).all()), "Gradient w.r.t initial positions must be finite"
    assert bool(jnp.any(d_loss_d_pos_0 != 0.0)), "Gradient w.r.t initial positions must be non-zero"

