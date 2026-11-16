"""JAX-only consistency checks (no HOOMD needed, run on CPU or GPU).

  * forces match an independent float64 all-pairs NumPy implementation
  * neighbor list contains every interacting pair (guards the periodic-box bug)
  * Newton's third law: total force sums to ~0
  * forces are unchanged when the whole system is shifted by a periodic amount
  * the step reproduces x + mu*F*dt exactly and applies noise additively
  * a short stochastic rollout stays in the box, is finite, and sets no overflow flag

Run with::

    pytest rubisco_epyc1_nucleation_growth/jax_md/tests/test_jax_internal.py -v -s
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("jax")
pytest.importorskip("jax_md")

from condensate_ml import RubiscoEpyc1Simulation  # noqa: E402

CONDITIONS = [(40.0, -0.5, 101), (33.0, -1.2, 103), (29.0, -2.5, 105)]  # box, epsilon, seed
PATCH_SIGMA, PATCH_RCUT = 1.0, 3.0


def min_image(d, box):
    return d - box * np.round(d / box)


def brute_force_components(sim, pos, epsilon):
    """Independent O(N^2) float64 implementation of the same model."""
    pos = np.asarray(pos, np.float64)
    box = sim.box
    t = np.asarray(sim.type_ids)
    d = min_image(pos[:, None, :] - pos[None, :, :], box)
    r2 = (d**2).sum(-1)
    eps = np.asarray(sim.epsilon_table, np.float64)[t[:, None], t[None, :]]
    s2 = np.asarray(sim.sigma_table, np.float64)[t[:, None], t[None, :]] ** 2
    rc = np.asarray(sim.cutoff_table, np.float64)[t[:, None], t[None, :]]
    active = np.asarray(sim.nonbonded_pair_mask) & (r2 < rc**2)
    steric = (np.where(active, eps / s2 * np.exp(-0.5 * r2 / s2), 0.0)[..., None] * d).sum(1)

    is_t, is_p = t == 3, t == 1
    tp = (is_t[:, None] & is_p[None, :]) | (is_p[:, None] & is_t[None, :])
    active = tp & (r2 < PATCH_RCUT**2)
    attraction = (np.where(active, epsilon / PATCH_SIGMA**2 * np.exp(-0.5 * r2 / PATCH_SIGMA**2), 0.0)[..., None] * d).sum(1)

    bonded = np.zeros_like(pos)
    for (i, j), k, r0 in zip(np.asarray(sim.bond_pairs), np.asarray(sim.bond_k, np.float64), np.asarray(sim.bond_r0, np.float64)):
        dv = min_image(pos[i] - pos[j], box)
        r = np.linalg.norm(dv)
        f = -k * (r - r0) * dv / r
        bonded[i] += f
        bonded[j] -= f
    return {"steric": steric, "sticker_patch": attraction, "bonded": bonded,
            "total": steric + attraction + bonded}


def make(box, seed, **kw):
    return RubiscoEpyc1Simulation(n_steps=1, box=box, write_every=1, seed=seed, **kw)


def perturbed(sim, sigma, seed):
    pos = np.asarray(sim.init(), np.float64)
    pos = pos + np.random.default_rng(seed).normal(0, sigma, pos.shape)
    return ((pos + sim.box / 2) % sim.box - sim.box / 2).astype(np.float32).astype(np.float64)


@pytest.mark.parametrize("sigma", (0.0, 0.2, 0.5))
@pytest.mark.parametrize("box,epsilon,seed", CONDITIONS)
def test_forces_match_brute_force(box, epsilon, seed, sigma):
    sim = make(box, seed)
    pos = perturbed(sim, sigma, seed)
    got = {k: np.asarray(v, np.float64) for k, v in sim.compute_force_components(pos, epsilon).items()}
    ref = brute_force_components(sim, pos, epsilon)
    for name in ("steric", "sticker_patch", "bonded", "total"):
        scale = np.linalg.norm(ref[name])
        err = np.linalg.norm(got[name] - ref[name])
        print(f"box={box} sigma={sigma} {name:14s} |F|={scale:9.3f} rel err={err / max(scale, 1e-12):.2e}")
        assert err <= 1e-4 * scale + 5e-3, f"{name} differs from brute force"


@pytest.mark.parametrize("box,epsilon,seed", CONDITIONS)
def test_neighbor_list_contains_all_interacting_pairs(box, epsilon, seed):
    import jax.numpy as jnp
    sim = make(box, seed)
    pos = perturbed(sim, 0.3, seed)
    nb = sim.neighbor_fn.allocate(jnp.asarray(pos + box / 2, jnp.float32), extra_capacity=32)
    assert not bool(nb.did_buffer_overflow)
    idx = np.asarray(nb.idx)
    n = len(pos)
    listed = {(i, int(j)) for i in range(n) for j in idx[i] if j < n}
    d = min_image(pos[:, None, :] - pos[None, :, :], box)
    r = np.sqrt((d**2).sum(-1))
    # 3.0 is the longest interaction cutoff in the model.
    need = {(i, j) for i, j in zip(*np.where((r < 3.0) & ~np.eye(n, dtype=bool)))}
    missing = need - listed
    assert not missing, f"{len(missing)} interacting pairs missing from neighbor list"


@pytest.mark.parametrize("box,epsilon,seed", CONDITIONS)
def test_total_force_sums_to_zero(box, epsilon, seed):
    sim = make(box, seed)
    pos = perturbed(sim, 0.3, seed)
    for name, f in sim.compute_force_components(pos, epsilon).items():
        f = np.asarray(f, np.float64)
        print(f"{name}: |sum F| = {np.abs(f.sum(0)).max():.2e} (max |F| {np.abs(f).max():.2e})")
        assert np.abs(f.sum(0)).max() < 1e-3 * max(np.abs(f).max(), 1.0)


@pytest.mark.parametrize("box,epsilon,seed", CONDITIONS)
def test_forces_invariant_under_periodic_translation(box, epsilon, seed):
    sim = make(box, seed)
    pos = perturbed(sim, 0.3, seed)
    shift = np.array([0.37 * box, -0.21 * box, 0.49 * box])
    moved = ((pos + shift + box / 2) % box - box / 2).astype(np.float32).astype(np.float64)
    a = np.asarray(sim.compute_forces(pos, epsilon), np.float64)
    b = np.asarray(sim.compute_forces(moved, epsilon), np.float64)
    err = np.linalg.norm(a - b) / np.linalg.norm(a)
    print(f"translation invariance rel err = {err:.2e}")
    assert err < 1e-3


def test_step_is_position_plus_mobility_force_dt_plus_noise():
    sim = make(40.0, 101)
    pos = perturbed(sim, 0.2, 7)
    eps = -1.0
    force = np.asarray(sim.compute_forces(pos, eps), np.float64)
    mu = np.asarray(sim.mobilities, np.float64).reshape(len(pos), -1)
    noise = np.random.default_rng(3).normal(0, 1e-3, pos.shape)
    got = np.asarray(sim.brownian_step(pos, eps, noise=noise), np.float64)
    want = pos + mu * force * sim.dt + noise
    want = (want + 20.0) % 40.0 - 20.0
    diff = (got - want + 20.0) % 40.0 - 20.0
    assert np.abs(diff).max() < 5e-6


def test_mobilities_follow_hoomd_gammas():
    sim = make(40.0, 101)
    mu = np.asarray(sim.mobilities, np.float64).reshape(len(sim.type_ids), -1)[:, 0]
    t = np.asarray(sim.type_ids)
    # gamma(R)=gamma(P)=5, gamma(S)=gamma(T)=1, mobility = 1/gamma at kT=1
    expected = np.where((t == 0) | (t == 1), 0.2, 1.0)
    np.testing.assert_allclose(mu, expected, rtol=1e-6)


def test_short_stochastic_rollout_is_sane():
    sim = RubiscoEpyc1Simulation(n_steps=200, box=29.0, write_every=100, seed=105)
    frames = np.asarray(sim({"epsilon": -2.5}))
    assert np.isfinite(frames).all()
    assert frames.max() < 14.5 + 1e-4 and frames.min() >= -14.5 - 1e-4
    assert not bool(sim.last_neighbor_overflow)
