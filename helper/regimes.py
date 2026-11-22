"""The five simulated regimes and the function that runs one of them.

Shared by ``generate.py``, ``generate_plots.py`` and ``tests/test_long_simulations.py`` so the
parameters are defined in exactly one place.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Regime:
    """One simulation condition.

    ``epsilon`` is the positive sticker-patch binding strength; the simulation
    converts it to the attractive (negative) Gaussian well depth internally.
    """

    name: str
    box: float
    epsilon: float
    seed: int
    n_steps: int = 100_000
    write_every: int = 500

    @property
    def n_frames(self) -> int:
        """Number of saved frames, including the initial one."""
        return self.n_steps // self.write_every + 1


REGIMES = (
    Regime("control", box=40.0, epsilon=0.5, seed=101),
    Regime("early", box=36.0, epsilon=0.8, seed=102),
    Regime("nucleation", box=33.0, epsilon=1.2, seed=103),
    Regime("weak_coarsening", box=31.0, epsilon=1.8, seed=104),
    Regime("coarsening", box=29.0, epsilon=2.5, seed=105),
)


def artifact_directory(output_dir: str | Path, kind: str) -> Path:
    """Directory the simulation writes ``kind`` ('traj', 'xyz', 'data' or 'plots') files to."""
    import jax

    device_label = "gpu" if any(device.platform == "gpu" for device in jax.devices()) else "cpu"
    return Path(output_dir) / f"{kind}_jaxmd_{device_label}"


def run_regime(regime: Regime, output_dir: str | Path):
    """Run one regime, write its GSD/XYZ/CSV/plots to ``output_dir``, and return (trajectory, seconds)."""
    import jax.numpy as jnp

    from condensate_ml import RubiscoEpyc1Simulation

    print(f"Starting {regime.name}: BOX={regime.box}, EPS={regime.epsilon}, SEED={regime.seed}")
    simulation = RubiscoEpyc1Simulation(
        n_steps=regime.n_steps,
        box=regime.box,
        write_every=regime.write_every,
        seed=regime.seed,
        checkpoint_blocks=True,
        output_dir=output_dir,
        output_name=regime.name,
    )
    start = time.perf_counter()
    trajectory = simulation({"epsilon": jnp.asarray(regime.epsilon, dtype=jnp.float32)})
    trajectory.block_until_ready()  # wait for asynchronous accelerator work
    elapsed = time.perf_counter() - start
    print(
        f"Finished {regime.name} in {elapsed:.2f} seconds. "
        f"Shape={trajectory.shape}, expected_frames={regime.n_frames}"
    )
    print(f"GSD written to: {artifact_directory(output_dir, 'traj') / f'{regime.name}.gsd'}")
    return trajectory, elapsed
