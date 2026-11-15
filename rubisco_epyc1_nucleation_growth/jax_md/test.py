import time

import jax
import jax.numpy as jnp

from condensate_ml import RubiscoEpyc1Simulation


param_example = {
    "name": "control",
    "BOX": 40.0,
    "EPS": 0.5,
    "N_STEPS": 100000,
    "WRITE_EVERY": 500,
    "SEED": 101,
}

# JAX uses explicit PRNG keys internally.
# EPS is a positive binding strength, while the simulation expects:
# negative epsilon = attraction
epsilon = jnp.asarray(
    -param_example["EPS"],
    dtype=jnp.float32,
)

print("JAX version:", jax.__version__)
print("JAX devices:", jax.devices())

simulation = RubiscoEpyc1Simulation(
    n_steps=param_example["N_STEPS"],
    box=param_example["BOX"],
    write_every=param_example["WRITE_EVERY"],
    seed=param_example["SEED"],
    checkpoint_blocks=True,
)

start = time.perf_counter()

trajectory = simulation({"epsilon": epsilon})

# JAX execution is asynchronous on accelerators. This makes sure the
# computation has actually completed before stopping the timer.
trajectory.block_until_ready()

elapsed = time.perf_counter() - start

print(f"Simulation completed in {elapsed:.2f} seconds")
print(f"Trajectory shape: {trajectory.shape}")