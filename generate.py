from condensate_ml import RubiscoEpyc1Simulation

import time
from pathlib import Path
import os

import jax.numpy as jnp
from dotenv import load_dotenv


dotenv_path = Path(__file__).resolve().parent.parent.parent / ".env"
load_dotenv(dotenv_path=dotenv_path)

output_dir = os.getenv("OUTPUT_DIR")

if output_dir is None:
    raise RuntimeError("OUTPUT_DIR was not found in the environment or .env file")


params_list = [
    {"name": "control", "BOX": 40, "EPS": 0.5, "N_STEPS": 100000, "WRITE_EVERY": 500, "SEED": 101},
    {"name": "early", "BOX": 36, "EPS": 0.8, "N_STEPS": 100000, "WRITE_EVERY": 500, "SEED": 102},
    {"name": "nucleation", "BOX": 33, "EPS": 1.2, "N_STEPS": 100000, "WRITE_EVERY": 500, "SEED": 103},
    {"name": "weak_coarsening", "BOX": 31, "EPS": 1.8, "N_STEPS": 100000, "WRITE_EVERY": 500, "SEED": 104},
    {"name": "coarsening", "BOX": 29, "EPS": 2.5, "N_STEPS": 100000, "WRITE_EVERY": 500, "SEED": 105},
]


trajectories = {}
elapsed_by_scenario = {}

for params in params_list:
    name = params["name"]
    print(f"Starting {name}: BOX={params['BOX']}, EPS={params['EPS']}, SEED={params['SEED']}")
    simulation = RubiscoEpyc1Simulation(
        n_steps=params["N_STEPS"],
        box=params["BOX"],
        write_every=params["WRITE_EVERY"],
        seed=params["SEED"],
        checkpoint_blocks=True,
        output_dir=output_dir,
        output_name=name,
    )
    epsilon = jnp.asarray(params["EPS"], dtype=jnp.float32)
    start = time.perf_counter()
    trajectory = simulation({"epsilon": epsilon})
    trajectory.block_until_ready()
    elapsed = time.perf_counter() - start
    trajectories[name] = trajectory
    elapsed_by_scenario[name] = elapsed
    expected_frames = params["N_STEPS"] // params["WRITE_EVERY"] + 1
    print(
        f"Finished {name} in {elapsed:.2f} seconds. "
        f"Shape={trajectory.shape}, expected_frames={expected_frames}"
    )
    print(f"GSD written to: {Path(output_dir) / f'{name}.gsd'}")
