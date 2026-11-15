from condensate_ml import RubiscoEpyc1TimeStepper
import torch
import time
torch.manual_seed(102)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
param_example = {
    "name": "control",
    "BOX": 40,
    "EPS": 0.5,
    "N_STEPS": 100000,
    "WRITE_EVERY": 500,
    "SEED": 101
}

epsilon = torch.tensor(-param_example.get("EPS"), device=device, dtype=torch.float32, requires_grad=True)

stepper = RubiscoEpyc1TimeStepper(
    n_steps=param_example.get("N_STEPS"),
    box=param_example.get("BOX"),
    write_every=param_example.get("WRITE_EVERY"),
    seed=param_example.get("SEED"),
    device=device,
    grad_mode="checkpoint",      # default; keeps memory bounded for long rollouts
    compile_step=True
)
start = time.time()
trajectory = stepper({"epsilon": epsilon})
elapsed = time.time() - start

print(f"Simulation completed in {elapsed:.2f} seconds")
print(f"Trajectory shape: {trajectory.shape}")
