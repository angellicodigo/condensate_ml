# condensate_ml

**`condensate_ml`** is a high-performance, differentiable Brownian dynamics simulation library implemented in [JAX](https://github.com/google/jax) and [JAX MD](https://github.com/jax-md/jax-md). It simulates the liquid-liquid phase separation (LLPS) and biomolecular condensation of **Rubisco** and **EPYC1** proteins into algal pyrenoid condensates.

Because the entire simulation pipeline is implemented as a pure JAX function, trajectories can be **differentiated end-to-end** using reverse-mode automatic differentiation (`jax.grad`, `jax.vjp`), enabling gradient-based parameter estimation, inverse design, and machine learning integration directly through molecular trajectories.

---

## Key Highlights

- **Differentiable Through Time:** Backpropagate gradients directly through thousands of Brownian dynamics timesteps using reverse-mode automatic differentiation.
- **Hierarchical Checkpointing:** Employs block-level rematerialization (`jax.checkpoint`) across scan intervals to prevent GPU Out-Of-Memory (OOM) errors during long trajectory rollouts.
- **Closed-Form Analytical Forces:** Evaluates harmonic bonds, multi-type Gaussian steric repulsions, and sticker-patch attractions analytically in [src/forces.py](file:///C:/Users/angel/VSCode/GitHub/condensate_ml/src/forces.py) to avoid expensive higher-order autodiff overhead.
- **O(N) Spatial Cell Lists:** Accelerates nonbonded neighbor search using dense periodic cell lists from `jax_md.partition.neighbor_list` with dynamic overflow reallocation.
- **HOOMD-blue Compatibility:** Validated against HOOMD-blue molecular dynamics simulations; exports standard GSD trajectories and analysis tables.

---

## Physical Model

The simulation models the coarse-grained multivalent binding of Rubisco complexes and EPYC1 disordered repeat proteins:

```
          [Rubisco Core] (Type 0)
             /    |    \
          [P]    [P]   [P]  (Type 1 Patches: 8 per Rubisco)
           :
           : (Attraction: -epsilon * exp(-(r/sigma)^2))
           :
       ---[Sticker]---[Spacer]---[Sticker]---[Spacer]--- (EPYC1 Chain)
          (Type 2)    (Type 3)   (Type 2)    (Type 3)
```

- **Particles:**
  - **Rubisco:** 24 core beads (Type 0) each rigidly tethered to 8 binding patches (Type 1) arranged with cubic symmetry ($24 + 24 \times 8 = 216$ beads).
  - **EPYC1:** 48 flexible polymer chains of 12 coarse-grained beads each (alternating 6 Type 3 spacers and 6 Type 2 stickers; $48 \times 12 = 576$ beads).
  - **Total particles:** 792 beads.
- **Potentials:**
  - **Bonds:** Harmonic springs $V(r) = \frac{1}{2} k (r - r_0)^2$ for polymer backbones and patch tethers.
  - **Steric Repulsion:** Soft Gaussian pair repulsion $U(r) = \epsilon_{ij} e^{-(r/\sigma_{ij})^2}$ with type-dependent parameters and bond exclusion masks.
  - **Specific Attraction:** Gaussian sticker-patch attraction $U(r) = -\epsilon_{\text{bind}} e^{-(r/\sigma)^2}$ acting exclusively between Type 1 (Rubisco patches) and Type 2 (EPYC1 stickers).
- **Integrator:** Overdamped Langevin / Euler-Maruyama Brownian dynamics with particle-dependent hydrodynamic drag ($\gamma_{\text{Rubisco}}$ vs. $\gamma_{\text{chain}}$) and thermal noise.

---

## Repository Structure

```text
condensate_ml/
├── src/                          # Core simulation package
│   ├── simulation.py             # RubiscoEpyc1Simulation class & compiled lax.scan rollout
│   ├── forces.py                 # Vectorized analytical bonded & nonbonded force kernels
│   ├── constants.py              # Physical constants, default parameters, and geometries
│   ├── topology.py               # Pure topology builders (types, mobilities, bond masks)
│   ├── neighbor_list.py          # Adapter for JAX-MD periodic cell lists & coordinates
│   ├── initial_snapshot.py       # Deterministic initial coordinate generator
│   └── output.py                 # GSD, XYZ trajectory writers and CSV analysis exporters
├── helper/                       # Analysis and validation utilities
│   ├── regime_plots.py           # Multi-regime condensation plotting helpers
│   ├── trajectory_analysis.py    # GSD cluster detection and percolation analysis
│   ├── validate_differentiability.py # Automated loss and finite-difference validation
│   ├── validate_forces.py        # Force consistency benchmarks
│   └── validate_snapshot.py      # Initial structure validation
├── tests/                        # Comprehensive pytest test suite
│   ├── test_differentiable.py    # Reverse-mode autodiff tests
│   ├── test_forces.py            # Force evaluation and HOOMD parity tests
│   ├── test_internal_functions.py# Invariance, Newton's 3rd law, and cell list tests
│   ├── test_initial_conditions.py# Coordinate geometry tests
│   ├── test_long_simulations.py  # Long rollout sanity tests
│   └── test_one_step.py          # Single-step integrator tests
├── generate.py                   # Multi-regime simulation rollout script
├── generate_plots.py             # Multi-regime condensation figure generator
├── DOCUMENTATION.md              # Detailed architecture, mathematical specs, and API guide
├── pyproject.toml                # Package metadata and build definitions
└── environment.yml               # Complete Conda environment specification
```

---

## Installation

### Prerequisites
- Python $\ge$ 3.10
- CUDA 12.x (for GPU acceleration, optional but recommended)

### 1. Conda Environment (Recommended)

```bash
# Clone the repository
git clone https://github.com/your-username/condensate_ml.git
cd condensate_ml

# If running on an HPC cluster with CUDA 12.6:
# ml cudatoolkit/12.6
# export CONDA_OVERRIDE_CUDA="12.6"

# Create and activate the conda environment
conda env create -f environment.yml
conda activate condensate

# Install in editable mode
pip install -e .
```

### 2. Minimal pip Installation

```bash
pip install -e .

# With CUDA 12 support:
pip install -e ".[cuda12]"
```

---

## Quickstart

### 1. Basic Simulation Rollout

```python
import jax.numpy as jnp
from condensate_ml import RubiscoEpyc1Simulation

# Initialize simulation parameters
sim = RubiscoEpyc1Simulation(
    n_steps=10000,
    box=40.0,
    write_every=500,
    seed=42,
    checkpoint_blocks=True,
)

# Run simulation with sticker-patch binding affinity epsilon = 1.2 kT
epsilon = jnp.asarray(1.2, dtype=jnp.float32)
trajectory = sim({"epsilon": epsilon})

print("Trajectory shape (n_frames, n_particles, 3):", trajectory.shape)
# Output: (21, 792, 3)
```

### 2. Reverse-Mode Autodiff through Simulation Trajectory

Optimize or take gradients of an objective function with respect to physical simulation parameters (such as `epsilon`):

> [!NOTE]
> **Parameterization & Gradient Sign:**
> The simulation converts input `epsilon` to an attractive well depth via $-\left|\epsilon\right|$ internally, allowing both positive binding affinities ($\epsilon > 0$) and negative HOOMD-style potential depths ($\epsilon < 0$) to produce attractive forces. Because of the $-|\epsilon|$ mapping, $\frac{\partial (-\left|\epsilon\right|)}{\partial \epsilon} = -\operatorname{sgn}(\epsilon)$, so the gradient with respect to a positive $\epsilon$ has the opposite sign to a negative $\epsilon$. When training neural networks or running gradient-based optimization on binding strength, parameterize $\epsilon$ as positive (e.g. through softplus, ReLU, or positive initialization) so that $\frac{\partial \text{contacts}}{\partial \epsilon} > 0$ consistently reflects that increased binding affinity yields increased condensation contacts.

```python
import jax
import jax.numpy as jnp
from condensate_ml import RubiscoEpyc1Simulation

sim = RubiscoEpyc1Simulation(
    n_steps=500,
    box=40.0,
    write_every=50,
    checkpoint_blocks=True,
)

def loss_fn(eps):
    trajectory = sim({"epsilon": eps})
    final_frame = trajectory[-1]
    # Example objective: minimize final mean-squared distance from origin
    return jnp.mean(final_frame ** 2)

# Compute loss and gradient in a single reverse-mode pass
value, grad = jax.value_and_grad(loss_fn)(jnp.asarray(1.5, dtype=jnp.float32))

print(f"Loss: {value:.4f}, dLoss/dEpsilon: {grad:.6f}")
```

### 3. Running Regime Sweeps and Generating Figures

To run the five characteristic physical regimes (`control`, `early`, `nucleation`, `weak_coarsening`, `coarsening`) and generate condensation plots:

```bash
# Configure your destination directory in .env:
# OUTPUT_DIR=/path/to/output

python generate.py
python generate_plots.py
```

---

## Running the Test Suite

Run unit, force, and differentiability tests using `pytest`:

```bash
# Run differentiability and force tests
pytest tests/test_differentiable.py tests/test_internal_functions.py -v

# Run all non-HOOMD tests
pytest tests/ -k "not hoomd" -v
```
