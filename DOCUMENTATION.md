# Source Code Architecture (`src/`)

An overview of the core differentiable Brownian dynamics simulation codebase for Rubisco–EPYC1 condensate formation.

```text
src/
├── __init__.py               # Package root; exports RubiscoEpyc1Simulation
├── constants.py              # Physical parameters, particle counts, and interaction tables
├── forces.py                 # Differentiable analytical force functions (bonded, steric, attraction)
├── initial_snapshot.py       # Deterministic initial 3D coordinates generator
├── neighbor_list.py          # Coordinate transformation adapter for JAX-MD cell-lists
├── output.py                 # Trajectory writing (GSD, XYZ), cluster analysis, and metric plots
├── simulation.py             # Main RubiscoEpyc1Simulation class & compiled lax.scan rollout
└── topology.py               # Pure topology builders (type IDs, mobilities, bond exclusion masks)
```

---

## simulation.py
This file contains the `RubiscoEpyc1Simulation` that hold these variables:

Input parameters:
- n_steps - Total number of Brownian dynamics integration steps to execute in the simulation.
- dt - Integration timestep size for Brownian dynamics.
- kT - Thermal energy scale ($k_B T$) controlling thermal noise fluctuations.
- box - Side length of the cubic periodic simulation box.
- write_every - Timestep interval at which simulation trajectory frames are recorded.
- seed - Random seed used for initial coordinate generation and thermal noise sampling.
- dtype - Floating-point data type used for simulation array computations.
- _extra_capacity - Extra neighbor slot padding allocated per particle to prevent neighbor list overflow during dynamics.
- _max_capacity_retries - Maximum number of neighbor list capacity increase and reallocation retries allowed upon overflow.
- last_neighbor_overflow - Boolean flag tracking whether a neighbor list overflow occurred during the most recent simulation run.
- output_dir - Directory path where trajectory files and analysis figures are saved.
- output_name - Base filename used when exporting simulation trajectories and analysis outputs.

From `constant.py`:
- n_rubisco - Total number of Rubisco core particles in the system.
- n_patches_per_rubisco - Number of binding patch sites rigidly tethered to each Rubisco core particle.
- n_chains - Total number of flexible EPYC1 polymer chains in the simulation.
- n_beads_per_chain - Number of coarse-grained beads comprising each EPYC1 polymer chain.
- n_particles - Total count of all particles in the system, equal to `n_rubisco + n_rubisco * n_patches_per_rubisco + n_chains * n_beads_per_chain`.
- sticker_local_indices - In a EPYC1 chain, it alternates between spacers and stickers. sticker_local_indices a tuple of a list of 6 indices of stickers.
- patch_dirs - 2D array of dimensions `(n_patches_per_rubisco, 3)` containing normalized unit vectors that define the spatial orientation of binding patches around each Rubisco core.

Other important variables are built from functions in `topology.py`:
- particle_labels - 1D list of length `n_particles` containing descriptive string labels identifying each particle's role and identity in the system. E.g. `['R', 'R', ..., 'P', 'P', ..., 'S', 'T', 'S', 'T', 'S', 'T', ...]`.
- type_ids - 1D array of dimensions `(n_particles,)` containing the integer type identifier of each particle for mapping to interaction parameter tables. E.g. `[0, 0, ..., 1, 1, ..., 2, 3, 2, 3, 2, 3, ...]`
- mobilities - 2D array of dimensions `(n_particles, 1)` defining the hydrodynamic inverse-friction mobility for each particle. Each value is between and including 0 and 1. 

- bond_pairs - 2D array of dimensions `(n_bonds, 2)` defining particle index pairs for harmonic bonds, where `n_bonds = n_chains * (n_beads_per_chain - 1) + n_rubisco * n_patches_per_rubisco`.
- bond_k - 1D array of dimensions `(n_bonds,)` specifying harmonic spring force constants for each bonded pair, where `n_bonds = n_chains * (n_beads_per_chain - 1) + n_rubisco * n_patches_per_rubisco`.
- bond_r0 - 1D array of dimensions `(n_bonds,)` specifying equilibrium rest bond lengths for each bonded pair, where `n_bonds = n_chains * (n_beads_per_chain - 1) + n_rubisco * n_patches_per_rubisco`.

- nonbonded_pair_mask - 2D boolean mask matrix of dimensions `(n_particles, n_particles)` indicating which particle pairs are excluded from nonbonded interactions due to bonded connectivity.
- epsilon_table - 2D matrix of dimensions `(n_types, n_types)` defining steric repulsion energy well depths between particle types (where `n_types = 4`). Each epsilon value is between and including 0 and 1.
- sigma_table - 2D matrix of dimensions `(n_types, n_types)` defining steric collision diameters between particle types (where `n_types = 4`).
- cutoff_table - 2D matrix of dimensions `(n_types, n_types)` defining pairwise spatial cutoff distances beyond which nonbonded forces are zeroed (where `n_types = 4`).

Lastly, variables created from JAX and JAX-MD:
- displacement - JAX-MD periodic displacement function computing minimum-image displacement vectors between positions.
- neighbor_fn - JAX-MD neighbor list generator function used to allocate and update interacting particle pairs.
- compiled_simulate_fn - JIT-compiled simulation rollout function executed via `jax.lax.scan` across time blocks.

Here is example code to use `RubiscoEpyc1Simulation`

```python
params = {
    "name": "control", 
    "BOX": 40, 
    "EPS": 0.5, 
    "N_STEPS": 100000, 
    "WRITE_EVERY": 500, 
    "SEED": 101
}

simulation = RubiscoEpyc1Simulation(
    n_steps=params["N_STEPS"],
    box=params["BOX"],
    write_every=params["WRITE_EVERY"],
    seed=params["SEED"],
    checkpoint_blocks=True,
    output_dir=output_dir,
    output_name=name,
)
```

Caling RubiscoEpyc1Simulation() first calls the __init__() method.

```python
epsilon = jnp.asarray(params["EPS"], dtype=jnp.float32)
trajectory = simulation({"epsilon": epsilon})
```

This calls the method __call__(), which executes the differentiable simulation rollout and returns sampled trajectory frames. Let's look at what this method does.

```python
def __call__(self, fields: Mapping[str, Any]) -> jax.Array:
    """Executes the differentiable simulation rollout and returns sampled trajectory frames.

    Inputs:
        fields: Mapping containing required 'epsilon' and optional 'positions_0'.
    Outputs:
        Sampled trajectory array shaped (n_frames, n_particles, 3).
    """
    epsilon = fields.get("epsilon")

    if epsilon is None:
        raise KeyError("JAX MD stepper requires fields['epsilon']")

    positions = self.init(fields)
    binding_strength = jnp.abs(jnp.asarray(epsilon, dtype=self.dtype))
    epsilon = -binding_strength
    key = jax.random.PRNGKey(self.seed)

    for attempt in range(self._max_capacity_retries + 1):
        neighbors = allocate_neighbors(
            self.neighbor_fn, positions, self.box, self._extra_capacity
        )
        frames, overflow = self.compiled_simulate_fn(positions, epsilon, key, neighbors)

        if isinstance(overflow, jax.core.Tracer):
            break

        self.last_neighbor_overflow = bool(overflow)

        if not self.last_neighbor_overflow:
            break

        if attempt < self._max_capacity_retries:
            self._extra_capacity = max(2 * self._extra_capacity, 16)
    else:
        warnings.warn(
            "Neighbor list overflowed even after growing its capacity; some interacting pairs "
            "were dropped. Increase neighbor_extra_capacity.",
            RuntimeWarning,
            stacklevel=2,
        )

    if self.output_dir is not None and not isinstance(frames, jax.core.Tracer):
        self.write_outputs(frames, binding_strength)

    return frames
```

A lot is happening in this method. Here, we call init(), allocate_neighbors(), compiled_simulate(), write_outputs() (which we will ignore).

`positions = self.init(fields)` does two things. It first checks if initial positions are given. If not given, then it calls `generate_initial_coordinates()`, which creates a particle coordinates array with the shape `(n_particles, 3)`.

`neighbors = allocate_neighbors(self.neighbor_fn, positions, self.box, self._extra_capacity)`. This line basically uses neighbor_fn defined by JAX-MD to create neighbors. The only thing we do is change the coordinate system.
- In the HOOMD Simulation: Particle coordinates are origin-centered: $[-
  \frac{\text{box}}{2}, \, \frac{\text{box}}{2})$.
- In JAX-MD's cell-list: Requires coordinates in the non-negative interval: $[0, \,
  \text{box})$.
- If you pass centered coordinates directly to self.neighbor_fn.allocate(positions), JAX-MD's spatial cell grid receives negative coordinates, placing particles in invalid cell bins.

`frames, overflow = self.compiled_simulate_fn(positions, epsilon, key, neighbors)`. Here, `compiled_simulate_fn` called `simulate()`, which returned `rollout()` with all the variables, and all of this was inputted into a `jax.jit()`. This basically compiled `rollout()` in a way s.t. it's very efficient to compute. We then pass into `rollout()` the parameters `positions`, `epsilon`, `key`, and `neighbors`.

```python
def simulate(self, checkpoint_blocks: bool):
    """Constructs the pure JAX-compilable scan function advancing the Brownian dynamics trajectory.

    Inputs:
        checkpoint_blocks: Whether to checkpoint scan intervals during reverse autodiff.
    Outputs:
        Rollout callable returning a tuple of (trajectory_frames, did_overflow).
    """
    displacement = self.displacement
    dt = self.dt
    kT = self.kT
    block_size = self.write_every
    n_blocks = self.n_steps // self.write_every
    mobility = self.mobilities
    epsilon_table = self.epsilon_table
    sigma_table = self.sigma_table
    cutoff_table = self.cutoff_table
    type_ids = self.type_ids
    exclusions = self.nonbonded_pair_mask
    bond_pairs, bond_k, bond_r0 = self.bond_pairs, self.bond_k, self.bond_r0
    box = self.box

    def wrap_centered(positions):
        """Wraps coordinates into the origin-centered periodic box interval.

        Inputs:
            positions: Particle coordinates array.
        Outputs:
            Periodic wrapped coordinates mapped to [-box / 2, box / 2).
        """
        return (positions + box * 0.5) % box - box * 0.5

    def one_step(state, _):
        """Advances the simulation state by one overdamped Euler-Maruyama Brownian dynamics step.

        Inputs:
            state: State tuple of (positions, neighbors, key, epsilon).
            _: Unused scan step index.
        Outputs:
            Tuple of (updated_state, None).
        """
        positions, neighbors, key, epsilon = state
        neighbors = neighbors.update(shift_to_neighbor_list_coordinates(positions, box))
        key, noise_key = jax.random.split(key)

        total = forces.calculate_total_force(
            positions, neighbors.idx, type_ids, epsilon_table, sigma_table,
            cutoff_table, exclusions, bond_pairs, bond_k, bond_r0,
            epsilon, PATCH_SIGMA, displacement,
        )
        noise = jnp.sqrt(2.0 * kT * mobility * dt) * jax.random.normal(
            noise_key, positions.shape, dtype=positions.dtype
        )
        next_positions = wrap_centered(positions + mobility * total * dt + noise)
        return (next_positions, neighbors, key, epsilon), None

    def integrate_block(state, _):
        """Advances the simulation through one output block interval.

        Inputs:
            state: State tuple passed through the timestep scan.
            _: Unused outer scan index.
        Outputs:
            Tuple of (updated_state, final_block_positions).
        """
        state, _ = jax.lax.scan(one_step, state, None, length=block_size)
        return state, state[0]

    if checkpoint_blocks:
        integrate_block = jax.checkpoint(integrate_block)

    def rollout(positions, epsilon, key, neighbors):
        """Runs all observation blocks and concatenates the resulting trajectory frames.

        Inputs:
            positions: Starting coordinates array shaped (n_particles, 3).
            epsilon: Signed sticker-patch attraction energy parameter.
            key: JAX PRNG key for thermal noise.
            neighbors: Preallocated dense neighbor list object.
        Outputs:
            Tuple of (trajectory_frames, overflow_flag).
        """
        state = (positions, neighbors, key, epsilon)
        state, block_frames = jax.lax.scan(integrate_block, state, None, length=n_blocks)
        return jnp.concatenate((positions[None], block_frames), axis=0), state[1].did_buffer_overflow

    return rollout
```

