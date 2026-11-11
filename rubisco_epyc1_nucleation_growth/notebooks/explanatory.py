# /// script
# [tool.marimo-studio]
# default = "animation"
# ///

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import json
    import numpy as np
    import gsd.hoomd
    import hoomd
    import hoomd.md as md
    import os
    from dotenv import load_dotenv
    from pathlib import Path
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

    dotenv_path = Path(__file__).resolve().parent.parent.parent / ".env"
    load_dotenv(dotenv_path=dotenv_path)
    return Path, hoomd, md, mo, mpimg, np, os, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Purpose

    The purpose of this notebook is to:
    1. Explain how the code in the "simulate_hoomd_patchy_full.py" works
    2. Explain the physics involved in modeling the pyrenoid and cholorplast.
    3. Explain how we translated "simulate_hoodmd_patchy_full.py" to a differentiable simulation that uses Pytorch tensors.

    We will start by analyzing lines of code in the "simulate_hood_patchy_full.py" file. The Python cells below will consist of code copied from the "simulate_hood_patchy_full.py" file.
    """)
    return


@app.cell
def _():
    params_list = [
      {
        "name": "control",
        "BOX": 40,
        "EPS": 0.5,
        "N_STEPS": 100000,
        "WRITE_EVERY": 500,
        "SEED": 101
      },
      {
        "name": "early",
        "BOX": 36,
        "EPS": 0.8,
        "N_STEPS": 100000,
        "WRITE_EVERY": 500,
        "SEED": 102
      },
      {
        "name": "nucleation",
        "BOX": 33,
        "EPS": 1.2,
        "N_STEPS": 100000,
        "WRITE_EVERY": 500,
        "SEED": 103
      },
      {
        "name": "weak_coarsening",
        "BOX": 31,
        "EPS": 1.8,
        "N_STEPS": 100000,
        "WRITE_EVERY": 500,
        "SEED": 104
      },
      {
        "name": "coarsening",
        "BOX": 29,
        "EPS": 2.5,
        "N_STEPS": 100000,
        "WRITE_EVERY": 500,
        "SEED": 105
      }
    ]
    return (params_list,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Each parameter in "params" represents a different regime, or a specific stage of behavior in the system.
    - Control: no binding between molecules, which leads to no condensate formations. This serves as a reference point.
    - Early: small number of condensates start to form, but are sparse and unstable. There is a constant rate of condensate formation.
    - Nucleation: condensates starts to become more stable. The rate of condesate formations increases sharply.
    - Weak/Early Coarsening: smaller condensates start connecting to each other to form a larger connected component. The rate of condensate formation starts to decline.
    - Coarsening: a highly and unified connected network. The rate of condensate formations rapidly declines to near 0.

    Each regime follows different physics and can be modeled by plotting the number of clusters over time.
    """)
    return


@app.cell
def _(params_list):
    # ============================================================
    # INPUT PARAMS
    # ============================================================
    # In simulate_hoomd_patchy_full.py, it's designed to input one element in params_list at a time.
    params = params_list[0]

    NAME = params["name"]
    BOX = float(params["BOX"]) # Represents the length of the simulation cube that follows periodic boundary conditions
    # Binding strength between EPYC1 sticker beads (T) and Rubisco patch particles (P)
    # Larger KON -> more persistent binding
    KON = float(params.get("KON", params.get("EPS"))) 
    KOFF = float(params.get("KOFF", 0.0)) # This is never used in the file

    SEED = int(params.get("SEED", 42))
    N_STEPS = int(params.get("N_STEPS", 10000))
    WRITE_EVERY = int(params.get("WRITE_EVERY", 100))
    DT = float(params.get("DT", 0.0005)) # The timestep
    return BOX, DT, KON, SEED


@app.cell
def _(KON):
    # ============================================================
    # MODEL PARAMETERS
    # ============================================================
    N_R = 24 # 24 Rubisco particles
    N_C = 48 # 48 EPYC1 polymers

    # Each EPYC1 chain has 12 beads. Alternating between stickers and spacers
    N_BEADS = 12 
    STICKER_INDEX = [1, 3, 5, 7, 9, 11]

    # Each Rubisco has 8 surface patches that are arranged symmetrically at radius 2.5
    # from the core of the Rubisco
    PATCHES = 8 
    PATCH_RADIUS = 2.5 

    # There is a tether from the patch to the core that acts like a spring
    CHAIN_K = 10.0 # Spring stiffness 
    CHAIN_R0 = 1.2 # Length where there is equilibrium

    PATCH_TETHER_K = 50.0

    # Sticker–patch attraction
    PATCH_EPSILON = -abs(KON)
    PATCH_SIGMA = 1.0
    PATCH_RCUT = 3.0

    CONTACT_CUTOFF = 3.0
    return (
        CHAIN_K,
        CHAIN_R0,
        N_BEADS,
        N_C,
        N_R,
        PATCHES,
        PATCH_EPSILON,
        PATCH_RADIUS,
        PATCH_RCUT,
        PATCH_SIGMA,
        PATCH_TETHER_K,
        STICKER_INDEX,
    )


@app.cell(hide_code=True)
def _():
    # from manim import Scene, Square, Circle, PINK, Create, Transform, tempconfig

    # class SquareToCircle(Scene):
    #     def construct(self):
    #         square = Square()
    #         circle = Circle()
    #         circle.set_fill(PINK, opacity=0.5)
    #         self.play(Create(square))
    #         self.play(Transform(square, circle))
    #         self.wait()


    # with tempconfig({
    #     "disable_caching": True,
    #     "verbosity": "WARNING",
    # }):
    #     scene = SquareToCircle()
    #     scene.render()

    # video_path = scene.renderer.file_writer.final_file_path
    # mo.video(str(video_path))
    return


@app.cell
def _(SEED, hoomd):
    # ============================================================
    # GPU DEVICE
    # ============================================================
    # device = hoomd.device.GPU()
    device = hoomd.device.CPU() # Uses HOOMD, a Python package that runs simulations of particle systems
    sim = hoomd.Simulation(device=device, seed=SEED)
    '''
    R: Rubisco core center.
    P: Rubisco surface patch.
    S: EPYC1 spacer bead.
    T: EPYC1 sticker bead.
    '''
    particle_types = ["R", "P", "S", "T"]
    return particle_types, sim


@app.cell
def _(np):
    # ============================================================
    # PATCH GEOMETRY
    # ============================================================
    patch_dirs = np.array(
        [
            [1, 1, 1],
            [1, 1, -1],
            [1, -1, 1],
            [1, -1, -1],
            [-1, 1, 1],
            [-1, 1, -1],
            [-1, -1, 1],
            [-1, -1, -1],
        ],
        dtype=float,
    )
    patch_dirs /= np.linalg.norm(patch_dirs, axis=1, keepdims=True)
    return (patch_dirs,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Here we are treating the particles as patchy particles. This means that the Rubisco has a core center (R) and highly interactive surface patches (P) tethered to the center.

    ```
    patch_dirs = np.array(
        [
            [1, 1, 1],
            [1, 1, -1],
            [1, -1, 1],
            [1, -1, -1],
            [-1, 1, 1],
            [-1, 1, -1],
            [-1, -1, 1],
            [-1, -1, -1],
        ]
    )
    ```
    In this matrix, each row is 8 3D directional vectors pointing at the 8 corners of the cube. We normalized these directional vectors to unit length.
    """)
    return


@app.cell
def _(np):
    def wrap_centered(x, box):
        return ((x + box / 2) % box) - box / 2


    def make_grid_positions(n, box, margin=6.0):
        n_side = int(np.ceil(n ** (1 / 3)))
        vals = np.linspace(-box / 2 + margin, box / 2 - margin, n_side)
        pts = np.array(np.meshgrid(vals, vals, vals)).T.reshape(-1, 3)
        return pts[:n].copy()

    return make_grid_positions, wrap_centered


@app.cell
def _(
    BOX,
    CHAIN_R0,
    N_BEADS,
    N_C,
    N_R,
    PATCHES,
    PATCH_RADIUS,
    SEED,
    STICKER_INDEX,
    hoomd,
    make_grid_positions,
    np,
    particle_types,
    patch_dirs,
    sim,
    wrap_centered,
):
    # ============================================================
    # BUILD INITIAL SNAPSHOT
    # ============================================================
    snap = hoomd.Snapshot() # An empty container where the simulation will occur
    print(sim.device)
    if sim.device.communicator.rank == 0: # You would have this line of code if using a GPU
        rng = np.random.default_rng(SEED)

        n_patch = N_R * PATCHES # Total number of patches per number of Rubisco
        n_chain = N_C * N_BEADS # Total number of beads per number of EPYC1
        n_total = N_R + n_patch + n_chain # Total number of particles

        snap.configuration.box = [BOX, BOX, BOX, 0, 0, 0] # First 3 numbers are the side lengths. The last 3 are the box tilt.
        snap.particles.N = n_total
        snap.particles.types = particle_types # Converts ["R", "P", "S", "T"] -> [0, 1, 2, 3]

        pos = np.zeros((n_total, 3), dtype=float) # Stores the initial position of every particle
        typeid = np.zeros(n_total, dtype=np.uint32) # Stores the particle type of each particle

        # -----------------------------
        # Rubisco centers
        # -----------------------------
        # Puts 24 Rubisco centers on a regular 3D grid, keeping them at least some margin units from the box edge.
        rubisco_centers = make_grid_positions(N_R, BOX, margin=7.0)
        rubisco_centers += rng.normal(scale=0.20, size=rubisco_centers.shape) # Adds some noise to the centers
        rubisco_centers = wrap_centered(rubisco_centers, BOX) # Any centers that are outside the box wraps back in

        idx = 0
        for r in range(N_R):
            pos[r] = rubisco_centers[r]
            typeid[idx] = 0
            idx += 1

        # -----------------------------
        # Rubisco patch particles
        # IMPORTANT: do NOT wrap patch particles independently.
        # They must remain local to their Rubisco center to avoid
        # huge patch-tether bonds across the periodic boundary.
        # -----------------------------
        patch_start = idx # Indices 0 to idx of pos is for the rubisco. Now we are deaing with the patch particles
        for r in range(N_R):
            center = rubisco_centers[r]
            for p in range(PATCHES):
                ppos = center + PATCH_RADIUS * patch_dirs[p] # Question: why do we not add noise to the pos of the Rubisco patches?
                pos[idx] = ppos
                typeid[idx] = 1
                idx += 1

        # -----------------------------
        # EPYC1 chains
        # -----------------------------
        # Here we represent an EPYC1 chain as 12 beads aligned in a straight line centered at chain_centers[c]. 
        # The other end of the straight line is pointing in a random direction.
        chain_start = N_R + n_patch # Find the first empty index that is not meant for the Rubisco

        chain_centers = make_grid_positions(N_C, BOX, margin=5.0)
        chain_centers += rng.normal(scale=0.25, size=chain_centers.shape)
        chain_centers = wrap_centered(chain_centers, BOX)

        for c in range(N_C):
            center = chain_centers[c]
            direction = rng.normal(size=3)
            direction /= np.linalg.norm(direction)

            for bead in range(N_BEADS):
                bead_pos = center + (bead - (N_BEADS - 1) / 2) * CHAIN_R0 * direction
                bead_pos = wrap_centered(bead_pos, BOX)

                pos[idx] = bead_pos
                # A particle type is 3 if corresponds to a sticker (T) if the bead index is in [1, 3, 5, 7, 9, 11], otherwise 
                # set the particle type to be 2 (S)
                typeid[idx] = 3 if bead in STICKER_INDEX else 2 
                idx += 1

        snap.particles.position[:] = pos
        snap.particles.typeid[:] = typeid

        # -----------------------------
        # EPYC1 chain bonds
        # -----------------------------
        # Now we are creating the bonds between the EPYC1 particles within a chain
        chain_bonds = []
        for c in range(N_C):
            base = chain_start + c * N_BEADS
            for bead in range(N_BEADS - 1):
                chain_bonds.append([base + bead, base + bead + 1])

        # -----------------------------
        # Patch tether bonds
        # These keep patch particles attached to Rubisco centers.
        # -----------------------------
        patch_tethers = []
        for r in range(N_R):
            for p in range(PATCHES):
                patch_tethers.append([r, patch_start + r * PATCHES + p])

        # -----------------------------
        # Finalize the snapshot
        # -----------------------------
        all_bonds = chain_bonds + patch_tethers

        snap.bonds.N = len(all_bonds)
        snap.bonds.types = ["chain", "patch_tether"]
        bond_typeid = np.zeros(len(all_bonds), dtype=np.uint32) # len(all_bonds) are for "chain", the rest is for "patch_tether"
        bond_typeid[len(chain_bonds):] = 1

        snap.bonds.typeid[:] = bond_typeid
        snap.bonds.group[:] = np.array(all_bonds, dtype=np.uint32)

    sim.create_state_from_snapshot(snap)
    return


@app.cell
def _(md, particle_types):
    # ============================================================
    # NEIGHBOR LIST
    # ============================================================
    # HOOMD divides the box into cells and only checks pairs of particles in nearby cells, which makes the simulation much faster than checking every particle against every other particle.
    # buffer=0.4 means that particles within 0.4 or less distance away are computed 
    nl = md.nlist.Cell(buffer=0.4)

    # ============================================================
    # BOUNDED SOFT REPULSION
    # Gaussian repulsion is bounded and avoids LJ blow-ups.
    # ============================================================
    # A repulsion force prevents unrealistic overlap
    soft_rep = md.pair.Gaussian(nlist=nl)

    for type_a1 in particle_types:
        for type_a2 in particle_types:
            soft_rep.params[(type_a1, type_a2)] = dict(epsilon=0.0, sigma=1.0)
            soft_rep.r_cut[(type_a1, type_a2)] = 0.0

    soft_rep.params[("R", "R")] = dict(epsilon=2.0, sigma=1.5)
    soft_rep.r_cut[("R", "R")] = 3.0

    for pair in [("S", "S"), ("T", "T"), ("S", "T")]:
        soft_rep.params[pair] = dict(epsilon=1.0, sigma=1.0)
        soft_rep.r_cut[pair] = 2.5

    for pair in [("R", "S"), ("R", "T")]:
        soft_rep.params[pair] = dict(epsilon=1.5, sigma=1.5)
        soft_rep.r_cut[pair] = 3.0

    for pair in [("P", "P"), ("P", "S"), ("P", "R")]:
        soft_rep.params[pair] = dict(epsilon=0.2, sigma=0.8)
        soft_rep.r_cut[pair] = 2.0
    return nl, soft_rep


@app.cell
def _(PATCH_EPSILON, PATCH_RCUT, PATCH_SIGMA, md, nl, particle_types):
    # ============================================================
    # STICKER-PATCH ATTRACTION
    # Force-based interaction. No discrete binding flags.
    # Negative Gaussian epsilon gives attraction.
    # ============================================================
     # A bound sticker and occupied patch are coupled by an effective spring
    patch_att = md.pair.Gaussian(nlist=nl) # Also uses the neighbor list

    for a in particle_types:
        for b in particle_types:
            patch_att.params[(a, b)] = dict(epsilon=0.0, sigma=1.0)
            patch_att.r_cut[(a, b)] = 0.0

    patch_att.params[("T", "P")] = dict(epsilon=PATCH_EPSILON, sigma=PATCH_SIGMA)
    patch_att.r_cut[("T", "P")] = PATCH_RCUT
    return (patch_att,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Remember that
    ```
    PATCH_EPSILON = -abs(KON)
    ```
    Because it's negative, the Gaussian force between a Rubisco surface patch and a EPYC1 sticker bead is attractive, not repulsive like the previous Python cell.
    """)
    return


@app.cell
def _(CHAIN_K, CHAIN_R0, PATCH_RADIUS, PATCH_TETHER_K, md):
    # ============================================================
    # BOND FORCES
    # ============================================================
    # Adjacent EPYC1 beads are connected by harmonic springs
    harmonic = md.bond.Harmonic()
    harmonic.params["chain"] = dict(k=CHAIN_K, r0=CHAIN_R0)
    harmonic.params["patch_tether"] = dict(k=PATCH_TETHER_K, r0=PATCH_RADIUS)
    return (harmonic,)


@app.cell
def _(DT, harmonic, hoomd, md, patch_att, sim, soft_rep):
    # ============================================================
    # BROWNIAN DYNAMICS
    # ============================================================
    integrator = md.Integrator(dt=DT)

    all_particles = hoomd.filter.All()
    # Means all particles follows Brownian dynamics. This means that particles will experience random nudges determined by kT plus friction
    # kT: Temperature of the simulation (energy)
    brownian = md.methods.Brownian(filter=all_particles, kT=1.0) 

    # Drag/Friction coefficients per particle type
    # The smaller the number, the particle diffuses faster
    brownian.gamma["R"] = 5.0
    brownian.gamma["P"] = 5.0
    brownian.gamma["S"] = 1.0
    brownian.gamma["T"] = 1.0

    integrator.methods.append(brownian)
    integrator.forces.append(soft_rep)
    integrator.forces.append(patch_att)
    integrator.forces.append(harmonic)

    sim.operations.integrator = integrator
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The rest of the code involves running the simulation
    ```
    sim.run(N_STEPS)
    ```
    and saving the results in four different folders.

    - CSV timecourses with
        - step
        - time
        - sigma
        - lambda
        - eta
        - number of clusters
        - largest cluster size
        - largest cluster fraction
        - number of bonds.

    - JPEG plots of each of those main observables.

    - GSD files that store the trajectories of each particle in the HOOMD simulation

    - XYZ trajectories with particle types:
        - R = Rubisco
        - S = spacer
        - T = sticker

    Let's look at the images. When we make this simulation differentiable, we can use these files from the original simulation to validate our results of the new simulation.
    """)
    return


@app.cell
def _(Path, mpimg, os, plt):
    dir = Path(os.getenv("plots_patchy_gpu"))

    scenarios = ["coarsening", "control", "early", "nucleation", "weak_coarsening"]
    metrics   = ["bonds", "clusters", "fraction", "largest"]

    images = {
        f"{s}_{m}": mpimg.imread(dir / f"{s}_{m}.jpeg")
        for s in scenarios
        for m in metrics
    }

    n_rows, n_cols = len(scenarios), len(metrics)

    # Leave room on top and left for headers
    left, right   = 0.01, 0.99
    bottom, top   = 0, 0.90

    fig = plt.figure(figsize=(25, 25))
    gs = fig.add_gridspec(
        n_rows, n_cols,
        left=left, right=right, bottom=bottom, top=top,
    )

    # Column centers and row centers in figure coordinates
    col_w = (right - left) / n_cols
    row_h = (top - bottom) / n_rows

    # Top headers (metrics)
    for j, m in enumerate(metrics):
        x = left + (j + 0.5) * col_w
        fig.text(x, 0.965, m.title(), ha="center", va="center",
                 fontsize=17, fontweight="bold")

    # Left headers (scenarios, vertical)
    for i, s in enumerate(scenarios):
        y = bottom + (n_rows - i - 0.5) * row_h
        fig.text(0.01, y, s.title(), ha="center", va="center",
                 fontsize=17, fontweight="bold", rotation=90)

    # Images
    for i, s in enumerate(scenarios):
        for j, m in enumerate(metrics):
            ax = fig.add_subplot(gs[i, j])
            ax.imshow(images[f"{s}_{m}"])
            ax.axis("off")
        
    fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Differentiable Simulation
    """)
    return


if __name__ == "__main__":
    app.run()
