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
    import sys
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
    return Path, gsd, hoomd, md, mo, mpimg, np, os, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Purpose

    The purpose of this notebook is to:
    1. Explain how the code in the "simulate_hoomd_patchy_full.py" works
    2. Explain the physics involved in modeling the pyrenoid and cholorplast.
    3. Explain how we translated "simulate_hood_patchy_full.py" into a differentiable version of the original simulation.

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
    # Using a repulsion force to prevent unrealistic overlap of the particles
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
    hoodmd_plots = Path(os.getenv("plots_patchy_gpu"))

    scenarios = ["coarsening", "control", "early", "nucleation", "weak_coarsening"]
    metrics   = ["bonds", "clusters", "fraction", "largest"]

    images = {
        f"{s}_{m}": mpimg.imread(hoodmd_plots / f"{s}_{m}.jpeg")
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
    return hoodmd_plots, metrics


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Differentiable Simulation
    To make this simulation differentiable, we will use a Python package called "jax-md" (https://github.com/jax-md/jax-md). JAX is a Python library by Google that accelerates numerical linear algebra computations. The code is implemented in `RubiscoEpyc1Simulation` that can be imported by doing `from condensate_ml import RubiscoEpyc1Simulation`. I already ran `RubiscoEpyc1Simulation` with each regime in `params_lst` and saved plots in `os.getenv("plots_jaxmd_gpu")`.
    """)
    return


@app.cell
def _(Path, hoodmd_plots, metrics, mpimg, os, plt):
    jaxmd_plots = Path(os.getenv("plots_jaxmd_gpu"))

    images1 = {
        f"hoodmd_{metric1}": mpimg.imread(
            hoodmd_plots / f"coarsening_{metric1}.jpeg"
        )
        for metric1 in metrics
    }

    images1.update({
        f"jaxmd_{metric1}": mpimg.imread(
            jaxmd_plots / f"coarsening_{metric1}.jpeg"
        )
        for metric1 in metrics
    })

    fig1 = plt.figure(figsize=(25, 10))

    grid1 = fig1.add_gridspec(
        2,
        len(metrics),
        left=0.08,
        right=0.99,
        bottom=0.06,
        top=0.88,
        wspace=0.15,
        hspace=0.12,
    )

    for column1, metric1 in enumerate(metrics):
        axis1_top = fig1.add_subplot(grid1[0, column1])
        axis1_bottom = fig1.add_subplot(grid1[1, column1])

        axis1_top.imshow(images1[f"hoodmd_{metric1}"])
        axis1_bottom.imshow(images1[f"jaxmd_{metric1}"])

        axis1_top.axis("off")
        axis1_bottom.axis("off")

        axis1_top.set_title(
            metric1.title(),
            fontsize=17,
            fontweight="bold",
        )

    fig1.text(
        0.025,
        0.67,
        "HOODMD",
        ha="center",
        va="center",
        fontsize=17,
        fontweight="bold",
        rotation=90,
    )

    fig1.text(
        0.025,
        0.30,
        "JAX-MD",
        ha="center",
        va="center",
        fontsize=17,
        fontweight="bold",
        rotation=90,
    )

    fig1.suptitle(
        "Coarsening",
        fontsize=20,
        fontweight="bold",
    )

    fig1
    return (jaxmd_plots,)


@app.cell
def _(hoodmd_plots, jaxmd_plots, metrics, mpimg, plt):
    images2 = {
        f"hoodmd_{metric2}": mpimg.imread(
            hoodmd_plots / f"control_{metric2}.jpeg"
        )
        for metric2 in metrics
    }

    images2.update({
        f"jaxmd_{metric2}": mpimg.imread(
            jaxmd_plots / f"control_{metric2}.jpeg"
        )
        for metric2 in metrics
    })

    fig2 = plt.figure(figsize=(25, 10))

    grid2 = fig2.add_gridspec(
        2,
        len(metrics),
        left=0.08,
        right=0.99,
        bottom=0.06,
        top=0.88,
        wspace=0.15,
        hspace=0.12,
    )

    for column2, metric2 in enumerate(metrics):
        axis2_top = fig2.add_subplot(grid2[0, column2])
        axis2_bottom = fig2.add_subplot(grid2[1, column2])

        axis2_top.imshow(images2[f"hoodmd_{metric2}"])
        axis2_bottom.imshow(images2[f"jaxmd_{metric2}"])

        axis2_top.axis("off")
        axis2_bottom.axis("off")

        axis2_top.set_title(
            metric2.title(),
            fontsize=17,
            fontweight="bold",
        )

    fig2.text(
        0.025,
        0.67,
        "HOODMD",
        ha="center",
        va="center",
        fontsize=17,
        fontweight="bold",
        rotation=90,
    )

    fig2.text(
        0.025,
        0.30,
        "JAX-MD",
        ha="center",
        va="center",
        fontsize=17,
        fontweight="bold",
        rotation=90,
    )

    fig2.suptitle(
        "Control",
        fontsize=20,
        fontweight="bold",
    )

    fig2
    return


@app.cell
def _(hoodmd_plots, jaxmd_plots, metrics, mpimg, plt):
    images3 = {
        f"hoodmd_{metric3}": mpimg.imread(
            hoodmd_plots / f"early_{metric3}.jpeg"
        )
        for metric3 in metrics
    }

    images3.update({
        f"jaxmd_{metric3}": mpimg.imread(
            jaxmd_plots / f"early_{metric3}.jpeg"
        )
        for metric3 in metrics
    })

    fig3 = plt.figure(figsize=(25, 10))

    grid3 = fig3.add_gridspec(
        2,
        len(metrics),
        left=0.08,
        right=0.99,
        bottom=0.06,
        top=0.88,
        wspace=0.15,
        hspace=0.12,
    )

    for column3, metric3 in enumerate(metrics):
        axis3_top = fig3.add_subplot(grid3[0, column3])
        axis3_bottom = fig3.add_subplot(grid3[1, column3])

        axis3_top.imshow(images3[f"hoodmd_{metric3}"])
        axis3_bottom.imshow(images3[f"jaxmd_{metric3}"])

        axis3_top.axis("off")
        axis3_bottom.axis("off")

        axis3_top.set_title(
            metric3.title(),
            fontsize=17,
            fontweight="bold",
        )

    fig3.text(
        0.025,
        0.67,
        "HOODMD",
        ha="center",
        va="center",
        fontsize=17,
        fontweight="bold",
        rotation=90,
    )

    fig3.text(
        0.025,
        0.30,
        "JAX-MD",
        ha="center",
        va="center",
        fontsize=17,
        fontweight="bold",
        rotation=90,
    )

    fig3.suptitle(
        "Early",
        fontsize=20,
        fontweight="bold",
    )

    fig3
    return


@app.cell
def _(hoodmd_plots, jaxmd_plots, metrics, mpimg, plt):
    images4 = {
        f"hoodmd_{metric4}": mpimg.imread(
            hoodmd_plots / f"nucleation_{metric4}.jpeg"
        )
        for metric4 in metrics
    }

    images4.update({
        f"jaxmd_{metric4}": mpimg.imread(
            jaxmd_plots / f"nucleation_{metric4}.jpeg"
        )
        for metric4 in metrics
    })

    fig4 = plt.figure(figsize=(25, 10))

    grid4 = fig4.add_gridspec(
        2,
        len(metrics),
        left=0.08,
        right=0.99,
        bottom=0.06,
        top=0.88,
        wspace=0.15,
        hspace=0.12,
    )

    for column4, metric4 in enumerate(metrics):
        axis4_top = fig4.add_subplot(grid4[0, column4])
        axis4_bottom = fig4.add_subplot(grid4[1, column4])

        axis4_top.imshow(images4[f"hoodmd_{metric4}"])
        axis4_bottom.imshow(images4[f"jaxmd_{metric4}"])

        axis4_top.axis("off")
        axis4_bottom.axis("off")

        axis4_top.set_title(
            metric4.title(),
            fontsize=17,
            fontweight="bold",
        )

    fig4.text(
        0.025,
        0.67,
        "HOODMD",
        ha="center",
        va="center",
        fontsize=17,
        fontweight="bold",
        rotation=90,
    )

    fig4.text(
        0.025,
        0.30,
        "JAX-MD",
        ha="center",
        va="center",
        fontsize=17,
        fontweight="bold",
        rotation=90,
    )

    fig4.suptitle(
        "Nucleation",
        fontsize=20,
        fontweight="bold",
    )

    fig4
    return


@app.cell
def _(hoodmd_plots, jaxmd_plots, metrics, mpimg, plt):
    images5 = {
        f"hoodmd_{metric5}": mpimg.imread(
            hoodmd_plots / f"weak_coarsening_{metric5}.jpeg"
        )
        for metric5 in metrics
    }

    images5.update({
        f"jaxmd_{metric5}": mpimg.imread(
            jaxmd_plots / f"weak_coarsening_{metric5}.jpeg"
        )
        for metric5 in metrics
    })

    fig5 = plt.figure(figsize=(25, 10))

    grid5 = fig5.add_gridspec(
        2,
        len(metrics),
        left=0.08,
        right=0.99,
        bottom=0.06,
        top=0.88,
        wspace=0.15,
        hspace=0.12,
    )

    for column5, metric5 in enumerate(metrics):
        axis5_top = fig5.add_subplot(grid5[0, column5])
        axis5_bottom = fig5.add_subplot(grid5[1, column5])

        axis5_top.imshow(images5[f"hoodmd_{metric5}"])
        axis5_bottom.imshow(images5[f"jaxmd_{metric5}"])

        axis5_top.axis("off")
        axis5_bottom.axis("off")

        axis5_top.set_title(
            metric5.title(),
            fontsize=17,
            fontweight="bold",
        )

    fig5.text(
        0.025,
        0.67,
        "HOODMD",
        ha="center",
        va="center",
        fontsize=17,
        fontweight="bold",
        rotation=90,
    )

    fig5.text(
        0.025,
        0.30,
        "JAX-MD",
        ha="center",
        va="center",
        fontsize=17,
        fontweight="bold",
        rotation=90,
    )

    fig5.suptitle(
        "Weak Coarsening",
        fontsize=20,
        fontweight="bold",
    )

    fig5
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # How is this simulation used with a Machine Learning model?
    The consequence of using JAX MD is that JAX uses JAX arrays instead of Pytorch tensors. Below is an example of a JAX neural network passing its output into our simulation. Here we use the package `flax` to easily create a JAX neural network, and `optax` package for optimizing said neural network. These two packages make creating a neural network similar if we just used Pytorch.
    """)
    return


@app.cell
def _():
    import jax
    import jax.numpy as jnp
    import optax
    from flax import nnx
    from condensate_ml import RubiscoEpyc1Simulation

    # ---------------------------------------------------------------------
    # 1. Differentiable Rubisco simulation
    # ---------------------------------------------------------------------

    simulation = RubiscoEpyc1Simulation(
        n_steps=32,
        box=40.0,
        write_every=8,
        seed=101,
        checkpoint_blocks=True,
        output_dir=None,  # Disable GSD/CSV/plot output during optimization
    )


    # ---------------------------------------------------------------------
    # 2. Flax neural network
    # ---------------------------------------------------------------------

    class EpsilonNetwork(nnx.Module):
        """Small coordinate/features-to-epsilon network."""

        def __init__(
            self,
            n_features: int = 4,
            n_hidden: int = 32,
            *,
            rngs: nnx.Rngs,
        ):
            self.layer1 = nnx.Linear(
                n_features,
                n_hidden,
                rngs=rngs,
            )

            self.layer2 = nnx.Linear(
                n_hidden,
                n_hidden,
                rngs=rngs,
            )

            self.output_layer = nnx.Linear(
                n_hidden,
                1,
                rngs=rngs,
            )

        def __call__(self, features: jax.Array) -> jax.Array:
            hidden = nnx.tanh(self.layer1(features))
            hidden = nnx.tanh(self.layer2(hidden))
            raw_epsilon = self.output_layer(hidden)

            return raw_epsilon.squeeze()


    # Initialize the Flax model.
    model = EpsilonNetwork(
        n_features=4,
        n_hidden=32,
        rngs=nnx.Rngs(0),
    )


    # ---------------------------------------------------------------------
    # 3. Differentiable soft-contact observable
    # ---------------------------------------------------------------------

    def soft_sticker_patch_contacts(
        trajectory: jax.Array,
        sticker_indices: jax.Array,
        patch_indices: jax.Array,
        box: float,
        width: float = 0.25,
    ) -> jax.Array:
        """Return a differentiable average sticker-patch contact score."""
        sticker_positions = trajectory[:, sticker_indices, :]
        patch_positions = trajectory[:, patch_indices, :]

        delta = (
            sticker_positions[:, :, None, :]
            - patch_positions[:, None, :, :]
        )

        delta = delta - box * jnp.round(delta / box)

        distance_squared = jnp.sum(delta**2, axis=-1)

        contact_scores = jnp.exp(
            -distance_squared / (2.0 * width**2)
        )

        return jnp.mean(contact_scores)


    # Example input features for one simulation.
    features = jnp.asarray(
        [1.0, 0.5, 0.2, -0.3],
        dtype=jnp.float32,
    )


    # ---------------------------------------------------------------------
    # 4. Neural network -> epsilon -> simulation -> loss
    # ---------------------------------------------------------------------

    def loss_fn(
        model: EpsilonNetwork,
        features: jax.Array,
    ):
        """Run the neural network and differentiable simulation."""
        raw_epsilon = model(features)

        # EPS is represented as a positive binding strength, but the
        # simulation expects signed epsilon:
        # negative epsilon = attraction.
        epsilon = -jax.nn.softplus(raw_epsilon)

        trajectory = simulation({
            "epsilon": epsilon,
        })

        contact_score = soft_sticker_patch_contacts(
            trajectory=trajectory,
            sticker_indices=simulation.sticker_indices,
            patch_indices=simulation.patch_indices,
            box=simulation.box,
        )

        # We want to maximize contacts, so minimize negative contacts.
        loss = -contact_score

        return loss, {
            "epsilon": epsilon,
            "contact_score": contact_score,
        }


    # ---------------------------------------------------------------------
    # 5. Flax/Optax optimizer
    # ---------------------------------------------------------------------

    optimizer = nnx.ModelAndOptimizer(
        model,
        optax.adam(learning_rate=1e-3),
    )


    # ---------------------------------------------------------------------
    # 6. Differentiable training step
    # ---------------------------------------------------------------------

    def train_step(
        model: EpsilonNetwork,
        optimizer: nnx.ModelAndOptimizer,
        features: jax.Array,
    ):
        """Compute gradients through the simulation and update the network."""
        loss_gradient = nnx.grad(
            loss_fn,
            has_aux=True,
        )

        gradients, auxiliary = loss_gradient(
            model,
            features,
        )

        optimizer.update(gradients)

        return auxiliary

    for iteration in range(10):
        auxiliary = train_step(
            model,
            optimizer,
            features,
        )

        if iteration % 1 == 0:
            loss, current_auxiliary = loss_fn(
                model,
                features,
            )

            print(
                f"iteration={iteration + 1} "
                f"loss={float(loss):.6f} "
                f"epsilon={float(current_auxiliary['epsilon']):.6f} "
                f"contacts={float(current_auxiliary['contact_score']):.6f}"
            )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Confocal Images
    We can convert an instance of the JAX MD simulation into a confocal image using the `microsim` package (https://github.com/tlambert03/microsim).
    """)
    return


@app.cell
def _(Path, gsd, np, os, plt):
    from microsim import schema as ms


    # ============================================================
    # USER SETTINGS
    # ============================================================

    regime = "control"
    frame_index = 25

    # Particle type rendered in green:
    # R = Rubisco centers
    # P = Rubisco patches
    # S = EPYC1 spacers
    # T = EPYC1 stickers
    green_particle_type = "T"

    trajectory_dir = os.getenv("traj_jaxmd_gpu")

    gsd_path = Path(trajectory_dir) / f"{regime}.gsd"


    # ============================================================
    # READ ONE GSD FRAME
    # ============================================================

    with gsd.hoomd.open(name=str(gsd_path), mode="r") as gsd_trajectory:
        n_frames = len(gsd_trajectory)

        print(f"GSD path: {gsd_path}")
        print(f"Number of frames: {n_frames}")

        if frame_index < 0 or frame_index >= n_frames:
            raise IndexError(
                f"frame_index={frame_index} is invalid; "
                f"trajectory contains {n_frames} frames"
            )

        frame = gsd_trajectory[frame_index]

    positions = np.asarray(
        frame.particles.position,
        dtype=np.float32,
    )

    type_ids = np.asarray(
        frame.particles.typeid,
    )

    box = float(frame.configuration.box[0])

    print(f"Selected regime: {regime}")
    print(f"Selected frame: {frame_index}")
    print(f"Simulation step: {frame.configuration.step}")
    print(f"Number of particles: {len(positions)}")
    print(f"Box length: {box}")


    # ============================================================
    # CREATE MICROSim TRUTH VOLUMES
    # ============================================================

    # Truth-space dimensions: z, y, x
    truth_shape = (64, 128, 128)
    nz, ny, nx = truth_shape

    # Synthetic physical size of the simulation box in micrometers.
    physical_box_um = 6.4
    voxel_size_um = physical_box_um / box

    print(f"Voxel size: {voxel_size_um:.4f} micrometers")


    # ------------------------------------------------------------
    # Magenta channel: synthetic chloroplast-like volume
    # ------------------------------------------------------------

    z_grid, y_grid, x_grid = np.ogrid[:nz, :ny, :nx]

    center_z = (nz - 1) / 2.0
    center_y = (ny - 1) / 2.0
    center_x = (nx - 1) / 2.0

    radius_z = 0.42 * nz
    radius_y = 0.45 * ny
    radius_x = 0.45 * nx

    ellipsoid_distance = (
        ((z_grid - center_z) / radius_z) ** 2
        + ((y_grid - center_y) / radius_y) ** 2
        + ((x_grid - center_x) / radius_x) ** 2
    )

    chloroplast_mask = ellipsoid_distance <= 1.0

    # This is a visualization proxy for chlorophyll autofluorescence.
    magenta_truth = (
        chloroplast_mask.astype(np.float32) * 3.0
    )


    # ------------------------------------------------------------
    # Green channel: selected particle type
    # ------------------------------------------------------------

    particle_type_ids = {
        "R": 0,
        "P": 1,
        "S": 2,
        "T": 3,
    }

    if green_particle_type not in particle_type_ids:
        raise ValueError(
            f"Unknown green_particle_type={green_particle_type!r}. "
            f"Choose from {tuple(particle_type_ids)}."
        )

    selected_type_id = particle_type_ids[green_particle_type]

    selected_positions = positions[
        type_ids == selected_type_id
    ]

    print(
        f"Green particle type: {green_particle_type}; "
        f"selected particles: {len(selected_positions)}"
    )

    green_truth = np.zeros(
        truth_shape,
        dtype=np.float32,
    )

    green_brightness = 500.0

    for x_position, y_position, z_position in selected_positions:
        x_index = int(
            ((float(x_position) + box / 2.0) / box) * nx
        )
        y_index = int(
            ((float(y_position) + box / 2.0) / box) * ny
        )
        z_index = int(
            ((float(z_position) + box / 2.0) / box) * nz
        )

        if (
            0 <= x_index < nx
            and 0 <= y_index < ny
            and 0 <= z_index < nz
        ):
            green_truth[z_index, y_index, x_index] += green_brightness


    # ============================================================
    # RUN MICROSim CONFOCAL RENDERING
    # ============================================================

    def render_confocal_channel(truth):
        """Render one voxelized fluorophore volume with microsim."""
        microsim_simulation = ms.Simulation.from_ground_truth(
            truth,
            scale=(
                voxel_size_um,
                voxel_size_um,
                voxel_size_um,
            ),
            output_space={
                "downscale": 1,
            },
            modality=ms.Confocal(
                pinhole_au=1.0,
            ),
            settings=ms.Settings(
                random_seed=0,
            ),
        )

        return microsim_simulation.run()


    print("Rendering magenta channel...")
    magenta_image = render_confocal_channel(magenta_truth)

    print("Rendering green channel...")
    green_image = render_confocal_channel(green_truth)

    print("Magenta microsim output:", magenta_image.dims, magenta_image.shape)
    print("Green microsim output:", green_image.dims, green_image.shape)


    # ============================================================
    # EXTRACT ONE 2D CONFOCAL PLANE
    # ============================================================

    def extract_middle_plane(image):
        """Extract one 2D image plane from a microsim output."""
        array = np.asarray(image)
        dimensions = list(image.dims)

        # Remove a named channel dimension.
        if "c" in dimensions:
            channel_axis = dimensions.index("c")
            array = np.take(array, indices=0, axis=channel_axis)
            dimensions.pop(channel_axis)

        # Handle microsim versions with an unnamed channel axis.
        elif array.ndim == 4:
            possible_channel_axes = [
                axis
                for axis, size in enumerate(array.shape)
                if size <= 4
            ]

            if not possible_channel_axes:
                raise ValueError(
                    f"Could not identify channel axis in shape {array.shape}"
                )

            channel_axis = possible_channel_axes[-1]
            array = np.take(array, indices=0, axis=channel_axis)
            dimensions.pop(channel_axis)

        # Select the middle z-plane.
        if "z" in dimensions:
            z_axis = dimensions.index("z")
            array = np.take(
                array,
                indices=array.shape[z_axis] // 2,
                axis=z_axis,
            )
        elif array.ndim == 3:
            # If no named z dimension exists, assume the first axis is z.
            array = array[array.shape[0] // 2]

        if array.ndim != 2:
            raise ValueError(
                f"Expected a 2D plane, but got shape {array.shape}. "
                f"Original dims={image.dims}, original shape={image.shape}"
            )

        return array


    def normalize_for_display(image):
        """Normalize an image using its 99.5th percentile."""
        image = np.asarray(image)
        maximum = np.percentile(image, 99.5)

        if maximum <= 0:
            return np.zeros_like(image, dtype=np.float32)

        return np.clip(image / maximum, 0.0, 1.0)


    magenta_plane = extract_middle_plane(magenta_image)
    green_plane = extract_middle_plane(green_image)

    magenta_display = normalize_for_display(magenta_plane)
    green_display = normalize_for_display(green_plane)


    # ============================================================
    # COMBINE INTO PURPLE/GREEN RGB IMAGE
    # ============================================================

    rgb_image = np.zeros(
        (*magenta_display.shape, 3),
        dtype=np.float32,
    )

    # Magenta = red + blue.
    rgb_image[..., 0] = magenta_display
    rgb_image[..., 2] = magenta_display

    # Green = selected particle signal.
    rgb_image[..., 1] = green_display


    # ============================================================
    # DISPLAY
    # ============================================================

    fig_confocal, axis_confocal = plt.subplots(
        figsize=(8, 8),
    )

    axis_confocal.imshow(rgb_image)
    axis_confocal.set_title(
        f"{regime.title()} — frame {frame_index}\n"
        f"Green: {green_particle_type}"
    )
    axis_confocal.axis("off")

    fig_confocal
    return


if __name__ == "__main__":
    app.run()
