import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gsd.hoomd
import hoomd
import hoomd.md as md
from pathlib import Path
import os
from dotenv import load_dotenv
load_dotenv()


# ============================================================
# INPUT PARAMS
# ============================================================
params = json.load(open(sys.argv[1]))

NAME = params["name"]
BOX = float(params["BOX"])
KON = float(params.get("KON", params.get("EPS")))
KOFF = float(params.get("KOFF", 0.0))

SEED = int(params.get("SEED", 42))
N_STEPS = int(params.get("N_STEPS", 10000))
WRITE_EVERY = int(params.get("WRITE_EVERY", 100))
DT = float(params.get("DT", 0.0005))

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
CHAIN_K = 10.0 
CHAIN_R0 = 1.2 

PATCH_TETHER_K = 50.0

PATCH_EPSILON = -abs(KON)
PATCH_SIGMA = 1.0
PATCH_RCUT = 3.0

CONTACT_CUTOFF = 3.0

# ============================================================
# OUTPUT FOLDERS
# ============================================================
output_base = Path(os.environ.get("OUTPUT_DIR"))
traj_dir = output_base / "traj_patchy_gpu"
xyz_dir = output_base / "xyz_patchy_gpu"
data_dir = output_base / "data_patchy_gpu"
plots_dir = output_base / "plots_patchy_gpu"

for folder in (traj_dir, xyz_dir, data_dir, plots_dir):
    folder.mkdir(parents=True, exist_ok=True)

GSD_PATH = str(traj_dir / f"{NAME}.gsd")
XYZ_PATH = str(xyz_dir / f"{NAME}.xyz")
CSV_PATH = str(data_dir / f"{NAME}.csv")

# ============================================================
# GPU DEVICE
# ============================================================
device = hoomd.device.GPU()
sim = hoomd.Simulation(device=device, seed=SEED)
'''
R: Rubisco core center.
P: Rubisco surface patch.
S: EPYC1 spacer bead.
T: EPYC1 sticker bead.
'''
particle_types = ["R", "P", "S", "T"]

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


def wrap_centered(x, box):
    return ((x + box / 2) % box) - box / 2


def make_grid_positions(n, box, margin=6.0):
    n_side = int(np.ceil(n ** (1 / 3)))
    vals = np.linspace(-box / 2 + margin, box / 2 - margin, n_side)
    pts = np.array(np.meshgrid(vals, vals, vals)).T.reshape(-1, 3)
    return pts[:n].copy()


# ============================================================
# BUILD INITIAL SNAPSHOT
# ============================================================
snap = hoomd.Snapshot()

if sim.device.communicator.rank == 0:
    rng = np.random.default_rng(SEED)

    n_patch = N_R * PATCHES
    n_chain = N_C * N_BEADS
    n_total = N_R + n_patch + n_chain

    snap.configuration.box = [BOX, BOX, BOX, 0, 0, 0]
    snap.particles.N = n_total
    snap.particles.types = particle_types

    pos = np.zeros((n_total, 3), dtype=float)
    typeid = np.zeros(n_total, dtype=np.uint32)

    # -----------------------------
    # Rubisco centers
    # -----------------------------
    rubisco_centers = make_grid_positions(N_R, BOX, margin=7.0)
    rubisco_centers += rng.normal(scale=0.20, size=rubisco_centers.shape)
    rubisco_centers = wrap_centered(rubisco_centers, BOX)

    idx = 0
    for r in range(N_R):
        pos[idx] = rubisco_centers[r]
        typeid[idx] = 0
        idx += 1

    # -----------------------------
    # Rubisco patch particles
    # IMPORTANT: do NOT wrap patch particles independently.
    # They must remain local to their Rubisco center to avoid
    # huge patch-tether bonds across the periodic boundary.
    # -----------------------------
    patch_start = idx
    for r in range(N_R):
        center = rubisco_centers[r]
        for p in range(PATCHES):
            ppos = center + PATCH_RADIUS * patch_dirs[p]
            pos[idx] = ppos
            typeid[idx] = 1
            idx += 1

    # -----------------------------
    # EPYC1 chains
    # -----------------------------
    chain_start = N_R + n_patch

    chain_centers = make_grid_positions(N_C, BOX, margin=5.0)
    chain_centers += rng.normal(scale=0.25, size=chain_centers.shape)
    chain_centers = wrap_centered(chain_centers, BOX)

    for c in range(N_C):
        center = chain_centers[c]
        direction = rng.normal(size=3)
        direction /= np.linalg.norm(direction)

        for b in range(N_BEADS):
            bead_pos = center + (b - (N_BEADS - 1) / 2) * CHAIN_R0 * direction
            bead_pos = wrap_centered(bead_pos, BOX)

            pos[idx] = bead_pos
            typeid[idx] = 3 if b in STICKER_INDEX else 2
            idx += 1

    snap.particles.position[:] = pos
    snap.particles.typeid[:] = typeid

    # -----------------------------
    # EPYC1 chain bonds
    # -----------------------------
    chain_bonds = []
    for c in range(N_C):
        base = chain_start + c * N_BEADS
        for b in range(N_BEADS - 1):
            chain_bonds.append([base + b, base + b + 1])

    # -----------------------------
    # Patch tether bonds
    # These keep patch particles attached to Rubisco centers.
    # -----------------------------
    patch_tethers = []
    for r in range(N_R):
        for p in range(PATCHES):
            patch_tethers.append([r, patch_start + r * PATCHES + p])

    all_bonds = chain_bonds + patch_tethers

    snap.bonds.N = len(all_bonds)
    snap.bonds.types = ["chain", "patch_tether"]

    bond_typeid = np.zeros(len(all_bonds), dtype=np.uint32)
    bond_typeid[len(chain_bonds):] = 1

    snap.bonds.typeid[:] = bond_typeid
    snap.bonds.group[:] = np.array(all_bonds, dtype=np.uint32)

sim.create_state_from_snapshot(snap)

# ============================================================
# NEIGHBOR LIST
# ============================================================
nl = md.nlist.Cell(buffer=0.4)

# ============================================================
# BOUNDED SOFT REPULSION
# Gaussian repulsion is bounded and avoids LJ blow-ups.
# ============================================================
soft_rep = md.pair.Gaussian(nlist=nl)

for a in particle_types:
    for b in particle_types:
        soft_rep.params[(a, b)] = dict(epsilon=0.0, sigma=1.0)
        soft_rep.r_cut[(a, b)] = 0.0

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

# ============================================================
# STICKER-PATCH ATTRACTION
# Force-based interaction. No discrete binding flags.
# Negative Gaussian epsilon gives attraction.
# ============================================================
patch_att = md.pair.Gaussian(nlist=nl)

for a in particle_types:
    for b in particle_types:
        patch_att.params[(a, b)] = dict(epsilon=0.0, sigma=1.0)
        patch_att.r_cut[(a, b)] = 0.0

patch_att.params[("T", "P")] = dict(epsilon=PATCH_EPSILON, sigma=PATCH_SIGMA)
patch_att.r_cut[("T", "P")] = PATCH_RCUT

# ============================================================
# BOND FORCES
# ============================================================
harmonic = md.bond.Harmonic()
harmonic.params["chain"] = dict(k=CHAIN_K, r0=CHAIN_R0)
harmonic.params["patch_tether"] = dict(k=PATCH_TETHER_K, r0=PATCH_RADIUS)

# ============================================================
# BROWNIAN DYNAMICS
# ============================================================
integrator = md.Integrator(dt=DT)

all_particles = hoomd.filter.All()
brownian = md.methods.Brownian(filter=all_particles, kT=1.0)

brownian.gamma["R"] = 5.0
brownian.gamma["P"] = 5.0
brownian.gamma["S"] = 1.0
brownian.gamma["T"] = 1.0

integrator.methods.append(brownian)
integrator.forces.append(soft_rep)
integrator.forces.append(patch_att)
integrator.forces.append(harmonic)

sim.operations.integrator = integrator

# ============================================================
# GSD OUTPUT
# Robust version: write frame 0 manually, then append frames
# ============================================================
hoomd.write.GSD.write(
    state=sim.state,
    filename=GSD_PATH,
    mode="wb",
)

writer = hoomd.write.GSD(
    filename=GSD_PATH,
    trigger=hoomd.trigger.Periodic(WRITE_EVERY),
    mode="ab",
)
sim.operations.writers.append(writer)

# ============================================================
# RUN
# ============================================================
print(f"Running {NAME}: BOX={BOX}, epsilon={PATCH_EPSILON}, KOFF(recorded)={KOFF}, DT={DT}")
sim.run(N_STEPS)

# Make sure all buffered trajectory data are written before analysis
try:
    writer.flush()
except Exception:
    pass

print(f"Finished simulation: {GSD_PATH}")

if (not os.path.exists(GSD_PATH)) or os.path.getsize(GSD_PATH) == 0:
    raise RuntimeError(f"GSD output is empty: {GSD_PATH}")


# ============================================================
# ANALYSIS HELPERS
# ============================================================
def minimum_image(delta, box):
    return delta - box * np.round(delta / box)


def union_find(n, edges):
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for a, b in edges:
        union(a, b)

    groups = {}
    for i in range(n):
        root = find(i)
        groups.setdefault(root, []).append(i)

    sizes = sorted([len(v) for v in groups.values()], reverse=True)
    largest = sizes[0] if sizes else 0
    fraction = largest / n
    return len(sizes), largest, fraction


# ============================================================
# POSTPROCESS: GSD -> XYZ, CSV, PLOTS
# Contacts are defined as T-P pairs closer than CONTACT_CUTOFF.
# Graph nodes are Rubisco centers and EPYC1 chains.
# ============================================================
traj = gsd.hoomd.open(GSD_PATH, mode="r")

records = []

with open(XYZ_PATH, "w") as xyz:
    for frame_i, frame in enumerate(traj):
        step = frame.configuration.step
        box = frame.configuration.box[0]
        positions = frame.particles.position
        typeids = frame.particles.typeid

        labels = [particle_types[t] for t in typeids]

        xyz.write(f"{len(positions)}\n")
        xyz.write(f"step {step}\n")
        for lab, p in zip(labels, positions):
            xyz.write(f"{lab} {p[0]} {p[1]} {p[2]}\n")

        patch_start = N_R
        chain_start = N_R + N_R * PATCHES

        patch_indices = np.arange(patch_start, chain_start)

        sticker_indices = []
        sticker_chain_ids = []

        for c in range(N_C):
            base = chain_start + c * N_BEADS
            for b in STICKER_INDEX:
                sticker_indices.append(base + b)
                sticker_chain_ids.append(c)

        sticker_indices = np.array(sticker_indices, dtype=int)
        sticker_chain_ids = np.array(sticker_chain_ids, dtype=int)

        patch_pos = positions[patch_indices]
        sticker_pos = positions[sticker_indices]

        delta = sticker_pos[:, None, :] - patch_pos[None, :, :]
        delta = minimum_image(delta, box)
        dist = np.linalg.norm(delta, axis=-1)

        contacts = np.argwhere(dist < CONTACT_CUTOFF)
        n_contacts = len(contacts)

        graph_edges = []
        for s_local, p_local in contacts:
            c = int(sticker_chain_ids[s_local])
            r = int(p_local // PATCHES)
            graph_edges.append((N_R + c, r))

        n_clusters, largest, fraction = union_find(N_R + N_C, graph_edges)

        records.append(
            {
                "frame": frame_i,
                "step": step,
                "time": step * DT,
                "box": BOX,
                "kon_or_epsilon": KON,
                "epsilon_used": PATCH_EPSILON,
                "koff_recorded": KOFF,
                "contacts": n_contacts,
                "bonds": n_contacts,
                "clusters": n_clusters,
                "largest": largest,
                "fraction": fraction,
            }
        )

df = pd.DataFrame(records)
df.to_csv(CSV_PATH, index=False)

if len(df) > 0:
    for key, ylabel in [
        ("bonds", "contacts / bonds"),
        ("clusters", "number of clusters"),
        ("largest", "largest cluster size"),
        ("fraction", "largest cluster fraction"),
    ]:
        plt.figure()
        plt.plot(df["time"], df[key])
        plt.xlabel("time")
        plt.ylabel(ylabel)
        plt.title(f"{NAME}: {ylabel} vs time")
        plt.tight_layout()
        plt.savefig(plots_dir / f"{NAME}_{key}.jpeg", dpi=180)
        plt.close()
else:
    print("WARNING: No trajectory frames found — skipping plots")

print(f"Wrote GSD: {GSD_PATH}")
print(f"Wrote XYZ: {XYZ_PATH}")
print(f"Wrote CSV: {CSV_PATH}")
print("Wrote plots in plots_patchy_gpu/")
