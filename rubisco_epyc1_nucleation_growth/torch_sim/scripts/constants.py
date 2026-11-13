"""Model constants for the Rubisco-EPYC1 patchy Brownian Dynamics model.

Everything here mirrors simulate_hoomd_patchy_full.py. Importing this module has no
side effects (no argv parsing, no device selection, no file I/O).
"""

from __future__ import annotations

import numpy as np

# --- composition -----------------------------------------------------------------------
N_R = 24                      # Rubisco scaffolds
N_C = 48                      # EPYC1 chains (2:1 EPYC1:Rubisco)
N_BEADS = 12                  # beads per EPYC1 chain
STICKER_INDEX = [1, 3, 5, 7, 9, 11]   # local bead indices that are stickers (6 per chain)
PATCHES = 8                   # patches per Rubisco (1.5:1 sticker:patch sites overall)
PATCH_RADIUS = 2.5            # core-to-patch distance

N_PARTICLES = N_R + N_R * PATCHES + N_C * N_BEADS   # 792
N_GRAPH_NODES = N_R + N_C                             # 72 macromolecules (cluster-graph nodes)

# --- bonded interactions ---------------------------------------------------------------
CHAIN_K = 10.0
CHAIN_R0 = 1.2
PATCH_TETHER_K = 50.0

# --- sticker-patch attraction (Gaussian, force-based; no discrete KOFF) ------------------
PATCH_SIGMA = 1.0
PATCH_RCUT = 3.0

# --- analysis --------------------------------------------------------------------------
CONTACT_CUTOFF = 3.0

# --- dynamics --------------------------------------------------------------------------
GAMMA_RUBISCO = 5.0           # friction for R and P particles
GAMMA_CHAIN = 1.0             # friction for S and T particles
DT = 0.0005                   # shared by every run
CHECKPOINT_EVERY = 32

# --- particle types (same ordering as the HOOMD particle_types list) ---------------------
TYPE_NAMES = ["R", "P", "S", "T"]
TYPE_INDEX = {name: i for i, name in enumerate(TYPE_NAMES)}

# Bounded Gaussian steric parameters: (epsilon, sigma, r_cut). Unlisted pairs do not
# interact (epsilon=0, r_cut=0), exactly as in the HOOMD script. Pairs are symmetric.
STERIC_PAIR_PARAMS: dict[tuple[str, str], tuple[float, float, float]] = {
    ("R", "R"): (2.0, 1.5, 3.0),
    ("S", "S"): (1.0, 1.0, 2.5),
    ("T", "T"): (1.0, 1.0, 2.5),
    ("S", "T"): (1.0, 1.0, 2.5),
    ("R", "S"): (1.5, 1.5, 3.0),
    ("R", "T"): (1.5, 1.5, 3.0),
    ("P", "P"): (0.2, 0.8, 2.0),
    ("P", "S"): (0.2, 0.8, 2.0),
    ("P", "R"): (0.2, 0.8, 2.0),
}

# --- patch geometry: 8 cube corners, normalised -----------------------------------------
PATCH_DIRS = np.array(
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
PATCH_DIRS /= np.linalg.norm(PATCH_DIRS, axis=1, keepdims=True)
