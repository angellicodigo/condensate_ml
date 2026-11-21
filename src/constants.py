"""Physical parameters, particle counts, and interaction tables for Rubisco-EPYC1 simulation."""

from __future__ import annotations

import numpy as np

# ============================================================
# PARTICLE AND MOLECULAR COUNTS
# ============================================================
N_R = 24  # Number of Rubisco core complexes
N_C = 48  # Number of EPYC1 polymer chains
N_BEADS = 12  # Number of beads per EPYC1 chain (alternating spacers and stickers)
STICKER_INDEX = (1, 3, 5, 7, 9, 11)  # Local 0-based bead indices within each chain that act as stickers

# ============================================================
# RUBISCO PATCH GEOMETRY
# ============================================================
PATCHES = 8  # Number of surface patches per Rubisco core
PATCH_RADIUS = 2.5  # Radial distance from Rubisco core center to each attached patch

# ============================================================
# HARMONIC BOND PARAMETERS
# ============================================================
CHAIN_K = 10.0  # Spring constant (stiffness) for EPYC1 chain backbone bonds
CHAIN_R0 = 1.2  # Equilibrium rest length for consecutive chain beads
PATCH_TETHER_K = 50.0  # Spring constant for core-to-patch harmonic tethers

# ============================================================
# STICKER-PATCH ATTRACTION PARAMETERS
# ============================================================
PATCH_SIGMA = 1.0  # Gaussian well width for sticker-patch attraction
PATCH_RCUT = 3.0  # Interaction cutoff distance for sticker-patch attraction

# ============================================================
# BROWNIAN DYNAMICS INTEGRATION PARAMETERS
# ============================================================
DT = 0.0005  # Brownian dynamics integration timestep
KBT = 1.0  # Thermal energy in reduced units (k_B * T)

# ============================================================
# HYDRODYNAMIC FRICTION COEFFICIENTS (GAMMA)
# ============================================================
GAMMA_RUBISCO = 5.0  # Friction coefficient for Rubisco core centers and surface patches
GAMMA_CHAIN = 1.0  # Friction coefficient for EPYC1 chain beads (spacers and stickers)

# ============================================================
# PARTICLE TYPE DEFINITIONS
# ============================================================
# R: Rubisco core center
# P: Rubisco surface patch
# S: EPYC1 spacer bead
# T: EPYC1 sticker bead
TYPE_NAMES = ("R", "P", "S", "T")
TYPE_INDEX = {name: index for index, name in enumerate(TYPE_NAMES)}

# ============================================================
# STERIC AND NONBONDED PAIR PARAMETERS: (epsilon, sigma, cutoff)
# ============================================================
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
    ("P", "T"): (0.0, PATCH_SIGMA, PATCH_RCUT),  # Steric repulsion is zero; cutoff matches attraction
}

# ============================================================
# PATCH DIRECTION VECTORS (CUBIC SYMMETRY)
# ============================================================
# 8 unit vectors defining symmetric patch directions on each Rubisco core
PATCH_DIRS = np.asarray(
    [
        [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
        [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1],
    ],
    dtype=np.float64,
)
PATCH_DIRS /= np.linalg.norm(PATCH_DIRS, axis=1, keepdims=True)
