from __future__ import annotations

import numpy as np

N_R = 24
N_C = 48
N_BEADS = 12
STICKER_INDEX = (1, 3, 5, 7, 9, 11)
PATCHES = 8
PATCH_RADIUS = 2.5

CHAIN_K = 10.0
CHAIN_R0 = 1.2
PATCH_TETHER_K = 50.0

PATCH_SIGMA = 1.0
PATCH_RCUT = 3.0
DT = 0.0005

GAMMA_RUBISCO = 5.0
GAMMA_CHAIN = 1.0

TYPE_NAMES = ("R", "P", "S", "T")
TYPE_INDEX = {name: index for index, name in enumerate(TYPE_NAMES)}

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

PATCH_DIRS = np.asarray(
    [
        [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
        [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1],
    ],
    dtype=np.float64,
)
PATCH_DIRS /= np.linalg.norm(PATCH_DIRS, axis=1, keepdims=True)
