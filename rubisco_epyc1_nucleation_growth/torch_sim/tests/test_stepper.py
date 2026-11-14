"""Initial-snapshot parity test: original HOOMD simulation vs RubiscoEpyc1TimeStepper.

Question this file answers
--------------------------
If HOOMD and the torch stepper are given the same conditions (BOX, EPS) and the same SEED,
do they start from the same configuration? Only if they do can later comparisons (forces,
dynamics, statistics) mean anything.

How it works (read top to bottom, there are three sections)
-----------------------------------------------------------
1. HOOMD side. The snapshot-building code of ``molecular_sim/simulate_hoomd_patchy_full.py``
   is executed as written (it is cut out of that file, not copied here) on a CPU device. The
   state is created in HOOMD and the snapshot is read back from it. No GPU, no output
   folders, no simulation steps.
2. Torch side. ``RubiscoEpyc1TimeStepper(box=BOX, seed=SEED).init({})`` gives the starting
   positions; the particle types and bonds come from the stepper's own tables.
3. Comparison. Both sides are put into the same small ``Snapshot`` container and compared:
   box, particle count and type names, type of every particle, bonds, position of every
   particle, the full particle-to-particle distance matrix, and geometric facts that must
   hold for any correct layout (patch tethers 2.5 long, chain bonds 1.2 long, ...).

Conditions come from ``molecular_sim/params.json`` (the five real runs).

Run (from anywhere; ``-s`` prints the per-check report)::

    pytest rubisco_epyc1_nucleation_growth/torch_sim/tests/test_stepper.py -v -s

Needs torch, nefi, hoomd, numpy and pytest. The optional GSD test also needs gsd and the
OUTPUT_DIR environment variable used by the HOOMD script. Tests that need a missing package
are skipped, and the pure-NumPy ``test_helper_*`` tests always run.

Note: ``condensate_ml`` is imported normally if installed. If it is not installed the
``scripts/`` folder is loaded under that name, so ``pip install -e .`` is not required here.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

# ----------------------------------------------------------------------------------------
# Paths and tolerances
# ----------------------------------------------------------------------------------------
TORCH_SIM_DIR = Path(__file__).resolve().parent.parent          # .../torch_sim
PROJECT_DIR = TORCH_SIM_DIR.parent                                # .../rubisco_epyc1_nucleation_growth
SCRIPTS_DIR = TORCH_SIM_DIR / "scripts"                           # importable as condensate_ml
HOOMD_SCRIPT = PROJECT_DIR / "molecular_sim" / "simulate_hoomd_patchy_full.py"
HOOMD_PARAMS = PROJECT_DIR / "molecular_sim" / "params.json"

# Positions are stored as float32 by HOOMD and (by default) by the stepper, so identical
# layouts agree to about 1e-6 for coordinates of size ~20. These tolerances leave margin.
POSITION_TOL = 1e-5       # per-particle position error (minimum-image, length units)
DISTANCE_TOL = 1e-4       # any pairwise distance
GEOMETRY_TOL = 1e-4       # bond length vs its rest length

# Model facts (from simulate_hoomd_patchy_full.py) used for absolute sanity checks.
EXPECTED_TYPE_COUNTS = {"R": 24, "P": 192, "S": 288, "T": 288}
EXPECTED_BOND_COUNTS = {"chain": 528, "patch_tether": 192}
EXPECTED_BOND_LENGTH = {"chain": 1.2, "patch_tether": 2.5}   # CHAIN_R0, PATCH_RADIUS


@dataclass
class Snapshot:
    """A starting configuration in a form that does not depend on where it came from."""

    source: str                  # "HOOMD" or "torch" (used in messages)
    box_full: np.ndarray         # (6,) = [Lx, Ly, Lz, xy, xz, yz]
    type_names: list[str]        # e.g. ["R", "P", "S", "T"]; typeid indexes into this
    typeid: np.ndarray           # (N,) integer type of each particle
    positions: np.ndarray        # (N, 3) float64
    bonds: np.ndarray            # (M, 2) integer particle numbers joined by a bond
    bond_typeid: np.ndarray      # (M,) integer type of each bond
    bond_type_names: list[str]   # e.g. ["chain", "patch_tether"]

    @property
    def box(self) -> float:
        return float(self.box_full[0])

    def type_of(self, particle: int) -> str:
        return self.type_names[int(self.typeid[particle])]


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str


# ========================================================================================
# 1. HOOMD SIDE
# ========================================================================================
def _cut(text: str, start_marker: str, end_marker: str) -> str:
    """Return the part of ``text`` from start_marker up to (not including) end_marker."""
    start = text.find(start_marker)
    end = text.find(end_marker, start + 1) if start >= 0 else -1
    if start < 0 or end < 0:
        raise RuntimeError(
            f"Could not find {start_marker!r} ... {end_marker!r} in {HOOMD_SCRIPT.name}. "
            "The HOOMD script was edited; update the markers in _cut() calls in this test."
        )
    return text[start:end]


def build_hoomd_snapshot(box: float, seed: int, kon: float) -> Snapshot:
    """Run the HOOMD script's own snapshot code on a CPU device and read the state back.

    Two pieces of the script are executed unchanged:
      * the MODEL PARAMETERS block (N_R, N_C, PATCHES, PATCH_RADIUS, CHAIN_R0, ...)
      * everything from ``particle_types = [...]`` through the end of the snapshot builder
    The script lines that need a GPU, environment variables or output folders are skipped.
    """
    return _build_hoomd_snapshot_cached(float(box), int(seed), float(kon))


@lru_cache(maxsize=None)
def _build_hoomd_snapshot_cached(box: float, seed: int, kon: float) -> Snapshot:
    hoomd = pytest.importorskip("hoomd")
    source = HOOMD_SCRIPT.read_text()

    model_parameters = _cut(source, "# MODEL PARAMETERS", "# OUTPUT FOLDERS")
    snapshot_builder = _cut(
        source, 'particle_types = ["R", "P", "S", "T"]', "sim.create_state_from_snapshot(snap)"
    )

    namespace = {"np": np, "hoomd": hoomd, "KON": kon}      # KON is read by MODEL PARAMETERS
    exec(compile(model_parameters, f"{HOOMD_SCRIPT.name} [model parameters]", "exec"), namespace)

    sim = hoomd.Simulation(device=hoomd.device.CPU(), seed=seed)
    namespace.update(BOX=box, SEED=seed, sim=sim)           # normally read from params.json
    exec(compile(snapshot_builder, f"{HOOMD_SCRIPT.name} [snapshot builder]", "exec"), namespace)

    sim.create_state_from_snapshot(namespace["snap"])       # the line the script runs next
    snap = sim.state.get_snapshot()                         # what HOOMD actually holds

    return Snapshot(
        source="HOOMD",
        box_full=np.array(snap.configuration.box, dtype=np.float64),
        type_names=list(snap.particles.types),
        typeid=np.array(snap.particles.typeid, dtype=np.int64),
        positions=np.array(snap.particles.position, dtype=np.float64),
        bonds=np.array(snap.bonds.group, dtype=np.int64),
        bond_typeid=np.array(snap.bonds.typeid, dtype=np.int64),
        bond_type_names=list(snap.bonds.types),
    )


def load_gsd_frame0(path: Path) -> Snapshot:
    """Frame 0 of a GSD file written by a real HOOMD run (the script writes it before running)."""
    gsd_hoomd = pytest.importorskip("gsd.hoomd")
    with gsd_hoomd.open(str(path), mode="r") as trajectory:
        frame = trajectory[0]
        return Snapshot(
            source=f"HOOMD GSD ({path.name})",
            box_full=np.array(frame.configuration.box, dtype=np.float64),
            type_names=list(frame.particles.types),
            typeid=np.array(frame.particles.typeid, dtype=np.int64),
            positions=np.array(frame.particles.position, dtype=np.float64),
            bonds=np.array(frame.bonds.group, dtype=np.int64),
            bond_typeid=np.array(frame.bonds.typeid, dtype=np.int64),
            bond_type_names=list(frame.bonds.types),
        )


# ========================================================================================
# 2. TORCH SIDE
# ========================================================================================
def import_condensate_ml():
    """Import condensate_ml; if it is not pip-installed, load scripts/ under that name."""
    pytest.importorskip("torch")
    pytest.importorskip("nefi")
    try:
        import condensate_ml
        return condensate_ml
    except ModuleNotFoundError as err:
        if err.name != "condensate_ml":
            raise                      # a real missing dependency inside the package
    spec = importlib.util.spec_from_file_location(
        "condensate_ml", SCRIPTS_DIR / "__init__.py", submodule_search_locations=[str(SCRIPTS_DIR)]
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["condensate_ml"] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop("condensate_ml", None)
        raise
    return module


def build_torch_snapshot(box: float, seed: int) -> Snapshot:
    """Starting configuration of RubiscoEpyc1TimeStepper(box=box, seed=seed)."""
    import torch

    condensate_ml = import_condensate_ml()
    from condensate_ml.constants import TYPE_NAMES

    stepper = condensate_ml.RubiscoEpyc1TimeStepper(
        n_steps=1, box=box, write_every=1, seed=seed, device=torch.device("cpu")
    )
    positions = stepper.init({}).detach().cpu().numpy().astype(np.float64)   # (792, 3)

    chain = stepper.bond_pairs_by_type["chain"].cpu().numpy()
    tether = stepper.bond_pairs_by_type["patch_tether"].cpu().numpy()
    return Snapshot(
        source="torch",
        box_full=np.array([box, box, box, 0.0, 0.0, 0.0]),
        type_names=list(TYPE_NAMES),
        typeid=stepper.particle_type_ids.cpu().numpy().astype(np.int64),
        positions=positions,
        bonds=np.concatenate([chain, tether]).astype(np.int64),
        bond_typeid=np.concatenate(
            [np.zeros(len(chain), dtype=np.int64), np.ones(len(tether), dtype=np.int64)]
        ),
        bond_type_names=["chain", "patch_tether"],     # same names and order as the HOOMD script
    )


# ========================================================================================
# 3. COMPARISON HELPERS (pure NumPy; each check returns one CheckResult)
# ========================================================================================
def min_image(delta: np.ndarray, box: float) -> np.ndarray:
    """Shortest periodic version of displacement vectors (works for any array shape)."""
    return delta - box * np.round(delta / box)


def pairwise_distances(positions: np.ndarray, box: float) -> np.ndarray:
    """(N, N) minimum-image distance between every pair of particles."""
    delta = min_image(positions[:, None, :] - positions[None, :, :], box)
    return np.sqrt((delta**2).sum(axis=-1))


def _first(items, limit=5) -> str:
    items = list(items)
    text = ", ".join(str(x) for x in items[:limit])
    return text + (f", ... ({len(items)} total)" if len(items) > limit else "")


def check_box(a: Snapshot, b: Snapshot) -> CheckResult:
    same = a.box_full.shape == b.box_full.shape and np.allclose(a.box_full, b.box_full, atol=1e-6)
    cubic = np.allclose(a.box_full[:3], a.box_full[0]) and np.allclose(a.box_full[3:], 0.0)
    detail = f"{a.source} box {np.round(a.box_full, 4).tolist()}  vs  {b.source} box {np.round(b.box_full, 4).tolist()}"
    if not cubic:
        detail += "  (HOOMD box is not a cube with zero tilt; min-image checks assume a cube)"
    return CheckResult("box", bool(same and cubic), detail)


def check_counts_and_names(a: Snapshot, b: Snapshot) -> CheckResult:
    problems = []
    if len(a.typeid) != len(b.typeid):
        problems.append(f"particle count {len(a.typeid)} vs {len(b.typeid)}")
    if a.type_names != b.type_names:
        problems.append(f"particle type names {a.type_names} vs {b.type_names}")
    if a.bond_type_names != b.bond_type_names:
        problems.append(f"bond type names {a.bond_type_names} vs {b.bond_type_names}")
    detail = "; ".join(problems) or f"{len(a.typeid)} particles, types {a.type_names}, bond types {a.bond_type_names}"
    return CheckResult("particle count and type names", not problems, detail)


def check_particle_types(a: Snapshot, b: Snapshot) -> CheckResult:
    if a.typeid.shape != b.typeid.shape:
        return CheckResult("type of every particle", False, "different particle counts")
    wrong = np.flatnonzero(a.typeid != b.typeid)
    if len(wrong) == 0:
        return CheckResult("type of every particle", True, f"all {len(a.typeid)} particle types identical")
    shown = [f"#{i}: {a.source} {a.type_of(i)} vs {b.source} {b.type_of(i)}" for i in wrong[:5]]
    return CheckResult("type of every particle", False, f"{len(wrong)} differ; first: " + _first(shown))


def check_bonds(a: Snapshot, b: Snapshot) -> CheckResult:
    """Same set of (particle, particle, bond type) triples, ignoring row order and direction."""
    def as_set(s: Snapshot):
        ends = np.sort(s.bonds, axis=1)
        return {(int(i), int(j), s.bond_type_names[int(t)]) for (i, j), t in zip(ends, s.bond_typeid)}

    set_a, set_b = as_set(a), as_set(b)
    only_a, only_b = sorted(set_a - set_b), sorted(set_b - set_a)
    counts = {name: sum(1 for x in set_a if x[2] == name) for name in a.bond_type_names}
    same_order = a.bonds.shape == b.bonds.shape and np.array_equal(a.bonds, b.bonds)
    if not only_a and not only_b:
        return CheckResult("bonds", True, f"identical bond sets {counts}; same row order: {same_order}")
    detail = (f"only in {a.source}: {len(only_a)} ({_first(only_a, 3)}); "
              f"only in {b.source}: {len(only_b)} ({_first(only_b, 3)})")
    return CheckResult("bonds", False, detail)


def check_positions(a: Snapshot, b: Snapshot, tol: float = POSITION_TOL) -> CheckResult:
    """Position of every particle, compared as the shortest periodic displacement."""
    if a.positions.shape != b.positions.shape:
        return CheckResult("position of every particle", False,
                           f"shapes {a.positions.shape} vs {b.positions.shape}")
    error = np.linalg.norm(min_image(a.positions - b.positions, a.box), axis=1)
    worst = int(error.argmax())
    per_type = ", ".join(
        f"{name} {error[a.typeid == i].max():.1e}" for i, name in enumerate(a.type_names) if (a.typeid == i).any()
    )
    detail = (f"max error {error.max():.2e} (particle #{worst}, type {a.type_of(worst)}), "
              f"mean {error.mean():.2e}, tolerance {tol:.0e}; max by type: {per_type}")
    return CheckResult("position of every particle", bool(error.max() <= tol), detail)


def check_distance_matrix(a: Snapshot, b: Snapshot, tol: float = DISTANCE_TOL) -> CheckResult:
    """All N*N particle-to-particle distances. Unlike positions, this ignores a global shift."""
    if a.positions.shape != b.positions.shape:
        return CheckResult("distances between all particles", False, "different particle counts")
    diff = np.abs(pairwise_distances(a.positions, a.box) - pairwise_distances(b.positions, b.box))
    i, j = np.unravel_index(diff.argmax(), diff.shape)
    detail = (f"max difference {diff.max():.2e} (pair #{i}-#{j}), mean {diff.mean():.2e}, "
              f"tolerance {tol:.0e}")
    return CheckResult("distances between all particles", bool(diff.max() <= tol), detail)


def check_composition(s: Snapshot) -> CheckResult:
    """Absolute sanity: the expected number of particles of each type and bonds of each type."""
    problems, parts = [], []
    for i, name in enumerate(s.type_names):
        n = int((s.typeid == i).sum())
        parts.append(f"{name}={n}")
        if n != EXPECTED_TYPE_COUNTS.get(name, n):
            problems.append(f"{name}: {n} particles, expected {EXPECTED_TYPE_COUNTS[name]}")
    for i, name in enumerate(s.bond_type_names):
        n = int((s.bond_typeid == i).sum())
        parts.append(f"{name} bonds={n}")
        if n != EXPECTED_BOND_COUNTS.get(name, n):
            problems.append(f"{name}: {n} bonds, expected {EXPECTED_BOND_COUNTS[name]}")
    return CheckResult(f"composition [{s.source}]", not problems, "; ".join(problems) or ", ".join(parts))


def check_geometry(s: Snapshot) -> CheckResult:
    """Absolute sanity: every bond starts at its rest length and nothing is NaN."""
    problems, parts = [], []
    if not np.isfinite(s.positions).all():
        problems.append("non-finite positions")
    for t, name in enumerate(s.bond_type_names):
        expected = EXPECTED_BOND_LENGTH.get(name)
        pairs = s.bonds[s.bond_typeid == t]
        if expected is None or len(pairs) == 0:
            continue
        length = np.linalg.norm(min_image(s.positions[pairs[:, 0]] - s.positions[pairs[:, 1]], s.box), axis=1)
        parts.append(f"{name} length {length.min():.4f}..{length.max():.4f} (rest {expected})")
        worst = np.abs(length - expected).max()
        if worst > GEOMETRY_TOL:
            problems.append(f"{name}: up to {worst:.2e} away from rest length {expected}")
    return CheckResult(f"bond lengths [{s.source}]", not problems, "; ".join(problems) or "; ".join(parts))


def compare_snapshots(hoomd_snap: Snapshot, torch_snap: Snapshot) -> list[CheckResult]:
    """Run every check, in the order a person would debug them."""
    return [
        check_box(hoomd_snap, torch_snap),
        check_counts_and_names(hoomd_snap, torch_snap),
        check_particle_types(hoomd_snap, torch_snap),
        check_bonds(hoomd_snap, torch_snap),
        check_positions(hoomd_snap, torch_snap),
        check_distance_matrix(hoomd_snap, torch_snap),
        check_composition(hoomd_snap),
        check_composition(torch_snap),
        check_geometry(hoomd_snap),
        check_geometry(torch_snap),
    ]


def format_report(title: str, results: list[CheckResult]) -> str:
    width = max(len(r.name) for r in results)
    lines = [title, "-" * len(title)]
    lines += [f"{'PASS' if r.passed else 'FAIL'}  {r.name:<{width}}  {r.detail}" for r in results]
    return "\n".join(lines)


# ========================================================================================
# TESTS
# ========================================================================================
# Conditions to compare: the five real runs. KON falls back to EPS exactly as in the HOOMD script.
CONDITIONS = json.loads(HOOMD_PARAMS.read_text())
CONDITION_IDS = [c["name"] for c in CONDITIONS]


def _condition_args(cond: dict) -> tuple[float, int, float]:
    return float(cond["BOX"]), int(cond.get("SEED", 42)), float(cond.get("KON", cond.get("EPS")))


@pytest.mark.parametrize("cond", CONDITIONS, ids=CONDITION_IDS)
def test_stepper(cond):
    """HOOMD initial snapshot == RubiscoEpyc1TimeStepper initial snapshot (same BOX and SEED)."""
    box, seed, kon = _condition_args(cond)
    hoomd_snap = build_hoomd_snapshot(box, seed, kon)
    torch_snap = build_torch_snapshot(box, seed)

    results = compare_snapshots(hoomd_snap, torch_snap)
    report = format_report(f"{cond['name']}: BOX={box}, SEED={seed}", results)
    print("\n" + report)
    assert all(r.passed for r in results), "\n" + report


def test_stepper_different_seed_is_detected():
    """Control: a different torch seed must make the position checks FAIL.

    Without this, a comparison that always passed (for example because both sides were
    accidentally the same object) would look identical to a correct one.
    """
    box, seed, kon = _condition_args(CONDITIONS[0])
    hoomd_snap = build_hoomd_snapshot(box, seed, kon)
    other_seed = build_torch_snapshot(box, seed + 1)

    positions = check_positions(hoomd_snap, other_seed)
    distances = check_distance_matrix(hoomd_snap, other_seed)
    print("\n" + format_report("control: torch seed + 1", [positions, distances]))
    assert not positions.passed, "positions matched even though the seed differs"
    assert not distances.passed, "distances matched even though the seed differs"


@pytest.mark.parametrize("cond", CONDITIONS, ids=CONDITION_IDS)
def test_stepper_matches_existing_hoomd_gsd(cond):
    """Same comparison against frame 0 of a GSD written by a real production HOOMD run.

    Needs the OUTPUT_DIR environment variable used by the HOOMD script; looks for
    $OUTPUT_DIR/traj_patchy_gpu/<name>.gsd and skips if it is not there.
    """
    output_dir = os.environ.get("OUTPUT_DIR")
    if not output_dir:
        pytest.skip("OUTPUT_DIR is not set")
    path = Path(output_dir) / "traj_patchy_gpu" / f"{cond['name']}.gsd"
    if not path.exists():
        pytest.skip(f"{path} not found")

    box, seed, _ = _condition_args(cond)
    gsd_snap = load_gsd_frame0(path)
    torch_snap = build_torch_snapshot(box, seed)

    results = compare_snapshots(gsd_snap, torch_snap)
    report = format_report(f"{cond['name']} vs existing GSD frame 0: BOX={box}, SEED={seed}", results)
    print("\n" + report)
    assert all(r.passed for r in results), "\n" + report


# ----------------------------------------------------------------------------------------
# Tests of the comparison helpers themselves. Tiny made-up snapshot, NumPy only, so they run
# without HOOMD or torch and show what each check does and does not catch.
# ----------------------------------------------------------------------------------------
def _tiny_snapshot(source: str = "A") -> Snapshot:
    return Snapshot(
        source=source,
        box_full=np.array([10.0, 10.0, 10.0, 0, 0, 0]),
        type_names=["R", "P", "S", "T"],
        typeid=np.array([0, 1, 2, 3]),
        positions=np.array([[0.0, 0.0, 0.0], [2.5, 0.0, 0.0], [-3.0, 1.0, 0.0], [-1.8, 1.0, 0.0]]),
        bonds=np.array([[0, 1], [2, 3]]),
        bond_typeid=np.array([1, 0]),
        bond_type_names=["chain", "patch_tether"],
    )


def _shifted(snapshot: Snapshot, offset, source="B") -> Snapshot:
    return Snapshot(**{**snapshot.__dict__, "source": source,
                       "positions": snapshot.positions + np.asarray(offset, dtype=float)})


def test_helper_identical_snapshots_pass():
    a, b = _tiny_snapshot("A"), _tiny_snapshot("B")
    for check in (check_box, check_counts_and_names, check_particle_types, check_bonds,
                  check_positions, check_distance_matrix):
        assert check(a, b).passed, check.__name__


def test_helper_one_moved_particle_fails_positions_and_distances():
    a, b = _tiny_snapshot("A"), _tiny_snapshot("B")
    b.positions[2, 0] += 1e-3
    assert not check_positions(a, b).passed
    assert not check_distance_matrix(a, b).passed


def test_helper_whole_box_shift_is_not_an_error():
    a = _tiny_snapshot("A")
    b = _shifted(a, [10.0, 0.0, -10.0])            # same layout, other periodic image
    assert check_positions(a, b).passed
    assert check_distance_matrix(a, b).passed


def test_helper_rigid_translation_fails_positions_but_not_distances():
    a = _tiny_snapshot("A")
    b = _shifted(a, [1.0, 0.0, 0.0])               # same shape of layout, different place
    assert not check_positions(a, b).passed        # starting points differ
    assert check_distance_matrix(a, b).passed      # internal structure is identical


def test_helper_wrong_type_is_reported_with_particle_number():
    a, b = _tiny_snapshot("A"), _tiny_snapshot("B")
    b.typeid[3] = 2
    result = check_particle_types(a, b)
    assert not result.passed and "#3" in result.detail


def test_helper_missing_bond_fails_but_row_order_and_direction_do_not_matter():
    a, b = _tiny_snapshot("A"), _tiny_snapshot("B")
    b.bonds, b.bond_typeid = b.bonds[::-1, ::-1].copy(), b.bond_typeid[::-1].copy()
    assert check_bonds(a, b).passed                # reordered and flipped: still the same bonds
    b.bonds, b.bond_typeid = b.bonds[:1], b.bond_typeid[:1]
    assert not check_bonds(a, b).passed            # one bond dropped


def test_helper_geometry_flags_wrong_bond_length():
    a = _tiny_snapshot("A")
    assert check_geometry(a).passed                # tether 2.5 long, chain bond 1.2 long
    a.positions[1, 0] = 3.0                        # tether now 3.0 long
    assert not check_geometry(a).passed


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-s"]))
