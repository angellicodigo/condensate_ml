# Storage and data plan for the differentiable Rubisco–EPYC1 pipeline

Status: brainstorm / proposal, 2026-11-12. Sizes marked *(est.)* are back-of-envelope, not measured.

## 1. What NeFTY does (from NeFTY-code and the HF dataset card)

The key point: **NeFTY never stores anything differentiable.** Gradients are recomputed on the fly by
running the differentiable solver inside every optimisation step.

| Thing | How NeFTY handles it |
|---|---|
| Ground-truth data | One `torch.save` dict per dataset (`dataset.pt`): `k_gt (N,64,64,16)`, `t0_gt`, `surface_gt (N,101,64,64)`, `surface_gt_phiflow`, `surface_gt_ours`, `resolution`. 3.8 GB for 1,000 samples. |
| Hosting | Hugging Face dataset repo, SHA-256 checksums on the card, `scripts/download_data.py` fetches to fixed local paths. |
| Small test set | 8-sample mini dataset (31 MB) tracked in git so tests and quick configs need no download. |
| Regeneration | `python -m dataset.synthetic --seed 42` reproduces `k_gt` exactly; the PhiFlow surfaces depend on the GPU/CPU used (so bitwise regeneration is *not* promised). |
| Per-run output | `args.txt`, `best.tar`, `final.tar`, then `metrics.txt` and PNGs. `best.tar` is chosen by surface-temperature PSNR, i.e. **no ground-truth labels are used for selection**. |
| Memory of gradients | Hand-written discrete adjoint: 21.9 MB vs 18.63 GB for plain autograd on one specimen (paper Table 4). |
| Training mode | Per-specimen test-time optimisation, no training set. Each run starts from a fresh network and fits one measurement. |
| Batch bookkeeping | `tasks.json` records which specimens were selected, so reports are reproducible. |

Takeaway for us: the heavy things to store are **inputs/targets and checkpoints**, not gradients.

## 2. What we actually have today

* HOOMD runs, 5 conditions, **one seed each** (`control`, `early`, `nucleation`, `weak_coarsening`, `coarsening`).
  Logs show GSD/XYZ/CSV were written to `/scratch/network/al5098/simulations/...`, which is **outside the repo**.
  In the repo we only have plots, logs and parameter files.
* Observables per run: 201 frames × `contacts, bonds, clusters, largest, fraction`.
* The five conditions change `BOX` and `EPS` together (box 40→29 while eps 0.5→2.5), so density and
  binding strength are confounded. Five single-seed points are a thin label set for any learned model.
* No torch-vs-HOOMD **statistical** comparison exists yet. The unit tests only check force implementations
  against autograd of the torch potentials, which cannot catch a mismatch with HOOMD.

## 3. Size of the data (est.)

* One trajectory, float32, 201 frames × 792 × 3: **~1.9 MB**. Writing *every* step of a 100k-step run would be ~950 MB.
* One OVITO XYZ of the same run: ~5 MB (text). CSV: a few KB.
* So 10,000 runs of stored trajectories ≈ 20 GB. **Disk is not the bottleneck.**
* GPU memory for backprop: one `(792,792,3)` float32 tensor is 7.5 MB. If roughly ten of these are alive per step,
  a 32-step checkpoint block needs on the order of 2–3 GB, and the saved block-boundary states are negligible.
  Measure this before relying on it. The real limits are **compute** (≈2× forward) and **gradient quality**.

## 4. Two routes, and why the choice decides what to store

`CONTINUATION_CONTEXT.txt` plans a PyTorch **surrogate** trained on simulation outputs. The torch stepper adds a second route.

| | A. Direct autograd through `RubiscoEpyc1TimeStepper` (NeFTY-shaped) | B. Surrogate trained on simulation outputs |
|---|---|---|
| Needs a big dataset? | No. Needs targets, e.g. HOOMD observable curves | Yes: a large sweep of (parameters → observables), replicated |
| Gradient source | Autograd + checkpointing, noise as reparameterised input | Surrogate network |
| Main risk | Gradient variance and chaos over long horizons | Surrogate error, coverage of parameter space |
| Storage | Targets, checkpoints, restart configs | Sweep dataset (below) |

They are compatible: use A for training, use B (or cheap HOOMD sweeps) for validation and for exploring the design space.

## 5. Proposed layout

### Tier A: in git (small)
* `params/*.json`, `manifest.json` per run (see below), CSV time series, `*_summary.json`.
* A **mini dataset** (a few runs, ≤ 30 MB) for tests, as NeFTY does.

### Tier B: shared storage / HF dataset (medium)
* **One file per run**, not one giant dict (NeFTY's evaluators load the whole 3.8 GB file per process, which bounds host memory):
  `runs/<id>.pt` with `positions (T,792,3) float32`, `params`, `seed`, `dt`, `write_every`, `engine` (`hoomd` | `torch`).
* `index.csv` or Parquet: `id, regime, BOX, EPS, seed, engine, n_frames, path, sha256`. DataLoaders read the index and load lazily.
* Observable tensor for targets: `obs (n_runs, T, 4)` = contacts, clusters, largest, fraction on a common time grid, plus per-condition mean and std over seeds.
* Keep **GSD on cluster scratch** as the raw HOOMD record and convert to the arrays above with a script. Scratch is usually not backed up; confirm the Research Computing policy.

### Tier C: checkpoints (per training run)
Same dict shape NeFTY uses: `model_state_dict`, `optimizer_state_dict`, `scheduler_state_dict`, `global_step`, `args`.
Save `best.tar` by a **label-free** criterion (e.g. distance of simulated observables to the target curves), as NeFTY does with surface PSNR.

### Run manifest (every simulation and training run)
`git commit`, `params`, `seed`, `dt`, `torch`/`nefi`/`hoomd` versions, device name, wall time, input file checksums.
A trajectory is *not* reproducible bit-for-bit (GPU nondeterminism, chaotic dynamics), so store the outputs you rely on rather than only the seed.

## 6. Ideas worth trying

1. **Restart library.** `RubiscoEpyc1TimeStepper` already accepts `fields["positions_0"]`. Store HOOMD (or torch) snapshots at several times
   (e.g. steps 0, 20k, 50k, 80k; 792×3 floats = 9.5 KB each). Train on **short windows** (2–10k steps) starting from realistic configurations
   instead of backpropagating 100k steps from the lattice. Particle ordering is identical in HOOMD and torch.
2. **Ensemble losses.** Noise does not depend on `epsilon`, so the pathwise gradient is valid, but single-trajectory gradients are noisy.
   Average over K seeds, optionally with common random numbers across optimiser steps, and compare to **seed-averaged** targets, never one HOOMD trajectory.
3. **Per-sticker epsilon.** The attraction code broadcasts, so an `(n_stickers, 1)` epsilon should already work mechanically.
   That gives a NeFTY-like "network maps sticker features → local binding strength" use case. Needs a test before relying on it.
4. **Differentiable cluster proxies.** Hard clusters use union-find. Build a soft 72×72 adjacency from `compute_soft_contacts` and use a spectral quantity
   (e.g. graph-Laplacian eigenvalues) as a smooth stand-in for cluster count / largest cluster. Validate against the hard metrics.
5. **Design of experiments.** Replace the coupled BOX/EPS ladder with a grid over density (box) × epsilon × ≥3 seeds so the effects separate.
6. **Gradient regression fixtures.** Store a few `(epsilon, loss, grad)` triples with noise disabled so refactors can be checked.

## 7. Open questions for you
* What does the network output: a single `epsilon`, a per-sticker profile, an initial configuration, or box/density too?
* What is the target: HOOMD curves, literature/experimental observables, or both?
* How many replicate seeds per condition are acceptable for HOOMD sweeps?
