import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")

with app.setup:
    import os
    import time
    from dataclasses import dataclass
    from pathlib import Path
    from typing import Mapping

    import marimo as mo
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt
    import pandas as pd

    from condensate_ml.constants import N_C, N_R

    # ------------------------------------------------------------------
    # The five simulated regimes (also used by tests/test_long_simulations.py)
    # ------------------------------------------------------------------
    @dataclass(frozen=True)
    class Regime:
        """One simulation condition.

        ``epsilon`` is the sticker-patch binding strength (positive = attractive).
        """

        name: str
        box: float
        epsilon: float
        seed: int
        n_steps: int = 100_000
        write_every: int = 500

        @property
        def n_frames(self) -> int:
            """Number of saved frames, including the initial one."""
            return self.n_steps // self.write_every + 1

    REGIMES = (
        Regime("control", box=40.0, epsilon=0.5, seed=101),
        Regime("early", box=36.0, epsilon=0.8, seed=102),
        Regime("nucleation", box=33.0, epsilon=1.2, seed=103),
        Regime("weak_coarsening", box=31.0, epsilon=1.8, seed=104),
        Regime("coarsening", box=29.0, epsilon=2.5, seed=105),
    )

    REGIME_COLORS = {
        "control": "#526579",
        "early": "#188C8D",
        "nucleation": "#7351AB",
        "weak_coarsening": "#C98A2F",
        "coarsening": "#B64F52",
    }
    REGIME_LABELS = {
        "control": "Control",
        "early": "Early",
        "nucleation": "Nucleation",
        "weak_coarsening": "Weak coarsening",
        "coarsening": "Coarsening",
    }
    # One HOOMD reference image per regime and metric: <regime>_<metric>.jpeg
    COMPARISON_METRICS = ("bonds", "clusters", "fraction", "largest")

    # ------------------------------------------------------------------
    # Locating files
    # ------------------------------------------------------------------
    def get_output_dir() -> Path:
        """Read OUTPUT_DIR from the environment or from a .env file next to this notebook."""
        from dotenv import load_dotenv

        load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")
        output_dir = os.getenv("OUTPUT_DIR")
        if output_dir is None:
            raise RuntimeError("OUTPUT_DIR was not found in the environment or .env file")
        return Path(output_dir)

    def artifact_directory(output_dir: str | Path, kind: str) -> Path:
        """Directory the simulation writes ``kind`` ('traj', 'xyz', 'data' or 'plots') files to."""
        import jax

        device_label = "gpu" if any(device.platform == "gpu" for device in jax.devices()) else "cpu"
        return Path(output_dir) / f"{kind}_jaxmd_{device_label}"

    # ------------------------------------------------------------------
    # Running a regime
    # ------------------------------------------------------------------
    def run_regime(regime: Regime, output_dir: str | Path):
        """Run one regime, write its GSD/XYZ/CSV/plots to ``output_dir``, return (trajectory, seconds)."""
        import jax.numpy as jnp

        from condensate_ml import RubiscoEpyc1Simulation

        print(f"Starting {regime.name}: BOX={regime.box}, EPS={regime.epsilon}, SEED={regime.seed}")
        simulation = RubiscoEpyc1Simulation(
            n_steps=regime.n_steps,
            box=regime.box,
            write_every=regime.write_every,
            checkpoint_blocks=True,
        )
        start = time.perf_counter()
        trajectory = simulation(
            {
                "epsilon": jnp.asarray(regime.epsilon, dtype=jnp.float32),
                "seed": regime.seed,
                "output_dir": output_dir,
                "output_name": regime.name,
            }
        )
        trajectory.block_until_ready()  # wait for asynchronous accelerator work
        elapsed = time.perf_counter() - start
        print(
            f"Finished {regime.name} in {elapsed:.2f} seconds. "
            f"Shape={trajectory.shape}, expected_frames={regime.n_frames}"
        )
        print(f"GSD written to: {artifact_directory(output_dir, 'traj') / f'{regime.name}.gsd'}")
        return trajectory, elapsed

    # ------------------------------------------------------------------
    # Summary plots (all regimes on one axis)
    # ------------------------------------------------------------------
    def read_regime_analysis(csv_path: str | Path) -> pd.DataFrame:
        """Read one post-simulation analysis CSV and validate its plot columns."""
        data = pd.read_csv(csv_path)
        required_columns = {"time", "clusters", "fraction"}
        missing_columns = required_columns.difference(data.columns)
        if missing_columns:
            raise ValueError(f"{csv_path} is missing required columns: {sorted(missing_columns)}")
        return data

    def load_regime_analysis(
        csv_paths_by_regime: Mapping[str, str | Path],
    ) -> dict[str, pd.DataFrame]:
        """Load analysis data for each named regime in a deterministic order."""
        unknown_regimes = set(csv_paths_by_regime).difference(REGIME_COLORS)
        if unknown_regimes:
            raise ValueError(f"Unknown regime names: {sorted(unknown_regimes)}")
        return {
            regime: read_regime_analysis(csv_paths_by_regime[regime])
            for regime in REGIME_COLORS
            if regime in csv_paths_by_regime
        }

    def plot_regime_metric(
        data_by_regime: Mapping[str, pd.DataFrame],
        metric: str,
        ylabel: str,
        output_path: str | Path,
    ) -> Path:
        """Plot one metric for all regimes and save it as a JPEG."""
        if metric not in {"clusters", "fraction"}:
            raise ValueError(f"Unsupported regime metric: {metric}")

        figure, axis = plt.subplots(figsize=(12, 7))
        for regime, data in data_by_regime.items():
            axis.plot(
                data["time"],
                data[metric],
                color=REGIME_COLORS[regime],
                linewidth=2.5,
                label=REGIME_LABELS[regime],
            )
        axis.set_xlabel("Time")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.2)
        axis.legend(frameon=False, ncol=len(data_by_regime), loc="upper center")
        figure.tight_layout()

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=180)
        plt.close(figure)
        return output_path

    def write_regime_summary_plots(
        csv_paths_by_regime: Mapping[str, str | Path],
        plots_dir: str | Path,
    ) -> tuple[Path, Path]:
        """Write the largest-cluster and connected-component regime plots."""
        data_by_regime = load_regime_analysis(csv_paths_by_regime)
        plots_dir = Path(plots_dir)
        largest_fraction_path = plot_regime_metric(
            data_by_regime=data_by_regime,
            metric="fraction",
            ylabel=f"Largest-cluster fraction: f_max = N_max / {N_R + N_C}",
            output_path=plots_dir / "largest_cluster_fraction.jpeg",
        )
        cluster_count_path = plot_regime_metric(
            data_by_regime=data_by_regime,
            metric="clusters",
            ylabel="Number of connected components",
            output_path=plots_dir / "number_of_clusters.jpeg",
        )
        return largest_fraction_path, cluster_count_path

    # ------------------------------------------------------------------
    # HOOMD-versus-JAX-MD comparison figures
    # ------------------------------------------------------------------
    def reference_plots_complete(reference_dir: Path) -> bool:
        """True if every HOOMD reference plot needed for the comparison figures exists."""
        return all(
            (reference_dir / f"{regime.name}_{metric}.jpeg").exists()
            for regime in REGIMES
            for metric in COMPARISON_METRICS
        )

    def write_regime_comparison_plots(
        reference_plots_dir: str | Path,
        jaxmd_plots_dir: str | Path,
        output_dir: str | Path,
    ) -> tuple[Path, ...]:
        """Write one HOOMD-versus-JAX-MD figure per regime, with each metric's plots stacked."""
        reference_plots_dir = Path(reference_plots_dir)
        jaxmd_plots_dir = Path(jaxmd_plots_dir)
        output_dir = Path(output_dir)
        output_paths = []

        for regime in REGIME_COLORS:
            reference_images = {
                metric: mpimg.imread(reference_plots_dir / f"{regime}_{metric}.jpeg")
                for metric in COMPARISON_METRICS
            }
            jaxmd_images = {
                metric: mpimg.imread(jaxmd_plots_dir / f"{regime}_{metric}.jpeg")
                for metric in COMPARISON_METRICS
            }

            figure = plt.figure(figsize=(16, 8))
            grid = figure.add_gridspec(
                2,
                len(COMPARISON_METRICS),
                left=0.08,
                right=0.99,
                bottom=0.06,
                top=0.88,
                wspace=0.15,
                hspace=0.12,
            )
            for column, metric in enumerate(COMPARISON_METRICS):
                reference_axis = figure.add_subplot(grid[0, column])
                jaxmd_axis = figure.add_subplot(grid[1, column])
                reference_axis.imshow(reference_images[metric])
                jaxmd_axis.imshow(jaxmd_images[metric])
                reference_axis.axis("off")
                jaxmd_axis.axis("off")
                reference_axis.set_title(metric.title(), fontsize=13, fontweight="bold")

            figure.text(0.025, 0.67, "HOOMD", ha="center", va="center", rotation=90,
                        fontsize=13, fontweight="bold")
            figure.text(0.025, 0.30, "JAX-MD", ha="center", va="center", rotation=90,
                        fontsize=13, fontweight="bold")
            figure.suptitle(REGIME_LABELS[regime], fontsize=16, fontweight="bold")
            figure.tight_layout(rect=(0.04, 0.0, 1.0, 0.9))

            output_path = output_dir / f"{regime}_comparison.jpeg"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(output_path, dpi=180)
            plt.close(figure)
            output_paths.append(output_path)

        return tuple(output_paths)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # HOOMD vs JAX-MD: condensation regimes

    This notebook runs the JAX-MD Rubisco/EPYC1 simulation for the **five regimes**
    (`control`, `early`, `nucleation`, `weak_coarsening`, `coarsening`), writes every output file,
    and builds the figures that compare the regimes with each other and with the HOOMD reference.

    **Two ways to run it, same files either way**

    | How | Command | What you get |
    |---|---|---|
    | Batch job (`job.slurm`) | `python comparison.py` | Files are written; progress is printed to the job log. Nothing is displayed. |
    | Interactive | `marimo edit comparison.py` | The same files are written, **and** every figure is shown below its cell. |

    The notebook notices which way it was started (`mo.app_meta().mode`) and only displays figures
    when you are working interactively.
    """)
    return


@app.cell(hide_code=True)
def _():
    interactive = mo.app_meta().mode != "script"
    mo.md(
        f"""
        ## 1. How was the notebook started?

        `mo.app_meta().mode` is `"script"` when you run `python comparison.py` (for example from
        `job.slurm`) and `"edit"` or `"run"` when you open it in marimo.

        Current mode: **{mo.app_meta().mode}**, so figures will
        **{"be displayed" if interactive else "only be written to disk"}**.
        """
    )
    return (interactive,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 2. Where do the files go?

    `get_output_dir()` reads the environment variable `OUTPUT_DIR` (or the same line in a `.env`
    file next to this notebook), for example `OUTPUT_DIR=/scratch/me/condensates`. It raises an
    error if neither is set, so a job can never silently write to the wrong place.

    The simulation then creates four sub-folders inside it. The suffix is `gpu` if JAX sees a
    GPU, otherwise `cpu`:

    | Folder | Contents |
    |---|---|
    | `traj_jaxmd_gpu/` | `<regime>.gsd` trajectories (HOOMD format) |
    | `xyz_jaxmd_gpu/` | `<regime>.xyz` frames for visualization |
    | `data_jaxmd_gpu/` | `<regime>.csv` cluster statistics per frame |
    | `plots_jaxmd_gpu/` | per-regime plots, the two summary plots, and the comparison figures |
    """)
    return


@app.cell
def _():
    output_dir = get_output_dir()
    print(f"Output directory: {output_dir}")
    return (output_dir,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 3. The five regimes

    Each regime is a `Regime` record: box length, sticker-patch binding strength `epsilon`
    (larger = stickier), random seed, number of steps and how often a frame is saved.
    A smaller box at the same particle number means a more concentrated system, and a larger
    `epsilon` means stronger binding, so the regimes go from dilute and weakly bound (`control`)
    to dense and strongly bound (`coarsening`).
    """)
    return


@app.cell
def _(interactive):
    _regime_table = mo.ui.table(
        [
            {
                "regime": regime.name,
                "box": regime.box,
                "epsilon": regime.epsilon,
                "seed": regime.seed,
                "steps": regime.n_steps,
                "frames": regime.n_frames,
            }
            for regime in REGIMES
        ],
        selection=None,
    )
    _regime_table if interactive else None
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 4. Run the simulations

    `run_regime()` builds a `RubiscoEpyc1Simulation`, calls it with the regime's `epsilon` and
    `seed`, and waits for the GPU to finish (`block_until_ready`) so the timing is honest.
    Because `output_dir` and `output_name` are passed, the simulation itself writes the GSD, XYZ,
    CSV and per-regime plots (`write_outputs`). The loop below runs the five regimes one after the other.
    """)
    return


@app.cell
def _(output_dir):
    elapsed_seconds = {}
    for _regime in REGIMES:
        _trajectory, elapsed_seconds[_regime.name] = run_regime(_regime, output_dir)
    return (elapsed_seconds,)


@app.cell
def _(elapsed_seconds, interactive):
    _timing_table = mo.ui.table(
        [{"regime": name, "seconds": round(seconds, 2)} for name, seconds in elapsed_seconds.items()],
        selection=None,
    )
    _timing_table if interactive else None
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 5. Summary plots: all regimes on one axis

    `write_regime_summary_plots()` reads each regime's CSV with `read_regime_analysis()` (and checks
    that the columns `time`, `clusters` and `fraction` exist) and draws two figures, one line per regime:

    * **Largest-cluster fraction** `f_max = N_max / 72`. A cluster is a set of Rubisco and EPYC1
      molecules connected by sticker-patch contacts; 72 is the total number of molecules (24 Rubisco + 48 chains).
      Coarsening keeps growing this value, nucleation levels off.
    * **Number of connected components**, which falls as molecules join clusters.

    Both are saved as JPEG files in `plots_jaxmd_<device>/`.
    """)
    return


@app.cell
def _(elapsed_seconds, output_dir):
    _finished = list(elapsed_seconds)  # makes this cell wait for the simulations above
    csv_paths = {
        regime.name: artifact_directory(output_dir, "data") / f"{regime.name}.csv"
        for regime in REGIMES
    }
    plots_dir = artifact_directory(output_dir, "plots")
    largest_fraction_path, cluster_count_path = write_regime_summary_plots(csv_paths, plots_dir)
    print(f"Largest-cluster plot written to: {largest_fraction_path}")
    print(f"Cluster-count plot written to: {cluster_count_path}")
    return cluster_count_path, largest_fraction_path, plots_dir


@app.cell
def _(cluster_count_path, interactive, largest_fraction_path):
    mo.vstack(
        [
            mo.md("### Largest-cluster fraction"),
            mo.image(src=str(largest_fraction_path)),
            mo.md("### Number of connected components"),
            mo.image(src=str(cluster_count_path)),
        ]
    ) if interactive else None
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 6. HOOMD vs JAX-MD comparison

    If the environment variable `plots_patchy_gpu` points to the folder with the **HOOMD** reference
    plots (`<regime>_bonds.jpeg`, `_clusters`, `_fraction`, `_largest`), `reference_plots_complete()` checks that
    all 20 files exist. If so, `write_regime_comparison_plots()` makes one figure per regime with
    HOOMD on the top row and JAX-MD on the bottom row for each metric, and saves it as
    `<regime>_comparison.jpeg`. If the variable is not set, or files are missing, this step is skipped.
    Since HOOMD and JAX-MD use different random numbers, the curves should look alike in shape and
    timescale, not match point for point.
    """)
    return


@app.cell
def _(plots_dir):
    reference_dir = os.getenv("plots_patchy_gpu")
    comparison_paths = ()
    if reference_dir and reference_plots_complete(Path(reference_dir)):
        comparison_paths = write_regime_comparison_plots(Path(reference_dir), plots_dir, plots_dir)
        for _path in comparison_paths:
            print(f"Comparison plot written to: {_path}")
    else:
        print("HOOMD reference plots not found (set plots_patchy_gpu); skipping comparison figures.")
    return (comparison_paths,)


@app.cell
def _(comparison_paths, interactive):
    mo.vstack(
        [mo.vstack([mo.md(f"### {_path.stem}"), mo.image(src=str(_path))]) for _path in comparison_paths]
        or [mo.md("*No comparison figures: HOOMD reference plots were not found.*")]
    ) if interactive else None
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Done

    Every regime's trajectory, XYZ frames, CSV statistics and plots are in the output directory, together
    with the summary and comparison figures. To change a regime, edit the `REGIMES` list in the setup
    cell at the top of this file.
    """)
    return


if __name__ == "__main__":
    app.run()
