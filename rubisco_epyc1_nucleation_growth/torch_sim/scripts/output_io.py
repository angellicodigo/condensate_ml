"""Input/output helpers: export XYZ, CSV, summary JSON, and diagnostic plots.

All functions write to disk and return nothing (or a small summary dict). They
are the "boundary" side effects of the simulation pipeline.
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence

import numpy as np
import pandas as pd
import torch

from .geometry import apply_minimum_image_displacement
from .post_simulation_analysis import compute_cluster_graph_metrics, compute_hard_contacts


def export_ovito_xyz(
    filepath: str,
    trajectory: torch.Tensor,
    particle_labels: Sequence[str],
    dt: float,
    write_every: int,
) -> None:
    """Export a multi-frame trajectory to OVITO-compatible XYZ format.

    Each frame writes the particle count, a step comment line, then one line
    per particle with its label and coordinates.
    """
    parent_dir = os.path.dirname(os.path.abspath(filepath))
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    traj_cpu = trajectory.detach().cpu().numpy()
    n_frames, n_particles, _ = traj_cpu.shape

    with open(filepath, "w") as fh:
        for frame_idx in range(n_frames):
            step = frame_idx * write_every
            fh.write(f"{n_particles}\n")
            fh.write(f"step {step}\n")
            coords = traj_cpu[frame_idx]
            for label, pos in zip(particle_labels, coords):
                fh.write(f"{label} {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}\n")


def compute_frame_metrics(
    pos_frame: torch.Tensor,
    sticker_indices: torch.Tensor,
    patch_indices: torch.Tensor,
    sticker_chain_ids: torch.Tensor,
    patch_rubisco_ids: torch.Tensor,
    box: float,
    step: int,
    sim_time: float,
    frame_idx: int,
    name: str,
    kon: float,
    epsilon_used: float,
    koff: float,
    n_rubisco: int,
    n_chains: int,
) -> dict:
    """Compute contact and cluster metrics for a single trajectory frame.

    Returns a dict with all columns needed for the CSV output.
    """
    contacts_binary = compute_hard_contacts(
        pos_frame, sticker_indices, patch_indices, box, cutoff=3.0
    )
    n_contacts = int(contacts_binary.sum().item())
    n_clusters, largest_size, fraction = compute_cluster_graph_metrics(
        contacts_binary, sticker_chain_ids, patch_rubisco_ids, n_rubisco, n_chains
    )

    return {
        "frame": frame_idx,
        "step": step,
        "time": sim_time,
        "box": box,
        "kon_or_epsilon": kon,
        "epsilon_used": epsilon_used,
        "koff_recorded": koff,
        "contacts": n_contacts,
        "bonds": n_contacts,
        "clusters": n_clusters,
        "largest": largest_size,
        "fraction": fraction,
    }


def write_csv_file(csv_filepath: str, records: list[dict]) -> None:
    """Write a list of per-frame metric dicts to a CSV file."""
    csv_dir = os.path.dirname(os.path.abspath(csv_filepath))
    if csv_dir:
        os.makedirs(csv_dir, exist_ok=True)

    df = pd.DataFrame(records)
    df.to_csv(csv_filepath, index=False)


def compile_simulation_summary(
    name: str,
    box: float,
    kon: float,
    epsilon_used: float,
    koff: float,
    dt: float,
    n_steps: int,
    df: pd.DataFrame,
    n_rubisco: int,
    n_chains: int,
) -> dict:
    """Compile the final summary dict from the per-frame metrics DataFrame."""
    final_contacts = int(df["contacts"].iloc[-1]) if len(df) > 0 else 0
    final_clusters = int(df["clusters"].iloc[-1]) if len(df) > 0 else (n_rubisco + n_chains)
    final_largest = int(df["largest"].iloc[-1]) if len(df) > 0 else 1
    final_fraction = float(df["fraction"].iloc[-1]) if len(df) > 0 else 1.0 / (n_rubisco + n_chains)

    return {
        "name": name,
        "box": box,
        "kon": kon,
        "epsilon_used": epsilon_used,
        "dt": dt,
        "n_steps": n_steps,
        "frames_written": len(df),
        "final_contacts": final_contacts,
        "final_clusters": final_clusters,
        "final_largest_cluster": final_largest,
        "final_fraction": final_fraction,
    }


def write_summary_json(summary_filepath: str, summary: dict) -> None:
    """Write the simulation summary dict to a JSON file."""
    summary_dir = os.path.dirname(os.path.abspath(summary_filepath))
    if summary_dir:
        os.makedirs(summary_dir, exist_ok=True)

    with open(summary_filepath, "w") as fh:
        json.dump(summary, fh, indent=2)


def _draw_line_plot_pillow(
    x: Sequence[float],
    y: Sequence[float],
    xlabel: str,
    ylabel: str,
    title: str,
    save_path: str,
    width: int = 1200,
    height: int = 800,
) -> None:
    """Render a 2-D line plot to JPEG using Pillow with no C-backend dependencies."""
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    margin_left = 120
    margin_right = 60
    margin_top = 80
    margin_bottom = 100

    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom

    draw.line(
        [(margin_left, margin_top), (margin_left, height - margin_bottom)],
        fill=(0, 0, 0),
        width=2,
    )
    draw.line(
        [(margin_left, height - margin_bottom), (width - margin_right, height - margin_bottom)],
        fill=(0, 0, 0),
        width=2,
    )

    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)

    if len(x_arr) > 0:
        x_min, x_max = float(x_arr.min()), float(x_arr.max())
        y_min, y_max = float(y_arr.min()), float(y_arr.max())
        if x_max == x_min:
            x_max += 1.0
        if y_max == y_min:
            y_max += 1.0

        pts = []
        for xi, yi in zip(x_arr, y_arr):
            px = margin_left + ((xi - x_min) / (x_max - x_min)) * plot_w
            py = (height - margin_bottom) - ((yi - y_min) / (y_max - y_min)) * plot_h
            pts.append((px, py))

        if len(pts) > 1:
            draw.line(pts, fill=(31, 119, 180), width=3)
        elif len(pts) == 1:
            draw.ellipse(
                [pts[0][0] - 3, pts[0][1] - 3, pts[0][0] + 3, pts[0][1] + 3],
                fill=(31, 119, 180),
            )

    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    draw.text((width // 2 - 60, margin_top // 2), title, fill=(0, 0, 0), font=font)
    draw.text((width // 2 - 20, height - margin_bottom // 2), xlabel, fill=(0, 0, 0), font=font)
    draw.text((margin_left // 4, height // 2), ylabel, fill=(0, 0, 0), font=font)

    img.save(save_path, "JPEG", quality=95)


def create_single_diagnostic_plot(
    df: pd.DataFrame,
    plots_dir: str,
    name: str,
    metric_key: str,
    ylabel: str,
) -> None:
    """Generate one diagnostic JPEG plot for a single metric column."""
    save_path = os.path.join(plots_dir, f"{name}_{metric_key}.jpeg")
    _draw_line_plot_pillow(
        df["time"],
        df[metric_key],
        "time",
        ylabel,
        f"{name}: {ylabel} vs time",
        save_path,
    )


def export_csv_and_summary(
    csv_filepath: str,
    summary_filepath: str,
    name: str,
    box: float,
    kon: float,
    epsilon_used: float,
    koff: float,
    dt: float,
    n_steps: int,
    write_every: int,
    trajectory: torch.Tensor,
    sticker_indices: torch.Tensor,
    patch_indices: torch.Tensor,
    sticker_chain_ids: torch.Tensor,
    patch_rubisco_ids: torch.Tensor,
    n_rubisco: int = 24,
    n_chains: int = 48,
) -> pd.DataFrame:
    """Compute per-frame metrics and write CSV plus summary JSON.

    Thin orchestration wrapper: compute per-frame metrics, write CSV,
    compile the summary dict, write summary JSON, return the DataFrame.
    """
    csv_dir = os.path.dirname(os.path.abspath(csv_filepath))
    if csv_dir:
        os.makedirs(csv_dir, exist_ok=True)
    summary_dir = os.path.dirname(os.path.abspath(summary_filepath))
    if summary_dir:
        os.makedirs(summary_dir, exist_ok=True)

    records = []
    n_frames = trajectory.shape[0]

    for frame_idx in range(n_frames):
        step = frame_idx * write_every
        sim_time = step * dt
        pos_frame = trajectory[frame_idx]

        record = compute_frame_metrics(
            pos_frame,
            sticker_indices,
            patch_indices,
            sticker_chain_ids,
            patch_rubisco_ids,
            box,
            step,
            sim_time,
            frame_idx,
            name,
            kon,
            epsilon_used,
            koff,
            n_rubisco,
            n_chains,
        )
        records.append(record)

    write_csv_file(csv_filepath, records)
    df = pd.DataFrame(records)

    summary = compile_simulation_summary(
        name, box, kon, epsilon_used, koff, dt, n_steps, df, n_rubisco, n_chains
    )
    write_summary_json(summary_filepath, summary)

    return df


def generate_all_diagnostic_plots(df: pd.DataFrame, plots_dir: str, name: str) -> None:
    """Generate 4 time-series diagnostic JPEG plots matching HOOMD reference outputs."""
    if plots_dir:
        os.makedirs(plots_dir, exist_ok=True)
    if len(df) == 0:
        return

    plots_by_metric = [
        ("bonds", "contacts / bonds"),
        ("clusters", "number of clusters"),
        ("largest", "largest cluster size"),
        ("fraction", "largest cluster fraction"),
    ]

    for metric_key, ylabel in plots_by_metric:
        create_single_diagnostic_plot(df, plots_dir, name, metric_key, ylabel)
