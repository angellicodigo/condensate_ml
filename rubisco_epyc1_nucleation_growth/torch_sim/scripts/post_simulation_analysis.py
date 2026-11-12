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
    contact_cutoff: float,
) -> dict:
    """Compute contact and cluster metrics for a single trajectory frame."""
    contacts_binary = compute_hard_contacts(
        pos_frame, sticker_indices, patch_indices, box, cutoff=contact_cutoff
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
    n_rubisco: int,
    n_chains: int,
    contact_cutoff: float,
) -> pd.DataFrame:
    """Compute per-frame metrics and write CSV plus summary JSON."""
    records = []
    for frame_idx in range(trajectory.shape[0]):
        step = frame_idx * write_every
        records.append(
            compute_frame_metrics(
                trajectory[frame_idx],
                sticker_indices,
                patch_indices,
                sticker_chain_ids,
                patch_rubisco_ids,
                box,
                step,
                step * dt,
                frame_idx,
                name,
                kon,
                epsilon_used,
                koff,
                n_rubisco,
                n_chains,
                contact_cutoff,
            )
        )

    write_csv_file(csv_filepath, records)
    df = pd.DataFrame(records)

    summary = compile_simulation_summary(
        name, box, kon, epsilon_used, koff, dt, n_steps, df, n_rubisco, n_chains
    )
    write_summary_json(summary_filepath, summary)
    return df