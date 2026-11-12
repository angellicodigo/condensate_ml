def build_mobilities_by_particle(
    n_rubisco: int,
    n_patches_per_rubisco: int,
    n_chains: int,
    n_beads_per_chain: int,
    gamma_rubisco: float = 5.0,
    gamma_chain: float = 1.0,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build mobility factors mu = 1 / gamma per particle, shape (n_particles, 1).

    Rubisco centers and patches (the first n_rubisco * (1 + patches) particles)
    use gamma_rubisco; chain beads use gamma_chain.
    """
    n_slow = n_rubisco * (1 + n_patches_per_rubisco)
    n_particles = n_slow + n_chains * n_beads_per_chain
    mobilities = torch.full((n_particles, 1), 1.0 / gamma_chain, device=device, dtype=dtype)
    mobilities[:n_slow] = 1.0 / gamma_rubisco
    return mobilities


def build_nonbonded_pair_mask(
    bond_pairs_by_type: Mapping[str, torch.Tensor],
    n_particles: int,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Boolean (N, N) matrix: True where a pair interacts via pair forces.

    Directly bonded pairs and self-pairs are False, matching HOOMD's default
    ``exclusions=('bond',)`` for the neighbor list.
    """
    mask = torch.ones((n_particles, n_particles), dtype=torch.bool, device=device)
    for pairs in bond_pairs_by_type.values():
        mask[pairs[:, 0], pairs[:, 1]] = False
        mask[pairs[:, 1], pairs[:, 0]] = False
    mask.fill_diagonal_(False)
    return mask