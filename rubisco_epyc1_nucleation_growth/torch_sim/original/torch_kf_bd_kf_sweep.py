import os, math, time, random, json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

DT = 0.004
N_EQ = 2000
N_RUN = 8000
STEPS = N_EQ + N_RUN
SAVE_EVERY = 40
XYZ_EVERY = 40

BOX = 55.0
N_R = 24
N_C = 36

CHAIN = ["S","T","S","T","S","T","S","T","S"]
N_BEADS = len(CHAIN)
STICKERS = [i for i,x in enumerate(CHAIN) if x=="T"]
N_STICKERS = len(STICKERS)

RUBISCO_MOBILITY = 0.01
CHAIN_MOBILITY = 0.02
THERMAL_NOISE_R = 0.02
THERMAL_NOISE_C = 0.03
ROT_SCALE = 0.05

CHAIN_BOND_LEN = 1.15
CHAIN_BOND_K = 25.0
PATCH_BOND_K = 14.0
PATCH_BOND_R0 = 1.10
PATCH_BREAK_R = 2.8

RUBISCO_REPULSION_R = 4.0
BEAD_RUBISCO_REPULSION_R = 0.95
REPULSION_K = 6.0

K_OFF = 0.01
PATCHES = 6
PATCH_RADIUS = 2.0
THETA_C_DEG = 95.0

SETS = [
    ("set00", 1.0, 1.5, 0.2),
    ("set01", 1.2, 1.8, 0.5),
    ("set02", 1.5, 2.0, 1.0),
    ("set03", 2.0, 2.2, 5.0),
    ("set04", 2.5, 2.5, 5.0),
    ("set05", 3.0, 3.5, 5.0),
]

os.makedirs("data_bd", exist_ok=True)
os.makedirs("traj_bd", exist_ok=True)
os.makedirs("plots_bd", exist_ok=True)

def rand_pos(n):
    return torch.rand((n, 3), device=device) * BOX

def pbc(d):
    return d - BOX * torch.round(d / BOX)

def rand_rot():
    v = torch.randn(3, device=device)
    v = v / (torch.norm(v) + 1e-8)
    theta = torch.randn(1, device=device) * 0.15
    K = torch.tensor(
        [[0.0, -v[2].item(), v[1].item()],
         [v[2].item(), 0.0, -v[0].item()],
         [-v[1].item(), v[0].item(), 0.0]],
        device=device, dtype=torch.float32
    )
    I = torch.eye(3, device=device)
    return I + torch.sin(theta) * K + (1 - torch.cos(theta)) * (K @ K)

def make_patches():
    pts = []
    for i in range(PATCHES):
        phi = 2 * np.pi * i / PATCHES
        pts.append([math.cos(phi) * PATCH_RADIUS, math.sin(phi) * PATCH_RADIUS, 0.0])
    return torch.tensor(pts, device=device, dtype=torch.float32)

PATCH_REF = make_patches()

def clusters(bind_cpu):
    parent = list(range(N_R + N_C))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a, b):
        pa, pb = find(a), find(b)
        if pa != pb:
            parent[pb] = pa
    for c in range(N_C):
        for s in range(N_STICKERS):
            r = int(bind_cpu[c, s])
            if r >= 0:
                union(N_R + c, r)
    groups = {}
    for i in range(len(parent)):
        root = find(i)
        groups.setdefault(root, []).append(i)
    sizes = sorted([len(v) for v in groups.values()], reverse=True)
    largest = sizes[0] if sizes else 0
    frac = largest / len(parent) if len(parent) > 0 else 0.0
    return len(sizes), largest, frac

def write_xyz(xyz, step, rub, chain):
    xyz.write(f"{N_R + N_C * N_BEADS}\n")
    xyz.write(f"step {step} time {step * DT:.5f}\n")
    rub_cpu = rub.detach().cpu().numpy()
    chain_cpu = chain.detach().cpu().numpy()
    for p in rub_cpu:
        xyz.write(f"R {p[0]} {p[1]} {p[2]}\n")
    for c in range(N_C):
        for b in range(N_BEADS):
            xyz.write(f"{CHAIN[b]} {chain_cpu[c,b,0]} {chain_cpu[c,b,1]} {chain_cpu[c,b,2]}\n")

def chain_spring_forces(chain):
    forces = torch.zeros_like(chain)
    for b in range(N_BEADS - 1):
        d = chain[:, b + 1, :] - chain[:, b, :]
        d = pbc(d)
        r = torch.norm(d, dim=1, keepdim=True) + 1e-8
        f = -CHAIN_BOND_K * (r - CHAIN_BOND_LEN) * d / r
        forces[:, b, :] += f
        forces[:, b + 1, :] -= f
    return forces

def rubisco_repulsion(rub):
    d = rub[:, None, :] - rub[None, :, :]
    d = pbc(d)
    r = torch.norm(d, dim=-1) + 1e-8
    mask = (r < RUBISCO_REPULSION_R)
    mask.fill_diagonal_(False)
    pref = REPULSION_K * (RUBISCO_REPULSION_R - r) / r
    pref = torch.where(mask, pref, torch.zeros_like(pref))
    return (pref.unsqueeze(-1) * d).sum(dim=1)

def bead_rubisco_repulsion(chain, rub):
    beads = chain.reshape(-1, 3)
    d = beads[:, None, :] - rub[None, :, :]
    d = pbc(d)
    r = torch.norm(d, dim=-1) + 1e-8
    mask = (r < BEAD_RUBISCO_REPULSION_R)
    pref = REPULSION_K * (BEAD_RUBISCO_REPULSION_R - r) / r
    pref = torch.where(mask, pref, torch.zeros_like(pref))
    bead_force = (pref.unsqueeze(-1) * d).sum(dim=1).reshape(N_C, N_BEADS, 3)
    rub_force = -(pref.unsqueeze(-1) * d).sum(dim=0)
    return bead_force, rub_force

for name, SIGMA, LAMBDA, K_ON in SETS:
    print("\n===================================================")
    print("RUN:", name)
    print("SIGMA =", SIGMA)
    print("LAMBDA =", LAMBDA)
    print("ETA =", (N_R + N_C) / BOX**3)
    print("K_ON =", K_ON)
    print("K_OFF =", K_OFF)
    print("THETA_C_DEG =", THETA_C_DEG)
    print("===================================================")

    rub = rand_pos(N_R)
    rot = torch.stack([rand_rot() for _ in range(N_R)], dim=0)
    chain = rand_pos(N_C).unsqueeze(1).repeat(1, N_BEADS, 1)

    bind = torch.full((N_C, N_STICKERS), -1, dtype=torch.long, device=device)
    patch_occ = torch.full((N_R, PATCHES), -1, dtype=torch.long, device=device)

    eta = (N_R + N_C) / BOX**3
    records = []
    t0 = time.time()

    xyz = open(f"traj_bd/{name}.xyz", "w")

    for step in range(STEPS):
        patch_pos = rub[:, None, :] + torch.einsum("pj,rij->rpi", PATCH_REF, rot)
        patch_norm = patch_pos - rub[:, None, :]
        patch_norm = patch_norm / (torch.norm(patch_norm, dim=-1, keepdim=True) + 1e-8)

        chain_force = chain_spring_forces(chain)
        rub_force = rubisco_repulsion(rub)
        bead_rep, rub_rep = bead_rubisco_repulsion(chain, rub)
        chain_force += bead_rep
        rub_force += rub_rep

        for c in range(N_C):
            for s_local, bead_idx in enumerate(STICKERS):
                r_idx = int(bind[c, s_local].item())
                if r_idx < 0:
                    continue
                occ_patches = torch.nonzero(patch_occ[r_idx] == c, as_tuple=False).flatten()
                if occ_patches.numel() == 0:
                    bind[c, s_local] = -1
                    continue
                p_idx = int(occ_patches[0].item())

                bead_pos = chain[c, bead_idx, :]
                ppos = patch_pos[r_idx, p_idx, :]
                d = bead_pos - ppos
                d = pbc(d)
                r = torch.norm(d) + 1e-8

                if (r.item() > PATCH_BREAK_R) or (random.random() < K_OFF * DT):
                    bind[c, s_local] = -1
                    patch_occ[r_idx, p_idx] = -1
                    continue

                f = -PATCH_BOND_K * (r - PATCH_BOND_R0) * d / r
                chain_force[c, bead_idx, :] += f
                rub_force[r_idx, :] -= f

        chain = (chain + CHAIN_MOBILITY * chain_force + THERMAL_NOISE_C * torch.randn_like(chain)) % BOX
        rub = (rub + RUBISCO_MOBILITY * rub_force + THERMAL_NOISE_R * torch.randn_like(rub)) % BOX

        for i in range(N_R):
            rot[i] = rand_rot() @ rot[i]

        if step >= N_EQ:
            patch_pos = rub[:, None, :] + torch.einsum("pj,rij->rpi", PATCH_REF, rot)
            patch_norm = patch_pos - rub[:, None, :]
            patch_norm = patch_norm / (torch.norm(patch_norm, dim=-1, keepdim=True) + 1e-8)

            sticker_pos = chain[:, STICKERS, :]
            sticker_flat = sticker_pos.reshape(-1, 3)
            all_patch_pos = patch_pos.reshape(-1, 3)
            all_patch_norm = patch_norm.reshape(-1, 3)

            delta = sticker_flat[:, None, :] - all_patch_pos[None, :, :]
            delta = pbc(delta)
            dist = torch.norm(delta, dim=-1) + 1e-8

            shell = (dist > SIGMA) & (dist < SIGMA * LAMBDA)
            d_hat = delta / dist.unsqueeze(-1)
            cosang = torch.sum(all_patch_norm[None, :, :] * (-d_hat), dim=-1)
            angle_ok = cosang > math.cos(math.radians(THETA_C_DEG))
            valid = shell & angle_ok

            rand = torch.rand_like(dist)
            proposals = (rand < K_ON * DT) & valid
            inds = torch.nonzero(proposals, as_tuple=False)

            for s_idx, p_idx in inds.tolist():
                c = s_idx // N_STICKERS
                s_local = s_idx % N_STICKERS
                r = p_idx // PATCHES
                p = p_idx % PATCHES
                if bind[c, s_local] >= 0:
                    continue
                if patch_occ[r, p] >= 0:
                    continue
                bind[c, s_local] = r
                patch_occ[r, p] = c

        if step % SAVE_EVERY == 0:
            bind_cpu = bind.detach().cpu().numpy()
            ncl, largest, frac = clusters(bind_cpu)
            bonds = int((bind >= 0).sum().item())
            records.append({
                "step": step,
                "time": step * DT,
                "sigma": SIGMA,
                "lambda": LAMBDA,
                "eta": eta,
                "n_clusters": ncl,
                "largest_cluster": largest,
                "largest_fraction": frac,
                "n_bonds": bonds
            })
            print(f"step={step:5d} sigma={SIGMA:.3f} lambda={LAMBDA:.3f} eta={eta:.6f} clusters={ncl:4d} largest={largest:4d} frac={frac:.4f} bonds={bonds:4d}")

        if step % XYZ_EVERY == 0:
            write_xyz(xyz, step, rub, chain)

    xyz.close()

    df = pd.DataFrame(records)
    csv_path = f"data_bd/{name}.csv"
    df.to_csv(csv_path, index=False)

    plt.figure()
    plt.plot(df["time"], df["n_clusters"])
    plt.xlabel("time"); plt.ylabel("number of clusters"); plt.title(f"{name}: clusters vs time")
    plt.tight_layout(); plt.savefig(f"plots_bd/{name}_clusters.jpeg", dpi=300); plt.close()

    plt.figure()
    plt.plot(df["time"], df["largest_cluster"])
    plt.xlabel("time"); plt.ylabel("largest cluster"); plt.title(f"{name}: largest cluster vs time")
    plt.tight_layout(); plt.savefig(f"plots_bd/{name}_largest.jpeg", dpi=300); plt.close()

    plt.figure()
    plt.plot(df["time"], df["largest_fraction"])
    plt.xlabel("time"); plt.ylabel("largest cluster fraction"); plt.title(f"{name}: largest fraction vs time")
    plt.tight_layout(); plt.savefig(f"plots_bd/{name}_frac.jpeg", dpi=300); plt.close()

    plt.figure()
    plt.plot(df["time"], df["n_bonds"])
    plt.xlabel("time"); plt.ylabel("number of bonds"); plt.title(f"{name}: bonds vs time")
    plt.tight_layout(); plt.savefig(f"plots_bd/{name}_bonds.jpeg", dpi=300); plt.close()

    elapsed = time.time() - t0
    summary = {
        "name": name,
        "sigma": SIGMA,
        "lambda": LAMBDA,
        "eta": eta,
        "final_clusters": int(df["n_clusters"].iloc[-1]),
        "final_largest_cluster": int(df["largest_cluster"].iloc[-1]),
        "final_largest_fraction": float(df["largest_fraction"].iloc[-1]),
        "final_bonds": int(df["n_bonds"].iloc[-1]),
        "elapsed_seconds": elapsed
    }
    with open(f"data_bd/{name}_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    print(f"Finished {name} in {elapsed:.2f} s")

print("DONE")
