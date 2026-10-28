#!/usr/bin/env python3
import os, json, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

N_R=24
N_C=48
V_R=8
V_E=6

def union_find(n, edges):
    parent=list(range(n))
    def find(x):
        while parent[x]!=x:
            parent[x]=parent[parent[x]]
            x=parent[x]
        return x
    def union(a,b):
        ra,rb=find(a),find(b)
        if ra!=rb: parent[rb]=ra
    for a,b in edges: union(a,b)
    groups={}
    for i in range(n):
        groups.setdefault(find(i),[]).append(i)
    sizes=sorted((len(v) for v in groups.values()),reverse=True)
    return len(sizes), sizes[0], sizes[0]/n

def simulate_set(cfg, reps=200):
    rng=np.random.default_rng(cfg.get("SEED",1))
    p=float(cfg["p"])
    rows=[]
    # Stub matching with finite site valency:
    # each Rubisco has 8 available sites; each EPYC1 has 6.
    for rep in range(reps):
        r_sites=np.repeat(np.arange(N_R),V_R)
        e_sites=np.repeat(np.arange(N_C),V_E)
        rng.shuffle(r_sites)
        rng.shuffle(e_sites)
        m=min(len(r_sites),len(e_sites))
        active=rng.random(m)<p
        pairs=list(zip(r_sites[:m][active], e_sites[:m][active]))
        # remove duplicate molecule-level edges for graph connectivity
        edges={(int(r),N_R+int(e)) for r,e in pairs}
        ncl,largest,frac=union_find(N_R+N_C,edges)
        rows.append({
            "replicate":rep,
            "p_site_occupancy":p,
            "occupied_site_pairs":int(active.sum()),
            "unique_edges":len(edges),
            "clusters":ncl,
            "largest":largest,
            "fraction":frac
        })
    return pd.DataFrame(rows)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--params",default="percolation_params.json")
    ap.add_argument("--reps",type=int,default=500)
    args=ap.parse_args()
    cfgs=json.load(open(args.params))
    os.makedirs("data",exist_ok=True); os.makedirs("plots",exist_ok=True)
    summary=[]
    for cfg in cfgs:
        df=simulate_set(cfg,args.reps)
        name=cfg["name"]
        df.to_csv(f"data/{name}.csv",index=False)
        summary.append({
            "name":name,"p":cfg["p"],
            "mean_fraction":df["fraction"].mean(),
            "sd_fraction":df["fraction"].std(),
            "mean_clusters":df["clusters"].mean(),
            "mean_largest":df["largest"].mean()
        })
        for key in ["fraction","clusters","largest","unique_edges"]:
            plt.figure()
            plt.hist(df[key],bins=30)
            plt.xlabel(key); plt.ylabel("replicates"); plt.title(f"{name}: {key}")
            plt.tight_layout(); plt.savefig(f"plots/{name}_{key}.png",dpi=180); plt.close()
    sdf=pd.DataFrame(summary)
    sdf.to_csv("data/summary.csv",index=False)
    plt.figure()
    plt.plot(sdf["p"],sdf["mean_fraction"],marker="o")
    plt.xlabel("site occupation probability p")
    plt.ylabel("mean largest cluster fraction")
    plt.tight_layout(); plt.savefig("plots/percolation_curve.png",dpi=180); plt.close()

if __name__=="__main__":
    main()
