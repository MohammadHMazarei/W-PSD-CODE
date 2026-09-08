"""Full-budget confirmation for representative rho-epsilon settings.

Usage:
    python code/ablation/run_rho_epsilon_confirmation.py
"""
from __future__ import annotations
import csv, json, os, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from common.algorithm import RunConfig, run_alternating_optimisation
from common.data import build_marginals, load_awa_dataset
from common.metrics import embed, l1_coupling_displacement, mean_per_animal_wasserstein, precision_at_k

CONFIGS = [
    (0.005, 0.200, "strong-anchoring"),
    (0.050, 0.050, "paper-setting"),
    (0.500, 0.010, "high-drift-grid-best"),
    (0.500, 0.005, "high-rho-low-epsilon"),
]

def main():
    counts, _, _ = load_awa_dataset(str(HERE / "data" / "AwA2"))
    Pobs, px, py = build_marginals(counts)
    n, m = Pobs.shape
    rows=[]
    for rho, eps, label in CONFIGS:
        cfg=RunConfig(lam=0.001,eps=eps,rho=rho,q=2,alpha0=0.02,decay=0.99,
                      num_outer=60,num_inner=10,tau=1e-8,coupling="kl",init="identity")
        r=run_alternating_optimisation(Pobs,px,py,cfg)
        phi,psi=embed(r.G,n,m,2)
        row={"setting":label,"rho":rho,"epsilon":eps,
             "p_at_5":precision_at_k(phi,psi,counts,5),
             "p_at_10":precision_at_k(phi,psi,counts,10),
             "p_at_20":precision_at_k(phi,psi,counts,20),
             "mean_delta_x_w2":mean_per_animal_wasserstein(r.G,Pobs,px,py,n,m,2),
             "l1_gamma_pobs":l1_coupling_displacement(r.Gamma,Pobs),
             "iterations":r.iterations,"time_seconds":r.time_seconds,
             "final_free_energy":r.history[-1]}
        rows.append(row); print(row)
    out=HERE/"results"; out.mkdir(exist_ok=True)
    (out/"rho_epsilon_confirmation.json").write_text(json.dumps(rows,indent=2))
    with (out/"rho_epsilon_confirmation.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

if __name__=="__main__": main()
