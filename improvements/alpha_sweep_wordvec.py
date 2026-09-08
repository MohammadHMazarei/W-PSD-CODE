import sys, os, json
import numpy as np
from scipy import stats
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from common.data import load_awa_dataset, build_marginals
from common.algorithm import RunConfig, run_alternating_optimisation
from common.metrics import embed, precision_at_k
from common.aux_geometry import name_distance_matrix, normalise_structure
from ablation.run_sparsity_ablation import mask_positives, heldout_precision

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "AwA2")
full, class_names, pred_names = load_awa_dataset(DATA_DIR)
D_x = normalise_structure(name_distance_matrix(class_names))
D_y = normalise_structure(name_distance_matrix(pred_names))

seeds = 3
keep = 0.30
base_held = []
alpha_results = {a: [] for a in (0.1, 0.2, 0.3, 0.4, 0.5)}

for seed in range(seeds):
    obs = mask_positives(full, keep, seed)
    Pobs, px, py = build_marginals(obs)
    n, m = Pobs.shape

    cfg = RunConfig(num_outer=60, q=2, init="identity", coupling="fixed", eps=0.0)
    G = run_alternating_optimisation(Pobs, px, py, cfg).G
    phi, psi = embed(G, n, m, 2)
    base_held.append(heldout_precision(phi, psi, full, obs, 10))

    for a in alpha_results:
        cfg = RunConfig(num_outer=60, q=2, init="identity", coupling="fgw",
                        fgw_D_x=D_x, fgw_D_y=D_y, fgw_alpha=a)
        G = run_alternating_optimisation(Pobs, px, py, cfg).G
        phi, psi = embed(G, n, m, 2)
        alpha_results[a].append(heldout_precision(phi, psi, full, obs, 10))

base_held = np.array(base_held)
print(f"PSD-CODE held-out P@10: {base_held.mean():.4f} +/- {base_held.std():.4f}")
for a, vals in alpha_results.items():
    v = np.array(vals)
    _, p = stats.ttest_rel(v, base_held)
    print(f"FGW alpha={a}: held-out P@10 = {v.mean():.4f} +/- {v.std():.4f}  (p={p:.4f})")
