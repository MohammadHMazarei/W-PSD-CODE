"""
Improvement study: tests whether the two extensions proposed in the
Future Work paragraph -- (1) a Fused Gromov-Wasserstein coupling using
EXOGENOUS auxiliary geometry (pretrained word-vector distances between
entity names, common/aux_geometry.py) and (2) a debiased entropic
coupling (Feydy et al., 2019) -- actually improve on PSD-CODE, unlike
the plain OT/entropic/KL couplings analysed in Sections
Why the Coupling Update Adds No Signal and Dependence on the Co-occurrence Structure

Two evaluations, both reusing the exact masking/held-out protocol of
ablation/run_sparsity_ablation.py so results are directly comparable to
Table 4:

  1. Full-information run (no masking) at each of a small grid of
     hyperparameters, to find a reasonable operating point cheaply.
  2. The proper test: sparsity ablation at keep=0.30 (the ablation
     study's middle density), 5 seeds, paired t-test against PSD-CODE,
     for the best-performing FGW/debiased configurations from step 1
     plus a warm-up (curriculum) variant of the existing KL coupling.

Usage:
    python run_improvement_study.py --stage grid
    python run_improvement_study.py --stage ablation
"""
import argparse, json, os, sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from common.data import load_awa_dataset, build_marginals
from common.algorithm import RunConfig, run_alternating_optimisation
from common.metrics import embed, precision_at_k
from common.aux_geometry import (name_distance_matrix, normalise_structure,
                                 wordnet_distance_matrix)
from ablation.run_sparsity_ablation import mask_positives, heldout_precision

DEFAULT_DATA = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "data", "AwA2")
OUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "results")


def full_info_grid(data_dir, use_wordnet=False):
    counts, class_names, pred_names = load_awa_dataset(data_dir)
    Pobs, px, py = build_marginals(counts)
    n, m = Pobs.shape
    if use_wordnet:
        D_x = normalise_structure(wordnet_distance_matrix(class_names))
    else:
        D_x = normalise_structure(name_distance_matrix(class_names))
    D_y = normalise_structure(name_distance_matrix(pred_names))
    tag = "FGW-WordNet" if use_wordnet else "FGW"

    configs = [("PSD-CODE", dict(coupling="fixed", eps=0.0))]
    for alpha in (0.1, 0.3, 0.5, 0.7, 0.9):
        configs.append((f"{tag} (alpha={alpha})",
                        dict(coupling="fgw", fgw_D_x=D_x, fgw_D_y=D_y,
                             fgw_alpha=alpha)))
    for eps in (0.005, 0.01, 0.03, 0.05):
        configs.append((f"Debiased entropic (eps={eps})",
                        dict(coupling="entropic_debiased", eps=eps)))
    configs.append(("W-PSD-CODE (KL, eps=0.05) [existing]",
                    dict(coupling="kl", eps=0.05, rho=0.05)))
    configs.append(("W-PSD-CODE (KL, eps=0.05) + warmup=20",
                    dict(coupling="kl", eps=0.05, rho=0.05, warmup_iters=20)))

    rows = []
    for label, kw in configs:
        cfg = RunConfig(num_outer=60, q=2, init="identity", **kw)
        result = run_alternating_optimisation(Pobs, px, py, cfg)
        phi, psi = embed(result.G, n, m, cfg.q)
        p5 = precision_at_k(phi, psi, counts, k=5)
        p10 = precision_at_k(phi, psi, counts, k=10)
        p20 = precision_at_k(phi, psi, counts, k=20)
        print(f"{label:42s} P@5={p5:.4f}  P@10={p10:.4f}  P@20={p20:.4f}"
              f"  iters={result.iterations}  time={result.time_seconds:.2f}s")
        rows.append({"label": label, "P@5": p5, "P@10": p10, "P@20": p20,
                     "iterations": result.iterations})

    os.makedirs(OUT_DIR, exist_ok=True)
    out_name = "improvement_grid_wordnet.json" if use_wordnet else "improvement_grid.json"
    with open(os.path.join(OUT_DIR, out_name), "w") as f:
        json.dump(rows, f, indent=2)


def sparsity_test(data_dir, seeds, methods, keep, out_stem, use_wordnet=False):
    full, class_names, pred_names = load_awa_dataset(data_dir)
    if use_wordnet:
        D_x = normalise_structure(wordnet_distance_matrix(class_names))
    else:
        D_x = normalise_structure(name_distance_matrix(class_names))
    D_y = normalise_structure(name_distance_matrix(pred_names))

    resolved = []
    for label, kw in methods:
        kw = dict(kw)
        if kw.get("coupling") == "fgw":
            kw["fgw_D_x"] = D_x
            kw["fgw_D_y"] = D_y
        resolved.append((label, kw))

    acc = {}
    for seed in range(seeds):
        obs = mask_positives(full, keep, seed)
        Pobs, px, py = build_marginals(obs)
        n, m = Pobs.shape
        for label, kw in resolved:
            cfg = RunConfig(num_outer=60, q=2, init="identity", **kw)
            G = run_alternating_optimisation(Pobs, px, py, cfg).G
            phi, psi = embed(G, n, m, cfg.q)
            acc.setdefault(label, {"all": [], "held": []})
            acc[label]["all"].append(precision_at_k(phi, psi, full, 10))
            acc[label]["held"].append(heldout_precision(phi, psi, full, obs, 10))

    base_all = np.array(acc["PSD-CODE"]["all"])
    base_held = np.array(acc["PSD-CODE"]["held"])
    print(f"\n### keep={keep:.0%}, {seeds} masking seeds")
    for label in acc:
        av = np.array(acc[label]["all"]); hv = np.array(acc[label]["held"])
        if label == "PSD-CODE":
            p_all = p_held = None
        else:
            _, p_all = stats.ttest_rel(av, base_all)
            _, p_held = stats.ttest_rel(hv, base_held)
        print(f"  {label:42s} P@10 all = {av.mean():.4f}+/-{av.std():.4f} "
              f"(p={p_all})  held = {hv.mean():.4f}+/-{hv.std():.4f} (p={p_held})")

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, f"{out_stem}.json"), "w") as f:
        json.dump({k: {kk: [float(x) for x in vv] for kk, vv in v.items()}
                   for k, v in acc.items()}, f, indent=2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=DEFAULT_DATA)
    p.add_argument("--stage", choices=["grid", "ablation"], required=True)
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--keep", type=float, default=0.30)
    p.add_argument("--fgw-alpha", type=float, default=0.5)
    p.add_argument("--debiased-eps", type=float, default=0.01)
    p.add_argument("--out-stem", default="improvement_ablation")
    p.add_argument("--use-wordnet", action="store_true",
                    help="use WordNet taxonomic distance for D_x (classes) "
                         "instead of the generic word-vector distance")
    a = p.parse_args()

    if a.stage == "grid":
        full_info_grid(a.data_dir, use_wordnet=a.use_wordnet)
    else:
        methods = [
            ("PSD-CODE", dict(coupling="fixed", eps=0.0)),
            ("W-PSD-CODE (KL, eps=0.05) [existing]", dict(coupling="kl", eps=0.05, rho=0.05)),
            (f"FGW{'-WordNet' if a.use_wordnet else ''} (alpha={a.fgw_alpha})",
             dict(coupling="fgw", fgw_alpha=a.fgw_alpha)),
        ]
        sparsity_test(a.data_dir, a.seeds, methods, a.keep, a.out_stem,
                      use_wordnet=a.use_wordnet)


if __name__ == "__main__":
    main()
