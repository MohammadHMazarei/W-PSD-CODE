"""
Sparsity ablation (Table 3).

The motivation for a geometry-aware coupling is that in SPARSE
co-occurrence data the empirical coupling carries little information about
unobserved pairs.  This script tests that claim directly: a fraction
``keep`` of the observed positive pairs is retained for training and the
rest is masked out, and retrieval is then evaluated against the FULL
ground-truth attribute matrix.  Results are averaged over several masking
seeds so that the reported differences can be compared against their own
run-to-run standard deviation.

Usage:
    python run_sparsity_ablation.py [--data-dir DIR] [--seeds 3]
"""
import argparse, json, os, sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from common.data import load_awa_dataset, build_marginals
from common.algorithm import RunConfig, run_alternating_optimisation
from common.metrics import embed, precision_at_k

DEFAULT_DATA = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "data", "AwA2")

METHODS = [
    ("PSD-CODE", dict(coupling="fixed", eps=0.0)),
    ("W-PSD-CODE (entropic)", dict(coupling="entropic", eps=0.01)),
    ("W-PSD-CODE (KL, eps=0.03)", dict(coupling="kl", eps=0.03, rho=0.05)),
    ("W-PSD-CODE (KL, eps=0.05)", dict(coupling="kl", eps=0.05, rho=0.05)),
]


def mask_positives(full, keep, seed):
    """Keep a random ``keep`` fraction of the positive pairs; never leave a
    row or column empty, so the marginals stay strictly positive."""
    rng = np.random.default_rng(seed)
    ones = np.argwhere(full > 0)
    sel = rng.random(len(ones)) < keep
    obs = np.zeros_like(full)
    obs[ones[sel, 0], ones[sel, 1]] = 1.0
    for i in range(full.shape[0]):
        if obs[i].sum() == 0:
            obs[i, ones[ones[:, 0] == i][0, 1]] = 1.0
    for j in range(full.shape[1]):
        if obs[:, j].sum() == 0:
            cand = ones[ones[:, 1] == j]
            if len(cand):
                obs[cand[0, 0], j] = 1.0
    return obs


def heldout_precision(phi, psi, full, obs, k=10):
    """P@k restricted to pairs that were NOT observed during training."""
    d2 = ((phi[:, None, :] - psi[None, :, :]) ** 2).sum(-1)
    d2 = np.where(obs > 0, np.inf, d2)
    hits = []
    for x in range(full.shape[0]):
        if ((full[x] > 0) & (obs[x] == 0)).sum() == 0:
            continue
        hits.append(full[x, np.argsort(d2[x])[:k]].sum() / k)
    return float(np.mean(hits))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=DEFAULT_DATA)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--out-dir", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "results"))
    a = p.parse_args()

    full, _, _ = load_awa_dataset(a.data_dir)
    table = {}

    for keep in (0.15, 0.30, 0.50):
        acc = {}
        for seed in range(a.seeds):
            obs = mask_positives(full, keep, seed)
            Pobs, px, py = build_marginals(obs)
            n, m = Pobs.shape
            for label, kw in METHODS:
                cfg = RunConfig(num_outer=60, q=2, init="identity", **kw)
                G = run_alternating_optimisation(Pobs, px, py, cfg).G
                phi, psi = embed(G, n, m, cfg.q)
                acc.setdefault(label, {"all": [], "held": []})
                acc[label]["all"].append(precision_at_k(phi, psi, full, 10))
                acc[label]["held"].append(
                    heldout_precision(phi, psi, full, obs, 10))
        density = float((mask_positives(full, keep, 0) > 0).mean())
        print(f"\n### keep={keep:.0%}  (training density {density:.1%}), q=2, "
              f"{a.seeds} masking seeds")
        for label in acc:
            av, hv = acc[label]["all"], acc[label]["held"]
            print(f"  {label:28s} P@10 all = {np.mean(av):.4f} +/- "
                  f"{np.std(av):.4f}   held-out = {np.mean(hv):.4f} +/- "
                  f"{np.std(hv):.4f}")
        table[f"{keep:.2f}"] = {
            "density": density,
            "results": {k: {kk: [float(x) for x in vv]
                            for kk, vv in v.items()} for k, v in acc.items()},
        }

    os.makedirs(a.out_dir, exist_ok=True)
    with open(os.path.join(a.out_dir, "sparsity_ablation.json"), "w") as f:
        json.dump(table, f, indent=2)


if __name__ == "__main__":
    main()
