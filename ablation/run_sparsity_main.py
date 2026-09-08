"""Main sparsity experiment for W-PSD-CODE.

Randomly retain a fraction of positive class--attribute pairs for training,
then evaluate retrieval against the complete AwA2 binary matrix. The held-out
P@10 score is the primary metric because it measures recovery of positives
that were not supplied to the model.

The experiment compares the fixed-coupling PSD-CODE baseline with the selected
KL-anchored W-PSD-CODE setting (rho=0.05, eps=0.05). Three masking seeds are
used at six retained-positive fractions with a 20-outer-iteration screening
budget.

Usage:
    python run_sparsity_main.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from common.algorithm import RunConfig, run_alternating_optimisation
from common.data import build_marginals, load_awa_dataset
from common.metrics import embed, l1_coupling_displacement, precision_at_k

OUTER_ITERS = 20
INNER_ITERS = 10

DEFAULT_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "AwA2")
KEEP_LEVELS = (0.05, 0.10, 0.15, 0.30, 0.50, 0.75)
METHODS = {
    "PSD-CODE": dict(coupling="fixed", eps=0.0),
    "W-PSD-CODE (KL, eps=0.05, rho=0.05)": dict(coupling="kl", eps=0.05, rho=0.05),
}


def mask_positives(full: np.ndarray, keep: float, seed: int) -> np.ndarray:
    """Retain a random fraction of positive pairs and enforce nonempty rows/cols."""
    rng = np.random.default_rng(seed)
    ones = np.argwhere(full > 0)
    selected = rng.random(len(ones)) < keep
    obs = np.zeros_like(full)
    obs[ones[selected, 0], ones[selected, 1]] = 1.0
    # Keep the KL reference well-defined by ensuring positive marginals.
    for i in range(full.shape[0]):
        if obs[i].sum() == 0:
            row_pos = ones[ones[:, 0] == i]
            obs[i, row_pos[0, 1]] = 1.0
    for j in range(full.shape[1]):
        if obs[:, j].sum() == 0:
            col_pos = ones[ones[:, 1] == j]
            obs[col_pos[0, 0], j] = 1.0
    return obs


def heldout_precision(phi: np.ndarray, psi: np.ndarray, full: np.ndarray,
                      obs: np.ndarray, k: int = 10) -> float:
    """P@k using only positive pairs that were absent from the training mask."""
    d2 = ((phi[:, None, :] - psi[None, :, :]) ** 2).sum(axis=-1)
    d2 = np.where(obs > 0, np.inf, d2)
    vals = []
    for x in range(full.shape[0]):
        heldout_pos = (full[x] > 0) & (obs[x] == 0)
        if not np.any(heldout_pos):
            continue
        topk = np.argsort(d2[x])[:k]
        vals.append(float(full[x, topk].sum() / k))
    return float(np.mean(vals))


def run_one(full: np.ndarray, obs: np.ndarray, cfg_kwargs: dict):
    Pobs, px, py = build_marginals(obs)
    n, m = Pobs.shape
    cfg = RunConfig(num_outer=OUTER_ITERS, num_inner=INNER_ITERS, q=2, init="identity", **cfg_kwargs)
    result = run_alternating_optimisation(Pobs, px, py, cfg)
    phi, psi = embed(result.G, n, m, cfg.q)
    return {
        "all_p10": precision_at_k(phi, psi, full, 10),
        "heldout_p10": heldout_precision(phi, psi, full, obs, 10),
        "l1_to_pobs": l1_coupling_displacement(result.Gamma, Pobs),
        "iterations": int(result.iterations),
        "runtime_seconds": float(result.time_seconds),
    }


def summarize(xs):
    arr = np.asarray(xs, dtype=float)
    return {"mean": float(arr.mean()), "std": float(arr.std(ddof=1)), "values": [float(x) for x in arr]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=DEFAULT_DATA)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--keep-levels", type=str, default=None, help="comma-separated kept-positive fractions, e.g. 0.05,0.10")
    ap.add_argument("--outer-iters", type=int, default=20)
    ap.add_argument("--inner-iters", type=int, default=10)
    ap.add_argument("--out-dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results"))
    args = ap.parse_args()

    full, _, _ = load_awa_dataset(args.data_dir)
    keep_levels = tuple(float(x) for x in args.keep_levels.split(",")) if args.keep_levels else KEEP_LEVELS
    total_pos = int((full > 0).sum())
    global OUTER_ITERS, INNER_ITERS
    OUTER_ITERS = args.outer_iters
    INNER_ITERS = args.inner_iters

    out = {
        "dataset_shape": list(full.shape),
        "full_positive_pairs": total_pos,
        "full_density": float((full > 0).mean()),
        "keep_levels": list(keep_levels),
        "seeds": list(range(args.seeds)),
        "protocol": "random masking of positive pairs; held-out P@10 is primary",
        "outer_iterations": args.outer_iters,
        "inner_iterations": args.inner_iters,
        "methods": list(METHODS.keys()),
        "results": {},
    }

    for keep in keep_levels:
        keep_key = f"{keep:.2f}"
        out["results"][keep_key] = {
            "kept_positive_fraction": keep,
            "training_positive_count_mean": None,
            "training_density": None,
            "methods": {},
        }
        seed_counts = []
        seed_densities = []
        for seed in range(args.seeds):
            obs = mask_positives(full, keep, seed)
            seed_counts.append(int((obs > 0).sum()))
            seed_densities.append(float((obs > 0).mean()))
            print(f"keep={keep:.0%} seed={seed+1}/{args.seeds}: {seed_counts[-1]} positives")
            for label, cfg_kwargs in METHODS.items():
                t0 = time.time()
                metrics = run_one(full, obs, cfg_kwargs)
                metrics["wall_clock_seconds"] = float(time.time() - t0)
                bucket = out["results"][keep_key]["methods"].setdefault(label, {
                    "all_p10": [], "heldout_p10": [], "l1_to_pobs": [],
                    "iterations": [], "runtime_seconds": []
                })
                for key in ("all_p10", "heldout_p10", "l1_to_pobs", "iterations", "runtime_seconds"):
                    bucket[key].append(metrics[key])
                print(f"  {label}: all={metrics['all_p10']:.4f} held={metrics['heldout_p10']:.4f} L1={metrics['l1_to_pobs']:.4f}")
        out["results"][keep_key]["training_positive_count_mean"] = float(np.mean(seed_counts))
        out["results"][keep_key]["training_density"] = float(np.mean(seed_densities))

    # Add mean/std summaries alongside raw values.
    for cell in out["results"].values():
        for method_bucket in cell["methods"].values():
            for key in ("all_p10", "heldout_p10", "l1_to_pobs", "runtime_seconds"):
                method_bucket[key + "_summary"] = summarize(method_bucket[key])
            method_bucket["iterations_mean"] = float(np.mean(method_bucket["iterations"]))

    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, "sparsity_main.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {path}")


if __name__ == "__main__":
    main()
