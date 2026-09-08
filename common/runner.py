"""Shared driver used by every run_*.py script."""
from __future__ import annotations

import os

from .data import load_awa_dataset, build_marginals
from .algorithm import run_alternating_optimisation
from .metrics import (embed, precision_at_k, mean_per_animal_wasserstein,
                      l1_coupling_displacement)
from .report import save_json


def run_and_report(cfg, label, data_dir, out_dir, stem, save_arrays=False):
    counts, class_names, pred_names = load_awa_dataset(data_dir)
    Pobs, pbar_x, pbar_y = build_marginals(counts)
    n, m = Pobs.shape

    result = run_alternating_optimisation(Pobs, pbar_x, pbar_y, cfg)
    phi, psi = embed(result.G, n, m, cfg.q)

    out = {
        "Method": label,
        "Mean P@5": precision_at_k(phi, psi, counts, k=5),
        "Mean P@10": precision_at_k(phi, psi, counts, k=10),
        "Mean P@20": precision_at_k(phi, psi, counts, k=20),
        "Mean Delta_x (W2)": mean_per_animal_wasserstein(
            result.G, Pobs, pbar_x, pbar_y, n, m, cfg.q),
        "L1(Gamma,Pobs)": l1_coupling_displacement(result.Gamma, Pobs),
        "Iterations": result.iterations,
        "Time (s)": result.time_seconds,
        "history": result.history,
    }
    os.makedirs(out_dir, exist_ok=True)
    save_json(out, os.path.join(out_dir, f"{stem}.json"))
    if save_arrays:
        import numpy as np
        np.save(os.path.join(out_dir, f"{stem}_G.npy"), result.G)
        np.save(os.path.join(out_dir, f"{stem}_Gamma.npy"), result.Gamma)
    print({k: v for k, v in out.items() if k != "history"})
    return out
