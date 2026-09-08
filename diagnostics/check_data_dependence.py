"""
Diagnostic / Remark rem:marginal_only of the paper.

CLAIM.  If the coupling term of the free energy is the plain entropy
-eps H(Gamma) (or eps = 0, exact OT), then the entire objective

    J(G,Gamma) = <Gamma,C(G)> + sum_x pbar_x log Z_G(x)
                 + lambda tr(G) - eps H(Gamma)

depends on the observed co-occurrence matrix Pobs ONLY through its
marginals pbar_x, pbar_y.  Consequently the algorithm, started from
G^(0) = I_N as stated in Algorithm 1, returns exactly the same Gram
matrix for Pobs and for ANY other coupling with the same marginals.

TEST.  Run the algorithm twice, once on the real AwA2 matrix and once on
the independence coupling pbar_x (x) pbar_y, which has identical
marginals but no co-occurrence structure whatsoever, and compare the
resulting Gram matrices.

EXPECTED.  Difference exactly 0 for coupling="exact" and "entropic";
strictly positive for the KL-anchored coupling used by W-PSD-CODE.

Usage:
    python check_data_dependence.py [--data-dir DIR]
"""
import argparse, os, sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from common.data import load_awa_dataset, build_marginals
from common.algorithm import RunConfig, run_alternating_optimisation

DEFAULT_DATA = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "data", "AwA2")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=DEFAULT_DATA)
    p.add_argument("--num-outer", type=int, default=35)
    a = p.parse_args()

    counts, _, _ = load_awa_dataset(a.data_dir)
    Pobs, pbar_x, pbar_y = build_marginals(counts)

    # A different coupling with exactly the same marginals.
    Pindep = np.outer(pbar_x, pbar_y)
    assert np.allclose(Pindep.sum(1), pbar_x)
    assert np.allclose(Pindep.sum(0), pbar_y)

    settings = [
        ("exact OT      (eps=0)      ", dict(coupling="exact", eps=0.0)),
        ("entropic      (eps=0.01)   ", dict(coupling="entropic", eps=0.01)),
        ("KL-anchored   (eps=0.05)   ", dict(coupling="kl", eps=0.05, rho=0.05)),
    ]

    print(f"{'coupling update':30s} max |G(Pobs) - G(independence)|")
    print("-" * 62)
    for label, kw in settings:
        cfg = RunConfig(num_outer=a.num_outer, init="identity", **kw)
        G_data = run_alternating_optimisation(Pobs, pbar_x, pbar_y, cfg).G
        G_null = run_alternating_optimisation(Pindep, pbar_x, pbar_y, cfg).G
        print(f"{label:30s} {np.abs(G_data - G_null).max():.3e}")


if __name__ == "__main__":
    main()
