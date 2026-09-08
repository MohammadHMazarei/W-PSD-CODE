"""Reproduces Table 3: "Isolating the coupling from the embedding".

Runs the same four methods and configurations as Table 2 (run_all.py /
psd_code, wpsd_code_exact_ot, wpsd_code_sinkhorn, wpsd_code_kl) but adds
a second readout for each: rank attributes directly by the final
learned coupling Gamma*(x, .), with no embedding step at all, instead
of by embedding distance. This isolates whether a method's failure (or
success) comes from what the M-step does with the coupling, or from the
coupling itself having become uninformative.

For PSD-CODE and the KL-anchored variant, Gamma* is at or extremely
close to Pobs itself, so the coupling-only column is close to reading
off the ground-truth matrix directly and is reported for completeness
rather than as a non-trivial diagnostic (see the manuscript's footnote
on Table 3). The exact-OT and entropic rows, where Gamma* has moved
substantially from Pobs, are the informative comparisons.

Usage:
    python run_isolate_mechanism.py [--data-dir DIR] [--out-dir DIR]
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CODE_ROOT = os.path.join(HERE, "..")
sys.path.insert(0, CODE_ROOT)

from common.data import load_awa_dataset, build_marginals
from common.algorithm import RunConfig, run_alternating_optimisation
from common.metrics import embed, precision_at_k, precision_at_k_from_scores

DATA_DIR = os.path.join(CODE_ROOT, "data", "AwA2")
OUT_DIR = os.path.join(CODE_ROOT, "results")

# Exactly the four configurations used for Table 2 in
# psd_code/run_psd_code.py, wpsd_code_exact_ot/run_wpsd_exact.py,
# wpsd_code_sinkhorn/run_wpsd_sinkhorn.py, wpsd_code_kl/run_wpsd_kl.py.
CONFIGS = [
    ("PSD-CODE (fixed-lambda baseline)",
     RunConfig(lam=0.001, eps=0.0, q=2, alpha0=0.02, decay=0.99,
               num_outer=60, num_inner=10, coupling="fixed")),
    ("W-PSD-CODE (exact OT)",
     RunConfig(lam=0.001, eps=0.0, q=2, alpha0=0.02, decay=0.99,
               num_outer=60, num_inner=10, coupling="exact")),
    ("W-PSD-CODE (entropic)",
     RunConfig(lam=0.001, eps=0.01, q=2, alpha0=0.02, decay=0.99,
               num_outer=60, num_inner=10, coupling="entropic")),
    ("W-PSD-CODE (KL-anchored)",
     RunConfig(lam=0.001, eps=0.05, rho=0.05, q=2, alpha0=0.02, decay=0.99,
               num_outer=60, num_inner=10, coupling="kl")),
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=DATA_DIR)
    p.add_argument("--out-dir", default=OUT_DIR)
    p.add_argument("--k", type=int, default=10)
    a = p.parse_args()

    counts, class_names, pred_names = load_awa_dataset(a.data_dir)
    Pobs, pbar_x, pbar_y = build_marginals(counts)
    n, m = Pobs.shape

    rows = []
    for label, cfg in CONFIGS:
        result = run_alternating_optimisation(Pobs, pbar_x, pbar_y, cfg)
        phi, psi = embed(result.G, n, m, cfg.q)

        embedding_p_at_k = precision_at_k(phi, psi, counts, k=a.k)
        coupling_only_p_at_k = precision_at_k_from_scores(
            result.Gamma, counts, k=a.k)
        l1_gamma_pobs = float(abs(result.Gamma - Pobs).sum())

        row = {
            "Method": label,
            f"Embedding P@{a.k}": embedding_p_at_k,
            f"Coupling-only P@{a.k}": coupling_only_p_at_k,
            "L1(Gamma,Pobs)": l1_gamma_pobs,
        }
        rows.append(row)
        print(f"{label:38s} Embedding P@{a.k}={embedding_p_at_k:.3f}  "
              f"Coupling-only P@{a.k}={coupling_only_p_at_k:.3f}  "
              f"L1(Gamma,Pobs)={l1_gamma_pobs:.4f}")

    os.makedirs(a.out_dir, exist_ok=True)
    out_path = os.path.join(a.out_dir, "isolate_mechanism.json")
    with open(out_path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nSaved {out_path}")

    # Also emit a LaTeX tabular snippet matching Table 3's layout, purely
    # as a visual cross-check against the manuscript (not \input by it).
    tex_path = os.path.join(a.out_dir, "table_isolate_mechanism.tex")
    with open(tex_path, "w") as f:
        f.write("\\begin{tabular}{lcc}\n\\toprule\n")
        f.write(f"Method & Embedding P@{a.k} & Coupling-only P@{a.k} \\\\\n")
        f.write("\\midrule\n")
        for row in rows:
            f.write(f"{row['Method']} & {row[f'Embedding P@{a.k}']:.3f} & "
                     f"{row[f'Coupling-only P@{a.k}']:.3f} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    print(f"Saved {tex_path}")


if __name__ == "__main__":
    main()
