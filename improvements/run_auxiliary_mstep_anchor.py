"""Reproduces Table 10: "A Second Attempted Fix: Auxiliary M-Step Anchoring".

Adds mu * ||G_XX - K_x||_F^2 to the free energy and 2*mu*(G_XX - K_x) to
the corresponding block of the M-step gradient, where K_x is a
class-similarity kernel built from WordNet taxonomic path-distance
between AwA2 class names (common/aux_geometry.py -- the same auxiliary
source used for the FGW extension in run_improvement_study.py, here
reused as an M-step regulariser instead of an OT cost). mu=0 reproduces
Table 2 exactly; every other term, and the E-step, are unchanged.

This does NOT modify common/objective.py, common/psd_utils.py, or
common/algorithm.py (which every other already-verified table depends
on with mu implicitly 0): the anchored free energy, gradient, and
projected-gradient step are self-contained wrappers defined below that
call straight through to the unmodified shared functions when mu=0.

Usage:
    python run_auxiliary_mstep_anchor.py [--data-dir DIR] [--out-dir DIR]
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CODE_ROOT = os.path.join(HERE, "..")
sys.path.insert(0, CODE_ROOT)

from common.data import load_awa_dataset, build_marginals
from common.aux_geometry import wordnet_taxonomic_distance
from common.objective import free_energy, gradient
from common.psd_utils import psd_project, init_gram
from common.ot_solvers import kl_ot_coupling, reference_coupling, log_reference
from common.metrics import embed, precision_at_k

DATA_DIR = os.path.join(CODE_ROOT, "data", "AwA2")
OUT_DIR = os.path.join(CODE_ROOT, "results")

MU_GRID = [0.0, 1e-4, 0.01, 0.05, 0.1, 0.5]


def build_class_similarity_kernel(class_names, target_diag_scale):
    """K_x: WordNet taxonomic similarity between AwA2 class names,
    scaled to match the magnitude the unregularised G_XX already
    converges to (paper's Section "Auxiliary M-Step Anchoring").

    wordnet_taxonomic_distance returns a DISTANCE matrix in [0, 1]
    (0 = identical synset). We convert to a similarity (1 - distance,
    diagonal exactly 1) and then rescale by ``target_diag_scale`` -- the
    mean diagonal entry of the unregularised (mu=0) G_XX, ~0.991 per the
    manuscript -- so K_x's diagonal sits at the same magnitude as the
    quantity it is regularising toward, rather than at an arbitrary
    scale of 1.
    """
    D = wordnet_taxonomic_distance(class_names)
    K_raw = 1.0 - D
    np.fill_diagonal(K_raw, 1.0)
    return K_raw * target_diag_scale


def free_energy_with_anchor(G, Gamma, pbar_x, pbar_y, lam, eps, n, m, mu,
                             K_x, log_Q=None):
    base = free_energy(G, Gamma, pbar_x, pbar_y, lam, eps, n, m, log_Q)
    if mu <= 0:
        return base
    G_XX = G[:n, :n]
    return base + mu * float(np.sum((G_XX - K_x) ** 2))


def gradient_with_anchor(G, Gamma, pbar_x, pbar_y, lam, n, m, mu, K_x):
    grad = gradient(G, Gamma, pbar_x, pbar_y, lam, n, m).copy()
    if mu > 0:
        G_XX = G[:n, :n]
        grad[:n, :n] += 2.0 * mu * (G_XX - K_x)
    return grad


def pgd_step_with_anchor(G, Gamma, pbar_x, pbar_y, lam, eps, n, m, alpha,
                          mu, K_x, log_Q=None, c1=1e-4, max_backtrack=25):
    """Same Armijo-backtracked projected-gradient step as
    common.psd_utils.pgd_step, with the mu * ||G_XX - K_x||^2 anchor
    term added to both the objective used for the backtracking line
    search and the gradient. Identical to pgd_step when mu=0."""
    J0 = free_energy_with_anchor(G, Gamma, pbar_x, pbar_y, lam, eps, n, m,
                                  mu, K_x, log_Q)
    grad = gradient_with_anchor(G, Gamma, pbar_x, pbar_y, lam, n, m, mu, K_x)
    grad_sq_norm = float(np.sum(grad * grad))

    step = alpha
    G_new = psd_project(G - step * grad)
    for _ in range(max_backtrack):
        J_new = free_energy_with_anchor(G_new, Gamma, pbar_x, pbar_y, lam,
                                         eps, n, m, mu, K_x, log_Q)
        if J_new <= J0 - c1 * step * grad_sq_norm or grad_sq_norm == 0.0:
            return G_new, step, J_new
        step *= 0.5
        G_new = psd_project(G - step * grad)

    J_new = free_energy_with_anchor(G_new, Gamma, pbar_x, pbar_y, lam, eps,
                                     n, m, mu, K_x, log_Q)
    return G_new, step, J_new


def run_anchored(Pobs, pbar_x, pbar_y, coupling, mu, K_x, lam=0.001,
                  eps=0.05, rho=0.05, q=2, alpha0=0.02, decay=0.99,
                  num_outer=60, num_inner=10, tau=1e-8):
    """Minimal re-implementation of common.algorithm.run_alternating_
    optimisation restricted to coupling in {"fixed", "kl"} (the two
    couplings tested in Table 10), with the mu-anchored M-step above.
    Behaves identically to the shared implementation when mu=0.
    """
    if coupling not in ("fixed", "kl"):
        raise ValueError("this script only supports coupling in "
                          "{'fixed', 'kl'}, matching Table 10")

    n, m = Pobs.shape
    log_Q = None
    if coupling == "kl":
        log_Q = log_reference(reference_coupling(Pobs, pbar_x, pbar_y, rho))
        eps_used = eps
    else:
        eps_used = 0.0

    G = init_gram(Pobs, mode="identity")
    Gamma = Pobs.copy()
    alpha = alpha0
    prev_J = None

    for k in range(num_outer):
        from common.objective import cost_matrix
        C = cost_matrix(G, n, m)
        if coupling == "kl":
            Gamma = kl_ot_coupling(pbar_x, pbar_y, C, eps_used, log_Q)

        for _ in range(num_inner):
            G, alpha, J = pgd_step_with_anchor(
                G, Gamma, pbar_x, pbar_y, lam, eps_used, n, m, alpha,
                mu, K_x, log_Q)

        J = free_energy_with_anchor(G, Gamma, pbar_x, pbar_y, lam,
                                     eps_used, n, m, mu, K_x, log_Q)
        alpha *= decay
        if prev_J is not None and abs(J - prev_J) / (1.0 + abs(prev_J)) < tau:
            break
        prev_J = J

    return G, Gamma


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=DATA_DIR)
    p.add_argument("--out-dir", default=OUT_DIR)
    p.add_argument("--mu", type=float, nargs="+", default=MU_GRID)
    p.add_argument("--k", type=int, default=10)
    a = p.parse_args()

    counts, class_names, pred_names = load_awa_dataset(a.data_dir)
    Pobs, pbar_x, pbar_y = build_marginals(counts)
    n, m = Pobs.shape

    # Unregularised (mu=0) run first, to find the diagonal magnitude that
    # G_XX already converges to, and to scale K_x to match it (see
    # build_class_similarity_kernel's docstring).
    G0, _ = run_anchored(Pobs, pbar_x, pbar_y, "fixed", 0.0, None)
    target_scale = float(np.mean(np.diag(G0[:n, :n])))
    print(f"Unregularised G_XX diagonal mean: {target_scale:.6f} "
          f"(paper reports 0.991)")

    K_x = build_class_similarity_kernel(class_names, target_scale)

    results = {"fixed": {}, "kl": {}}
    for coupling, row_label in [("fixed", "Fixed (PSD-CODE)"),
                                 ("kl", "KL-anchored")]:
        for mu in a.mu:
            G, Gamma = run_anchored(Pobs, pbar_x, pbar_y, coupling, mu, K_x)
            phi, psi = embed(G, n, m, q=2)
            p_at_k = precision_at_k(phi, psi, counts, k=a.k)
            results[coupling][str(mu)] = p_at_k
            print(f"{row_label:20s} mu={mu:<8g} P@{a.k}={p_at_k:.3f}")

    os.makedirs(a.out_dir, exist_ok=True)
    out_path = os.path.join(a.out_dir, "auxiliary_mstep.json")
    with open(out_path, "w") as f:
        json.dump({
            "description": "Table 10: P@10 under auxiliary M-step "
                            "anchoring toward a WordNet-taxonomic class "
                            "kernel K_x, at increasing anchor strength mu.",
            "target_diag_scale": target_scale,
            "mu_grid": a.mu,
            "k": a.k,
            "results": results,
        }, f, indent=2)
    print(f"\nSaved {out_path}")

    tex_path = os.path.join(a.out_dir, "table_auxiliary_mstep.tex")
    with open(tex_path, "w") as f:
        f.write("\\begin{tabular}{l" + "c" * len(a.mu) + "}\n\\toprule\n")
        f.write("Coupling & " + " & ".join(f"${mu:g}$" for mu in a.mu) +
                " \\\\\n\\midrule\n")
        for coupling, row_label in [("fixed", "Fixed (PSD-CODE)"),
                                     ("kl", "KL-anchored")]:
            vals = " & ".join(f"{results[coupling][str(mu)]:.3f}"
                               for mu in a.mu)
            f.write(f"{row_label} & {vals} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    print(f"Saved {tex_path}")


if __name__ == "__main__":
    main()
