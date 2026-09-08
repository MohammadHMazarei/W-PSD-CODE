"""
Held-out attribute-split experiment.

Exogenous M-step anchoring with WordNet taxonomy found that
an exogenous anchor can be genuinely independent of the co-occurrence
matrix and still fail, because taxonomic proximity is not the same
quantity as attribute-sharing. This script tests the fix that finding
implies: an exogenous anchor built from attribute annotations, which
IS directly relevant to attribute-sharing by construction, but from a
set of attributes disjoint from the ones used to fit and evaluate the
model, so the anchor is still independent of the evaluation target,
not circular.

Design
------
AwA2's 85 attributes are split into two conventional groups (following
the category structure introduced with the original Animals-with-
Attributes benchmark):

    APPEARANCE   (33 attrs): colour, texture/pattern, size/shape, body
                 parts, teeth/horns/claws (predicate indices 1-33)
    ECOLOGICAL   (52 attrs): locomotion/behaviour, diet, habitat,
                 character/social (predicate indices 34-85)

We run this in both directions:

  Direction A: fit & evaluate P@k on APPEARANCE attributes; build the
               exogenous class-similarity kernel K_x from ECOLOGICAL
               attributes only (cosine similarity of class profiles
               restricted to that held-out set).
  Direction B: fit & evaluate on ECOLOGICAL; anchor from APPEARANCE.

This is a genuinely different question: does ecological
similarity between species help predict shared visual appearance (and
vice versa)? Because K_x never sees the attributes used for fitting or
evaluation, this is not circular, unlike ranking Gamma against Pobs
directly, and not obviously mismatched to the task the way taxonomy was.

Usage:
    python run_heldout_attribute_split.py
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CODE_ROOT = os.path.join(HERE, "..")
sys.path.insert(0, CODE_ROOT)

from common.data import load_awa_dataset, build_marginals
from common.algorithm import RunConfig
from common.objective import free_energy, gradient, cost_matrix
from common.psd_utils import psd_project, init_gram
from common.ot_solvers import kl_ot_coupling, reference_coupling, log_reference
from common.metrics import embed, precision_at_k

DATA_DIR = os.path.join(CODE_ROOT, "data", "AwA2")
OUT_DIR = os.path.join(CODE_ROOT, "results")

# 1-indexed predicate ids from predicates.txt, converted to 0-indexed below.
APPEARANCE_IDS_1INDEXED = list(range(1, 34))   # black ... tusks (33 attrs)
ECOLOGICAL_IDS_1INDEXED = list(range(34, 86))  # smelly ... domestic (52 attrs)


def build_cosine_kernel(counts_exo: np.ndarray, base_scale: float) -> np.ndarray:
    """Cosine similarity between class rows, restricted to the held-out
    attribute set, rescaled to the magnitude the unregularised G_XX
    already converges to."""
    X = counts_exo.astype(float)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    Xn = X / norms
    sim = Xn @ Xn.T  # in [0, 1] for non-negative binary vectors
    n = X.shape[0]
    K = base_scale * sim
    np.fill_diagonal(K, base_scale)
    return K


def run_with_exo_anchor(Pobs, pbar_x, pbar_y, cfg: RunConfig, K_x, mu: float):
    n, m = Pobs.shape
    log_Q = None
    if cfg.coupling == "kl":
        log_Q = log_reference(reference_coupling(Pobs, pbar_x, pbar_y, cfg.rho))

    G = init_gram(Pobs, mode="identity")
    Gamma = Pobs.copy()
    alpha = cfg.alpha0

    def exo_penalty(Gmat):
        diff = Gmat[:n, :n] - K_x
        return mu * float(np.sum(diff * diff))

    for k in range(cfg.num_outer):
        C = cost_matrix(G, n, m)
        if cfg.coupling == "kl":
            Gamma = kl_ot_coupling(pbar_x, pbar_y, C, cfg.eps, log_Q)
        elif cfg.coupling == "fixed":
            Gamma = Pobs.copy()
        else:
            raise ValueError("this script only supports 'fixed' and 'kl'")

        for _ in range(cfg.num_inner):
            J0 = free_energy(G, Gamma, pbar_x, pbar_y, cfg.lam, cfg.eps, n, m, log_Q) + exo_penalty(G)
            grad = gradient(G, Gamma, pbar_x, pbar_y, cfg.lam, n, m)
            if mu > 0:
                grad[:n, :n] += 2.0 * mu * (G[:n, :n] - K_x)
            grad_sq = float(np.sum(grad * grad))

            step = alpha
            G_new = psd_project(G - step * grad)
            for _ in range(25):
                J_new = free_energy(G_new, Gamma, pbar_x, pbar_y, cfg.lam, cfg.eps, n, m, log_Q) + exo_penalty(G_new)
                if J_new <= J0 - 1e-4 * step * grad_sq or grad_sq == 0.0:
                    break
                step *= 0.5
                G_new = psd_project(G - step * grad)
            G = G_new
        alpha *= cfg.decay

    return G, Gamma


def run_direction(name, counts_fit, counts_exo, mus):
    Pobs, pbar_x, pbar_y = build_marginals(counts_fit)
    n, m = Pobs.shape

    # calibrate K_x's scale against the mu=0 (unregularised) G_XX diagonal
    cfg0 = RunConfig(lam=0.001, eps=0.05, rho=0.05, q=2, alpha0=0.02,
                      decay=0.99, num_outer=60, num_inner=10, coupling="kl")
    G0, _ = run_with_exo_anchor(Pobs, pbar_x, pbar_y, cfg0, np.zeros((n, n)), 0.0)
    base_scale = float(np.diag(G0[:n, :n]).mean())

    K_x = build_cosine_kernel(counts_exo, base_scale)
    print(f"[{name}] n={n} m={m}  base_scale(diag G_XX at mu=0)={base_scale:.4f}  "
          f"K_x off-diag mean={(K_x.sum()-np.trace(K_x))/(n*n-n):.4f}")

    results = {}
    for coupling in ["fixed", "kl"]:
        cfg = RunConfig(lam=0.001, eps=0.05, rho=0.05, q=2, alpha0=0.02,
                         decay=0.99, num_outer=60, num_inner=10, coupling=coupling)
        results[coupling] = {}
        for mu in mus:
            G, Gamma = run_with_exo_anchor(Pobs, pbar_x, pbar_y, cfg, K_x, mu)
            phi, psi = embed(G, n, m, cfg.q)
            p5 = precision_at_k(phi, psi, counts_fit, k=5)
            p10 = precision_at_k(phi, psi, counts_fit, k=10)
            results[coupling][mu] = {"P@5": p5, "P@10": p10}
            print(f"  [{name}] coupling={coupling:5s} mu={mu:<8} P@5={p5:.3f} P@10={p10:.3f}")
    return results


def main():
    counts, class_names, pred_names = load_awa_dataset(DATA_DIR)
    n, m = counts.shape
    assert m == 85, f"expected 85 AwA2 attributes, got {m}"

    appearance_idx = [i - 1 for i in APPEARANCE_IDS_1INDEXED]
    ecological_idx = [i - 1 for i in ECOLOGICAL_IDS_1INDEXED]
    assert len(appearance_idx) == 33 and len(ecological_idx) == 52

    counts_appearance = counts[:, appearance_idx]
    counts_ecological = counts[:, ecological_idx]

    mus = [0.0, 0.001, 0.005, 0.01, 0.03, 0.05, 0.1, 0.3, 0.5]

    all_results = {}
    all_results["A_fit_appearance_exo_ecological"] = run_direction(
        "A: fit=appearance, exo=ecological", counts_appearance, counts_ecological, mus)
    all_results["B_fit_ecological_exo_appearance"] = run_direction(
        "B: fit=ecological, exo=appearance", counts_ecological, counts_appearance, mus)

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "heldout_attribute_split.json"), "w") as f:
        json.dump(all_results, f, indent=2)


if __name__ == "__main__":
    main()
