"""Full-budget null controls for the held-out attribute experiment.

Uses the already selected mu=0.01 and a specified subset of the primary test masks. For each seed we compare:
  1) correct held-out ecological anchor;
  2) class-permuted ecological anchor (geometry preserved, class alignment broken);
  3) density-matched random binary anchor (task-independent noise geometry).

All conditions share the exact same masked observations and configurable
optimization budget (60x10 for the reported control). Anchor scale is calibrated from the same mu=0 baseline as in the
primary run.
"""
import csv, json, os, sys, argparse
import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
CODE_ROOT = os.path.join(HERE, "..")
sys.path.insert(0, CODE_ROOT)
from common.data import load_awa_dataset, build_marginals
from common.algorithm import RunConfig
from common.metrics import embed, precision_at_k
from run_heldout_attribute_split import build_cosine_kernel, run_with_exo_anchor
from run_heldout_validation_test import mask_positives

DATA_DIR = os.path.join(CODE_ROOT, "data", "AwA2")
OUT_DIR = os.path.join(CODE_ROOT, "results")
DEFAULT_SEED_START = 2000
DEFAULT_N_SEEDS = 6
DEFAULT_KEEP = 0.90
DEFAULT_OUTER = 60
DEFAULT_INNER = 10

APPEARANCE = list(range(33))
ECOLOGICAL = list(range(33, 85))


def fixed_cfg(outer_iters, inner_iters):
    return RunConfig(lam=0.001, eps=0.05, rho=0.05, q=2,
                     alpha0=0.02, decay=0.99, num_outer=outer_iters,
                     num_inner=inner_iters, coupling="fixed")


def cosine_kernel_from_features(X, base_scale):
    X = np.asarray(X, dtype=float)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    Xn = X / norms
    K = base_scale * (Xn @ Xn.T)
    np.fill_diagonal(K, base_scale)
    return K


def one_seed(seed, keep, outer_iters, inner_iters, mu):
    counts, _, _ = load_awa_dataset(DATA_DIR)
    fit = counts[:, APPEARANCE]
    exo = counts[:, ECOLOGICAL]
    obs = mask_positives(fit, keep, seed)
    Pobs, px, py = build_marginals(obs)
    cfg = fixed_cfg(outer_iters, inner_iters)
    n = fit.shape[0]

    G0, _ = run_with_exo_anchor(Pobs, px, py, cfg, np.zeros((n, n)), 0.0)
    base_scale = float(np.diag(G0[:n, :n]).mean())
    phi0, psi0 = embed(G0, n, fit.shape[1], cfg.q)
    base = float(precision_at_k(phi0, psi0, fit, k=10))

    K_correct = build_cosine_kernel(exo, base_scale)

    rng_perm = np.random.default_rng(700000 + seed)
    perm = rng_perm.permutation(n)
    K_permuted = K_correct[np.ix_(perm, perm)]

    rng_noise = np.random.default_rng(800000 + seed)
    density = float(exo.mean())
    X_noise = (rng_noise.random(exo.shape) < density).astype(float)
    # Avoid zero-vector rows so cosine similarities remain defined.
    for i in range(X_noise.shape[0]):
        if X_noise[i].sum() == 0:
            X_noise[i, rng_noise.integers(0, X_noise.shape[1])] = 1.0
    K_noise = cosine_kernel_from_features(X_noise, base_scale)

    out = {"seed": int(seed), "baseline_P@10": base}
    for name, K in [("correct", K_correct), ("permuted", K_permuted), ("noise", K_noise)]:
        G, _ = run_with_exo_anchor(Pobs, px, py, cfg, K, mu)
        phi, psi = embed(G, n, fit.shape[1], cfg.q)
        p10 = float(precision_at_k(phi, psi, fit, k=10))
        out[f"{name}_P@10"] = p10
        out[f"{name}_gain"] = p10 - base
    return out


def summarize(rows, condition):
    d = np.asarray([r[f"{condition}_gain"] for r in rows], float)
    mean_d = float(d.mean())
    sd = float(d.std(ddof=1))
    se = sd / np.sqrt(len(d))
    t = float(mean_d / se) if se else np.inf
    ci = stats.t.ppf(0.975, len(d)-1) * se
    return {
        "condition": condition,
        "n": len(d),
        "mean_baseline_P@10": float(np.mean([r["baseline_P@10"] for r in rows])),
        "mean_anchor_P@10": float(np.mean([r[f"{condition}_P@10"] for r in rows])),
        "mean_gain_P@10": mean_d,
        "sd_gain_P@10": sd,
        "ci95_gain_P@10": [float(mean_d-ci), float(mean_d+ci)],
        "paired_t": t,
        "df": len(d)-1,
        "paired_t_p": float(2*stats.t.sf(abs(t), len(d)-1)) if np.isfinite(t) else 0.0,
        "positive_seeds": int(np.sum(d > 0)),
        "zero_seeds": int(np.sum(np.isclose(d, 0.0))),
        "negative_seeds": int(np.sum(d < 0)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-start", type=int, default=DEFAULT_SEED_START)
    ap.add_argument("--n-seeds", type=int, default=DEFAULT_N_SEEDS)
    ap.add_argument("--keep", type=float, default=DEFAULT_KEEP)
    ap.add_argument("--outer-iters", type=int, default=DEFAULT_OUTER)
    ap.add_argument("--inner-iters", type=int, default=DEFAULT_INNER)
    ap.add_argument("--mu", type=float, default=0.01)
    args = ap.parse_args()
    seeds = list(range(args.seed_start, args.seed_start + args.n_seeds))
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = []
    for seed in seeds:
        row = one_seed(seed, args.keep, args.outer_iters, args.inner_iters, args.mu)
        rows.append(row)
        print(row, flush=True)
        # Persist partial progress so an interrupted run still leaves usable results.
        csv_path = os.path.join(OUT_DIR, "heldout_anchor_controls.csv")
        fields = list(rows[0].keys())
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader(); w.writerows(rows)
    summary = {c: summarize(rows, c) for c in ["correct", "permuted", "noise"]}
    out = {
        "protocol": {"mu": args.mu, "test_seeds": seeds, "keep": args.keep,
                     "outer_iters": args.outer_iters, "inner_iters": args.inner_iters,
                     "correct_anchor": "held-out ecological attributes (33 appearance / 52 ecological split)",
                     "permuted_anchor": "same ecological geometry after class-label permutation",
                     "noise_anchor": "density-matched random binary 52-attribute matrix"},
        "summary": summary,
        "rows": rows,
    }
    with open(os.path.join(OUT_DIR, "heldout_anchor_controls.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("SUMMARY")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
