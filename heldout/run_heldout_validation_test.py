"""
Pre-registered-style held-out attribute validation/test experiment.

Protocol
--------
Primary direction:
  fit/evaluate on the 33 appearance attributes; build the exogenous anchor
  from the disjoint 52 ecological/behavioural attributes.

1. Validation:
   - use masking seeds 1000--1009;
   - evaluate the pre-specified mu grid;
   - select a single mu using mean P@10 on validation (ties -> smaller mu).

2. Test:
   - freeze the selected mu;
   - use independent masking seeds 2000--2029;
   - compare fixed PSD-CODE (mu=0) against anchored W-PSD-CODE (selected mu)
     on exactly the same masked observations for each seed.

Inference:
  - paired mean P@10 gain;
  - 95% CI for the paired mean difference;
  - Cohen's dz;
  - paired t-test (secondary);
  - paired sign-flip permutation test (Monte Carlo, primary nonparametric check);
  - per-seed differences are saved.

A reverse-direction analysis (fit=eccological, anchor=appearance) is also
run on the frozen mu as a secondary robustness check, but it is not used for
selection of the primary mu.

The held-out anchor is never constructed from the attributes used to fit or
evaluate the embedding.

Usage
-----
    python run_heldout_validation_test.py

Optional:
    python run_heldout_validation_test.py --validation-seeds 10 --test-seeds 30
    python run_heldout_validation_test.py --keep 0.90 --outer-iters 60 --inner-iters 10
"""
import argparse
import csv
import json
import os
import sys
from statistics import mean

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
CODE_ROOT = os.path.join(HERE, "..")
sys.path.insert(0, CODE_ROOT)

from common.data import load_awa_dataset, build_marginals
from common.algorithm import RunConfig
from common.metrics import embed, precision_at_k
from run_heldout_attribute_split import (
    APPEARANCE_IDS_1INDEXED,
    ECOLOGICAL_IDS_1INDEXED,
    build_cosine_kernel,
    run_with_exo_anchor,
)

DATA_DIR = os.path.join(CODE_ROOT, "data", "AwA2")
OUT_DIR = os.path.join(CODE_ROOT, "results")

DEFAULT_MUS = [0.0, 0.001, 0.005, 0.01, 0.03, 0.05, 0.1, 0.3, 0.5]
DEFAULT_VALIDATION_SEEDS = list(range(1000, 1010))
DEFAULT_TEST_SEEDS = list(range(2000, 2030))


def mask_positives(full: np.ndarray, keep: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    ones = np.argwhere(full > 0)
    selected = rng.random(len(ones)) < keep
    obs = np.zeros_like(full)
    obs[ones[selected, 0], ones[selected, 1]] = 1.0
    # Preserve nonempty rows and columns exactly as in the V5 protocol.
    for i in range(full.shape[0]):
        if obs[i].sum() == 0:
            row_pos = ones[ones[:, 0] == i]
            obs[i, row_pos[0, 1]] = 1.0
    for j in range(full.shape[1]):
        if obs[:, j].sum() == 0:
            col_pos = ones[ones[:, 1] == j]
            obs[col_pos[0, 0], j] = 1.0
    return obs


def fixed_cfg(outer_iters: int, inner_iters: int) -> RunConfig:
    return RunConfig(
        lam=0.001, eps=0.05, rho=0.05, q=2,
        alpha0=0.02, decay=0.99,
        num_outer=outer_iters, num_inner=inner_iters,
        coupling="fixed",
    )


def fit_score(
    counts_fit: np.ndarray,
    counts_exo: np.ndarray,
    keep: float,
    seed: int,
    mu: float,
    outer_iters: int,
    inner_iters: int,
):
    """Return one P@10 score plus the corresponding G and mask."""
    obs = mask_positives(counts_fit, keep, seed)
    Pobs, pbar_x, pbar_y = build_marginals(obs)
    cfg = fixed_cfg(outer_iters, inner_iters)

    # Scale the anchor from a mu=0 fit performed on the same masked training
    # matrix. This calibration uses no held-out/evaluation attributes.
    G0, _ = run_with_exo_anchor(
        Pobs, pbar_x, pbar_y, cfg, np.zeros((counts_fit.shape[0], counts_fit.shape[0])), 0.0
    )
    base_scale = float(np.diag(G0[:counts_fit.shape[0], :counts_fit.shape[0]]).mean())
    K_x = build_cosine_kernel(counts_exo, base_scale)

    G, _ = run_with_exo_anchor(Pobs, pbar_x, pbar_y, cfg, K_x, mu)
    phi, psi = embed(G, counts_fit.shape[0], counts_fit.shape[1], cfg.q)
    p10 = float(precision_at_k(phi, psi, counts_fit, k=10))
    return p10, G, obs


def select_mu_validation(
    counts_fit,
    counts_exo,
    mus,
    validation_seeds,
    keep,
    outer_iters,
    inner_iters,
):
    val = {float(mu): [] for mu in mus}
    # Cache baseline and anchor scale per seed so the validation grid only
    # repeats the M-step for each mu.
    for seed in validation_seeds:
        obs = mask_positives(counts_fit, keep, seed)
        Pobs, pbar_x, pbar_y = build_marginals(obs)
        cfg = fixed_cfg(outer_iters, inner_iters)
        n = counts_fit.shape[0]
        G0, _ = run_with_exo_anchor(Pobs, pbar_x, pbar_y, cfg, np.zeros((n, n)), 0.0)
        base_scale = float(np.diag(G0[:n, :n]).mean())
        K_x = build_cosine_kernel(counts_exo, base_scale)

        # Baseline score is identical across the mu grid for this seed.
        phi0, psi0 = embed(G0, n, counts_fit.shape[1], cfg.q)
        base_p10 = float(precision_at_k(phi0, psi0, counts_fit, k=10))

        for mu in mus:
            G, _ = run_with_exo_anchor(Pobs, pbar_x, pbar_y, cfg, K_x, float(mu))
            phi, psi = embed(G, n, counts_fit.shape[1], cfg.q)
            p10 = float(precision_at_k(phi, psi, counts_fit, k=10))
            val[float(mu)].append(p10)
        print(
            f"[validation] seed={seed} baseline P@10={base_p10:.4f} "
            + " ".join(f"mu={mu:g}:{np.mean(val[float(mu)]):.4f}" for mu in mus)
        )

    summary = []
    for mu in mus:
        vals = np.asarray(val[float(mu)], dtype=float)
        summary.append(
            {
                "mu": float(mu),
                "mean_P@10": float(vals.mean()),
                "std_P@10": float(vals.std(ddof=1)),
                "values": vals.tolist(),
            }
        )
    # Maximise mean validation P@10; exact ties favour the smaller mu.
    selected = sorted(summary, key=lambda r: (-r["mean_P@10"], r["mu"]))[0]
    return selected["mu"], summary


def paired_signflip_pvalue(diffs: np.ndarray, rng: np.random.Generator, n_perm: int = 100_000):
    diffs = np.asarray(diffs, dtype=float)
    observed = abs(float(diffs.mean()))
    # Vectorised Monte Carlo sign flips.
    batch = 10_000
    exceed = 0
    done = 0
    while done < n_perm:
        b = min(batch, n_perm - done)
        signs = rng.choice(np.array([-1.0, 1.0]), size=(b, diffs.size))
        means = np.abs((signs * diffs).mean(axis=1))
        exceed += int(np.count_nonzero(means >= observed))
        done += b
    return (exceed + 1.0) / (n_perm + 1.0)


def run_test(
    counts_fit,
    counts_exo,
    mu,
    test_seeds,
    keep,
    outer_iters,
    inner_iters,
):
    rows = []
    for seed in test_seeds:
        obs = mask_positives(counts_fit, keep, seed)
        Pobs, pbar_x, pbar_y = build_marginals(obs)
        cfg0 = fixed_cfg(outer_iters, inner_iters)
        n = counts_fit.shape[0]

        # Baseline and anchor share the exact same masked observation.
        G0, _ = run_with_exo_anchor(Pobs, pbar_x, pbar_y, cfg0, np.zeros((n, n)), 0.0)
        base_scale = float(np.diag(G0[:n, :n]).mean())
        K_x = build_cosine_kernel(counts_exo, base_scale)
        Gm, _ = run_with_exo_anchor(Pobs, pbar_x, pbar_y, cfg0, K_x, mu)

        phi0, psi0 = embed(G0, n, counts_fit.shape[1], cfg0.q)
        phim, psim = embed(Gm, n, counts_fit.shape[1], cfg0.q)
        p10_0 = float(precision_at_k(phi0, psi0, counts_fit, k=10))
        p10_m = float(precision_at_k(phim, psim, counts_fit, k=10))
        row = {
            "seed": int(seed),
            "baseline_P@10": p10_0,
            "anchored_P@10": p10_m,
            "gain_P@10": p10_m - p10_0,
        }
        rows.append(row)
        print(
            f"[test] seed={seed} baseline={p10_0:.4f} "
            f"anchored={p10_m:.4f} gain={row['gain_P@10']:+.4f}"
        )

    diffs = np.asarray([r["gain_P@10"] for r in rows], dtype=float)
    mean_diff = float(diffs.mean())
    sd_diff = float(diffs.std(ddof=1))
    se = sd_diff / np.sqrt(len(diffs))
    tstat = float(mean_diff / se) if se > 0 else np.inf
    tcrit = float(stats.t.ppf(0.975, len(diffs) - 1))
    ci_low, ci_high = mean_diff - tcrit * se, mean_diff + tcrit * se
    paired_t_p = float(2.0 * stats.t.sf(abs(tstat), len(diffs) - 1)) if np.isfinite(tstat) else 0.0
    dz = float(mean_diff / sd_diff) if sd_diff > 0 else np.inf
    perm_p = float(
        paired_signflip_pvalue(
            diffs, np.random.default_rng(20260822), n_perm=100_000
        )
    )

    summary = {
        "n_test_seeds": len(test_seeds),
        "mean_baseline_P@10": float(np.mean([r["baseline_P@10"] for r in rows])),
        "std_baseline_P@10": float(np.std([r["baseline_P@10"] for r in rows], ddof=1)),
        "mean_anchored_P@10": float(np.mean([r["anchored_P@10"] for r in rows])),
        "std_anchored_P@10": float(np.std([r["anchored_P@10"] for r in rows], ddof=1)),
        "mean_gain_P@10": mean_diff,
        "sd_gain_P@10": sd_diff,
        "ci95_gain_P@10": [float(ci_low), float(ci_high)],
        "paired_t": tstat,
        "paired_t_df": len(diffs) - 1,
        "paired_t_p": paired_t_p,
        "cohen_dz": dz,
        "paired_signflip_permutation_p": perm_p,
        "test_seeds": [int(s) for s in test_seeds],
        "rows": rows,
    }
    return summary


def save_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validation-seeds", type=int, default=len(DEFAULT_VALIDATION_SEEDS))
    ap.add_argument("--test-seeds", type=int, default=len(DEFAULT_TEST_SEEDS))
    ap.add_argument("--keep", type=float, default=0.90)
    ap.add_argument("--outer-iters", type=int, default=60)
    ap.add_argument("--inner-iters", type=int, default=10)
    ap.add_argument("--mus", type=float, nargs="+", default=DEFAULT_MUS)
    args = ap.parse_args()

    counts, class_names, pred_names = load_awa_dataset(DATA_DIR)
    appearance_idx = [i - 1 for i in APPEARANCE_IDS_1INDEXED]
    ecological_idx = [i - 1 for i in ECOLOGICAL_IDS_1INDEXED]
    appearance = counts[:, appearance_idx]
    ecological = counts[:, ecological_idx]

    val_seeds = list(range(1000, 1000 + args.validation_seeds))
    test_seeds = list(range(2000, 2000 + args.test_seeds))

    selected_mu, validation = select_mu_validation(
        appearance, ecological, args.mus, val_seeds, args.keep,
        args.outer_iters, args.inner_iters
    )
    print(f"\nSELECTED mu={selected_mu:g} from validation only.\n")

    primary = run_test(
        appearance, ecological, selected_mu, test_seeds,
        args.keep, args.outer_iters, args.inner_iters
    )

    # Secondary reverse direction using the same frozen mu. This does not
    # alter the selected parameter.
    secondary = run_test(
        ecological, appearance, selected_mu, test_seeds,
        args.keep, args.outer_iters, args.inner_iters
    )

    payload = {
        "protocol": {
            "validation_seeds": val_seeds,
            "test_seeds": test_seeds,
            "keep": args.keep,
            "mu_grid": [float(x) for x in args.mus],
            "selected_mu": float(selected_mu),
            "selection_rule": "maximize mean validation P@10; ties -> smaller mu",
            "primary_direction": "fit/evaluate appearance; anchor ecological",
            "secondary_direction": "fit/evaluate ecological; anchor appearance",
            "permutation_test": "Monte Carlo paired sign-flip, 100000 permutations, RNG seed 20260822",
        },
        "validation": validation,
        "primary_test": primary,
        "secondary_reverse_direction": secondary,
    }
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "heldout_attribute_validation_test.json"), "w") as f:
        json.dump(payload, f, indent=2)
    save_csv(
        os.path.join(OUT_DIR, "heldout_attribute_validation_test_primary.csv"),
        primary["rows"],
    )

    print("\nPRIMARY TEST SUMMARY")
    for k in [
        "n_test_seeds", "mean_baseline_P@10", "mean_anchored_P@10",
        "mean_gain_P@10", "sd_gain_P@10", "ci95_gain_P@10",
        "paired_t", "paired_t_df", "paired_t_p", "cohen_dz",
        "paired_signflip_permutation_p",
    ]:
        print(f"{k}: {primary[k]}")


if __name__ == "__main__":
    main()
