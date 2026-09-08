"""
Rho-epsilon tradeoff experiment for the KL-anchored W-PSD-CODE method.

Sweeps the reference mixing weight rho and KL/Sinkhorn temperature epsilon
on the full AwA/AwA2 class-attribute matrix.  Each cell starts from the
same identity Gram matrix and uses the deterministic optimisation settings
reported in the manuscript, so the grid isolates the effect of (rho, eps).

This is the cheap SCREENING sweep behind Figure 4 and uses a reduced
30-outer-iteration, 5-inner-step budget per cell by default, matching
the manuscript's Figure 4 caption exactly. Do NOT change this default to
60/10 -- that budget belongs to the separate, single-process FULL-BUDGET
CONFIRMATION of four representative cells (run_rho_epsilon_confirmation.py,
Table 5), which is a different script producing a different table.
Conflating the two silently changes every number in Figure 4 away from
what the manuscript's own prose describes ("P@10=0.732 at rho=0.50,
eps=0.01" only holds at the 30/5 screening budget).

Outputs:
  results/rho_epsilon_tradeoff.csv
  results/rho_epsilon_tradeoff.json
  results/rho_epsilon_tradeoff_summary.txt
  figures/rho_epsilon_tradeoff.pdf
  figures/rho_epsilon_tradeoff.png

Usage:
    python code/ablation/run_rho_epsilon_tradeoff.py
    python code/ablation/run_rho_epsilon_tradeoff.py --eps 0.005 0.01 0.03 0.05 0.10 0.20
    python code/ablation/run_rho_epsilon_tradeoff.py --rho 0.005 0.01 0.03 0.05 0.10 0.20 0.50 0.80
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import multiprocessing as mp
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from common.algorithm import RunConfig, run_alternating_optimisation
from common.data import build_marginals, load_awa_dataset
from common.metrics import (embed, l1_coupling_displacement,
                            precision_at_k)

DEFAULT_DATA = HERE / "data" / "AwA2"
DEFAULT_EPS = [0.005, 0.01, 0.03, 0.05, 0.10, 0.20]
DEFAULT_RHO = [0.005, 0.03, 0.05, 0.10, 0.20, 0.50]


def _run_one(args):
    data_dir, rho, eps, num_outer, num_inner = args
    counts, _, _ = load_awa_dataset(str(data_dir))
    Pobs, pbar_x, pbar_y = build_marginals(counts)
    n, m = Pobs.shape
    cfg = RunConfig(
                lam=0.001,
                eps=eps,
                rho=rho,
                q=2,
                alpha0=0.02,
                decay=0.99,
                num_outer=num_outer,
                num_inner=num_inner,
                tau=1e-8,
                coupling="kl",
                init="identity",
            )
    result = run_alternating_optimisation(Pobs, pbar_x, pbar_y, cfg)
    phi, psi = embed(result.G, n, m, cfg.q)
    row = {
        "rho": float(rho),
        "epsilon": float(eps),
        "p_at_5": precision_at_k(phi, psi, counts, 5),
        "p_at_10": precision_at_k(phi, psi, counts, 10),
        "p_at_20": precision_at_k(phi, psi, counts, 20),
        "l1_gamma_pobs": l1_coupling_displacement(result.Gamma, Pobs),
        "iterations": int(result.iterations),
        "time_seconds": float(result.time_seconds),
        "final_free_energy": float(result.history[-1]),
    }
    return row


def run_grid(data_dir: Path, eps_values: list[float], rho_values: list[float],
             num_outer: int = 30, num_inner: int = 5, workers: int = 4):
    jobs = [(data_dir, rho, eps, num_outer, num_inner)
            for rho in rho_values for eps in eps_values]
    if workers <= 1:
        rows = [_run_one(job) for job in jobs]
    else:
        with mp.Pool(processes=workers) as pool:
            rows = list(pool.imap(_run_one, jobs))
    rows.sort(key=lambda x: (x["rho"], x["epsilon"]))
    for r in rows:
        print(
            f"rho={r['rho']:>5.3f} eps={r['epsilon']:>6.3f} "
            f"P@10={r['p_at_10']:.4f} L1={r['l1_gamma_pobs']:.4f} "
            f"iters={r['iterations']:>2d}"
        )
    return rows


def matrix(rows, rho_values, eps_values, key):
    lookup = {(r["rho"], r["epsilon"]): r[key] for r in rows}
    return np.array([[lookup[(rho, eps)] for rho in rho_values]
                     for eps in eps_values], dtype=float)


def annotate_heatmap(ax, im, arr, fmt=".3f", fontsize=7.2):
    """Annotate cells with text whose color is chosen from the rendered cell luminance."""
    norm = im.norm
    cmap = im.cmap
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            rgba = cmap(norm(arr[i, j]))
            # Perceived luminance of the actual colormap color.
            luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
            color = "black" if luminance > 0.58 else "white"
            ax.text(
                j, i, format(arr[i, j], fmt),
                ha="center", va="center",
                fontsize=fontsize, fontweight="medium", color=color,
            )


def add_colorbar_note(cbar, text):
    """Put a compact horizontal semantic note above the colorbar."""
    cbar.ax.text(
        0.5, 1.04, text,
        transform=cbar.ax.transAxes,
        ha="center", va="bottom",
        fontsize=7.5, fontweight="medium",
    )


def plot_tradeoff(rows, rho_values, eps_values, out_pdf, out_png):
    specs = [
        ("p_at_10", "P@10", ".3f", "viridis", "higher is better"),
        ("l1_gamma_pobs", r"$L_1(\Gamma^*,P_{\rm obs})$", ".3f", "magma_r", "smaller is better"),
    ]

    # Extra horizontal space is intentional: the earlier version squeezed the
    # first colorbar label into the gap between the two panels.
    fig, axes = plt.subplots(
        1, 2, figsize=(10.8, 4.45),
        gridspec_kw={"wspace": 0.42},
    )

    for ax, (key, title, fmt, cmap, note) in zip(axes, specs):
        arr = matrix(rows, rho_values, eps_values, key)
        im = ax.imshow(
            arr, aspect="auto", origin="upper", cmap=cmap,
            interpolation="nearest",
        )
        ax.set_xticks(np.arange(len(rho_values)))
        ax.set_xticklabels(
            [f"{x:g}" for x in rho_values],
            rotation=45, ha="right", rotation_mode="anchor",
        )
        ax.set_yticks(np.arange(len(eps_values)))
        ax.set_yticklabels([f"{x:g}" for x in eps_values])
        ax.set_xlabel(r"reference mixing $\rho$", labelpad=6)
        ax.set_ylabel(r"KL temperature $\varepsilon$", labelpad=6)
        ax.set_title(title, pad=10)
        ax.tick_params(axis="both", which="major", labelsize=8.5)
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)

        annotate_heatmap(ax, im, arr, fmt)
        cbar = fig.colorbar(im, ax=ax, fraction=0.048, pad=0.035)
        cbar.ax.tick_params(labelsize=8)
        add_colorbar_note(cbar, note)

    fig.suptitle(
        r"KL-anchored W-PSD-CODE: the $\rho$--$\varepsilon$ tradeoff on AwA/AwA2",
        fontsize=13, y=0.98,
    )
    fig.subplots_adjust(left=0.08, right=0.97, bottom=0.19, top=0.86)
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_outputs(rows, rho_values, eps_values, results_dir, figures_dir,
                   num_outer, num_inner):
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    csv_path = results_dir / "rho_epsilon_tradeoff.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    json_path = results_dir / "rho_epsilon_tradeoff.json"
    payload = {
        "experiment": "rho-epsilon tradeoff",
        "description": "KL-anchored W-PSD-CODE sweep on full AwA/AwA2 matrix",
        "rho_values": rho_values,
        "epsilon_values": eps_values,
        "settings": {"lambda": 0.001, "q": 2, "num_outer": num_outer,
                     "num_inner": num_inner, "init": "identity"},
        "results": rows,
    }
    json_path.write_text(json.dumps(payload, indent=2))

    # Select by P@10 first; among near-best cells, prefer the one with less
    # coupling displacement.  This avoids declaring a tiny numerical gain to
    # be a meaningful optimum when it requires substantially larger drift.
    best = max(rows, key=lambda x: x["p_at_10"])
    best_p = best["p_at_10"]
    near = [x for x in rows if x["p_at_10"] >= best_p - 0.005]
    conservative = min(near, key=lambda x: (x["l1_gamma_pobs"], -x["p_at_10"]))

    text = []
    text.append("rho-epsilon tradeoff summary")
    text.append("================================")
    text.append(f"Best raw P@10: rho={best['rho']}, epsilon={best['epsilon']}, "
                f"P@10={best['p_at_10']:.4f}, L1={best['l1_gamma_pobs']:.4f}")
    text.append(f"Best conservative cell (within 0.005 P@10 of maximum): "
                f"rho={conservative['rho']}, epsilon={conservative['epsilon']}, "
                f"P@10={conservative['p_at_10']:.4f}, "
                f"L1={conservative['l1_gamma_pobs']:.4f}")
    text.append("")
    for rho in rho_values:
        rowset = [x for x in rows if x["rho"] == rho]
        top = max(rowset, key=lambda x: x["p_at_10"])
        text.append(
            f"rho={rho:g}: best epsilon={top['epsilon']:g}, "
            f"P@10={top['p_at_10']:.4f}, L1={top['l1_gamma_pobs']:.4f}"
        )
    (results_dir / "rho_epsilon_tradeoff_summary.txt").write_text("\n".join(text) + "\n")

    return csv_path, json_path, results_dir / "rho_epsilon_tradeoff_summary.txt", best, conservative


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=str(DEFAULT_DATA))
    p.add_argument("--out-dir", default=str(HERE / "results"))
    p.add_argument("--fig-dir", default=str(HERE.parent / "figures"))
    p.add_argument("--eps", nargs="+", type=float, default=DEFAULT_EPS)
    p.add_argument("--rho", nargs="+", type=float, default=DEFAULT_RHO)
    p.add_argument("--num-outer", type=int, default=30)
    p.add_argument("--num-inner", type=int, default=5)
    p.add_argument("--workers", type=int, default=4)
    a = p.parse_args()

    eps_values = sorted(set(a.eps))
    rho_values = sorted(set(a.rho))
    if any(e <= 0 for e in eps_values):
        raise SystemExit("All epsilon values must be > 0 for the KL-anchored update.")
    if any(r <= 0 or r >= 1 for r in rho_values):
        raise SystemExit("rho values must satisfy 0 < rho < 1 because Q must be strictly positive for the KL term.")

    rows = run_grid(Path(a.data_dir), eps_values, rho_values,
                    num_outer=a.num_outer, num_inner=a.num_inner, workers=a.workers)
    csv_path, json_path, summary_path, best, conservative = write_outputs(
        rows, rho_values, eps_values, Path(a.out_dir), Path(a.fig_dir),
        num_outer=a.num_outer, num_inner=a.num_inner)
    pdf_path = Path(a.fig_dir) / "rho_epsilon_tradeoff.pdf"
    png_path = Path(a.fig_dir) / "rho_epsilon_tradeoff.png"
    plot_tradeoff(rows, rho_values, eps_values, pdf_path, png_path)

    print("\nBest raw cell:", best)
    print("Best conservative cell:", conservative)
    print("Wrote:")
    for path in (csv_path, json_path, summary_path, pdf_path, png_path):
        print(" ", path)


if __name__ == "__main__":
    main()
