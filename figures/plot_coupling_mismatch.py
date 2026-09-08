"""Plot coupling-mismatch diagnostics.

Left: L1(Gamma_KL*, Pobs) vs P@10 for every cell of the saved rho-epsilon
screening grid.
Right: delta(Pobs, Gamma_C*) vs P@10 for the diagnostic subset for which the
exact OT plan at the final learned geometry was recomputed.

Usage:
    python code/figures/plot_coupling_mismatch.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
RESULTS = HERE / "results"
FIGURES = HERE.parent / "figures"


def _fit_line(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if np.allclose(x, x[0]):
        return None
    coef = np.polyfit(x, y, 1)
    xx = np.linspace(x.min(), x.max(), 100)
    yy = coef[0] * xx + coef[1]
    return xx, yy


def _spearman(x, y):
    rx = pd.Series(x).rank(method="average").to_numpy()
    ry = pd.Series(y).rank(method="average").to_numpy()
    if np.std(rx) == 0 or np.std(ry) == 0:
        return np.nan
    return float(np.corrcoef(rx, ry)[0, 1])


def _style_axes(ax, xlabel, ylabel):
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.grid(True, alpha=0.2, linewidth=0.7)
    ax.tick_params(labelsize=9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rho-eps-csv", type=Path,
                    default=RESULTS / "rho_epsilon_tradeoff.csv")
    ap.add_argument("--diagnostic-csv", type=Path,
                    default=RESULTS / "coupling_mismatch_diagnostics.csv")
    ap.add_argument("--out-dir", type=Path, default=FIGURES)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    grid = pd.read_csv(args.rho_eps_csv)
    diag = pd.read_csv(args.diagnostic_csv)

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.15), constrained_layout=True)

    # Same rho encoding in both panels.
    rho_vals = sorted(grid["rho"].unique())
    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(min(rho_vals), max(rho_vals))

    # Left: all 36 screening cells.
    ax = axes[0]
    for eps in sorted(grid["epsilon"].unique()):
        sub = grid[grid["epsilon"] == eps]
        sc = ax.scatter(sub["l1_gamma_pobs"], sub["p_at_10"],
                        c=sub["rho"], cmap=cmap, norm=norm,
                        s=42, alpha=0.9, edgecolors="none",
                        label=fr"$\varepsilon={eps:g}$")
    fit = _fit_line(grid["l1_gamma_pobs"], grid["p_at_10"])
    if fit is not None:
        ax.plot(*fit, linewidth=1.4, linestyle="--", alpha=0.65)
    rho_ref = 0.05
    eps_ref = 0.05
    ref = grid[(np.isclose(grid["rho"], rho_ref)) & (np.isclose(grid["epsilon"], eps_ref))]
    if len(ref):
        r = ref.iloc[0]
        ax.scatter([r["l1_gamma_pobs"]], [r["p_at_10"]],
                   marker="*", s=150, edgecolor="black", linewidth=0.8,
                   zorder=5)
    sp = _spearman(grid["l1_gamma_pobs"], grid["p_at_10"])
    _style_axes(ax, r"$L_1(\Gamma^*,P_{\rm obs})$",
                r"P@10")
    ax.set_title("Coupling displacement", fontsize=11)
    ax.text(0.03, 0.97, f"Spearman $\\rho_s={sp:.2f}$\n36 screening cells",
            transform=ax.transAxes, va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.82, edgecolor="0.8"))
    ax.legend(title="KL temperature", fontsize=7.5, title_fontsize=8,
              frameon=False, loc="lower right", ncol=2)
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax,
                        fraction=0.047, pad=0.02)
    cbar.set_label(r"reference mixing $\rho$", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    # Right: exact-OT mismatch diagnostic subset.
    ax = axes[1]
    for eps in sorted(diag["epsilon"].unique()):
        sub = diag[diag["epsilon"] == eps]
        ax.scatter(sub["delta_pobs_gamma_ot"], sub["p_at_10"],
                   c=sub["rho"], cmap=cmap, norm=norm,
                   s=56, alpha=0.95, edgecolors="none",
                   marker="o" if np.isclose(eps, 0.005) else "s")
    fit = _fit_line(diag["delta_pobs_gamma_ot"], diag["p_at_10"])
    if fit is not None:
        ax.plot(*fit, linewidth=1.4, linestyle="--", alpha=0.65)
    ref = diag[(np.isclose(diag["rho"], 0.05)) & (np.isclose(diag["epsilon"], 0.05))]
    if len(ref):
        r = ref.iloc[0]
        ax.scatter([r["delta_pobs_gamma_ot"]], [r["p_at_10"]],
                   marker="*", s=150, edgecolor="black", linewidth=0.8, zorder=5)
    sp = _spearman(diag["delta_pobs_gamma_ot"], diag["p_at_10"])
    _style_axes(ax, r"$\delta(P_{\rm obs},\Gamma_C^*)$",
                r"P@10")
    ax.set_title("Geometry mismatch", fontsize=11)
    ax.ticklabel_format(axis="x", style="sci", scilimits=(-4, -4))
    ax.text(0.03, 0.97, f"Spearman $\\rho_s={sp:.2f}$\n12 diagnostic cells",
            transform=ax.transAxes, va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.82, edgecolor="0.8"))
    ax.text(0.98, 0.04,
            r"$\Gamma_C^*$ = exact OT plan at final $C(G^*)$",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7.8,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.82, edgecolor="0.8"))

    out_pdf = args.out_dir / "coupling_mismatch_diagnostics.pdf"
    out_png = args.out_dir / "coupling_mismatch_diagnostics.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_pdf}")
    print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()
