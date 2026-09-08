"""
Reproduces Figure 1 (free-energy convergence traces).
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_convergence(histories: dict, out_path: str):
    """
    histories: dict mapping method label -> list of free-energy values
               (one per outer iteration), e.g.

               {"PSD-CODE": [...],
                "W-PSD-CODE (exact OT)": [...],
                "W-PSD-CODE (Sinkhorn)": [...]}
    """
    # Keys must match the exact method labels passed in via `histories`
    # (see psd_code/run_psd_code.py, wpsd_code_exact_ot/run_wpsd_exact.py,
    # wpsd_code_sinkhorn/run_wpsd_sinkhorn.py, wpsd_code_kl/run_wpsd_kl.py).
    # PSD-CODE and the KL-anchored variant keep Gamma extremely close to
    # Pobs (L1 displacement ~0.03), so their free-energy trajectories are
    # nearly identical (max difference ~0.0008 across all 60 iterations)
    # and one would otherwise be plotted directly on top of the other;
    # the KL-anchored trace uses a dashed linestyle and a sparser marker
    # so both remain visible even where the two curves coincide.
    style = {
        "PSD-CODE (fixed-lambda baseline)":
            dict(color="tab:blue", linestyle="-", marker="o", markevery=1),
        "W-PSD-CODE (exact OT)":
            dict(color="tab:orange", linestyle="-", marker="s", markevery=1),
        "W-PSD-CODE (entropic, eps=0.01)":
            dict(color="tab:green", linestyle="-", marker="^", markevery=1),
        "W-PSD-CODE (KL-anchored, eps=0.05)":
            dict(color="tab:red", linestyle="--", marker="d", markevery=3),
    }
    default_style = dict(color=None, linestyle="-", marker="o", markevery=1)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for label, hist in histories.items():
        iters = list(range(len(hist)))
        s = style.get(label, default_style)
        ax.plot(iters, hist, marker=s["marker"], markersize=4,
                markevery=s["markevery"], linewidth=1.4,
                linestyle=s["linestyle"], label=label, color=s["color"])

    ax.set_xlabel("Outer iteration")
    ax.set_ylabel("Free energy")
    ax.set_title("W-PSD-CODE")
    ax.legend(frameon=True)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_embedding(phi, psi, Gamma, class_names, pred_names, out_path: str,
                    n_animal_labels: int = 12, n_attr_labels: int = 18,
                    n_edges: int = 70, method_label: str = "W-PSD-CODE"):
    """
    Reproduces Figure 2: 2D scatter of the animal embedding phi (red
    squares) and attribute embedding psi (blue circles), with the
    n_edges heaviest OT coupling weights Gamma*(x,y) drawn as light
    grey edges (linewidth proportional to coupling weight).
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(phi[:, 0], phi[:, 1], marker="s", s=45, c="#d62728",
               label="animals", zorder=3)
    ax.scatter(psi[:, 0], psi[:, 1], marker="o", s=25, c="#1f77b4",
               label="attributes", zorder=3)

    for i, name in enumerate(class_names[:n_animal_labels]):
        ax.text(phi[i, 0], phi[i, 1], name, fontsize=7, zorder=4)
    for j, name in enumerate(pred_names[:n_attr_labels]):
        ax.text(psi[j, 0], psi[j, 1], name, fontsize=6, alpha=0.75, zorder=4)

    flat = np.argsort(Gamma.ravel())[-n_edges:]
    gmax = Gamma.max() if Gamma.max() > 0 else 1.0
    for idx in flat:
        i, j = divmod(int(idx), Gamma.shape[1])
        ax.plot([phi[i, 0], psi[j, 0]], [phi[i, 1], psi[j, 1]],
                color="0.70", linewidth=0.3 + 4 * Gamma[i, j] / gmax,
                alpha=0.25, zorder=1)

    ax.legend(loc="best", fontsize=8)
    ax.set_title(f"{method_label} embedding")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
