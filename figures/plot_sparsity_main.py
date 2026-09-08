"""Plot the main random-sparsity experiment from results/sparsity_main.json."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "code" / "results" / "sparsity_main.json"
OUT_PNG = ROOT / "figures" / "sparsity_main.png"
OUT_PDF = ROOT / "figures" / "sparsity_main.pdf"


def summary(data, method, key):
    vals = []
    errs = []
    for k in data["results"]:
        s = data["results"][k]["methods"][method][key + "_summary"]
        vals.append(s["mean"])
        errs.append(s["std"])
    return np.array(vals), np.array(errs)


def main():
    with open(RESULTS) as f:
        data = json.load(f)
    keeps = np.array([100 * float(k) for k in data["results"]])
    methods = list(data["methods"])
    labels = ["PSD-CODE", "W-PSD-CODE (KL)"]
    short = {methods[0]: "PSD-CODE", methods[1]: "W-PSD-CODE (KL)"}

    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.8))
    panels = [
        ("all_p10", "P@10 (all pairs)", "Higher is better"),
        ("heldout_p10", "P@10 (held-out positives)", "Higher is better"),
        ("l1_to_pobs", r"$L_1(\Gamma^*,P_{\mathrm{obs}})$", "Smaller means less drift"),
    ]

    for ax, (metric, ylabel, guide) in zip(axes, panels):
        for method, label in zip(methods, labels):
            y, e = summary(data, method, metric)
            if metric == "l1_to_pobs" and method == methods[0]:
                y = np.zeros_like(y)
                e = np.zeros_like(e)
            ax.errorbar(keeps, y, yerr=e, marker="o", markersize=4,
                        linewidth=1.8, capsize=3, label=label)
        ax.set_xlabel("Retained positive pairs (%)")
        ax.set_ylabel(ylabel)
        ax.set_xticks(keeps)
        ax.grid(True, alpha=0.20, linewidth=0.7)
        ax.set_title(guide, fontsize=10)

    axes[0].set_ylim(bottom=0)
    axes[1].set_ylim(bottom=0)
    axes[2].set_ylim(bottom=0)
    axes[0].legend(frameon=False, loc="lower right")

    fig.suptitle(
        "Random sparsity experiment on AwA/AwA2: does adaptive coupling help when observations are missing?",
        fontsize=12, y=1.03)
    fig.tight_layout()
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {OUT_PNG}\nsaved {OUT_PDF}")


if __name__ == "__main__":
    main()
