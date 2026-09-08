#!/usr/bin/env python3
"""Create a schematic coupling-polytope figure for W-PSD-CODE.

The 2x3 case has a 2D feasible set, allowing the coupling-polytope
constraint and the distinction between the fixed empirical coupling
P_obs and a geometry-selected OT coupling Gamma* to be visualised.

Outputs a vector PDF and a 300-dpi PNG suitable for manuscript inclusion,
under the repository-root ``figures/`` directory.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

# code/figures/make_coupling_polytope_figure.py -> repo root is two levels up.
REPO_ROOT = Path(__file__).resolve().parents[2]


def feasible_vertices(a1: float, b: np.ndarray) -> np.ndarray:
    """Return vertices in (Gamma_11, Gamma_12) coordinates for a 2x3 coupling."""
    # For Gamma = [[x1, x2, a1-x1-x2], [b1-x1, b2-x2, b3-(a1-x1-x2)]],
    # constraints are 0 <= x_j <= b_j and 0 <= a1-x1-x2 <= b3.
    candidates = []
    # Boundary intersections among x1=0, x1=b1, x2=0, x2=b2,
    # x1+x2=a1, x1+x2=a1-b3.
    lines = [
        (1.0, 0.0, 0.0),
        (1.0, 0.0, b[0]),
        (0.0, 1.0, 0.0),
        (0.0, 1.0, b[1]),
        (1.0, 1.0, a1),
        (1.0, 1.0, a1 - b[2]),
    ]
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            A = np.array([[lines[i][0], lines[i][1]], [lines[j][0], lines[j][1]]])
            if abs(np.linalg.det(A)) < 1e-12:
                continue
            rhs = np.array([lines[i][2], lines[j][2]])
            x1, x2 = np.linalg.solve(A, rhs)
            x3 = a1 - x1 - x2
            if (-1e-10 <= x1 <= b[0] + 1e-10 and
                -1e-10 <= x2 <= b[1] + 1e-10 and
                -1e-10 <= x3 <= b[2] + 1e-10):
                candidates.append((x1, x2))
    pts = np.unique(np.round(np.asarray(candidates), 10), axis=0)
    # sort counter-clockwise around centroid
    center = pts.mean(axis=0)
    order = np.argsort(np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0]))
    return pts[order]


def coupling_from_xy(x1: float, x2: float, a1: float, b: np.ndarray) -> np.ndarray:
    x3 = a1 - x1 - x2
    return np.array([[x1, x2, x3], [b[0] - x1, b[1] - x2, b[2] - x3]])


def barycentric_xy(G: np.ndarray) -> tuple[float, float]:
    return float(G[0, 0]), float(G[0, 1])


def main() -> None:
    out_dir = REPO_ROOT / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Schematic marginals (rows: X; columns: Y).
    a = np.array([0.55, 0.45])
    b = np.array([0.40, 0.35, 0.25])
    a1 = a[0]
    vertices = feasible_vertices(a1, b)

    # A feasible empirical coupling P_obs (interior point).
    P_obs = coupling_from_xy(0.18, 0.22, a1, b)

    # A geometry-aware OT optimum for the schematic cost below.
    # Lower costs for (x1,y1), (x1,y3), and (x2,y2), with the coupling
    # constraints forcing a trade-off along the polygon boundary.
    C = np.array([[0.15, 1.20, 0.30],
                  [1.10, 0.20, 1.00]])
    d = C[0] - C[1]
    # Objective in x1,x2 after eliminating x3:
    # const + (d1-d3)x1 + (d2-d3)x2.
    coeff = np.array([d[0] - d[2], d[1] - d[2]])
    scores = vertices @ coeff
    gstar_xy = vertices[np.argmin(scores)]
    G_star = coupling_from_xy(gstar_xy[0], gstar_xy[1], a1, b)

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    poly = Polygon(vertices, closed=True, facecolor="#EAF0F7", edgecolor="#4C6A85",
                   linewidth=1.8, zorder=1)
    ax.add_patch(poly)

    # Linear OT objective contours in the 2D slice.
    xmin, xmax = vertices[:, 0].min() - 0.05, vertices[:, 0].max() + 0.05
    ymin, ymax = vertices[:, 1].min() - 0.05, vertices[:, 1].max() + 0.05
    xx, yy = np.meshgrid(np.linspace(xmin, xmax, 300), np.linspace(ymin, ymax, 300))
    zz = coeff[0] * xx + coeff[1] * yy
    levels = np.linspace(zz.min() + 0.15, zz.max() - 0.15, 7)
    ax.contour(xx, yy, zz, levels=levels, colors="#98A4B3", linewidths=0.8, alpha=0.8, zorder=2)

    # Feasible point and OT solution.
    pxy = np.array(barycentric_xy(P_obs))
    gxy = np.array(barycentric_xy(G_star))
    ax.scatter(*pxy, s=95, marker="o", facecolor="#C95C5C", edgecolor="white",
               linewidth=1.2, zorder=5, label=r"empirical coupling $P_{\rm obs}$")
    ax.scatter(*gxy, s=125, marker="*", facecolor="#2F6B4F", edgecolor="white",
               linewidth=1.0, zorder=6, label=r"geometry-aware coupling $\Gamma_C^{\star}$")
    ax.annotate("", xy=gxy, xytext=pxy,
                arrowprops=dict(arrowstyle="->", lw=1.8, color="#2F6B4F"), zorder=4)

    # Label two conceptual regions.
    centroid = vertices.mean(axis=0)
    ax.text(centroid[0] - 0.13, centroid[1] + 0.17,
            r"all $\Gamma\in\Pi(a,b)$ with fixed marginals",
            ha="center", va="center", fontsize=9.2, color="#4C6A85")
    ax.text(pxy[0] + 0.08, pxy[1] + 0.02,
            r"fixed by observed counts",
            ha="right", va="center", fontsize=9.2, color="#C95C5C")
    ax.text(gxy[0] + 0.02, gxy[1] - 0.025,
            r"minimum transport cost",
            ha="right", va="bottom", fontsize=9.2, color="#2F6B4F")

    ax.set_xlabel(r"$\Gamma_{11}$", fontsize=10.5)
    ax.set_ylabel(r"$\Gamma_{12}$", fontsize=10.5)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.18, linewidth=0.7)
    ax.legend(loc="upper right", frameon=True, fontsize=9)
    ax.set_title(r"Coupling polytope: same marginals, different couplings", fontsize=12)
    ax.text(0.28, 0.99, r"schematic $2\times3$ slice", transform=ax.transAxes,
            ha="right", va="top", fontsize=8.8, color="#5D6670")

    # A compact formula box to make the connection to the paper explicit.
    formula = (r"$\Pi(a,b)=\{\Gamma\geq0:\Gamma\mathbf{1}=a,\;"
               r"\Gamma^{\top}\mathbf{1}=b\}$")
    ax.text(0.02, 0.02, formula, transform=ax.transAxes, fontsize=10,
            bbox=dict(boxstyle="round,pad=0.38", facecolor="white", edgecolor="#B5BDC7"),
            ha="left", va="bottom")

    fig.tight_layout()
    fig.savefig(out_dir / "coupling_polytope_schematic.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "coupling_polytope_schematic.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("P_obs =\n", P_obs)
    print("C =\n", C)
    print("Gamma* =\n", G_star)
    print("Figure written to", out_dir / "coupling_polytope_schematic.pdf")


if __name__ == "__main__":
    main()
