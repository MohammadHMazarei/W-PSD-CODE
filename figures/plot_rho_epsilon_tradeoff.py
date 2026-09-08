"""Regenerate the rho-epsilon tradeoff heatmap from an existing CSV result.

This script only re-plots the saved experiment results; it does NOT rerun the
expensive OT/PSD optimisation sweep.

Usage:
    python code/figures/plot_rho_epsilon_tradeoff.py
    python code/figures/plot_rho_epsilon_tradeoff.py --csv code/results/rho_epsilon_tradeoff.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
FIGURES = HERE.parent / "figures"
ABLATION = HERE / "ablation"
sys.path.insert(0, str(ABLATION))

from run_rho_epsilon_tradeoff import plot_tradeoff


def load_rows(path: Path):
    rows = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            row["rho"] = float(row["rho"])
            row["epsilon"] = float(row["epsilon"])
            for key in ("p_at_5", "p_at_10", "p_at_20", "l1_gamma_pobs",
                        "iterations", "time_seconds", "final_free_energy"):
                if key in row:
                    row[key] = float(row[key])
            rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        type=Path,
        default=HERE / "results" / "rho_epsilon_tradeoff.csv",
        help="saved rho-epsilon sweep CSV",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=FIGURES,
        help="directory for PDF and PNG figures",
    )
    args = parser.parse_args()

    rows = load_rows(args.csv)
    rho_values = sorted({r["rho"] for r in rows})
    eps_values = sorted({r["epsilon"] for r in rows})
    args.out_dir.mkdir(parents=True, exist_ok=True)

    out_pdf = args.out_dir / "rho_epsilon_tradeoff.pdf"
    out_png = args.out_dir / "rho_epsilon_tradeoff.png"
    plot_tradeoff(rows, rho_values, eps_values, out_pdf, out_png)
    print(f"Wrote {out_pdf}")
    print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()
