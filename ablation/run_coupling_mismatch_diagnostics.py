"""Coupling-mismatch diagnostics for the KL-anchored W-PSD-CODE sweep.

For every (rho, epsilon) setting, run the same screening optimisation used in
Section 6.4 and record:

  1. L1(Gamma_KL*, Pobs): displacement of the actual KL-anchored coupling;
  2. delta(Pobs, Gamma_C*): transport-cost gap between Pobs and the exact OT
     plan for the *final learned geometry* C(G*);
  3. P@10 of the resulting embedding.

The second quantity is intentionally computed with an exact OT plan even when
training used the KL-anchored coupling: it is the paper's diagnostic
\\delta(Pobs, Gamma_C*) from Eq. (mismatch), evaluated at the final geometry.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from common.algorithm import RunConfig, run_alternating_optimisation
from common.data import build_marginals, load_awa_dataset
from common.metrics import embed, l1_coupling_displacement, precision_at_k
from common.objective import cost_matrix
from common.ot_solvers import exact_ot_coupling
from common.ot_solvers import log_reference, reference_coupling

DEFAULT_RHO = (0.005, 0.03, 0.05, 0.10, 0.20, 0.50)
DEFAULT_EPS = (0.005, 0.01, 0.03, 0.05, 0.10, 0.20)
DEFAULT_DATA = HERE / "data" / "AwA2"


def run_grid(data_dir: Path, rho_values, eps_values, outer: int, inner: int):
    full, _, _ = load_awa_dataset(str(data_dir))
    Pobs, px, py = build_marginals(full)
    n, m = Pobs.shape
    rows = []
    total = len(rho_values) * len(eps_values)
    done = 0

    for rho in rho_values:
        Q = reference_coupling(Pobs, px, py, rho)
        log_Q = log_reference(Q)
        for eps in eps_values:
            done += 1
            print(f"[{done:02d}/{total}] rho={rho:g}, eps={eps:g}")
            cfg = RunConfig(
                eps=eps,
                rho=rho,
                q=2,
                lam=0.001,
                alpha0=0.02,
                decay=0.99,
                num_outer=outer,
                num_inner=inner,
                coupling="kl",
                init="identity",
            )
            t0 = time.time()
            result = run_alternating_optimisation(Pobs, px, py, cfg)
            phi, psi = embed(result.G, n, m, cfg.q)
            p10 = precision_at_k(phi, psi, full, 10)
            l1 = l1_coupling_displacement(result.Gamma, Pobs)

            C = cost_matrix(result.G, n, m)
            Gamma_ot = exact_ot_coupling(px, py, C)
            delta = float(np.sum((Pobs - Gamma_ot) * C))
            obs_cost = float(np.sum(Pobs * C))
            ot_cost = float(np.sum(Gamma_ot * C))

            rows.append({
                "rho": float(rho),
                "epsilon": float(eps),
                "p_at_10": float(p10),
                "l1_gamma_pobs": float(l1),
                "delta_pobs_gamma_ot": delta,
                "pobs_transport_cost": obs_cost,
                "exact_ot_transport_cost": ot_cost,
                "iterations": int(result.iterations),
                "runtime_seconds": float(result.time_seconds),
                "wall_clock_seconds": float(time.time() - t0),
            })
    return rows


def write_outputs(rows, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "coupling_mismatch_diagnostics.csv"
    json_path = out_dir / "coupling_mismatch_diagnostics.json"
    fields = list(rows[0].keys())
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    payload = {
        "protocol": "rho-epsilon screening sweep; P@10 evaluated on full AwA/AwA2; delta uses exact OT at final learned geometry",
        "rho_values": sorted({r["rho"] for r in rows}),
        "epsilon_values": sorted({r["epsilon"] for r in rows}),
        "rows": rows,
    }
    json_path.write_text(json.dumps(payload, indent=2))
    return csv_path, json_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--out-dir", type=Path, default=HERE / "results")
    ap.add_argument("--rho", nargs="+", type=float,
                    default=[0.005, 0.03, 0.05, 0.10, 0.20, 0.50])
    ap.add_argument("--eps", nargs="+", type=float,
                    default=[0.005, 0.05])
    ap.add_argument("--outer-iters", type=int, default=30)
    ap.add_argument("--inner-iters", type=int, default=5)
    args = ap.parse_args()

    rows = run_grid(args.data_dir, sorted(set(args.rho)), sorted(set(args.eps)),
                    args.outer_iters, args.inner_iters)
    csv_path, json_path = write_outputs(rows, args.out_dir)
    print(f"Saved {csv_path}")
    print(f"Saved {json_path}")




if __name__ == "__main__":
    main()
