"""W-PSD-CODE with the KL-anchored coupling update (the formulation of the paper).

Usage:
    python run_wpsd_kl.py [--data-dir DIR] [--out-dir DIR]
"""
import argparse, os, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from common.algorithm import RunConfig
from common.runner import run_and_report

DEFAULT_DATA = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "data", "AwA2")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=DEFAULT_DATA)
    p.add_argument("--out-dir", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "results"))
    a = p.parse_args()
    cfg = RunConfig(lam=0.001, eps=0.05, rho=0.05, q=2, alpha0=0.02, decay=0.99, num_outer=60, num_inner=10, coupling="kl")
    run_and_report(cfg, "W-PSD-CODE (KL-anchored, eps=0.05)", a.data_dir, a.out_dir, "wpsd_kl", save_arrays=True)


if __name__ == "__main__":
    main()
