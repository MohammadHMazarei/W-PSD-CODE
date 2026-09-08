"""W-PSD-CODE with the exact (unregularised) OT coupling update, eps = 0.

Usage:
    python run_wpsd_exact.py [--data-dir DIR] [--out-dir DIR]
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
    cfg = RunConfig(lam=0.001, eps=0.0, q=2, alpha0=0.02, decay=0.99, num_outer=60, num_inner=10, coupling="exact")
    run_and_report(cfg, "W-PSD-CODE (exact OT)", a.data_dir, a.out_dir, "wpsd_exact_ot", save_arrays=False)


if __name__ == "__main__":
    main()
