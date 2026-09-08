"""Sensitivity of the held-out attribute-split gain to the optimisation budget.

Reruns the primary held-out validation/test protocol (see
run_heldout_validation_test.py) at the frozen mu=0.01 on a small subset of
test seeds (2000-2004), under two optimisation budgets: the 20-outer/5-inner
screening budget used during development and the 60-outer/10-inner budget
used for the publication numbers. This checks whether the reported gain is
an artefact of the larger optimisation budget rather than the anchoring
mechanism itself.

Usage:
    python run_heldout_budget_sensitivity.py
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CODE_ROOT = os.path.join(HERE, "..")
sys.path.insert(0, CODE_ROOT)

from common.data import load_awa_dataset
from run_heldout_attribute_split import APPEARANCE_IDS_1INDEXED, ECOLOGICAL_IDS_1INDEXED
from run_heldout_validation_test import run_test

DATA_DIR = os.path.join(CODE_ROOT, "data", "AwA2")
OUT_DIR = os.path.join(CODE_ROOT, "results")


def main():
    counts, _, _ = load_awa_dataset(DATA_DIR)
    appearance = counts[:, [i - 1 for i in APPEARANCE_IDS_1INDEXED]]
    ecological = counts[:, [i - 1 for i in ECOLOGICAL_IDS_1INDEXED]]
    seeds = list(range(2000, 2005))

    all_results = {}
    for outer_iters, inner_iters in [(20, 5), (60, 10)]:
        r = run_test(appearance, ecological, 0.01, seeds, 0.90, outer_iters, inner_iters)
        all_results[f"{outer_iters}x{inner_iters}"] = r
        print(outer_iters, inner_iters,
              "mean gain", r["mean_gain_P@10"], "sd", r["sd_gain_P@10"],
              "mean baseline", r["mean_baseline_P@10"], "mean anchor", r["mean_anchored_P@10"])

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "holdout_budget_sensitivity.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
