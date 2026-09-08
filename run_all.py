"""
Runs every method reported in Table 2 with the settings of Section 6.1,
then writes Table 2 and Figures 1-2.

Usage:
    python run_all.py [--data-dir DIR]
"""
import argparse, json, os, subprocess, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.join(HERE, "..")
DEFAULT_DATA = os.path.join(HERE, "data", "AwA2")

SCRIPTS = [
    ("psd_code/run_psd_code.py", "psd_code"),
    ("wpsd_code_exact_ot/run_wpsd_exact.py", "wpsd_exact_ot"),
    ("wpsd_code_sinkhorn/run_wpsd_sinkhorn.py", "wpsd_sinkhorn"),
    ("wpsd_code_kl/run_wpsd_kl.py", "wpsd_kl"),
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=DEFAULT_DATA)
    args = p.parse_args()

    out_dir = os.path.join(HERE, "results")
    fig_dir = REPO_ROOT
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    for script, _ in SCRIPTS:
        subprocess.run([sys.executable, os.path.join(HERE, script),
                        "--data-dir", args.data_dir, "--out-dir", out_dir],
                       check=True)

    rows = []
    histories = {}
    for _, stem in SCRIPTS:
        with open(os.path.join(out_dir, f"{stem}.json")) as f:
            d = json.load(f)
        histories[d["Method"]] = d["history"]
        rows.append({k: v for k, v in d.items() if k != "history"})

    from common.report import print_table, to_latex_tabular
    print("\n=== Table 2 ===")
    print_table(rows)
    with open(os.path.join(out_dir, "table2.tex"), "w") as f:
        f.write(to_latex_tabular(rows))
    print(f"\nLaTeX table written to {os.path.join(out_dir, 'table2.tex')}")

    from common.plotting import plot_convergence, plot_embedding
    plot_convergence(histories, os.path.join(fig_dir, "wpsd_convergence.png"))

    G_path = os.path.join(out_dir, "wpsd_kl_G.npy")
    Gam_path = os.path.join(out_dir, "wpsd_kl_Gamma.npy")
    if os.path.isfile(G_path):
        from common.data import load_awa_dataset
        from common.metrics import embed
        counts, class_names, pred_names = load_awa_dataset(args.data_dir)
        n, m = counts.shape
        phi, psi = embed(np.load(G_path), n, m, q=2)
        plot_embedding(phi, psi, np.load(Gam_path), class_names, pred_names,
                       os.path.join(fig_dir, "embedding_plot.png"),
                       method_label="W-PSD-CODE (KL-anchored)")
    print("Figures written to", fig_dir)


if __name__ == "__main__":
    main()
