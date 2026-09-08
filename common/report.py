"""
Formats the results dict produced by each run_*.py script into the
Table 2 layout (both a plain-text table and a LaTeX tabular snippet
matching tab:results in the manuscript).
"""
from __future__ import annotations

import json


COLUMNS = ["Method", "Mean P@5", "Mean P@10", "Mean P@20",
           "Mean Delta_x (W2)", "L1(Gamma,Pobs)", "Iterations", "Time (s)"]


def save_json(results: dict, out_path: str):
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)


def print_table(rows: list[dict]):
    widths = [max(len(str(r[c])) for r in rows + [dict(zip(COLUMNS, COLUMNS))])
              for c in COLUMNS]
    header = " | ".join(c.ljust(w) for c, w in zip(COLUMNS, widths))
    print(header)
    print("-" * len(header))
    for r in rows:
        print(" | ".join(str(r[c]).ljust(w) for c, w in zip(COLUMNS, widths)))


def to_latex_tabular(rows: list[dict]) -> str:
    lines = [
        r"\begin{table}",
        r"\centering",
        r"\caption{Reproducible results on the AwA/AwA2 class--attribute "
        r"matrix ($q=2$, $\lambda=0.001$)}",
        r"\label{tab:results}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lccccccc}",
        r"\toprule",
        r"\textbf{Method} & \textbf{P@5}$\uparrow$ & \textbf{P@10}$\uparrow$"
        r" & \textbf{P@20}$\uparrow$"
        r" & \textbf{Mean $\Delta_x$ (W2)}"
        r" & \textbf{$L_1(\Gamma,\Pobs)$}"
        r" & \textbf{Iter.}"
        r" & \textbf{Time (s)} \\",
        r"\midrule",
    ]
    for r in rows:
        lines.append(
            f"{r['Method']} & {r['Mean P@5']:.4f} & {r['Mean P@10']:.4f} & "
            f"{r['Mean P@20']:.4f} & {r['Mean Delta_x (W2)']:.4f} & "
            f"{r['L1(Gamma,Pobs)']:.4f} & "
            f"{r['Iterations']} & {r['Time (s)']:.2f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"}", r"\end{table}"]
    return "\n".join(lines)
