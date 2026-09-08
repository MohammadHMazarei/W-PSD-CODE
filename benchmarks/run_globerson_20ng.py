"""
Generalisation benchmark: document-word co-occurrence on 20 Newsgroups,
following Globerson, Chechik, Pereira & Tishby (2007), Section 8.

W-PSD-CODE was proposed and tested only on the AwA/AwA2 class-attribute
matrix. Since the argument for a geometry-aware coupling is generic to
co-occurrence embedding, not specific to attribute retrieval, we test it
here on a second, unrelated domain: joint embedding of documents and
words, which is the original benchmark used to introduce PSD-CODE's
parent method (CODE) in Globerson et al. (2007).

We reuse THREE of the newsgroup subsets from Table 1 of that paper so the
resulting doc-doc numbers can be read against a published reference
point, and reuse the SAME algorithm implementation (common/algorithm.py)
and the SAME four-method comparison as the AwA2 experiments and the
sparsity ablation, with no changes to the core method.

Usage:
    python run_globerson_20ng.py [--data-dir DIR] [--seeds 5]
"""
import argparse, json, os, sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from common.newsgroups_data import (load_newsgroup_docs, build_doc_word_matrix,
                                     build_marginals, DEFAULT_DATA)
from common.algorithm import RunConfig, run_alternating_optimisation
from common.metrics import embed, doc_doc_measure, doc_word_measure, word_given_class_table

METHODS = [
    ("PSD-CODE", dict(coupling="fixed", eps=0.0)),
    ("W-PSD-CODE (exact OT)", dict(coupling="exact", eps=0.0)),
    ("W-PSD-CODE (entropic)", dict(coupling="entropic", eps=0.01)),
    ("W-PSD-CODE (KL, eps=0.05)", dict(coupling="kl", eps=0.05, rho=0.05)),
]

# Subsets reproduced from Table 1 of Globerson et al. (2007). The
# "published doc-doc" figures are that paper's CODE column (times 100,
# their Table 1), included here purely as an external sanity check that
# our re-implementation of the co-occurrence-embedding idea on this data
# is in the right ballpark -- W-PSD-CODE is not the same objective as
# their CODE, so exact agreement is not expected.
SUBSETS = [
    {
        "name": "politics (mideast vs misc)",
        "categories": ["talk.politics.mideast", "talk.politics.misc"],
        "published_code_docdoc": 0.85,
    },
    {
        "name": "windows/hardware (confusable pair)",
        "categories": ["comp.os.ms-windows.misc", "comp.sys.ibm.pc.hardware"],
        "published_code_docdoc": 0.68,
    },
    {
        "name": "atheism/graphics/crypt (triple)",
        "categories": ["alt.atheism", "comp.graphics", "sci.crypt"],
        "published_code_docdoc": 0.66,
    },
]


def run_one(categories, seed, docs_per_class, vocab_size, q):
    tokenised, labels = load_newsgroup_docs(categories, docs_per_class, seed)
    counts, vocab, labels = build_doc_word_matrix(
        tokenised, labels=labels, vocab_size=vocab_size)
    Pobs, px, py = build_marginals(counts)
    word_given_class, _ = word_given_class_table(counts, labels)

    n, m = Pobs.shape
    results = {}
    for label, kw in METHODS:
        cfg = RunConfig(num_outer=60, q=q, init="identity", **kw)
        out = run_alternating_optimisation(Pobs, px, py, cfg)
        phi, psi = embed(out.G, n, m, q)
        dd = doc_doc_measure(phi, labels)
        dw = doc_word_measure(phi, psi, labels, word_given_class)
        results[label] = {"doc_doc": dd, "doc_word": dw}
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=DEFAULT_DATA)
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--docs-per-class", type=int, default=40)
    p.add_argument("--vocab-size", type=int, default=200)
    p.add_argument("--q", type=int, default=2)
    p.add_argument("--out-dir", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "results"))
    p.add_argument("--subsets", default=None,
                    help="comma-separated 0-based subset indices to run "
                         "(default: all). Results are merged into the "
                         "existing globerson_20ng.json if present.")
    a = p.parse_args()

    subsets_to_run = SUBSETS
    if a.subsets is not None:
        idx = [int(i) for i in a.subsets.split(",")]
        subsets_to_run = [SUBSETS[i] for i in idx]

    out_path = os.path.join(a.out_dir, "globerson_20ng.json")
    table = {}
    if os.path.exists(out_path):
        with open(out_path) as f:
            table = json.load(f)

    for subset in subsets_to_run:
        acc = {label: {"doc_doc": [], "doc_word": []} for label, _ in METHODS}
        for seed in range(a.seeds):
            res = run_one(subset["categories"], seed, a.docs_per_class,
                          a.vocab_size, a.q)
            for label in acc:
                acc[label]["doc_doc"].append(res[label]["doc_doc"])
                acc[label]["doc_word"].append(res[label]["doc_word"])

        print(f"\n### {subset['name']}  "
              f"({', '.join(subset['categories'])}), "
              f"{a.docs_per_class} docs/class, vocab={a.vocab_size}, "
              f"q={a.q}, {a.seeds} seeds")
        base_dd = np.array(acc["PSD-CODE"]["doc_doc"])
        for label in acc:
            dd = np.array(acc[label]["doc_doc"])
            dw = np.array(acc[label]["doc_word"])
            if label == "PSD-CODE":
                p_str = "--"
            else:
                _, pval = stats.ttest_rel(dd, base_dd)
                p_str = f"{pval:.3f}"
            print(f"  {label:28s} doc-doc = {dd.mean():.4f} +/- {dd.std():.4f}"
                  f"   doc-word = {dw.mean():.4f} +/- {dw.std():.4f}"
                  f"   p(doc-doc vs PSD-CODE) = {p_str}")
        print(f"  [Globerson et al. 2007, Table 1, CODE doc-doc measure "
              f"on a comparable subset: {subset['published_code_docdoc']:.2f}]")

        table[subset["name"]] = {
            "categories": subset["categories"],
            "published_code_docdoc": subset["published_code_docdoc"],
            "results": {k: {kk: [float(x) for x in vv] for kk, vv in v.items()}
                        for k, v in acc.items()},
        }

    os.makedirs(a.out_dir, exist_ok=True)
    with open(os.path.join(a.out_dir, "globerson_20ng.json"), "w") as f:
        json.dump(table, f, indent=2)


if __name__ == "__main__":
    main()
