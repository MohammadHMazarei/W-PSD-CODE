"""
Data loading for W-PSD-CODE / PSD-CODE.

Expects the standard Animals-with-Attributes (AwA/AwA2) class-attribute
files under ``data_dir`` (default: /data):

    predicate-matrix-binary.txt   (n x m, whitespace-separated 0/1 matrix)
    classes.txt                   (n lines: "<id> <class-name>")
    predicates.txt                (m lines: "<id> <predicate-name>")

Common alternative filenames (predicate-matrix-binary.csv, classes.csv,
predicates.csv; comma-separated) are also tried.

If none of these files can be found, a synthetic binary co-occurrence
matrix with the same shape and density reported in the paper
(n=50, m=85, ~36.8% density) is generated instead, so that the pipeline
can still be exercised end to end. A clear warning is printed in that
case -- the resulting numbers will NOT match Table 2 of the paper,
which requires the real AwA/AwA2 files.
"""
from __future__ import annotations

import os
import warnings
import numpy as np

CANDIDATE_MATRIX_NAMES = [
    "predicate-matrix-binary.txt",
    "predicate-matrix-binary.csv",
    "predicate_matrix_binary.txt",
    "class_attribute_matrix.txt",
]
CANDIDATE_CLASS_NAMES = ["classes.txt", "classes.csv"]
CANDIDATE_PRED_NAMES = ["predicates.txt", "predicates.csv"]


def _find_file(data_dir: str, candidates: list[str]) -> str | None:
    for name in candidates:
        path = os.path.join(data_dir, name)
        if os.path.isfile(path):
            return path
    return None


def _load_matrix(path: str) -> np.ndarray:
    delim = "," if path.endswith(".csv") else None
    mat = np.loadtxt(path, delimiter=delim)
    return (mat > 0).astype(float)


def _load_names(path: str) -> list[str]:
    names = []
    delim = "," if path.endswith(".csv") else None
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(delim) if delim else line.split(None, 1)
            names.append(parts[-1].strip())
    return names


def _synthetic_dataset(n: int = 50, m: int = 85, density: float = 0.368,
                        seed: int = 0):
    warnings.warn(
        "No AwA/AwA2 class-attribute files found in the data directory. "
        "Falling back to a SYNTHETIC binary co-occurrence matrix with "
        f"n={n}, m={m}, density={density:.3f} so the pipeline can still "
        "be run end to end. Results produced from synthetic data will "
        "NOT reproduce Table 2 of the paper -- place the real AwA/AwA2 "
        "predicate-matrix-binary.txt (+ classes.txt, predicates.txt) in "
        "the data directory to reproduce the reported numbers.",
        stacklevel=2,
    )
    rng = np.random.default_rng(seed)
    counts = (rng.random((n, m)) < density).astype(float)
    # guarantee no empty row/column so marginals stay strictly positive
    for i in range(n):
        if counts[i].sum() == 0:
            counts[i, rng.integers(m)] = 1.0
    for j in range(m):
        if counts[:, j].sum() == 0:
            counts[rng.integers(n), j] = 1.0
    class_names = [f"class_{i}" for i in range(n)]
    pred_names = [f"attr_{j}" for j in range(m)]
    return counts, class_names, pred_names


def load_awa_dataset(data_dir: str = "/data"):
    """
    Returns
    -------
    counts : (n, m) float ndarray of {0,1} co-occurrence counts
    class_names : list[str] of length n
    pred_names  : list[str] of length m
    """
    matrix_path = _find_file(data_dir, CANDIDATE_MATRIX_NAMES)
    if matrix_path is None:
        return _synthetic_dataset()

    counts = _load_matrix(matrix_path)
    n, m = counts.shape

    class_path = _find_file(data_dir, CANDIDATE_CLASS_NAMES)
    pred_path = _find_file(data_dir, CANDIDATE_PRED_NAMES)
    class_names = _load_names(class_path) if class_path else [f"class_{i}" for i in range(n)]
    pred_names = _load_names(pred_path) if pred_path else [f"attr_{j}" for j in range(m)]

    if len(class_names) != n:
        class_names = [f"class_{i}" for i in range(n)]
    if len(pred_names) != m:
        pred_names = [f"attr_{j}" for j in range(m)]

    return counts, class_names, pred_names


def build_marginals(counts: np.ndarray):
    """
    Build the normalised empirical co-occurrence coupling Pobs and its
    marginals pbar_x, pbar_y, matching Eq. (eq:pobs) of the paper.
    """
    total = counts.sum()
    Pobs = counts / total
    pbar_x = Pobs.sum(axis=1)
    pbar_y = Pobs.sum(axis=0)
    return Pobs, pbar_x, pbar_y
