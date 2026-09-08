"""
Optimal-transport backend.

Uses the POT library (``pip install POT``) when it is available, and
otherwise falls back to a self-contained SciPy implementation so that the
whole package can be run in an environment without POT installed.

The fallback solves the exact OT problem as a linear program with the
HiGHS solver (the transportation LP is totally unimodular, so the LP
optimum is attained at a vertex of the coupling polytope and coincides
with the network-simplex solution used by ``ot.emd``).
"""
from __future__ import annotations

import numpy as np

try:  # pragma: no cover - depends on the environment
    import ot as _pot

    HAVE_POT = True
except Exception:  # pragma: no cover
    _pot = None
    HAVE_POT = False

from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _emd_scipy(a: np.ndarray, b: np.ndarray, M: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    M = np.asarray(M, dtype=float)
    n, m = M.shape
    a = a / a.sum()
    b = b / b.sum()

    rows, cols = [], []
    for i in range(n):
        for j in range(m):
            rows.append(i)
            cols.append(i * m + j)
    for j in range(m):
        for i in range(n):
            rows.append(n + j)
            cols.append(i * m + j)
    A = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n + m, n * m))

    res = linprog(M.ravel(), A_eq=A, b_eq=np.concatenate([a, b]),
                  bounds=(0, None), method="highs")
    if not res.success:
        raise RuntimeError(f"exact OT LP failed: {res.message}")
    return res.x.reshape(n, m)


def emd(a: np.ndarray, b: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Exact OT plan for cost ``M`` and marginals ``a``, ``b``."""
    if HAVE_POT:
        return _pot.emd(np.ascontiguousarray(a), np.ascontiguousarray(b),
                        np.ascontiguousarray(M))
    return _emd_scipy(a, b, M)


def emd2(a: np.ndarray, b: np.ndarray, M: np.ndarray) -> float:
    """Exact OT cost."""
    if HAVE_POT:
        return float(_pot.emd2(np.ascontiguousarray(a),
                               np.ascontiguousarray(b),
                               np.ascontiguousarray(M)))
    G = _emd_scipy(a, b, M)
    return float(np.sum(G * np.asarray(M, dtype=float)))
