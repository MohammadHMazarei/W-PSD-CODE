"""
E-step coupling updates.

Three coupling updates are provided, all of them exact block minimisers
of the free-energy objective with respect to Gamma for fixed G:

1. ``exact_ot_coupling``    (eps = 0)
       Gamma = argmin_{Gamma in Pi(px,py)} <Gamma, C>
   solved by the network simplex.

2. ``entropic_ot_coupling`` (eps > 0, no reference)
       Gamma = argmin <Gamma,C> - eps H(Gamma)
   the classical Sinkhorn problem.  NOTE: this update -- like (1) --
   depends on the data ONLY through the marginals px, py.  See
   ``diagnostics/check_data_dependence.py``.

3. ``kl_ot_coupling``       (eps > 0, reference coupling Q)
       Gamma = argmin <Gamma,C> + eps KL(Gamma || Q)
   the KL-anchored update used by W-PSD-CODE.  Setting Q = 1 (log_Q = 0)
   recovers (2) exactly, and Q = Pobs with eps -> infinity recovers the
   fixed empirical coupling of PSD-CODE, so (3) is a one-parameter family
   whose two endpoints are the OT coupling and the PSD-CODE coupling.

All Sinkhorn iterations are run in the log domain, which is numerically
stable for the small eps values used in the paper.
"""
from __future__ import annotations

import numpy as np
from scipy.special import logsumexp

from ._ot_backend import emd

LOG_FLOOR = 1e-300


def exact_ot_coupling(pbar_x: np.ndarray, pbar_y: np.ndarray,
                      C: np.ndarray) -> np.ndarray:
    """Exact (unregularised) OT plan, Eq. (eq:exact_ot_step)."""
    return emd(pbar_x, pbar_y, C)


def reference_coupling(Pobs: np.ndarray, pbar_x: np.ndarray,
                       pbar_y: np.ndarray, rho: float) -> np.ndarray:
    """
    Reference coupling Q = (1 - rho) Pobs + rho (px py^T), Eq. (eq:reference).

    The mixture with the independence coupling guarantees Q > 0 entrywise,
    so KL(Gamma || Q) is finite on the whole coupling polytope.  Without it,
    KL(.||Pobs) would be +infinity for any Gamma putting mass on a pair that
    was never observed -- which would defeat the purpose of letting the
    coupling discover unobserved relationships.
    """
    if not (0.0 < rho < 1.0):
        raise ValueError("rho must lie strictly between 0 and 1")
    Q = (1.0 - rho) * Pobs + rho * np.outer(pbar_x, pbar_y)
    return Q / Q.sum()


def log_reference(Q: np.ndarray) -> np.ndarray:
    return np.log(np.maximum(Q, LOG_FLOOR))


def kl_ot_coupling(pbar_x: np.ndarray, pbar_y: np.ndarray, C: np.ndarray,
                   eps: float, log_Q=None, num_iter: int = 3000,
                   stop_thr: float = 1e-10) -> np.ndarray:
    """
    Unique minimiser of  <Gamma,C> + eps * KL(Gamma || Q)  over
    Pi(pbar_x, pbar_y), Eq. (eq:e_step_kl).

    The solution has the Gibbs form Gamma = diag(u) (Q * exp(-C/eps)) diag(v)
    (Proposition prop:sinkhorn_form); u and v are obtained by log-domain
    Sinkhorn iterations on the dual potentials f = eps log u, g = eps log v.

    ``log_Q = None`` is treated as Q = 1, i.e. the plain entropic OT problem
    with -eps H(Gamma).
    """
    pbar_x = np.asarray(pbar_x, dtype=float).ravel()
    pbar_y = np.asarray(pbar_y, dtype=float).ravel()
    n, m = C.shape
    if log_Q is None:
        log_Q = np.zeros((n, m))

    log_K = log_Q - C / eps
    f = np.zeros(n)
    g = np.zeros(m)
    log_px = np.log(pbar_x)
    log_py = np.log(pbar_y)

    for _ in range(num_iter):
        f_prev = f
        f = eps * (log_px - logsumexp(log_K + g[None, :] / eps, axis=1))
        g = eps * (log_py - logsumexp(log_K + f[:, None] / eps, axis=0))
        if np.max(np.abs(f - f_prev)) < stop_thr:
            break

    return np.exp(log_K + f[:, None] / eps + g[None, :] / eps)


def entropic_ot_coupling(pbar_x: np.ndarray, pbar_y: np.ndarray,
                         C: np.ndarray, eps: float, num_iter: int = 3000,
                         stop_thr: float = 1e-10) -> np.ndarray:
    """Classical entropic OT: the ``log_Q = 0`` case of kl_ot_coupling."""
    return kl_ot_coupling(pbar_x, pbar_y, C, eps, None, num_iter, stop_thr)


# ---------------------------------------------------------------------------
# Fused Gromov-Wasserstein coupling.
#
# Unlike exact_ot_coupling / entropic_ot_coupling / kl_ot_coupling above,
# whose cost matrix C(G) is derived entirely from the current embedding
# (Section Why the Coupling Update Adds No Signal), this coupling additionally constrains the
# transport plan using two within-domain structure matrices D_x, D_y that
# are built from EXOGENOUS information -- pretrained word-vector distances
# between entity names (see common/aux_geometry.py) -- and are therefore
# not a function of G. This is the concrete instantiation of the
# Gromov-Wasserstein extension proposed in the paper's Future Work
# paragraph, using Vayer et al.'s (2019) fused Gromov-Wasserstein
# objective as implemented in the POT library.
# ---------------------------------------------------------------------------

def fgw_coupling(pbar_x: np.ndarray, pbar_y: np.ndarray, C: np.ndarray,
                 D_x: np.ndarray, D_y: np.ndarray, alpha: float) -> np.ndarray:
    """
    Gamma = argmin_{Gamma in Pi(px,py)}
                (1-alpha) <Gamma, C>
                + alpha sum_{x,x',y,y'} |D_x(x,x') - D_y(y,y')|^2 Gamma(x,y) Gamma(x',y')

    ``alpha`` trades off the cross-domain embedding cost C (as in the
    other three couplings) against the within-domain structure terms
    D_x, D_y. alpha=0 recovers exact_ot_coupling exactly.
    """
    import ot.gromov as gw
    pbar_x = np.asarray(pbar_x, dtype=float).ravel()
    pbar_y = np.asarray(pbar_y, dtype=float).ravel()
    Gamma = gw.fused_gromov_wasserstein(
        C, D_x, D_y, pbar_x, pbar_y,
        loss_fun="square_loss", alpha=alpha, symmetric=True,
        max_iter=2000, tol_rel=1e-9, tol_abs=1e-9)
    return np.asarray(Gamma)


# ---------------------------------------------------------------------------
# Debiased entropic coupling, following the Sinkhorn-divergence bias
# correction of Feydy et al. (2019) [genevay2018sinkhorn is the closest
# entry already in the bibliography for this line of work; the exact
# reference is Feydy, Seguy, Damodaran, Rolet, Fung, Trouve, Peyre,
# "Interpolating between Optimal Transport and MMD using Sinkhorn
# Divergences", AISTATS 2019].
#
# Plain entropic OT (entropic_ot_coupling / kl_ot_coupling with eps>0) is
# known to be a biased estimator of the true OT coupling: because both
# marginals are embedded in the SAME space here (phi(x) and psi(y) both
# live in R^q), we can compute the two "self" entropic couplings
# Gamma(px,px) and Gamma(py,py) using the same eps and the same
# within-domain squared-distance costs, and use their dual potentials to
# correct the cross-domain potentials before forming Gamma(px,py). This
# is the standard practical debiasing construction used in the Sinkhorn
# divergence literature; we flag explicitly that, unlike the three
# couplings above, we have not re-derived the block-minimisation and
# monotonicity guarantees this variant -- it is an
# empirical variant, not a proven block minimiser of J_eps.
# ---------------------------------------------------------------------------

def _sinkhorn_potentials(pbar_a: np.ndarray, pbar_b: np.ndarray,
                         C: np.ndarray, eps: float, num_iter: int = 3000,
                         stop_thr: float = 1e-10):
    """Return converged dual potentials (f, g) for entropic OT(pbar_a,
    pbar_b; C), without materialising the coupling matrix."""
    n, m = C.shape
    log_K = -C / eps
    f = np.zeros(n)
    g = np.zeros(m)
    log_pa = np.log(np.maximum(pbar_a, 1e-300))
    log_pb = np.log(np.maximum(pbar_b, 1e-300))
    for _ in range(num_iter):
        f_prev = f
        f = eps * (log_pa - logsumexp(log_K + g[None, :] / eps, axis=1))
        g = eps * (log_pb - logsumexp(log_K + f[:, None] / eps, axis=0))
        if np.max(np.abs(f - f_prev)) < stop_thr:
            break
    return f, g


def debiased_entropic_ot_coupling(pbar_x: np.ndarray, pbar_y: np.ndarray,
                                  C: np.ndarray, C_xx: np.ndarray,
                                  C_yy: np.ndarray, eps: float,
                                  num_iter: int = 3000) -> np.ndarray:
    """
    Sinkhorn-divergence-debiased coupling: subtract the self-transport
    potentials (f_aa on the x side, g_bb on the y side) from the
    cross-transport potentials (f_ab, g_ab) before forming Gamma, which
    cancels the entropic blur that plain entropic_ot_coupling exhibits
    (Feydy et al., 2019). ``C_xx``/``C_yy`` are the within-domain
    squared-distance matrices in the SAME embedding used for ``C``.
    """
    pbar_x = np.asarray(pbar_x, dtype=float).ravel()
    pbar_y = np.asarray(pbar_y, dtype=float).ravel()

    f_ab, g_ab = _sinkhorn_potentials(pbar_x, pbar_y, C, eps, num_iter)
    f_aa, _ = _sinkhorn_potentials(pbar_x, pbar_x, C_xx, eps, num_iter)
    _, g_bb = _sinkhorn_potentials(pbar_y, pbar_y, C_yy, eps, num_iter)

    f_deb = f_ab - f_aa
    g_deb = g_ab - g_bb

    log_K = -C / eps
    Gamma = np.exp(log_K + f_deb[:, None] / eps + g_deb[None, :] / eps)
    # The debiasing correction is not guaranteed to preserve the exact
    # marginals to machine precision; renormalise onto Pi(px,py) with a
    # short IPFP polish so the output remains a valid coupling matrix.
    for _ in range(50):
        Gamma *= (pbar_x / np.maximum(Gamma.sum(axis=1), 1e-300))[:, None]
        Gamma *= (pbar_y / np.maximum(Gamma.sum(axis=0), 1e-300))[None, :]
    return Gamma
