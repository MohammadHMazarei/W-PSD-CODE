"""PSD projection (Eq. eq:psd_projection), Gram-matrix initialisation,
and the projected-gradient step with Armijo backtracking
(Eq. eq:pgd_revised)."""
from __future__ import annotations

import numpy as np

from .objective import free_energy, gradient


def psd_project(A: np.ndarray) -> np.ndarray:
    """Proj_{S+^N}(A) via eigenvalue clipping (Eq. eq:psd_projection)."""
    A_sym = 0.5 * (A + A.T)
    eigvals, eigvecs = np.linalg.eigh(A_sym)
    eigvals_clipped = np.clip(eigvals, 0.0, None)
    return (eigvecs * eigvals_clipped) @ eigvecs.T


def init_gram(Pobs: np.ndarray, mode: str = "identity", rank: int = 8,
              noise: float = 1e-2) -> np.ndarray:
    """
    Initial Gram matrix G^(0).

    ``mode="identity"`` returns G^(0) = I_N, exactly as stated in
    Algorithm 1 of the paper.  This is the setting used for all reported
    results, so that the reported numbers correspond to the algorithm as
    written rather than to an undocumented warm start.

    ``mode="svd"`` returns a rank-``rank`` SVD factorisation of Pobs plus a
    small multiple of the identity.  This was the initialisation used in an
    earlier version of the code.  It is retained only for reproducing that
    earlier behaviour and is NOT used for the reported results: with the
    entropy-regularised (unanchored) objective it silently smuggles the
    observed co-occurrence structure into the run through the starting
    point, masking the fact that the objective itself does not depend on it
    (see ``diagnostics/check_data_dependence.py``).
    """
    n, m = Pobs.shape
    N = n + m
    if mode == "identity":
        return np.eye(N)
    if mode != "svd":
        raise ValueError(f"unknown init mode: {mode}")
    u, s, vt = np.linalg.svd(Pobs, full_matrices=False)
    r = min(rank, len(s))
    phi0 = u[:, :r] * np.sqrt(s[:r])
    psi0 = vt[:r, :].T * np.sqrt(s[:r])
    V0 = np.vstack([phi0, psi0])
    return V0 @ V0.T + noise * np.eye(N)


def pgd_step(G: np.ndarray, Gamma: np.ndarray, pbar_x: np.ndarray,
             pbar_y: np.ndarray, lam: float, eps: float, n: int, m: int,
             alpha: float, log_Q=None, c1: float = 1e-4,
             max_backtrack: int = 25):
    """
    One Armijo-backtracked projected-gradient step (Eq. eq:pgd_revised):

        G_{t+1} = Proj_{S+^N}(G_t - alpha_t * grad_G J_eps(G_t, Gamma))

    Returns the updated G, the step size actually used (a starting point for
    the next call), and the new objective value.
    """
    J0 = free_energy(G, Gamma, pbar_x, pbar_y, lam, eps, n, m, log_Q)
    grad = gradient(G, Gamma, pbar_x, pbar_y, lam, n, m)
    grad_sq_norm = float(np.sum(grad * grad))

    step = alpha
    G_new = psd_project(G - step * grad)
    for _ in range(max_backtrack):
        J_new = free_energy(G_new, Gamma, pbar_x, pbar_y, lam, eps, n, m, log_Q)
        if J_new <= J0 - c1 * step * grad_sq_norm or grad_sq_norm == 0.0:
            return G_new, step, J_new
        step *= 0.5
        G_new = psd_project(G - step * grad)

    J_new = free_energy(G_new, Gamma, pbar_x, pbar_y, lam, eps, n, m, log_Q)
    return G_new, step, J_new
