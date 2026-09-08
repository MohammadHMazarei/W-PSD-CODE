"""
The W-PSD-CODE free-energy objective J_eps(G, Gamma) (Eq. eq:wpsd_objective)
and its gradient with respect to the Gram matrix (Eqs. grad_xx_revised,
grad_yy_revised, grad_cross_revised).

    J_eps(G,Gamma) = <Gamma, C(G)>                        transport term
                   + sum_x pbar_x(x) log Z_G(x)           PSD-CODE log-partition
                   + lambda tr(G)                         trace regularisation
                   + eps KL(Gamma || Q)                   KL-anchored coupling

The last term replaces the plain negative entropy -eps H(Gamma) of the
earlier draft.  Passing ``log_Q = None`` reproduces that earlier form
exactly, since KL(Gamma || 1) = sum Gamma log Gamma = -H(Gamma); passing
``log_Q = log Q`` with Q built from Pobs is the formulation actually used
by W-PSD-CODE, and is what makes the objective depend on the observed
co-occurrence structure rather than on its marginals alone.

Gram-matrix block convention (Eq. eq:G_block):

    G = [[G_XX, B],
         [B.T,  G_YY]]

with G_XX (n x n), G_YY (m x m), B (n x m).  Only the diagonal entries of
G_XX and G_YY, and the full cross block B, ever receive a non-zero
gradient: the squared-distance cost

    C_xy(G) = G_xx + G_yy - 2 G_xy               (Eq. eq:cost_G)

does not depend on the off-diagonal entries of G_XX or G_YY, and neither
does the trace regulariser.
"""
from __future__ import annotations

import numpy as np
from scipy.special import xlogy


def split_blocks(G: np.ndarray, n: int, m: int):
    G_XX = G[:n, :n]
    G_YY = G[n:, n:]
    B = G[:n, n:]
    return G_XX, G_YY, B


def cost_matrix(G: np.ndarray, n: int, m: int) -> np.ndarray:
    """C_xy(G) = G_xx + G_yy - 2 G_xy   (Eq. eq:cost_G)"""
    G_XX, G_YY, B = split_blocks(G, n, m)
    diag_x = np.diag(G_XX).reshape(n, 1)
    diag_y = np.diag(G_YY).reshape(1, m)
    C = diag_x + diag_y - 2.0 * B
    return np.maximum(C, 0.0)  # guard against tiny negative numerical noise


def within_block_cost(G_block: np.ndarray) -> np.ndarray:
    """Squared-distance matrix WITHIN one entity set (X-X or Y-Y), using
    the same Gram-matrix convention as ``cost_matrix``: for a k x k block
    G_block, C(i,j) = G_block[i,i] + G_block[j,j] - 2 G_block[i,j]. Used
    by the debiased-entropic coupling (``common.ot_solvers``), which
    needs a self-transport cost in the same embedding space as C_xy.
    """
    d = np.diag(G_block).reshape(-1, 1)
    C = d + d.T - 2.0 * G_block
    return np.maximum(C, 0.0)


def partition_and_conditional(C: np.ndarray, pbar_y: np.ndarray):
    """
    Z_G(x) (Eq. eq:psd_partition_revised) and r_G(y|x)
    (Eq. eq:psd_conditional_revised), computed with a per-row max-shift
    for numerical stability (does not change the ratio r_G).
    """
    shift = C.min(axis=1, keepdims=True)
    unnorm = pbar_y[None, :] * np.exp(-(C - shift))
    Z_shifted = unnorm.sum(axis=1, keepdims=True)
    r_G = unnorm / Z_shifted
    log_Z = np.log(Z_shifted.squeeze(-1)) - shift.squeeze(-1)
    return log_Z, r_G


def entropy(Gamma: np.ndarray) -> float:
    """H(Gamma) = -sum Gamma*log(Gamma), with convention 0 log 0 = 0
    (Eq. eq:entropy_gamma)."""
    return float(-xlogy(Gamma, Gamma).sum())


def kl_to_reference(Gamma: np.ndarray, log_Q=None) -> float:
    """
    KL(Gamma || Q) = sum Gamma log(Gamma / Q), with 0 log 0 = 0
    (Eq. eq:kl_gamma).  ``log_Q = None`` means Q = 1, in which case this
    reduces to -H(Gamma).
    """
    val = float(xlogy(Gamma, Gamma).sum())
    if log_Q is not None:
        val -= float((Gamma * log_Q).sum())
    return val


def free_energy(G: np.ndarray, Gamma: np.ndarray, pbar_x: np.ndarray,
                pbar_y: np.ndarray, lam: float, eps: float,
                n: int, m: int, log_Q=None) -> float:
    """J_eps(G, Gamma), Eq. eq:wpsd_objective."""
    C = cost_matrix(G, n, m)
    log_Z, _ = partition_and_conditional(C, pbar_y)
    transport_term = float(np.sum(Gamma * C))
    log_partition_term = float(np.sum(pbar_x * log_Z))
    trace_term = lam * float(np.trace(G))
    coupling_term = eps * kl_to_reference(Gamma, log_Q) if eps > 0 else 0.0
    return transport_term + log_partition_term + trace_term + coupling_term


def gradient(G: np.ndarray, Gamma: np.ndarray, pbar_x: np.ndarray,
             pbar_y: np.ndarray, lam: float, n: int, m: int) -> np.ndarray:
    """
    Full (N x N) symmetric gradient of J_eps with respect to G, for fixed
    Gamma (Eqs. grad_xx_revised, grad_yy_revised, grad_cross_revised,
    block_cross_gradient, block_y_gradient).  The coupling term
    eps KL(Gamma || Q) does not contribute, since Gamma is held fixed
    during the M-step.
    """
    N = n + m
    C = cost_matrix(G, n, m)
    _, r_G = partition_and_conditional(C, pbar_y)
    R_G = pbar_x[:, None] * r_G  # Eq. eq:R_explicit

    grad = np.zeros((N, N))
    # X-diagonal block: lambda * I_n  (Eq. eq:grad_xx_revised)
    grad[:n, :n] = lam * np.eye(n)
    # Y-diagonal block: diag(pbar_y - R_G^T 1_n) + lambda * I_m
    # (Eqs. eq:grad_yy_revised, eq:block_y_gradient)
    y_diag = pbar_y - R_G.sum(axis=0)
    grad[n:, n:] = np.diag(y_diag) + lam * np.eye(m)
    # Cross block: 2 (R_G - Gamma)   (Eqs. eq:grad_cross_revised,
    # eq:block_cross_gradient)
    cross = 2.0 * (R_G - Gamma)
    grad[:n, n:] = cross
    grad[n:, :n] = cross.T
    return grad
