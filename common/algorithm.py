"""
Generic alternating-optimisation runner implementing Algorithm 1
(``alg:wpsd_revised``).  All methods reported in Table 2 come from this
same loop, selected by ``RunConfig.coupling``:

  * PSD-CODE (fixed-lambda baseline) : coupling="fixed"     (Gamma == Pobs always)
  * W-PSD-CODE exact OT : coupling="exact",    eps=0
  * W-PSD-CODE entropic : coupling="entropic", eps>0   (no data anchor)
  * W-PSD-CODE KL       : coupling="kl",       eps>0, rho in (0,1)

so the run_*.py scripts are thin configuration wrappers around this one
implementation.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .objective import free_energy, cost_matrix, within_block_cost
from .psd_utils import pgd_step, init_gram
from .ot_solvers import (exact_ot_coupling, entropic_ot_coupling,
                         kl_ot_coupling, reference_coupling, log_reference,
                         fgw_coupling, debiased_entropic_ot_coupling)


@dataclass
class RunConfig:
    lam: float = 0.001
    eps: float = 0.0
    rho: float = 0.05          # reference-coupling mixing weight
    q: int = 2
    alpha0: float = 0.02
    decay: float = 0.99
    num_outer: int = 60
    num_inner: int = 10
    tau: float = 1e-8
    coupling: str = "kl"       # "fixed"|"exact"|"entropic"|"kl"|"fgw"|"entropic_debiased"
    init: str = "identity"     # matches Algorithm 1
    warmup_iters: int = 0      # outer iterations run with coupling="fixed"
                                # before switching on the configured coupling
                                # (two-stage / curriculum variant)
    # Fused Gromov-Wasserstein-specific (coupling="fgw")
    fgw_D_x: np.ndarray = None
    fgw_D_y: np.ndarray = None
    fgw_alpha: float = 0.5


@dataclass
class RunResult:
    G: np.ndarray
    Gamma: np.ndarray
    history: list = field(default_factory=list)
    iterations: int = 0
    time_seconds: float = 0.0


def run_alternating_optimisation(Pobs: np.ndarray, pbar_x: np.ndarray,
                                 pbar_y: np.ndarray,
                                 cfg: RunConfig) -> RunResult:
    n, m = Pobs.shape

    valid_couplings = {"fixed", "exact", "entropic", "kl", "fgw", "entropic_debiased"}
    if cfg.coupling not in valid_couplings:
        raise ValueError(f"unknown coupling mode: {cfg.coupling}")
    if cfg.coupling in {"entropic", "kl", "entropic_debiased"} and cfg.eps <= 0:
        raise ValueError(f"coupling={cfg.coupling!r} requires eps > 0")
    if cfg.coupling == "fgw" and (cfg.fgw_D_x is None or cfg.fgw_D_y is None):
        raise ValueError("coupling='fgw' requires fgw_D_x and fgw_D_y")

    log_Q = None
    if cfg.coupling == "kl":
        log_Q = log_reference(reference_coupling(Pobs, pbar_x, pbar_y, cfg.rho))

    G = init_gram(Pobs, mode=cfg.init)
    Gamma = Pobs.copy()
    alpha = cfg.alpha0
    history = []
    prev_J = None

    t0 = time.time()
    k = 0
    for k in range(cfg.num_outer):
        C = cost_matrix(G, n, m)
        active_coupling = "fixed" if k < cfg.warmup_iters else cfg.coupling

        # ---- E-step: block minimisation over Gamma ----------
        if active_coupling == "exact":
            Gamma = exact_ot_coupling(pbar_x, pbar_y, C)
        elif active_coupling == "entropic":
            Gamma = entropic_ot_coupling(pbar_x, pbar_y, C, cfg.eps)
        elif active_coupling == "kl":
            Gamma = kl_ot_coupling(pbar_x, pbar_y, C, cfg.eps, log_Q)
        elif active_coupling == "fgw":
            Gamma = fgw_coupling(pbar_x, pbar_y, C, cfg.fgw_D_x, cfg.fgw_D_y,
                                 cfg.fgw_alpha)
        elif active_coupling == "entropic_debiased":
            C_xx = within_block_cost(G[:n, :n])
            C_yy = within_block_cost(G[n:, n:])
            Gamma = debiased_entropic_ot_coupling(pbar_x, pbar_y, C, C_xx,
                                                  C_yy, cfg.eps)
        # "fixed" -> Gamma stays equal to Pobs (PSD-CODE baseline)

        # ---- M-step: projected gradient descent over G ------
        for _ in range(cfg.num_inner):
            G, alpha, J = pgd_step(G, Gamma, pbar_x, pbar_y, cfg.lam,
                                   cfg.eps, n, m, alpha, log_Q)

        J = free_energy(G, Gamma, pbar_x, pbar_y, cfg.lam, cfg.eps, n, m, log_Q)
        history.append(J)
        alpha *= cfg.decay

        if prev_J is not None:
            if abs(J - prev_J) / (1.0 + abs(prev_J)) < cfg.tau:
                break
        prev_J = J

    runtime = time.time() - t0
    return RunResult(G=G, Gamma=Gamma, history=history,
                     iterations=k + 1, time_seconds=runtime)
