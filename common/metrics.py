"""
Evaluation metrics.
"""
from __future__ import annotations

import numpy as np
from ._ot_backend import emd2

from .objective import cost_matrix, partition_and_conditional


def embed(G: np.ndarray, n: int, m: int, q: int):
    """
    Leading-q eigenpairs of the optimised Gram matrix, split into the
    animal embedding phi (n x q) and attribute embedding psi (m x q).
    """
    G_sym = 0.5 * (G + G.T)
    eigvals, eigvecs = np.linalg.eigh(G_sym)
    order = np.argsort(eigvals)[::-1]
    eigvals = np.clip(eigvals[order[:q]], 0.0, None)
    eigvecs = eigvecs[:, order[:q]]
    coords = eigvecs * np.sqrt(eigvals)[None, :]
    phi = coords[:n]
    psi = coords[n:]
    return phi, psi


def precision_at_k(phi: np.ndarray, psi: np.ndarray, counts: np.ndarray,
                    k: int = 10) -> float:
    """Mean P@k over animals x."""
    n = phi.shape[0]
    d2 = ((phi[:, None, :] - psi[None, :, :]) ** 2).sum(axis=-1)  # n x m
    precisions = np.empty(n)
    for x in range(n):
        top_k = np.argsort(d2[x])[:k]
        hits = counts[x, top_k].sum()
        precisions[x] = hits / k
    return float(precisions.mean())


def precision_at_k_from_scores(scores: np.ndarray, counts: np.ndarray,
                                k: int = 10) -> float:
    """Mean P@k over animals x, ranking directly by a score matrix
    (higher score = higher rank) instead of by embedding distance.

    Used for the "coupling-only" readout in Table 3: pass the learned
    coupling ``Gamma`` (n x m) as ``scores`` to rank attributes by
    transported mass ``Gamma(x, .)`` with no embedding step at all,
    exactly as described in the manuscript's isolate-mechanism ablation.
    """
    n = scores.shape[0]
    precisions = np.empty(n)
    for x in range(n):
        top_k = np.argsort(scores[x])[::-1][:k]
        hits = counts[x, top_k].sum()
        precisions[x] = hits / k
    return float(precisions.mean())


def mean_per_animal_wasserstein(G_star: np.ndarray, Pobs: np.ndarray,
                                 pbar_x: np.ndarray, pbar_y: np.ndarray,
                                 n: int, m: int, q: int) -> float:
    """Mean Delta_x (W2) over animals x."""
    C = cost_matrix(G_star, n, m)
    _, r_G = partition_and_conditional(C, pbar_y)  # model conditional r_{G*}(y|x)
    _, psi = embed(G_star, n, m, q)
    C_attr = ((psi[:, None, :] - psi[None, :, :]) ** 2).sum(axis=-1)  # m x m

    deltas = np.empty(n)
    for x in range(n):
        p_obs_x = Pobs[x] / pbar_x[x]
        p_model_x = r_G[x]
        deltas[x] = emd2(p_obs_x, p_model_x, C_attr)
    return float(deltas.mean())


def l1_coupling_displacement(Gamma_star: np.ndarray, Pobs: np.ndarray) -> float:
    """L1(Gamma*, Pobs)."""
    return float(np.abs(Gamma_star - Pobs).sum())


# ---------------------------------------------------------------------------
# Metrics for the 20-Newsgroups (Globerson et al., 2007) benchmark.
#
# These follow Section 8.3 of Globerson, Chechik, Pereira & Tishby (2007)
# as closely as our smaller per-run sample sizes allow: the "doc-doc"
# measure quantifies whether same-newsgroup documents are embedded near
# each other, and the "doc-word" measure quantifies whether the words
# nearest a document in the embedding are representative of that
# document's newsgroup.
# ---------------------------------------------------------------------------

def doc_doc_measure(phi: np.ndarray, labels: list[str],
                     neighborhood_sizes=None) -> float:
    """
    Fraction of a document's k nearest OTHER documents that share its
    newsgroup label, averaged over documents and over a range of
    neighbourhood sizes k (Globerson et al., Section 8.3, "doc-doc").
    A random embedding has expected value 1 / (number of newsgroups).
    """
    n = phi.shape[0]
    labels = np.asarray(labels)
    d2 = ((phi[:, None, :] - phi[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(d2, np.inf)
    order = np.argsort(d2, axis=1)

    if neighborhood_sizes is None:
        neighborhood_sizes = range(1, n)

    same = (labels[order] == labels[:, None])  # n x (n-1), sorted by distance
    scores = []
    for k in neighborhood_sizes:
        if k >= n:
            continue
        scores.append(same[:, :k].mean())
    return float(np.mean(scores))


def doc_word_measure(phi: np.ndarray, psi: np.ndarray, labels: list[str],
                      word_given_class: np.ndarray, max_k: int = 100) -> float:
    """
    For each document, look at its k nearest words in the embedding and
    average their probability under the document's true newsgroup,
    normalised by the best achievable value for that k (Globerson et al.,
    Section 8.3, "doc-word"). ``word_given_class`` is a
    (n_classes, vocab_size) row-normalised ground-truth p(word|newsgroup)
    table built from raw counts and is NOT available to the embedding
    algorithm.
    """
    n = phi.shape[0]
    m = psi.shape[0]
    labels = np.asarray(labels)
    classes = sorted(set(labels.tolist()))
    class_index = {c: i for i, c in enumerate(classes)}

    d2 = ((phi[:, None, :] - psi[None, :, :]) ** 2).sum(-1)  # n x m
    order = np.argsort(d2, axis=1)

    max_k = min(max_k, m)
    scores = []
    for x in range(n):
        c = class_index[labels[x]]
        p_word = word_given_class[c]
        p_sorted_desc = np.sort(p_word)[::-1]
        for k in range(1, max_k + 1):
            achieved = p_word[order[x, :k]].mean()
            best = p_sorted_desc[:k].mean()
            if best > 0:
                scores.append(achieved / best)
    return float(np.mean(scores))


def word_given_class_table(counts: np.ndarray, labels: list[str]):
    """Ground-truth p(word | newsgroup), built from raw counts and used
    only for evaluation (never seen by the embedding algorithm)."""
    labels = np.asarray(labels)
    classes = sorted(set(labels.tolist()))
    table = np.zeros((len(classes), counts.shape[1]))
    for i, c in enumerate(classes):
        sub = counts[labels == c].sum(axis=0)
        total = sub.sum()
        table[i] = sub / total if total > 0 else sub
    return table, classes
