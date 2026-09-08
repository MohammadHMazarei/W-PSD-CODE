"""
Data loading for the second benchmark: document-word co-occurrence on
the 20 Newsgroups corpus, following the preprocessing and evaluation
protocol of Globerson et al. (2007), Section 8.

Expects raw ``20news-bydate-train`` message files under
``data_dir/<newsgroup-name>/<message-id>`` (one plain-text file per
message, standard UCI/qwone.com layout), bundled under
``data/20newsgroups/``.

We reproduce the preprocessing described in Section 8.1 of Globerson
et al.: strip headers, tokenise, remove the 100 globally most frequent
words, then keep the next ``vocab_size`` most frequent words as the
vocabulary. Each document contributes a normalised row to a raw
document-word count matrix; the resulting joint table n(doc, word) is
turned into a coupling exactly as ``build_marginals`` does for AwA2, so
the SAME W-PSD-CODE implementation (``common.algorithm``) can be run on
it unmodified.
"""
from __future__ import annotations

import os
import re
from collections import Counter

import numpy as np

DEFAULT_DATA = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "data", "20newsgroups")

_TOKEN_RE = re.compile(r"[A-Za-z]+")

# A short stoplist of closed-class words. Globerson et al. do not specify
# their exact stoplist beyond "the 100 most frequent words"; we combine
# that frequency-based rule with this small stoplist so that the vocabulary
# is not dominated by function words alone, which in our smaller per-class
# samples do not always coincide with the globally most frequent tokens.
_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "of", "to", "in", "on",
    "for", "with", "as", "is", "are", "was", "were", "be", "been", "being",
    "this", "that", "these", "those", "it", "its", "i", "you", "he", "she",
    "we", "they", "them", "his", "her", "their", "our", "your", "my",
    "at", "by", "from", "up", "down", "out", "so", "not", "no", "do",
    "does", "did", "have", "has", "had", "will", "would", "can", "could",
    "should", "may", "might", "must", "there", "here", "what", "which",
    "who", "whom", "when", "where", "why", "how", "all", "any", "some",
    "than", "then", "just", "about", "into", "over", "after", "before",
    "re", "subject", "lines", "organization", "writes", "article", "edu",
    "com", "one", "would", "also", "get", "like", "know", "think", "use",
    "used", "using", "new", "us", "au",
}


def _tokenise(text: str) -> list[str]:
    return [w.lower() for w in _TOKEN_RE.findall(text) if len(w) > 2]


def _read_message(path: str) -> str:
    with open(path, "r", encoding="latin-1", errors="replace") as f:
        raw = f.read()
    # Drop the header block (up to the first blank line), which mostly
    # carries routing metadata rather than topical content.
    parts = raw.split("\n\n", 1)
    body = parts[1] if len(parts) > 1 else raw
    return body


def load_newsgroup_docs(categories: list[str], docs_per_class: int,
                         seed: int, data_dir: str = DEFAULT_DATA):
    """
    Sample ``docs_per_class`` messages from each of ``categories``.

    Returns
    -------
    tokenised : list[list[str]]   tokens per sampled document
    labels    : list[str]         newsgroup name per sampled document, same
                                   order as ``tokenised``
    """
    rng = np.random.default_rng(seed)
    tokenised, labels = [], []
    for cat in categories:
        cat_dir = os.path.join(data_dir, cat)
        files = sorted(os.listdir(cat_dir))
        if len(files) < docs_per_class:
            raise ValueError(
                f"category {cat!r} has only {len(files)} files, "
                f"need {docs_per_class}")
        chosen = rng.choice(files, size=docs_per_class, replace=False)
        for fname in chosen:
            body = _read_message(os.path.join(cat_dir, fname))
            tokenised.append(_tokenise(body))
            labels.append(cat)
    return tokenised, labels


def build_doc_word_matrix(tokenised: list[list[str]], labels: list[str] = None,
                           vocab_size: int = 200, skip_top: int = 100):
    """
    Build the n(doc, word) count matrix following Globerson et al.,
    Section 8.1: remove the ``skip_top`` globally most frequent tokens
    (after stoplisting), then keep the next ``vocab_size`` most frequent
    tokens as the vocabulary.

    Documents whose token stream contains none of the retained vocabulary
    words (e.g. very short messages) are dropped, exactly as the sparsity
    ablation guarantees non-empty rows/columns before building marginals;
    otherwise the empirical marginal on that document would be zero and
    every OT/Sinkhorn solve in the alternating optimisation would be
    ill-posed for it.

    Returns
    -------
    counts : (n_kept, vocab_size) float ndarray
    vocab  : list[str] of length vocab_size
    kept_labels : list[str] of length n_kept, aligned with ``counts`` rows
                  (only returned if ``labels`` is not None)
    """
    global_counts = Counter()
    for toks in tokenised:
        global_counts.update(t for t in toks if t not in _STOPWORDS)

    ranked = [w for w, _ in global_counts.most_common()]
    vocab = ranked[skip_top:skip_top + vocab_size]
    index = {w: i for i, w in enumerate(vocab)}

    rows = []
    kept_labels = []
    for i, toks in enumerate(tokenised):
        row = np.zeros(len(vocab), dtype=float)
        for t in toks:
            j = index.get(t)
            if j is not None:
                row[j] += 1.0
        if row.sum() > 0:
            rows.append(row)
            if labels is not None:
                kept_labels.append(labels[i])

    counts = np.vstack(rows) if rows else np.zeros((0, len(vocab)))

    # Also drop any vocabulary word that ended up with zero mass after
    # dropping empty documents (guards the column marginal the same way).
    keep_cols = counts.sum(axis=0) > 0
    counts = counts[:, keep_cols]
    vocab = [w for w, keep in zip(vocab, keep_cols) if keep]

    if labels is not None:
        return counts, vocab, kept_labels
    return counts, vocab


def build_marginals(counts: np.ndarray):
    """Identical convention to ``common.data.build_marginals``."""
    total = counts.sum()
    Pobs = counts / total
    pbar_x = Pobs.sum(axis=1)
    pbar_y = Pobs.sum(axis=0)
    return Pobs, pbar_x, pbar_y
