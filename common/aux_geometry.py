"""
Auxiliary, EXOGENOUS structure matrices for the Fused Gromov-Wasserstein
coupling variant (``coupling="fgw"`` in ``common.algorithm``).

Section 7.10 of the paper (Why the Coupling Update Adds No Signal) shows that the plain OT
coupling adds no signal because its cost matrix is built entirely from
the current Gram matrix G, i.e. from information the model already has.
The natural fix, following the Future Work paragraph and
Vayer et al. (2019) [vayer2019fgw], is to additionally constrain the
coupling using a WITHIN-domain structure matrix built from something the
model has never seen. Here we use general-purpose pretrained word
vectors (spaCy's ``en_core_web_md``, static 300-d vectors trained on
Common Crawl) to build:

  * D_X : pairwise semantic distance between the NAMES of the row
          entities (AwA2 class names, or nothing useful for 20NG
          documents -- see below),
  * D_Y : pairwise semantic distance between the NAMES of the column
          entities (AwA2 attribute names, or 20NG vocabulary words).

These distances are computed from word meaning alone and are
independent of any co-occurrence statistics in Pobs, which is what makes
them a legitimate exogenous signal rather than a restatement of G.

For 20 Newsgroups, D_X (document-document structure) is not available
from names (documents don't have names), so we only constrain the word
side (D_Y) and pass an all-zero D_X, which reduces the row-side GW term
to a no-op while keeping the word-side term active.
"""
from __future__ import annotations

import os
import numpy as np


def _load_spacy():
    import spacy
    return spacy.load("en_core_web_md")


def name_distance_matrix(names: list[str], nlp=None) -> np.ndarray:
    """
    Pairwise cosine-distance matrix between pretrained word-vector
    representations of ``names``. Multi-word names (AwA2 uses
    "grizzly+bear" style class names) are split on non-alphabetic
    characters and averaged, exactly as spaCy averages token vectors for
    a multi-token span. A small manual dictionary additionally splits the
    handful of unspaced AwA2 attribute compounds (e.g. "toughskin" ->
    "tough skin") that are otherwise out-of-vocabulary for the static
    word-vector table, so their distance is informative rather than
    falling back to a neutral zero vector.
    """
    if nlp is None:
        nlp = _load_spacy()

    vecs = []
    missing = []
    for i, name in enumerate(names):
        clean = _COMPOUND_SPLITS.get(name, name)
        clean = clean.replace("+", " ").replace("_", " ").replace("-", " ")
        doc = nlp(clean)
        if doc.vector_norm > 0:
            vecs.append(doc.vector / doc.vector_norm)
        else:
            vecs.append(np.zeros(nlp.vocab.vectors_length))
            missing.append(name)
    if missing:
        import warnings
        warnings.warn(f"{len(missing)} names had no pretrained vector "
                       f"(out-of-vocabulary): {missing[:5]}"
                       f"{'...' if len(missing) > 5 else ''}")

    V = np.vstack(vecs)  # already L2-normalised rows (or zero rows)
    sim = V @ V.T
    dist = 1.0 - sim
    np.fill_diagonal(dist, 0.0)
    return dist


# Manual splits for the unspaced AwA2 attribute-name compounds that are
# out-of-vocabulary for spaCy's static word-vector table as single
# tokens. This is a small, fixed, dataset-specific lookup -- it does not
# use any information from the co-occurrence matrix itself, only the
# English meaning of the attribute name, so it does not reintroduce the
# circularity the FGW coupling is meant to avoid.
_COMPOUND_SPLITS = {
    "toughskin": "tough skin",
    "longleg": "long leg",
    "longneck": "long neck",
    "chewteeth": "chewing teeth",
    "meatteeth": "meat teeth",
    "buckteeth": "buck teeth",
    "strainteeth": "strain teeth",
    "bipedal": "two legged",
    "quadrapedal": "four legged",
    "newworld": "new world",
    "oldworld": "old world",
    "nestspot": "nest spot",
}


def normalise_structure(D: np.ndarray) -> np.ndarray:
    """Scale a structure matrix to have the same average magnitude as a
    typical embedding-space squared-distance matrix, so alpha in
    fused_gromov_wasserstein trades off two comparable quantities rather
    than two arbitrarily-scaled ones."""
    m = D.mean()
    return D / m if m > 0 else D


# ---------------------------------------------------------------------------
# WordNet taxonomic distance (AwA2 class names only).
#
# The generic word-vector distance above treats "grizzly+bear" and
# "polar+bear" as two points in a flat semantic space; it has no notion
# that they are both members of Ursidae specifically. AwA2's class names
# resolve directly to WordNet noun synsets (the dataset was built on top
# of the ImageNet/WordNet hierarchy), so we can instead measure distance
# along the actual is-a taxonomy -- a genuinely different, and arguably
# more appropriate, source of exogenous structure than flat word
# similarity for a set of biological categories.
# ---------------------------------------------------------------------------

def _resolve_synset(name: str, wn):
    """Map an AwA2-style class name ("grizzly+bear") to a WordNet noun
    synset. Tries the full compound first, then falls back to the head
    noun (last word) if the compound itself is not a WordNet lemma."""
    candidate = name.replace("+", "_")
    syns = wn.synsets(candidate, pos=wn.NOUN)
    if syns:
        return syns[0]
    head = name.split("+")[-1]
    syns = wn.synsets(head, pos=wn.NOUN)
    if syns:
        return syns[0]
    return None


DEFAULT_WORDNET_DATA = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "data", "wordnet")


def wordnet_distance_matrix(names: list[str], nltk_data_path: str = None) -> np.ndarray:
    """
    Pairwise taxonomic distance matrix using WordNet path similarity:
    distance(i,j) = 1 - path_similarity(synset_i, synset_j), where
    path_similarity = 1 / (1 + shortest_path_length) along the WordNet
    is-a hypernym graph. Falls back to the generic word-vector distance
    (name_distance_matrix) for any name that cannot be resolved to a
    WordNet noun synset, so the function always returns a complete
    matrix even on datasets where taxonomy is only partially available.
    ``nltk_data_path`` defaults to the copy of the WordNet corpus bundled
    under ``data/wordnet`` alongside the AwA2 data, so no network access
    is needed at run time.
    """
    import nltk
    nltk.data.path.insert(0, nltk_data_path or DEFAULT_WORDNET_DATA)
    from nltk.corpus import wordnet as wn

    synsets = [_resolve_synset(name, wn) for name in names]
    unresolved = [name for name, s in zip(names, synsets) if s is None]
    if unresolved:
        import warnings
        warnings.warn(f"{len(unresolved)} names could not be resolved to a "
                       f"WordNet synset and fall back to word-vector "
                       f"distance for those rows/columns: {unresolved}")

    k = len(names)
    dist = np.zeros((k, k))
    fallback_needed = bool(unresolved)
    fallback = name_distance_matrix(names) if fallback_needed else None

    for i in range(k):
        for j in range(i + 1, k):
            if synsets[i] is not None and synsets[j] is not None:
                sim = synsets[i].path_similarity(synsets[j])
                d = 1.0 - sim if sim is not None else 1.0
            else:
                d = fallback[i, j]
            dist[i, j] = dist[j, i] = d
    return dist


# ---------------------------------------------------------------------------
# WordNet taxonomic distance (path-length in the hypernym hierarchy),
# as a second, more targeted source of exogenous structure for the FGW
# coupling's X-side (AwA2 classes). Unlike the flat word-vector distance
# above, this uses real biological taxonomy -- e.g. "grizzly bear" and
# "polar bear" share a recent common ancestor (Ursidae) in a way that a
# generic word-embedding cosine distance does not explicitly encode.
#
# All 50 AwA/AwA2 classes are mammal species, so we disambiguate each
# class name to its animal WordNet sense by requiring the synset to be a
# hyponym of mammal.n.01 (the first synsets() hit for common English
# animal names is frequently the WRONG sense, e.g. "dolphin" -> the
# dolphinfish/mahi-mahi sense, or "mole" -> the espionage sense; both
# would otherwise be picked by a naive first-sense lookup).
# ---------------------------------------------------------------------------

_MANUAL_SYNSETS = {
    # A handful of AwA2 names where automatic disambiguation needs a
    # nudge even under the mammal.n.01 filter (checked by hand).
}


def _mammal_synset(name: str, wn):
    if name in _MANUAL_SYNSETS:
        return wn.synset(_MANUAL_SYNSETS[name])
    clean = name.replace("+", "_").replace("-", "_")
    candidates = wn.synsets(clean, pos=wn.NOUN)
    for s in candidates:
        hyper = set(h.name() for h in s.closure(lambda x: x.hypernyms()))
        if "mammal.n.01" in hyper:
            return s
    if "_" in clean:
        return _mammal_synset(clean.split("_")[-1], wn)
    return candidates[0] if candidates else None


def wordnet_taxonomic_distance(names: list[str],
                               nltk_data_path: str = None) -> np.ndarray:
    """
    Pairwise taxonomic distance matrix using WordNet's hypernym-tree
    shortest path between the (disambiguated) animal sense of each name
    in ``names``. Distance is 1 - path_similarity, where path_similarity
    in (0, 1] is 1 / (shortest_path_length + 1) along hypernym edges
    (nltk's ``Synset.path_similarity``); two identical synsets have
    distance 0, and taxonomically distant synsets approach distance 1.

    This is independent of any co-occurrence statistics in Pobs -- it
    depends only on the WordNet hierarchy -- so it is a legitimate
    exogenous signal for the FGW coupling, same as the word-vector
    distance in ``name_distance_matrix``, but encodes hierarchical
    biological relatedness rather than flat lexical-semantic similarity.
    """
    import nltk
    nltk.data.path.insert(0, nltk_data_path or DEFAULT_WORDNET_DATA)
    from nltk.corpus import wordnet as wn

    synsets = [_mammal_synset(n, wn) for n in names]
    missing = [n for n, s in zip(names, synsets) if s is None]
    if missing:
        import warnings
        warnings.warn(f"{len(missing)} names could not be resolved to a "
                       f"WordNet mammal synset: {missing}")

    k = len(names)
    dist = np.ones((k, k))
    for i in range(k):
        dist[i, i] = 0.0
        for j in range(i + 1, k):
            if synsets[i] is None or synsets[j] is None:
                d = 1.0
            else:
                sim = synsets[i].path_similarity(synsets[j])
                d = 1.0 - sim if sim is not None else 1.0
            dist[i, j] = dist[j, i] = d
    return dist
