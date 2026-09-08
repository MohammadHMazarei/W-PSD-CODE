# W-PSD-CODE

**Title:** W-PSD-CODE: Geometry-Adaptive Optimal Transport Coupling for Positive-Semidefinite
Co-occurrence Embedding

**Status:** Manuscript under peer review. This directory is the reproducibility code
accompanying the manuscript.

## Description

W-PSD-CODE is a geometry-adaptive extension of the positive-semidefinite co-occurrence
embedding (PSD-CODE) framework that represents the empirical co-occurrence distribution as an
element of the optimal-transport coupling polytope, and updates that coupling using the geometry
of the current embedding rather than holding it fixed. The method alternates between (1) an
optimal-transport step computing a geometry-aware coupling from the current squared-distance
matrix, and (2) a positive-semidefinite optimisation step updating the embedding Gram matrix,
with the coupling update regularised by a KL divergence to a reference coupling built from the
observed data.

The paper's central finding is a carefully diagnosed **negative result**: on the AwA2
class-attribute matrix, geometry-aware coupling alone matches but does not improve on
PSD-CODE's retrieval accuracy, and the paper analyses why, tests generalisation on a second,
unrelated benchmark (20 Newsgroups subsets from Globerson et al., 2007), and tests — and rules
out — two natural fixes (fused Gromov-Wasserstein coupling with auxiliary structure, and an
auxiliary M-step anchor toward a WordNet-taxonomic class kernel). Against that negative-result
backdrop, the paper also reports one **positive result**: a controlled held-out side-information
experiment in which one disjoint AwA2 attribute group is used for retrieval and another for the
embedding anchor. Selecting the anchor strength `mu` on 10 validation masks and evaluating on 30
independent test masks gives a mean P@10 gain of +0.0714 (95% CI [0.0654, 0.0774]), with all 30
test masks improving — indicating that coupling/embedding adaptation helps only when the
auxiliary geometry is genuinely held out from the fitting target and relevant to what is being
evaluated.

## What's included

- `data/AwA2/` — the Animals with Attributes 2 class–attribute matrix (50 classes, 85
  attributes) used for the paper's headline results, plus `data/20newsgroups/` (the secondary
  generalisation benchmark) and `data/wordnet/` (a bundled offline WordNet corpus used by one
  of the ruled-out fixes). All small, public-benchmark-derived data.
- `common/` — the core method implementation: the alternating algorithm (`algorithm.py`), the
  free-energy objective (`objective.py`), the OT coupling solvers (`ot_solvers.py`,
  `_ot_backend.py`), the PSD projection step (`psd_utils.py`), and the data loaders.
- `psd_code/`, `wpsd_code_exact_ot/`, `wpsd_code_sinkhorn/`, `wpsd_code_kl/` — the four
  compared methods (Table 2), each a thin script around `common/`.
- `diagnostics/`, `ablation/`, `benchmarks/`, `improvements/`, `isolate_mechanism/`,
  `heldout/`, `figures/` — every ablation, sensitivity sweep, the 20 Newsgroups benchmark, both
  ruled-out fixes, the coupling-vs-embedding isolation check, and the held-out attribute-anchor
  experiment behind the paper's positive result.
- `results/` — generated JSON/CSV/NPY outputs from the above, kept so every reported number can
  be inspected without rerunning anything.
- `run_all.py`, `requirements.txt` — top-level entry point and dependencies.

Nothing has been deliberately withheld from this directory: it is the complete code base behind
every reported result, not a limited excerpt.

## Running the core method

```bash
pip install -r requirements.txt
python run_all.py                              # Table 2, Fig. 7, Fig. 8
```

`POT` is optional; if absent, the code falls back to a SciPy/HiGHS linear-program solver for the
exact-OT step (see `common/_ot_backend.py`) — note that this fallback is what the manuscript's
published exact-OT numbers correspond to (see `REPRODUCTION_GUIDE.md` for details). `spacy` and
`nltk` are needed only for the two ruled-out-fix scripts under `improvements/`.

## Citation

The manuscript is currently under peer review. No citation is available yet; citation details
(including any DOI) will be added here once the manuscript is accepted.

## Contact

- Mohammadhossein Mazarei
- Hadi Sadoghi Yazdi (corresponding author) — h-sadoghi@um.ac.ir · [ORCID 0000-0002-6885-4956](https://orcid.org/0000-0002-6885-4956)

Department of Computer Engineering, Faculty of Engineering, Ferdowsi University of Mashhad,
Mashhad, Iran.

## License

The source code in this repository is released under the MIT License.
See the `LICENSE` file for details.

The datasets used in this work are publicly available from their
respective sources and remain subject to their original licenses and
terms of use. We do not claim ownership of these datasets.
