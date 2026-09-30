# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-09-30

Initial release: a pure-Rust port of the core algorithm of umap-learn
(baseline commit `3c6b0c2`; 0.5.12 plus subsequent correctness fixes).

### Added

- `csr`: CSR/COO sparse matrix types with scipy-compatible semantics
  (canonical ordering, duplicate summing, `eliminate_zeros`,
  transpose, elementwise ops, `maximum_with_transpose`).
- `distances`: the full dense metric registry from Python
  `umap/distances.py`, including distance-with-gradient functions for
  the SGD kernels (f32 arithmetic; integer-exact discrete metrics).
- `sparse_metrics`: sparse metrics and simplicial-set
  intersection/union operations from `umap/sparse.py`.
- `fuzzy`: `smooth_knn_dist` (with the repo's finite-filtering fix),
  `compute_membership_strengths`, `fuzzy_simplicial_set`, and discrete
  metric intersections.
- `layouts`: SGD optimization kernels (euclidean, generic output
  metrics, inverse transform) with densMAP support, ported with
  Python's per-sample RNG-state replication and Hogwild!-style
  parallel mutation.
- `spectral`: BFS connected components, spectral layout via dense
  self-adjoint eigendecomposition (faer) for `n <= 4096` and an
  unpreconditioned LOBPCG solver on `B = 2I − L` with guard vectors
  for larger graphs, multi-component layouts.
- `embedding`: `find_ab_params` (hand-rolled Levenberg–Marquardt),
  `simplicial_set_embedding` with random/spectral/pca initialization,
  transform initializers, exact brute-force kNN.
- `umap`: the top-level `Umap` model — `fit` (brute-force kNN for
  n < 4096), `fit_with_knn` (precomputed-KNN contract for large
  inputs), `transform`, `transform_graph`, and `inverse_transform`.
- `validation`: trustworthiness metric with exact brute-force
  neighborhoods.
- `utils`: leaf helpers (`fast_knn_indices`, `submatrix`, `norm`,
  `csr_unique`, `average_nn_distance`).
- `rng`: integer-exact port of numba's `tau_rand_int`/`tau_rand`
  leapfrog RNG; `TauRng` seeds via SplitMix64.

### Divergences from Python

All intentional divergences are documented in the README and the
crate-level documentation; the largest is the precomputed-KNN
contract replacing pynndescent.

### Testing

102 tests: unit tests, cross-validation reference tests generated
from the Python baseline, and end-to-end trustworthiness gate tests.
Zero clippy (pedantic) warnings.
