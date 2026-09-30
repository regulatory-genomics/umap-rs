# umap-rs

A pure-Rust port of the core algorithm of
[umap-learn](https://github.com/lmcinnes/umap) — Uniform Manifold
Approximation and Projection.

Ported from the umap-learn repository at baseline commit `3c6b0c2`
(0.5.12 plus subsequent correctness fixes, including the
`smooth_knn_dist` finite-filtering bug fix that is **not** in the PyPI
0.5.12 release).

## Installation

```toml
[dependencies]
umap-rs = "0.1"
```

MSRV: Rust 1.85.

## Usage

```rust
use umap_rs::umap::{Umap, UmapConfig};

let data: ndarray::Array2<f32> = /* your (n_samples, n_features) data */;

let model = Umap::with_config(UmapConfig {
    n_neighbors: 15,
    min_dist: 0.1,
    n_components: 2,
    random_state: Some(42), // None => parallel/nondeterministic SGD
    ..UmapConfig::default()
})
.fit(&data)
.expect("fit");

let embedding = model.embedding;          // (n_samples, 2)
let graph = model.graph;                  // fuzzy 1-skeleton (Python graph_)

// Embed new points into the fitted space.
let transformed = model.transform(&new_points).expect("transform");

// Project embedded points back into the input space.
let recovered = model.inverse_transform(&transformed).expect("inverse");
```

### Large inputs (precomputed KNN)

pynndescent is not ported. Raw dense inputs with n < 4096 rows use an
exact brute-force kNN automatically. For larger inputs, supply
precomputed k-nearest neighbors (indices/dists including the
self-neighbor at column 0):

```rust
let model = umap
    .fit_with_knn(Some(&data), &knn_indices, &knn_dists)
    .expect("fit");
```

The lower-level building blocks are public: `fuzzy` (smooth knn +
graph construction), `layouts` (SGD kernels, densMAP), `spectral`
(EVD-based initialization), `distances` / `sparse_metrics` (metric
registries), `csr` (sparse matrix types), `validation`
(trustworthiness), and `embedding` (simplicial set embedding).

## Comparison to the Python version

The port targets **statistical parity** with the Python baseline:

| Check | Tolerance | Status |
| --- | --- | --- |
| Distance metrics (`named_distances`, gradients) | ≤ 1e-6 relative | cross-validated (55 tests) |
| Sparse metric / set operations | ≤ 1e-6 relative | cross-validated (21 tests) |
| Fuzzy graph edge weights | ≤ 1e-5 | cross-validated |
| Layout SGD kernels (1 epoch) | < 1e-5 absolute | cross-validated |
| Layout SGD kernels (40 epochs) | < 1e-2 (float reassociation) | cross-validated |
| End-to-end trustworthiness (iris, n=150) | ± 0.02 | cross-validated |
| End-to-end trustworthiness (digits-scale, n=1797) | ± 0.02 | cross-validated |
| Single-threaded, same seed | reproducible | verified |

Python's own run-to-run variance at the ~1800-sample scale is ~1e-3
(numba's parallel `smooth_knn_dist` reductions are unsynchronized even
with a fixed `random_state`), which the ±0.02 end-to-end tolerance
covers.

### Intentional divergences

- **KNN search**: pynndescent is replaced by exact brute force
  (n < 4096) or the precomputed-KNN contract above.
- **Spectral initialization**: scipy `eigsh` (sparse Lanczos) is
  replaced by a full dense self-adjoint eigendecomposition
  ([faer](https://codeberg.org/sarah-quinones/faer)) for `n <= 4096`,
  and by an unpreconditioned LOBPCG solver on `B = 2I − L` (matvec
  only, O(nnz) memory) with guard vectors for larger graphs. For the
  normalized Laplacian (eigenvalues in [0, 2]) the bottom-k
  eigenvectors coincide; they are defined up to sign/rotation within
  degenerate eigenspaces, so parity is statistical. The LOBPCG
  iteration count is capped and the best available block is returned
  where Python falls back to random initialisation on ARPACK
  non-convergence.
- **`find_ab_params`**: scipy `curve_fit` is replaced by a compact
  hand-rolled Levenberg–Marquardt (agrees to < 1e-6; scipy itself
  stops slightly short of the true minimum at its default
  tolerances).
- **RNG**: the numba `tau_rand_int` leapfrog RNG is ported
  integer-exact (verified against numba), but `TauRng` seeds via
  SplitMix64 rather than numpy MT19937 — different streams than
  Python for the same seed.
- **Parallel SGD**: `parallel = true` uses rayon with Hogwild-style
  shared mutation, matching numba's unsynchronized float semantics
  (but not its exact thread interleavings). Serial runs
  (`random_state = Some(_)` for the fit path) are deterministic.
- **`inverse_transform`**: neighborhoods use brute-force exact kNN in
  the embedding space instead of scipy Delaunay triangulation + BFS.
- **Out of scope**: `plot.py`, `parametric_umap.py`, `aligned_umap.py`,
  supervised fitting (`y`), `update`, sparse input data, and model
  serialization.

## Tests

```bash
cargo test
```

Includes unit tests, cross-validation reference tests generated from
the Python baseline, and end-to-end gate tests (the digits-scale gate
takes ~2 minutes in debug mode).

## Benchmarks

```bash
cargo run --release --example bench
```

Times the fit phases (kNN, fuzzy graph, spectral init, SGD kernel),
end-to-end fit, `transform`, and `inverse_transform` on a deterministic
dataset, with trustworthiness as the quality column.

## License

BSD-3-Clause, matching umap-learn.
