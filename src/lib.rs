//! Rust port of the core algorithm of
//! [umap-learn](https://github.com/lmcinnes/umap) (baseline `3c6b0c2`:
//! 0.5.12 + the `smooth_knn_dist` finite-filtering fix and subsequent
//! correctness commits).
//!
//! # Scope
//!
//! Ported: `utils` leaf functions, `distances`, `sparse` (as
//! [`sparse_metrics`]), `layouts`, `spectral`, `fuzzy`/`embedding`, the
//! `umap_` core ([`Umap`]: `fit/fit_transform/transform/inverse_transform`)
//! and `validation` (trustworthiness).
//!
//! Not ported: `plot.py`, `parametric_umap.py`, `aligned_umap.py`, pynndescent,
//! supervised fitting (`y`), `update`, and model serialization.
//!
//! # The precomputed-KNN contract
//!
//! pynndescent is replaced by an exact brute-force kNN for raw dense inputs
//! with n < 4096 rows ([`Umap::fit`]). For larger inputs callers must supply
//! precomputed k-nearest neighbors (indices/dists including the self-neighbor
//! at column 0) via [`Umap::fit_with_knn`].
//!
//! # Parity
//!
//! The port targets *statistical* parity with the Python baseline: metrics
//! match to ≤ 1e-6 relative tolerance, graph edge weights to ≤ 1e-5,
//! end-to-end trustworthiness within ±0.02 of the Python result, and
//! single-threaded runs are seed-reproducible. Cross-validation tests compare
//! against reference values generated from the Python baseline (the repo
//! source, not the `PyPI` 0.5.12 release — see below).
//!
//! # Divergences (documented per module)
//!
//! - **KNN**: brute force / precomputed kNN instead of pynndescent.
//! - **Spectral init**: scipy `eigsh` (sparse Lanczos) replaced by a full
//!   dense self-adjoint eigendecomposition (`faer`); eigenvalues of the
//!   normalized Laplacian lie in [0, 2] so the bottom-k eigenvectors
//!   coincide, but eigenvectors are defined up to sign — parity is
//!   statistical.
//! - **`find_ab_params`**: scipy `curve_fit` replaced by a hand-rolled
//!   Levenberg–Marquardt (agrees to < 1e-6; scipy stops slightly short of
//!   the true minimum at its default tolerances).
//! - **RNG seeding**: `tau_rand_int` is ported integer-exact (verified
//!   against numba), but `TauRng` seeds via `SplitMix64` rather than numpy's
//!   MT19937 — different streams than Python for the same seed.
//! - **Parallel SGD**: `parallel=True` runs use rayon with Hogwild-style
//!   shared mutation, matching numba's unsynchronized float semantics (but
//!   not its exact thread interleavings). Serial runs are deterministic.
//! - **`inverse_transform`**: brute-force exact kNN neighborhoods in the
//!   embedding space instead of scipy Delaunay triangulation + BFS.
//! - **densMAP `graph_dists`**: hand-rolled Dijkstra (scipy `csgraph.dijkstra`
//!   equivalent values).
//!
//! # Example
//!
//! ```no_run
//! use umap_rs::umap::{Umap, UmapConfig};
//!
//! let data = umap_rs::ndarray::Array2::<f32>::zeros((100, 4));
//! let model = Umap::with_config(UmapConfig {
//!     n_neighbors: 10,
//!     n_epochs: Some(50),
//!     ..UmapConfig::default()
//! })
//! .fit(&data)
//! .expect("fit");
//! let embedding = model.embedding;
//! ```

pub mod csr;
pub mod distances;
pub mod embedding;
pub mod error;
pub mod fuzzy;
pub mod layouts;
pub mod rng;
pub mod sparse_metrics;
pub mod spectral;
pub mod umap;
pub mod utils;
pub mod validation;

pub use csr::{CooMatrix, CsrMatrix};
pub use error::{Result, UmapError};
pub use rng::{tau_rand, tau_rand_int};
pub use umap::{Umap, UmapConfig};

pub use ndarray;
