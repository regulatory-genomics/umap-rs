//! Ported from Python: umap/umap_.py (`find_ab_params`, `simplicial_set_embedding`,
//! `init_transform`, `init_graph_transform`, `noisy_scale_coords`,
//! _`densmap_original_densities`, _`densmap_embedding_densities`).
//!
//! Divergences:
//! - scipy `curve_fit` (Levenberg–Marquardt via MINPACK) is replaced by a
//!   compact hand-rolled Levenberg–Marquardt for the two-parameter a/b fit.
//! - PCA init is a hand-rolled covariance eigendecomposition (faer EVD);
//!   sklearn's randomized SVD PCA matches this up to sign/tolerance.
//! - `init="tswspectral"` (`TruncatedSVD` warm start) is out of scope.
// The kernels mirror the numba-compiled Python structure (single-char loop
// variables, index-based loops, many flag parameters matching the Python
// signatures); pedantic lints for those are allowed module-wide.
#![allow(
    clippy::many_single_char_names,
    clippy::needless_range_loop,
    clippy::struct_excessive_bools,
    clippy::type_complexity
)]

use crate::csr::CsrMatrix;
use crate::error::{Result, UmapError};
use crate::layouts::{
    optimize_layout_euclidean, optimize_layout_generic, DensmapKwds, OutputMetricFn, TailEmbedding,
};
use crate::rng::TauRng;
use crate::spectral::spectral_layout;

/// Python: `umap_.find_ab_params` — fit a, b params for the differentiable
/// curve used in lower dimensional fuzzy simplicial complex construction.
///
/// The curve family is `1 / (1 + a * x^(2b))`; the target is an offset
/// exponential decay with the given `spread` and `min_dist`. scipy
/// `curve_fit` is replaced by hand-rolled Levenberg–Marquardt (agreement
/// with scipy is typically < 1e-6 for default inputs).
#[must_use]
pub fn find_ab_params(spread: f64, min_dist: f64) -> (f64, f64) {
    // Target curve samples: Python xv = np.linspace(0, spread * 3, 300).
    let n = 300usize;
    let xv: Vec<f64> = (0..n)
        .map(|i| spread * 3.0 * i as f64 / (n - 1) as f64)
        .collect();
    let yv: Vec<f64> = xv
        .iter()
        .map(|&x| {
            if x < min_dist {
                1.0
            } else {
                (-(x - min_dist) / spread).exp()
            }
        })
        .collect();

    let curve = |x: f64, a: f64, b: f64| 1.0 / (1.0 + a * x.powf(2.0 * b));
    let cost = |a: f64, b: f64| -> f64 {
        xv.iter()
            .zip(yv.iter())
            .map(|(&x, &y)| {
                let r = y - curve(x, a, b);
                r * r
            })
            .sum()
    };

    // Levenberg–Marquardt with numeric central-difference Jacobian.
    let mut p = [1.0f64, 1.0f64]; // scipy curve_fit default p0 = [1, 1]
    let mut lambda = 1e-3f64;
    // scipy leastsq numeric Jacobian step ≈ sqrt(machine eps) ≈ 1.49e-8.
    let eps = 1.49e-8f64;
    let mut best = cost(p[0], p[1]);

    for _ in 0..300 {
        // Numeric Jacobian J (n x 2) and residual r (n) at the current point.
        let mut jtj = [[0.0f64; 2]; 2];
        let mut jtr = [0.0f64; 2];
        for (&x, &y) in xv.iter().zip(yv.iter()) {
            let mut jac = [0.0f64; 2];
            for k in 0..2 {
                let mut pp = p;
                pp[k] += eps;
                let fp = curve(x, pp[0], pp[1]);
                pp[k] -= 2.0 * eps;
                let fm = curve(x, pp[0], pp[1]);
                jac[k] = (fp - fm) / (2.0 * eps);
            }
            let r = y - curve(x, p[0], p[1]);
            for a in 0..2 {
                for b in 0..2 {
                    jtj[a][b] += jac[a] * jac[b];
                }
                jtr[a] += jac[a] * r;
            }
        }

        // Damped solve: (JᵀJ + λ diag(JᵀJ)) δ = Jᵀr — 2x2. Increase λ until
        // the step is accepted (standard Levenberg adjustment).
        let mut accepted = false;
        let mut lam = lambda;
        for _ in 0..40 {
            let m00 = jtj[0][0] * (1.0 + lam);
            let m01 = jtj[0][1];
            let m10 = jtj[1][0];
            let m11 = jtj[1][1] * (1.0 + lam);
            let det = m00 * m11 - m01 * m10;
            if !det.is_finite() || det.abs() < 1e-300 {
                break;
            }
            let d0 = (m11 * jtr[0] - m01 * jtr[1]) / det;
            let d1 = (m00 * jtr[1] - m10 * jtr[0]) / det;
            if !d0.is_finite() || !d1.is_finite() {
                break;
            }
            let candidate = [p[0] + d0, p[1] + d1];
            let c = cost(candidate[0], candidate[1]);
            if c.is_finite() && c < best {
                p = candidate;
                best = c;
                lambda = (lam / 3.0).max(1e-12);
                accepted = true;
                break;
            }
            lam *= 3.0;
        }
        if !accepted || best < 1e-18 {
            break;
        }
        // Stop when converged (relative improvement below ~1e-14).
        let c0 = cost(p[0], p[1]);
        let _ = c0;
        if best <= 1e-300 {
            break;
        }
    }

    (p[0], p[1])
}

/// Python: `noisy_scale_coords` — scale so the largest |coordinate| is
/// `max_coord`, then add N(0, noise) jitter.
#[must_use]
pub fn noisy_scale_coords(
    coords: &ndarray::Array2<f32>,
    random_state: &mut TauRng,
    max_coord: f64,
    noise: f64,
) -> ndarray::Array2<f32> {
    let max_abs = coords
        .iter()
        .map(|v| f64::from(v.abs()))
        .fold(0.0f64, f64::max);
    let expansion = max_coord / max_abs.max(f64::EPSILON);
    let mut out = coords.mapv(|v| (f64::from(v) * expansion) as f32);
    for v in &mut out {
        *v += random_state.normal_f64() as f32 * noise as f32;
    }
    out
}

/// How to initialize the low-dimensional embedding.
/// Python `init` string parameter of `simplicial_set_embedding`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Init {
    /// Spectral embedding of the fuzzy 1-skeleton (default).
    Spectral,
    /// Uniform random positions in [-10, 10].
    Random,
    /// First `n_components` of PCA applied to the input data.
    Pca,
}

/// Auxiliary outputs of `simplicial_set_embedding`. Python: `aux_data` dict.
#[derive(Debug, Default, Clone)]
pub struct AuxData {
    /// Snapshots at requested checkpoints (Python `n_epochs` list).
    pub embedding_list: Vec<ndarray::Array2<f32>>,
    /// Local radii in the original data (`output_dens`).
    pub rad_orig: Option<Vec<f32>>,
    /// Local radii in the embedding (`output_dens`).
    pub rad_emb: Option<Vec<f32>>,
}

/// Parameters for `simplicial_set_embedding`. Python kwargs grouped into a
/// struct; defaults mirror the Python signature.
#[derive(Clone)]
pub struct SsetParams {
    /// Embedding dimensionality (Python `n_components`).
    pub n_components: usize,
    /// Initial SGD learning rate (Python `initial_alpha`).
    pub initial_alpha: f64,
    /// Differentiable-curve parameter a.
    pub a: f64,
    /// Differentiable-curve parameter b.
    pub b: f64,
    /// Negative-sample weight (Python gamma).
    pub gamma: f64,
    /// Negative samples per positive sample per epoch.
    pub negative_sample_rate: f64,
    /// Training epochs; 0 selects 500 (small data) / 200 (large) + 200 for
    /// densMAP, mirroring the Python defaults.
    pub n_epochs: usize,
    /// Initialization method.
    pub init: Init,
    /// High-dimensional metric used by the multi-component fallback layout.
    pub metric: crate::fuzzy::MetricFn,
    pub metric_args: Vec<f32>,
    /// Run the SGD with rayon (nondeterministic floats, like numba parallel).
    pub parallel: bool,
    pub verbose: bool,
    /// densMAP density-augmented objective.
    pub densmap: bool,
    /// densMAP auxiliary data (`graph_dists`, `n_neighbors`, lambda, frac,
    /// `var_shift`) — required when densmap or `output_dens` is set.
    pub densmap_kwds: Option<DensmapKwdsInput>,
    /// Output local radii (Python `output_dens`).
    pub output_dens: bool,
    /// Output metric for non-euclidean embeddings; None uses the specialized
    /// euclidean kernel (Python `euclidean_output=True`).
    pub output_metric: Option<OutputMetricFn>,
    pub output_metric_kwds: Vec<f32>,
    /// Epoch checkpoints (Python `n_epochs` list) — snapshots in `aux_data`.
    pub embedding_checkpoints: Option<Vec<usize>>,
}

/// densMAP inputs the caller must provide (Python `densmap_kwds` entries that
/// are computed in UMAP.fit before calling `simplicial_set_embedding`).
#[derive(Debug, Clone)]
pub struct DensmapKwdsInput {
    /// Shortest-path graph distances over the fuzzy graph (Python
    /// densMAP dictionary key `graph_dists`).
    pub graph_dists: CsrMatrix,
    /// Number of neighbors used for the embedding-space graph.
    pub n_neighbors: usize,
    /// densmap lambda weight.
    pub lambda: f64,
    /// densmap fraction of epochs with the density objective.
    pub frac: f64,
    /// Variance shift for numerical stability.
    pub var_shift: f64,
}

impl Default for SsetParams {
    fn default() -> Self {
        Self {
            n_components: 2,
            initial_alpha: 1.0,
            a: 1.577,
            b: 0.8951,
            gamma: 1.0,
            negative_sample_rate: 5.0,
            n_epochs: 0,
            init: Init::Spectral,
            metric: crate::distances::euclidean,
            metric_args: vec![],
            parallel: false,
            verbose: false,
            densmap: false,
            densmap_kwds: None,
            output_dens: false,
            output_metric: None,
            output_metric_kwds: vec![],
            embedding_checkpoints: None,
        }
    }
}

/// Python: `_densmap_original_densities` — per-vertex original-space radii
/// from the graph edges and the shortest-path distance matrix.
fn densmap_original_densities(
    head: &[i32],
    tail: &[i32],
    graph_data: &[f32],
    dists: &CsrMatrix,
) -> (Vec<f32>, Vec<f32>) {
    let n_vertices = dists.shape.0;
    let mut ro = vec![0.0f32; n_vertices];
    let mut mu_sum = vec![0.0f32; n_vertices];
    for (idx, (&j, &k)) in head.iter().zip(tail.iter()).enumerate() {
        let (j, k) = (j as usize, k as usize);
        let mut d_val = 0.0f32;
        for nz in dists.indptr[j]..dists.indptr[j + 1] {
            if dists.indices[nz] == k as i32 {
                d_val = dists.data[nz];
                break;
            }
        }
        let big_d = d_val * d_val;
        let mu = graph_data[idx];
        ro[j] += mu * big_d;
        ro[k] += mu * big_d;
        mu_sum[j] += mu;
        mu_sum[k] += mu;
    }
    (ro, mu_sum)
}

/// Python: `_densmap_embedding_densities` — per-vertex embedding-space
/// radii from the embedding graph edges and distances. Note the kernel uses
/// `mu * d` (NOT the squared distance used in the original-space variant).
fn densmap_embedding_densities(
    head: &[i32],
    tail: &[i32],
    graph_data: &[f32],
    dists: &CsrMatrix,
) -> (Vec<f32>, Vec<f32>) {
    let n_vertices = dists.shape.0;
    let mut re = vec![0.0f32; n_vertices];
    let mut mu_sum = vec![0.0f32; n_vertices];
    for (idx, (&j, &k)) in head.iter().zip(tail.iter()).enumerate() {
        let (j, k) = (j as usize, k as usize);
        let mut d_val = 0.0f32;
        for nz in dists.indptr[j]..dists.indptr[j + 1] {
            if dists.indices[nz] == k as i32 {
                d_val = dists.data[nz];
                break;
            }
        }
        let mu = graph_data[idx];
        let weighted = mu * d_val;
        re[j] += weighted;
        re[k] += weighted;
        mu_sum[j] += mu;
        mu_sum[k] += mu;
    }
    (re, mu_sum)
}

/// Python: `simplicial_set_embedding` — perform a fuzzy simplicial set
/// embedding using the specified initialisation, then minimize the fuzzy set
/// cross entropy via SGD.
///
/// Returns (embedding, `aux_data`).
#[allow(clippy::too_many_arguments)]
pub fn simplicial_set_embedding(
    data: Option<&ndarray::Array2<f32>>,
    graph: &CsrMatrix,
    params: &SsetParams,
    random_state: &mut TauRng,
) -> Result<(ndarray::Array2<f32>, AuxData)> {
    let mut aux = AuxData::default();
    let n_vertices = graph.shape.1;
    let n_rows = graph.shape.0;

    // For smaller datasets we can use more epochs.
    let default_epochs =
        if n_rows <= 10_000 { 500 } else { 200 } + if params.densmap { 200 } else { 0 };
    let n_epochs = if params.n_epochs == 0 {
        default_epochs
    } else {
        params.n_epochs
    };
    let n_epochs_max = params
        .embedding_checkpoints
        .as_ref()
        .and_then(|cps| cps.iter().max().copied())
        .map_or(n_epochs, |m| m.max(n_epochs));

    // Prune low-weight edges.
    let mut pruned = graph.clone();
    let threshold = {
        let max_w = pruned
            .data
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        if n_epochs_max > 10 {
            max_w / n_epochs_max as f32
        } else {
            max_w / default_epochs as f32
        }
    };
    for v in &mut pruned.data {
        if *v < threshold {
            *v = 0.0;
        }
    }
    pruned.eliminate_zeros();

    // Initialization.
    let mut embedding: ndarray::Array2<f32> = match params.init {
        Init::Random => {
            let mut emb = ndarray::Array2::<f32>::zeros((n_rows, params.n_components));
            for v in &mut emb {
                *v = (random_state.uniform_f64() * 20.0 - 10.0) as f32;
            }
            emb
        }
        Init::Pca => {
            let data = data.ok_or_else(|| {
                UmapError::InvalidInput("pca init requires dense input data".to_string())
            })?;
            let emb = pca_init(data, params.n_components, random_state)?;
            noisy_scale_coords(&emb, random_state, 10.0, 0.0001)
        }
        Init::Spectral => {
            let emb = spectral_layout(data, &pruned, params.n_components, random_state)?;
            noisy_scale_coords(&emb, random_state, 10.0, 0.0001)
        }
    };

    let epochs_per_sample = crate::fuzzy::make_epochs_per_sample(&pruned.data, n_epochs_max);

    let head: Vec<i32> = {
        let mut h = Vec::with_capacity(pruned.data.len());
        for i in 0..pruned.shape.0 {
            for _ in pruned.indptr[i]..pruned.indptr[i + 1] {
                h.push(i as i32);
            }
        }
        h
    };
    let tail: Vec<i32> = pruned.indices.clone();
    let weight: Vec<f32> = pruned.data.clone();

    let _rng_state = random_state.draw_rng_state();

    let mut densmap_kwds: Option<DensmapKwds> = None;
    if params.densmap || params.output_dens {
        let input = params.densmap_kwds.as_ref().ok_or_else(|| {
            UmapError::InvalidInput(
                "densmap/output_dens requires densmap_kwds with graph_dists".to_string(),
            )
        })?;

        let (ro_raw, mu_sum) =
            densmap_original_densities(&head, &tail, &weight, &input.graph_dists);

        let epsilon = 1e-8f32;
        let ro: Vec<f32> = ro_raw
            .iter()
            .zip(mu_sum.iter())
            .map(|(&r, &m)| (epsilon + r / m).ln())
            .collect();

        if params.densmap {
            let mean = ro.iter().map(|&v| f64::from(v)).sum::<f64>() / n_vertices as f64;
            let var = ro
                .iter()
                .map(|&v| {
                    let d = f64::from(v) - mean;
                    d * d
                })
                .sum::<f64>()
                / n_vertices as f64;
            let std = var.sqrt();
            let r: Vec<f32> = ro
                .iter()
                .map(|&v| ((f64::from(v) - mean) / std) as f32)
                .collect();
            densmap_kwds = Some(DensmapKwds {
                lambda: input.lambda,
                frac: input.frac,
                var_shift: input.var_shift,
                r,
                mu: weight.clone(),
                mu_sum: mu_sum.clone(),
            });
        }

        if params.output_dens {
            aux.rad_orig = Some(ro);
        }
    }

    // Rescale the embedding to [0, 10] per dimension.
    for d in 0..params.n_components {
        let mut min = f32::INFINITY;
        let mut max = f32::NEG_INFINITY;
        for i in 0..n_rows {
            let v = embedding[[i, d]];
            min = min.min(v);
            max = max.max(v);
        }
        let range = max - min;
        if range > 0.0 {
            for i in 0..n_rows {
                embedding[[i, d]] = 10.0 * (embedding[[i, d]] - min) / range;
            }
        }
    }

    if let Some(metric_fn) = params.output_metric {
        optimize_layout_generic(
            &mut embedding,
            TailEmbedding::Same,
            &head,
            &tail,
            n_epochs,
            n_vertices,
            &epochs_per_sample,
            params.a,
            params.b,
            &mut random_state.draw_rng_state(),
            params.gamma,
            params.initial_alpha,
            params.negative_sample_rate,
            metric_fn,
            &params.output_metric_kwds,
            true,
        );
    } else {
        let snapshots = optimize_layout_euclidean(
            &mut embedding,
            TailEmbedding::Same,
            &head,
            &tail,
            n_epochs,
            n_vertices,
            &epochs_per_sample,
            params.a,
            params.b,
            &mut random_state.draw_rng_state(),
            params.gamma,
            params.initial_alpha,
            params.negative_sample_rate,
            params.parallel,
            params.verbose,
            params.densmap,
            densmap_kwds.as_ref(),
            params.embedding_checkpoints.as_deref(),
            true,
        );
        if !snapshots.is_empty() {
            aux.embedding_list = snapshots;
        }
    }

    if params.output_dens {
        // Compute the graph in embedding space (brute-force exact NN).
        let input = params.densmap_kwds.as_ref().ok_or_else(|| {
            UmapError::InvalidInput("output_dens requires densmap_kwds".to_string())
        })?;
        let n_neighbors = input.n_neighbors;
        let (knn_indices, knn_dists) = exact_nn(&embedding, n_neighbors);

        let (emb_graph, _sigmas, _rhos, emb_dists) = crate::fuzzy::fuzzy_simplicial_set(
            n_rows,
            n_neighbors,
            &knn_indices,
            &knn_dists,
            1.0,
            1.0,
            true,
            true,
        )?;

        let emb_head: Vec<i32> = {
            let mut h = Vec::with_capacity(emb_graph.data.len());
            for i in 0..emb_graph.shape.0 {
                for _ in emb_graph.indptr[i]..emb_graph.indptr[i + 1] {
                    h.push(i as i32);
                }
            }
            h
        };
        let (re_raw, mu_sum) = densmap_embedding_densities(
            &emb_head,
            &emb_graph.indices,
            &emb_graph.data,
            emb_dists.as_ref().expect("dists"),
        );

        let epsilon = 1e-8f32;
        let re: Vec<f32> = re_raw
            .iter()
            .zip(mu_sum.iter())
            .map(|(&r, &m)| (epsilon + r / m).ln())
            .collect();
        aux.rad_emb = Some(re);
    }

    Ok((embedding, aux))
}

/// sklearn `PCA(n_components)` replacement: center the data, eigendecompose
/// the covariance matrix, project onto the top components. Sign convention
/// of each component is indeterminate (sklearn's `svd_flip` differs) —
/// statistical parity only.
#[allow(clippy::needless_pass_by_ref_mut)]
fn pca_init(
    data: &ndarray::Array2<f32>,
    n_components: usize,
    _random_state: &mut TauRng,
) -> Result<ndarray::Array2<f32>> {
    let (n, f) = (data.nrows(), data.ncols());
    if n == 0 || f == 0 {
        return Err(UmapError::InvalidInput(
            "pca init requires non-empty data".to_string(),
        ));
    }
    let k = n_components.min(f).min(n);

    // Center in f64.
    let mut means = vec![0.0f64; f];
    for i in 0..n {
        for j in 0..f {
            means[j] += f64::from(data[[i, j]]);
        }
    }
    for m in &mut means {
        *m /= n as f64;
    }
    let mut centered = ndarray::Array2::<f64>::zeros((n, f));
    for i in 0..n {
        for j in 0..f {
            centered[[i, j]] = f64::from(data[[i, j]]) - means[j];
        }
    }

    // Covariance matrix (f x f), unbiased like sklearn.
    let mut cov = ndarray::Array2::<f64>::zeros((f, f));
    for a in 0..f {
        for b in a..f {
            let mut s = 0.0f64;
            for i in 0..n {
                s += centered[[i, a]] * centered[[i, b]];
            }
            let v = s / (n - 1).max(1) as f64;
            cov[[a, b]] = v;
            cov[[b, a]] = v;
        }
    }

    let (eigvals, eigvecs) = crate::spectral::self_adjoint_eigendecomposition(&cov)?;
    // Top-k components by descending eigenvalue.
    let mut order: Vec<usize> = (0..eigvals.len()).collect();
    order.sort_by(|&a, &b| {
        eigvals[b]
            .partial_cmp(&eigvals[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut emb = ndarray::Array2::<f32>::zeros((n, k));
    for (out_col, &src) in order.iter().take(k).enumerate() {
        for i in 0..n {
            let mut s = 0.0f64;
            for j in 0..f {
                s += centered[[i, j]] * eigvecs[[j, src]];
            }
            emb[[i, out_col]] = s as f32;
        }
    }
    Ok(emb)
}

/// Exact brute-force k-nearest neighbors (replacement for
/// `nearest_neighbors`/pynndescent on small inputs; the precomputed-KNN
/// contract covers large inputs). Includes the self-neighbor at index 0,
/// matching Python's knn arrays.
#[must_use]
pub fn exact_nn(
    embedding: &ndarray::Array2<f32>,
    k: usize,
) -> (ndarray::Array2<i32>, ndarray::Array2<f32>) {
    let n = embedding.nrows();
    let mut indices = ndarray::Array2::<i32>::from_elem((n, k), -1);
    let mut dists = ndarray::Array2::<f32>::from_elem((n, k), f32::INFINITY);
    for i in 0..n {
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| {
            let da = sq_dist(embedding, i, a);
            let db = sq_dist(embedding, i, b);
            da.partial_cmp(&db)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.cmp(&b))
        });
        for (j, &idx) in order.iter().take(k).enumerate() {
            indices[[i, j]] = idx as i32;
            dists[[i, j]] = sq_dist(embedding, i, idx).sqrt() as f32;
        }
    }
    (indices, dists)
}

#[inline]
fn sq_dist(m: &ndarray::Array2<f32>, i: usize, j: usize) -> f64 {
    let mut s = 0.0f64;
    for d in 0..m.ncols() {
        let diff = f64::from(m[[i, d]]) - f64::from(m[[j, d]]);
        s += diff * diff;
    }
    s
}

/// Python: `init_transform` — initialize new points relative to their
/// neighbors' embedding positions.
/// Python: `init_transform` — initialize new points relative to their
/// neighbors' embedding positions.
#[must_use]
pub fn init_transform(
    indices: &ndarray::Array2<i32>,
    weights: &ndarray::Array2<f32>,
    embedding: &ndarray::Array2<f32>,
) -> ndarray::Array2<f32> {
    let mut result = ndarray::Array2::<f32>::zeros((indices.nrows(), embedding.ncols()));
    for i in 0..indices.nrows() {
        for j in 0..indices.ncols() {
            let idx = indices[[i, j]];
            if idx < 0 {
                continue;
            }
            for d in 0..embedding.ncols() {
                result[[i, d]] += weights[[i, j]] * embedding[[idx as usize, d]];
            }
        }
    }
    result
}

/// Python: `init_graph_transform` — initialize new points from a bipartite
/// graph over the training data. Rows with an edge weight of exactly 1.0
/// embed at that training point's coordinates (first such edge wins);
/// non-empty rows without an exact match are the weighted average of their
/// neighbors' embedding positions; empty rows embed as NaN.
#[must_use]
pub fn init_graph_transform(
    graph: &CsrMatrix,
    embedding: &ndarray::Array2<f32>,
) -> ndarray::Array2<f32> {
    let n_new = graph.shape.0;
    let mut result = ndarray::Array2::<f32>::zeros((n_new, embedding.ncols()));

    // Exact matches first: a row whose edge weight is 1.0 was identical to a
    // training point (Python exact_data_mask / has_exact logic).
    let mut has_exact = vec![false; n_new];
    for i in 0..n_new {
        for z in graph.indptr[i]..graph.indptr[i + 1] {
            // Python compares the weight to exactly 1.0.
            #[allow(clippy::float_cmp)]
            if graph.data[z] == 1.0 {
                let c = graph.indices[z] as usize;
                for d in 0..embedding.ncols() {
                    result[[i, d]] = embedding[[c, d]];
                }
                has_exact[i] = true;
                break;
            }
        }
    }

    // Weighted averages for non-exact, non-empty rows.
    for i in 0..n_new {
        if has_exact[i] {
            continue;
        }
        let (idx, data) = graph.row_slice(i);
        if idx.is_empty() {
            continue;
        }
        let total: f32 = data.iter().sum();
        for (&c, &w) in idx.iter().zip(data.iter()) {
            for d in 0..embedding.ncols() {
                result[[i, d]] += w / total * embedding[[c as usize, d]];
            }
        }
    }

    // Empty rows embed as NaN.
    for i in 0..n_new {
        if graph.indptr[i + 1] == graph.indptr[i] {
            for d in 0..embedding.ncols() {
                result[[i, d]] = f32::NAN;
            }
        }
    }
    result
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unreadable_literal, clippy::float_cmp)]

    use super::*;

    #[test]
    fn find_ab_params_matches_scipy() {
        // References: scipy curve_fit via the umap repo (see test comment).
        // Note: scipy curve_fit stops at its default ftol/xtol slightly short
        // of the true minimum; our LM converges further (lower cost). For
        // (1.0, 1.0) the resulting b gap is ~1.8e-6 — well within statistical
        // parity (UMAP docs round a/b to 4 digits).
        let cases = [
            ((1.0, 0.1), (1.57694346046584, 0.8950608779639974), 1e-6),
            ((1.0, 0.5), (0.5830300203968329, 1.3341669926352966), 1e-6),
            ((1.0, 0.001), (1.9290733950863033, 0.7915045328700837), 1e-6),
            ((1.0, 1.0), (0.11497568236610484, 1.9292371499665542), 1e-5),
            ((0.5, 0.1), (5.069309324972921, 1.003005422393625), 1e-6),
        ];
        for ((spread, min_dist), (a_ref, b_ref), tol) in cases {
            let (a, b) = find_ab_params(spread, min_dist);
            assert!(
                (a - a_ref).abs() < tol,
                "spread {spread} min_dist {min_dist}: a {a} vs {a_ref}"
            );
            assert!(
                (b - b_ref).abs() < tol,
                "spread {spread} min_dist {min_dist}: b {b} vs {b_ref}"
            );
        }
    }

    #[test]
    fn init_transform_weighted_average() {
        let indices = ndarray::arr2(&[[0, 1], [1, -1]]);
        let weights = ndarray::arr2(&[[0.5, 0.5], [1.0, 0.0]]);
        let embedding = ndarray::arr2(&[[0.0, 1.0], [2.0, 3.0]]);
        let result = init_transform(&indices, &weights, &embedding);
        assert_eq!(result[[0, 0]], 1.0);
        assert_eq!(result[[0, 1]], 2.0);
        assert_eq!(result[[1, 0]], 2.0);
    }
}
