//! Ported from Python: umap/umap_.py (UMAP class core: fit, `fit_transform`,
//! transform, `inverse_transform`).
//!
//! Divergences:
//! - pynndescent is NOT ported. For raw dense inputs with n < 4096 the kNN
//!   graph is computed exactly by brute force; for larger inputs callers must
//!   supply precomputed kNN via [`Umap::fit_with_knn`].
//! - `inverse_transform` neighborhoods use brute-force exact nearest
//!   neighbors in the embedding space instead of scipy Delaunay triangulation
//!   + BFS (statistical parity only).
//! - Supervised fitting (`y`), `update`, model serialization, and sparse
//!   input data are out of scope for this version.
//! - densMAP `graph_dists` are computed with a hand-rolled Dijkstra over the
//!   fuzzy graph (scipy `csgraph.dijkstra` equivalent, same values).

use crate::csr::{CooMatrix, CsrMatrix};
use crate::embedding::{
    find_ab_params, init_graph_transform, simplicial_set_embedding, DensmapKwdsInput, Init,
    SsetParams,
};
use crate::error::{Result, UmapError};
use crate::fuzzy::{
    compute_membership_strengths, fuzzy_simplicial_set, make_epochs_per_sample, smooth_knn_dist,
};
use crate::layouts::{
    named_output_metric, optimize_layout_euclidean, optimize_layout_generic,
    optimize_layout_inverse, TailEmbedding,
};
use crate::rng::TauRng;

/// Default threshold: inputs with n rows below this use the exact brute-force
/// kNN path (Python uses pynndescent for everything; the precomputed-KNN
/// contract replaces it for large n).
const BRUTE_FORCE_MAX_N: usize = 4096;

/// Configuration for a UMAP model. Defaults mirror the Python signature.
#[derive(Debug, Clone)]
pub struct UmapConfig {
    /// Size of the local neighborhood (Python `n_neighbors`).
    pub n_neighbors: usize,
    /// Minimum distance between embedded points.
    pub min_dist: f64,
    /// Effective scale of embedded points.
    pub spread: f64,
    /// Embedding dimensionality (Python `n_components`).
    pub n_components: usize,
    /// Distance metric name (Python metric), resolved against
    /// `distances::named_distances`.
    pub metric: String,
    /// Extra metric arguments (Python `metric_kwds` values).
    pub metric_args: Vec<f32>,
    /// Training epochs; None selects 500 (n <= 10000) / 200 (+200 densMAP).
    pub n_epochs: Option<usize>,
    /// Initial SGD learning rate (Python `learning_rate` / `initial_alpha`).
    pub learning_rate: f64,
    /// Initialization method.
    pub init: Init,
    /// Random seed; None gives nondeterministic parallel runs.
    pub random_state: Option<u64>,
    /// Negative-sample weight (Python `repulsion_strength` / gamma).
    pub repulsion_strength: f64,
    /// Negative samples per positive sample per epoch.
    pub negative_sample_rate: f64,
    /// Local connectivity (Python `local_connectivity`).
    pub local_connectivity: f64,
    /// Fuzzy set operation mix (Python `set_op_mix_ratio`).
    pub set_op_mix_ratio: f64,
    /// "embedding" (default) or "graph" transform output.
    pub transform_mode: TransformMode,
    /// Optional non-euclidean output metric name for the SGD.
    pub output_metric: Option<String>,
    /// Output metric arguments.
    pub output_metric_kwds: Vec<f32>,
    /// densMAP density-augmented objective.
    pub densmap: bool,
    /// densMAP lambda weight (Python `lambda_dens`).
    pub densmap_lambda: f64,
    /// densMAP fraction of epochs with the density objective (Python `frac_dens`).
    pub densmap_frac: f64,
    /// Distances at or above this disconnect the graph (Python
    /// `disconnection_distance`; None = no cutoff).
    pub disconnection_distance: Option<f32>,
}

impl Default for UmapConfig {
    fn default() -> Self {
        Self {
            n_neighbors: 15,
            min_dist: 0.1,
            spread: 1.0,
            n_components: 2,
            metric: "euclidean".to_string(),
            metric_args: vec![],
            n_epochs: None,
            learning_rate: 1.0,
            init: Init::Spectral,
            random_state: Some(42),
            repulsion_strength: 1.0,
            negative_sample_rate: 5.0,
            local_connectivity: 1.0,
            set_op_mix_ratio: 1.0,
            transform_mode: TransformMode::Embedding,
            output_metric: None,
            output_metric_kwds: vec![],
            densmap: false,
            densmap_lambda: 2.0,
            densmap_frac: 0.3,
            disconnection_distance: None,
        }
    }
}

/// Python `transform_mode` parameter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransformMode {
    Embedding,
    Graph,
}

/// A fitted UMAP model.
#[derive(Debug, Clone)]
pub struct Umap {
    /// The configuration used for fitting.
    pub config: UmapConfig,
    /// The low-dimensional embedding (Python embedding_).
    pub embedding: ndarray::Array2<f32>,
    /// The fuzzy 1-skeleton of the fitted data (Python graph_).
    pub graph: CsrMatrix,
    /// Differentiable-curve parameter a (Python _a).
    pub a: f64,
    /// Differentiable-curve parameter b (Python _b).
    pub b: f64,
    /// The raw training data (Python _`raw_data`).
    pub raw_data: ndarray::Array2<f32>,
    /// Per-sample smooth-knn sigmas (Python _sigmas).
    pub sigmas: Vec<f32>,
    /// Per-sample smooth-knn rhos (Python _rhos).
    pub rhos: Vec<f32>,
    /// Auxiliary data from embedding (checkpoint snapshots, densMAP radii).
    pub aux_data: crate::embedding::AuxData,
}

impl Umap {
    /// Create an unfitted model with the given configuration; call
    /// [`Umap::fit`] or [`Umap::fit_with_knn`] to populate the fitted fields.
    #[must_use]
    pub fn with_config(config: UmapConfig) -> Self {
        Self {
            config,
            embedding: ndarray::Array2::zeros((0, 0)),
            graph: CsrMatrix::from_coo(&CooMatrix::new((0, 0))),
            a: 0.0,
            b: 0.0,
            raw_data: ndarray::Array2::zeros((0, 0)),
            sigmas: vec![],
            rhos: vec![],
            aux_data: crate::embedding::AuxData::default(),
        }
    }

    /// Fit the model on raw dense data. For n < 4096 the kNN graph is exact
    /// brute force; for larger n use [`Umap::fit_with_knn`].
    ///
    /// # Errors
    /// Returns an error on invalid metrics or insufficient data.
    pub fn fit(&self, data: &ndarray::Array2<f32>) -> Result<Umap> {
        let n = data.nrows();
        if n < 2 {
            return Err(UmapError::InvalidInput(
                "need at least 2 samples to fit".to_string(),
            ));
        }
        if n >= BRUTE_FORCE_MAX_N {
            return Err(UmapError::InvalidInput(format!(
                "n = {n} >= {BRUTE_FORCE_MAX_N}: supply precomputed kNN via fit_with_knn"
            )));
        }
        let metric_fn =
            crate::distances::named_distances(&self.config.metric).ok_or_else(|| {
                UmapError::InvalidInput(format!("unknown metric '{}'", self.config.metric))
            })?;
        let (knn_indices, knn_dists) = brute_force_knn(
            data,
            data,
            self.config.n_neighbors,
            metric_fn,
            &self.config.metric_args,
        );
        self.fit_with_knn(Some(data), &knn_indices, &knn_dists)
    }

    /// Fit with precomputed k-nearest neighbors (indices/dists include the
    /// self-neighbor at column 0). `data` is required for the spectral/pca
    /// initializations and multi-component layouts.
    ///
    /// # Errors
    /// Returns an error on invalid metrics or mismatched kNN shapes.
    pub fn fit_with_knn(
        &self,
        data: Option<&ndarray::Array2<f32>>,
        knn_indices: &ndarray::Array2<i32>,
        knn_dists: &ndarray::Array2<f32>,
    ) -> Result<Umap> {
        let cfg = &self.config;
        let n = knn_indices.nrows();
        let k = knn_indices.ncols();
        if knn_dists.nrows() != n || knn_dists.ncols() != k {
            return Err(UmapError::InvalidInput(
                "knn_indices and knn_dists shapes differ".to_string(),
            ));
        }
        let metric_fn = crate::distances::named_distances(&cfg.metric)
            .ok_or_else(|| UmapError::InvalidInput(format!("unknown metric '{}'", cfg.metric)))?;

        let mut random_state = TauRng::new(cfg.random_state.unwrap_or(42));

        // Prune disconnected neighbors (Python fit:
        // indices[dists >= disconnection_distance] = -1; default = no cutoff).
        let mut pruned_indices = knn_indices.clone();
        let mut pruned_dists = knn_dists.clone();
        if let Some(dc) = cfg.disconnection_distance {
            for i in 0..n {
                for j in 0..k {
                    if pruned_dists[[i, j]] >= dc {
                        pruned_indices[[i, j]] = -1;
                        pruned_dists[[i, j]] = f32::INFINITY;
                    }
                }
            }
        }

        // Fuzzy simplicial set of the kNN graph.
        let (graph, sigmas, rhos, _dists) = fuzzy_simplicial_set(
            n,
            k,
            &pruned_indices,
            &pruned_dists,
            cfg.set_op_mix_ratio,
            cfg.local_connectivity,
            true,
            false,
        )?;

        // a, b params for the differentiable curve.
        let (a, b) = find_ab_params(cfg.spread, cfg.min_dist);

        // Default epochs (Python fit rule).
        let densmap_extra = if cfg.densmap { 200 } else { 0 };
        let _default_epochs = (if n <= 10_000 { 500 } else { 200 }) + densmap_extra;

        let densmap_kwds = if cfg.densmap {
            Some(DensmapKwdsInput {
                graph_dists: dijkstra_all(&graph),
                n_neighbors: cfg.n_neighbors,
                lambda: cfg.densmap_lambda,
                frac: cfg.densmap_frac,
                var_shift: 0.1, // Python: constant offset to avoid log(0)
            })
        } else {
            None
        };

        let params = SsetParams {
            n_components: cfg.n_components,
            initial_alpha: cfg.learning_rate,
            a,
            b,
            gamma: cfg.repulsion_strength,
            negative_sample_rate: cfg.negative_sample_rate,
            n_epochs: cfg.n_epochs.unwrap_or(0),
            init: cfg.init,
            metric: metric_fn,
            metric_args: cfg.metric_args.clone(),
            parallel: cfg.random_state.is_none(),
            verbose: false,
            densmap: cfg.densmap,
            densmap_kwds,
            output_dens: false,
            output_metric: cfg
                .output_metric
                .as_ref()
                .and_then(|name| named_output_metric(name)),
            output_metric_kwds: cfg.output_metric_kwds.clone(),
            embedding_checkpoints: None,
        };

        let (embedding, aux) = simplicial_set_embedding(data, &graph, &params, &mut random_state)?;

        Ok(Umap {
            config: cfg.clone(),
            embedding,
            graph,
            a,
            b,
            raw_data: data
                .cloned()
                .unwrap_or_else(|| ndarray::Array2::zeros((0, 0))),
            sigmas,
            rhos,
            aux_data: aux,
        })
    }

    /// Embed new points into the fitted space. Python `transform`.
    ///
    /// # Errors
    /// Returns an error if the model was fitted without raw data, or the
    /// graph transform mode is requested.
    pub fn transform(&self, new_data: &ndarray::Array2<f32>) -> Result<ndarray::Array2<f32>> {
        if self.raw_data.nrows() == 0 {
            return Err(UmapError::InvalidInput(
                "model was fitted without raw data; transform unavailable".to_string(),
            ));
        }
        if self.config.transform_mode == TransformMode::Graph {
            return Err(UmapError::InvalidInput(
                "transform_mode == graph: use transform_graph instead".to_string(),
            ));
        }
        let cfg = &self.config;
        let metric_fn = crate::distances::named_distances(&cfg.metric)
            .ok_or_else(|| UmapError::InvalidInput(format!("unknown metric '{}'", cfg.metric)))?;

        let n_new = new_data.nrows();
        let k = cfg.n_neighbors;
        let mut random_state = TauRng::new(cfg.random_state.unwrap_or(42));

        // Exact kNN of the new points against the training data.
        let (mut indices, mut dists) =
            brute_force_knn(new_data, &self.raw_data, k, metric_fn, &cfg.metric_args);

        // Prune disconnected neighbors.
        if let Some(dc) = cfg.disconnection_distance {
            for i in 0..n_new {
                for j in 0..k {
                    if dists[[i, j]] >= dc {
                        indices[[i, j]] = -1;
                        dists[[i, j]] = f32::INFINITY;
                    }
                }
            }
        }

        // Adjusted local connectivity for the transform graph.
        let adjusted_local_connectivity = (cfg.local_connectivity - 1.0).max(0.0);
        let (sigmas, rhos) =
            smooth_knn_dist(&dists, k as f64, 64, adjusted_local_connectivity, 1.0);

        let (rows, cols, vals, _d) =
            compute_membership_strengths(&indices, &dists, &sigmas, &rhos, false, true);
        let graph =
            CooMatrix::from_triplets((n_new, self.raw_data.nrows()), rows, cols, vals).tocsr();

        let mut csr_graph = graph.clone();
        csr_graph.eliminate_zeros();
        let mut embedding = init_graph_transform(&csr_graph, &self.embedding);

        // Epochs: Python transform rule.
        let n_epochs = match cfg.n_epochs {
            None => {
                if graph.shape.0 <= 10_000 {
                    100
                } else {
                    30
                }
            }
            Some(e) => (e as f64 / 3.0) as usize,
        };

        // Prune + epochs per sample.
        let mut pruned = graph;
        let max_w = pruned
            .data
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        let threshold = max_w / n_epochs as f32;
        for v in &mut pruned.data {
            if *v < threshold {
                *v = 0.0;
            }
        }
        pruned.eliminate_zeros();
        let epochs_per_sample = make_epochs_per_sample(&pruned.data, n_epochs);

        let mut head = Vec::with_capacity(pruned.data.len());
        for i in 0..pruned.shape.0 {
            for _ in pruned.indptr[i]..pruned.indptr[i + 1] {
                head.push(i as i32);
            }
        }
        let tail: Vec<i32> = pruned.indices.clone();

        let mut rng_state = random_state.draw_rng_state();
        let mut tail_embedding = self.embedding.clone();

        if let Some(metric_name) = &cfg.output_metric {
            let output_metric_fn = named_output_metric(metric_name).ok_or_else(|| {
                UmapError::InvalidInput(format!("unknown output metric '{metric_name}'"))
            })?;
            optimize_layout_generic(
                &mut embedding,
                TailEmbedding::Distinct(&mut tail_embedding),
                &head,
                &tail,
                n_epochs,
                pruned.shape.1,
                &epochs_per_sample,
                self.a,
                self.b,
                &mut rng_state,
                cfg.repulsion_strength,
                cfg.learning_rate / 4.0,
                cfg.negative_sample_rate,
                output_metric_fn,
                &cfg.output_metric_kwds,
                false,
            );
        } else {
            optimize_layout_euclidean(
                &mut embedding,
                TailEmbedding::Distinct(&mut tail_embedding),
                &head,
                &tail,
                n_epochs,
                pruned.shape.1,
                &epochs_per_sample,
                self.a,
                self.b,
                &mut rng_state,
                cfg.repulsion_strength,
                cfg.learning_rate / 4.0,
                cfg.negative_sample_rate,
                cfg.random_state.is_none(),
                false,
                false,
                None,
                None,
                false,
            );
        }

        Ok(embedding)
    }

    /// Return the transform-mode fuzzy graph for new points (Python
    /// `transform` with `transform_mode == "graph"`).
    ///
    /// # Errors
    /// Returns an error if the model was fitted without raw data.
    pub fn transform_graph(&self, new_data: &ndarray::Array2<f32>) -> Result<CsrMatrix> {
        if self.raw_data.nrows() == 0 {
            return Err(UmapError::InvalidInput(
                "model was fitted without raw data; transform unavailable".to_string(),
            ));
        }
        let cfg = &self.config;
        let metric_fn = crate::distances::named_distances(&cfg.metric)
            .ok_or_else(|| UmapError::InvalidInput(format!("unknown metric '{}'", cfg.metric)))?;

        let n_new = new_data.nrows();
        let k = cfg.n_neighbors;

        let (mut indices, mut dists) =
            brute_force_knn(new_data, &self.raw_data, k, metric_fn, &cfg.metric_args);
        if let Some(dc) = cfg.disconnection_distance {
            for i in 0..n_new {
                for j in 0..k {
                    if dists[[i, j]] >= dc {
                        indices[[i, j]] = -1;
                        dists[[i, j]] = f32::INFINITY;
                    }
                }
            }
        }
        let adjusted_local_connectivity = (cfg.local_connectivity - 1.0).max(0.0);
        let (sigmas, rhos) =
            smooth_knn_dist(&dists, k as f64, 64, adjusted_local_connectivity, 1.0);
        let (rows, cols, vals, _d) =
            compute_membership_strengths(&indices, &dists, &sigmas, &rhos, false, true);
        let graph =
            CooMatrix::from_triplets((n_new, self.raw_data.nrows()), rows, cols, vals).tocsr();
        Ok(graph)
    }

    /// Project new points back into the input space. Python
    /// `inverse_transform`; neighborhoods use exact brute-force kNN in the
    /// embedding space (replacing scipy Delaunay + BFS).
    ///
    /// # Errors
    /// Returns an error if the model was fitted without raw data.
    pub fn inverse_transform(
        &self,
        embedding: &ndarray::Array2<f32>,
    ) -> Result<ndarray::Array2<f32>> {
        if self.raw_data.nrows() == 0 {
            return Err(UmapError::InvalidInput(
                "model was fitted without raw data; inverse_transform unavailable".to_string(),
            ));
        }
        let cfg = &self.config;
        let dist_fn = match &cfg.output_metric {
            Some(name) => named_output_metric(name).ok_or_else(|| {
                UmapError::InvalidInput(format!("unknown output metric '{name}'"))
            })?,
            None => named_output_metric("euclidean")
                .ok_or_else(|| UmapError::Computation("euclidean metric missing".to_string()))?,
        };

        let n_new = embedding.nrows();
        let mut random_state = TauRng::new(cfg.random_state.unwrap_or(42));
        let min_vertices = self.raw_data.ncols().min(self.raw_data.nrows()).max(1);

        // Distances from each new point to every embedding point, ordered by
        // distance (stable ties by index).
        let mut ordered: Vec<Vec<(usize, f32)>> = Vec::with_capacity(n_new);
        for i in 0..n_new {
            let xi: Vec<f32> = (0..embedding.ncols()).map(|d| embedding[[i, d]]).collect();
            let mut pairs: Vec<(usize, f32)> = (0..self.embedding.nrows())
                .map(|nb| {
                    let yi: Vec<f32> = (0..self.embedding.ncols())
                        .map(|d| self.embedding[[nb, d]])
                        .collect();
                    (nb, dist_fn(&xi, &yi, &cfg.output_metric_kwds).0)
                })
                .collect();
            pairs.sort_by(|a, b| {
                a.1.partial_cmp(&b.1)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then(a.0.cmp(&b.0))
            });
            ordered.push(pairs);
        }

        // Membership strengths of each edge: 1 / (1 + a d^(2b)).
        let mut graph = CooMatrix::new((n_new, self.raw_data.nrows()));
        for (i, pairs) in ordered.iter().enumerate() {
            for &(nb, d) in pairs.iter().take(min_vertices) {
                let w = 1.0 / (1.0 + self.a * f64::from(d).powf(2.0 * self.b));
                graph.row.push(i as i32);
                graph.col.push(nb as i32);
                graph.data.push(w as f32);
            }
        }

        // L1-normalize rows of the CSR for the initialization.
        let csr = graph.tocsr();
        let mut normalized = csr.clone();
        for i in 0..normalized.shape.0 {
            let sum: f32 = normalized.data[normalized.indptr[i]..normalized.indptr[i + 1]]
                .iter()
                .sum();
            if sum > 0.0 {
                for z in normalized.indptr[i]..normalized.indptr[i + 1] {
                    normalized.data[z] /= sum;
                }
            }
        }
        let mut inv = init_transform_indices(&normalized, &self.raw_data);

        // Epochs: same rule as transform.
        let n_epochs = match cfg.n_epochs {
            None => {
                if n_new <= 10_000 {
                    100
                } else {
                    30
                }
            }
            Some(e) => (e as f64 / 3.0) as usize,
        };
        // Note: epochs_per_sample from the UNnormalized weights, matching
        // Python's use of `graph.data`.
        let epochs_per_sample = make_epochs_per_sample(&graph.data, n_epochs);

        let head: Vec<i32> = graph.row.clone();
        let tail: Vec<i32> = graph.col.clone();
        let weight: Vec<f32> = graph.data.clone();
        let mut rng_state = random_state.draw_rng_state();

        let dist_fn_inv = crate::distances::named_distances_with_gradients(&cfg.metric)
            .map(|(_, grad)| grad)
            .ok_or_else(|| UmapError::InvalidInput(format!("unknown metric '{}'", cfg.metric)))?;

        let mut raw_data = self.raw_data.clone();
        optimize_layout_inverse(
            &mut inv,
            TailEmbedding::Distinct(&mut raw_data),
            &head,
            &tail,
            &weight,
            &self.sigmas,
            &self.rhos,
            n_epochs,
            self.raw_data.nrows(),
            &epochs_per_sample,
            self.a,
            self.b,
            &mut rng_state,
            cfg.repulsion_strength,
            cfg.learning_rate / 4.0,
            cfg.negative_sample_rate,
            dist_fn_inv,
            &cfg.metric_args,
            false,
        );

        Ok(inv)
    }
}

/// Brute-force k-nearest neighbors (with self-neighbor at column 0) of
/// `query` rows against `reference` rows using `metric_fn`.
#[must_use]
pub fn brute_force_knn(
    query: &ndarray::Array2<f32>,
    reference: &ndarray::Array2<f32>,
    k: usize,
    metric_fn: crate::distances::MetricFn,
    metric_args: &[f32],
) -> (ndarray::Array2<i32>, ndarray::Array2<f32>) {
    let n = query.nrows();
    let m = reference.nrows();
    let kk = k.min(m);
    let mut indices = ndarray::Array2::<i32>::from_elem((n, kk), -1);
    let mut dists = ndarray::Array2::<f32>::from_elem((n, kk), f32::INFINITY);
    for i in 0..n {
        let qi = row_as_slice(query, i);
        let mut pairs: Vec<(f32, usize)> = (0..m)
            .map(|j| {
                let rj = row_as_slice(reference, j);
                (metric_fn(qi, rj, metric_args), j)
            })
            .collect();
        pairs.sort_by(|a, b| {
            a.0.partial_cmp(&b.0)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.1.cmp(&b.1))
        });
        for (j, &(d, idx)) in pairs.iter().take(kk).enumerate() {
            indices[[i, j]] = idx as i32;
            dists[[i, j]] = d;
        }
    }
    (indices, dists)
}

/// Row of an owned standard-layout array as a contiguous slice. `ndarray`
/// owned arrays are always contiguous in C order, so this always succeeds
/// and never copies.
fn row_as_slice(m: &ndarray::Array2<f32>, i: usize) -> &[f32] {
    let ncols = m.ncols();
    &m.as_slice().expect("owned ndarray is contiguous")[i * ncols..(i + 1) * ncols]
}

/// Dijkstra single-source shortest paths over the graph for every vertex
/// (scipy `csgraph.dijkstra` equivalent). Returns a row-per-vertex CSR of
/// shortest-path distances (unreachable entries omitted).
fn dijkstra_all(graph: &CsrMatrix) -> CsrMatrix {
    let n = graph.shape.0;
    let mut all = Vec::with_capacity(n * n);
    let mut indices = Vec::with_capacity(n * n);
    let mut indptr = vec![0i32; n + 1];
    for src in 0..n {
        let mut dist = vec![f64::INFINITY; n];
        let mut visited = vec![false; n];
        dist[src] = 0.0;
        for _ in 0..n {
            // Extract-min (linear scan; fine for the small-data path).
            let mut u: i64 = -1;
            let mut best = f64::INFINITY;
            for (v, &d) in dist.iter().enumerate() {
                if !visited[v] && d < best {
                    best = d;
                    u = v as i64;
                }
            }
            if u < 0 {
                break;
            }
            let u = u as usize;
            visited[u] = true;
            for z in graph.indptr[u]..graph.indptr[u + 1] {
                let w = graph.indices[z] as usize;
                if !visited[w] {
                    let nd = dist[u] + f64::from(graph.data[z]);
                    if nd < dist[w] {
                        dist[w] = nd;
                    }
                }
            }
        }
        for (v, d) in dist.iter().enumerate() {
            if d.is_finite() {
                all.push(*d as f32);
                indices.push(v as i32);
            }
        }
        indptr[src + 1] = all.len() as i32;
    }
    let mut rows = Vec::with_capacity(all.len());
    for src in 0..n {
        for _ in indptr[src]..indptr[src + 1] {
            rows.push(src as i32);
        }
    }
    CooMatrix::from_triplets((n, n), rows, indices, all).tocsr()
}

/// Python `init_transform` over a variable-degree CSR graph (used by
/// `inverse_transform` after L1 normalization).
fn init_transform_indices(
    graph: &CsrMatrix,
    raw_data: &ndarray::Array2<f32>,
) -> ndarray::Array2<f32> {
    let mut result = ndarray::Array2::<f32>::zeros((graph.shape.0, raw_data.ncols()));
    for i in 0..graph.shape.0 {
        let (idx, data) = graph.row_slice(i);
        for (&c, &w) in idx.iter().zip(data.iter()) {
            for d in 0..raw_data.ncols() {
                result[[i, d]] += w * raw_data[[c as usize, d]];
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
    fn fit_transform_iris_is_sane() {
        // Tiny synthetic dataset with clear 2D structure.
        let n = 60;
        let mut data = ndarray::Array2::<f32>::zeros((n, 3));
        for i in 0..n {
            let cluster = i / 20;
            data[[i, 0]] = (i as f32) * 0.1 + (cluster as f32) * 50.0;
            data[[i, 1]] = ((i * 7) as f32) * 0.05 + (cluster as f32) * 30.0;
            data[[i, 2]] = ((i * 13) as f32) * 0.02;
        }
        let umap = Umap::with_config(UmapConfig {
            n_neighbors: 10,
            min_dist: 0.1,
            n_components: 2,
            n_epochs: Some(50),
            ..UmapConfig::default()
        });
        let model = umap.fit(&data).expect("fit");
        assert_eq!(model.embedding.dim(), (n, 2));
        assert!(model.embedding.iter().all(|v| v.is_finite()));
        assert!((model.a - 1.577).abs() < 1e-2);
        assert!((model.b - 0.8951).abs() < 1e-2);

        // Transform new points.
        let mut new_data = ndarray::Array2::<f32>::zeros((5, 3));
        for i in 0..5 {
            new_data[[i, 0]] = (i as f32) * 0.1 + 50.0;
            new_data[[i, 1]] = ((i * 7) as f32) * 0.05 + 30.0;
            new_data[[i, 2]] = ((i * 13) as f32) * 0.02;
        }
        let transformed = model.transform(&new_data).expect("transform");
        assert_eq!(transformed.dim(), (5, 2));
        assert!(transformed.iter().all(|v| v.is_finite()));

        // Inverse transform.
        let inv = model
            .inverse_transform(&model.embedding.slice(ndarray::s![0..5, ..]).to_owned())
            .expect("inverse");
        assert_eq!(inv.dim(), (5, 3));
    }

    #[test]
    fn brute_force_knn_includes_self() {
        let data = ndarray::arr2(&[[0.0, 0.0], [1.0, 0.0], [0.0, 2.0]]);
        let metric = crate::distances::named_distances("euclidean").expect("euclidean");
        let (indices, dists) = brute_force_knn(&data, &data, 2, metric, &[]);
        assert_eq!(indices[[1, 0]], 1); // self first
        assert_eq!(dists[[1, 0]], 0.0);
        assert_eq!(indices[[1, 1]], 0);
    }

    #[test]
    fn fit_rejects_large_n() {
        let data = ndarray::Array2::<f32>::zeros((BRUTE_FORCE_MAX_N + 1, 3));
        let umap = Umap::with_config(UmapConfig::default());
        assert!(umap.fit(&data).is_err());
    }
}
