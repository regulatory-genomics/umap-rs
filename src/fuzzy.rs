//! Ported from Python: umap/umap_.py (`smooth_knn_dist`,
//! `compute_membership_strengths`, `fuzzy_simplicial_set`, `fast_intersection`,
//! `reset_local_connectivity`, `discrete_metric_simplicial_set_intersection`,
//! `general_simplicial_set_intersection/union`, `make_epochs_per_sample`).
// The kernels mirror the numba-compiled Python structure (single-char loop
// variables, index-based loops, many flag parameters matching the Python
// signatures); pedantic lints for those are allowed module-wide.
#![allow(
    clippy::many_single_char_names,
    clippy::needless_range_loop,
    clippy::struct_excessive_bools,
    clippy::type_complexity
)]

use crate::csr::{CooMatrix, CsrMatrix};
use crate::error::{Result, UmapError};

/// Metric fn-pointer type — same contract as `crate::distances::MetricFn`
/// (fn-pointer type aliases are structural, so the two are interchangeable).
pub type MetricFn = fn(&[f32], &[f32], &[f32]) -> f32;

/// Python: `SMOOTH_K_TOLERANCE`.
pub(crate) const SMOOTH_K_TOLERANCE: f64 = 1e-5;
/// Python: `MIN_K_DIST_SCALE`.
pub(crate) const MIN_K_DIST_SCALE: f64 = 1e-3;
/// Python: `NPY_FLOATMAX = np.finfo(np.float32).max`.
pub(crate) const NPY_FLOATMAX: f64 = f32::MAX as f64;

/// Python: `umap_.smooth_knn_dist` — binary search for the continuous
/// distance to the kth nearest neighbor.
///
/// Returns (sigmas, rhos). Ported serially per row (Python uses numba
/// prange; per-row results are independent so serial is deterministic and
/// identical in value).
#[must_use]
pub fn smooth_knn_dist(
    distances: &ndarray::Array2<f32>,
    k: f64,
    n_iter: usize,
    local_connectivity: f64,
    bandwidth: f64,
) -> (Vec<f32>, Vec<f32>) {
    let target = k.log2() * bandwidth;
    let n_samples = distances.nrows();
    let mut rho = vec![0.0f32; n_samples];
    let mut result = vec![0.0f32; n_samples];

    // Pruned-by-disconnection neighbours arrive as inf; exclude them from
    // the mean used for the MIN_K_DIST_SCALE floor (Python: finite filter).
    let flat = distances.iter().copied();
    let finite: Vec<f64> = flat.filter(|d| d.is_finite()).map(f64::from).collect();
    let mean_distances = if finite.is_empty() {
        0.0
    } else {
        finite.iter().sum::<f64>() / finite.len() as f64
    };

    for i in 0..n_samples {
        let mut lo = 0.0f64;
        let mut hi = NPY_FLOATMAX;
        let mut mid = 1.0f64;

        let ith: Vec<f64> = distances
            .row(i)
            .iter()
            .copied()
            .filter(|d| d.is_finite())
            .map(f64::from)
            .collect();
        let non_zero: Vec<f64> = ith.iter().copied().filter(|&d| d > 0.0).collect();

        if non_zero.len() as f64 >= local_connectivity {
            let index = local_connectivity.floor() as usize;
            let interpolation = local_connectivity - index as f64;
            if index > 0 {
                rho[i] = non_zero[index - 1] as f32;
                if interpolation > SMOOTH_K_TOLERANCE {
                    rho[i] += (interpolation * (non_zero[index] - non_zero[index - 1])) as f32;
                }
            } else {
                rho[i] = (interpolation * non_zero[0]) as f32;
            }
        } else if !non_zero.is_empty() {
            rho[i] = non_zero.iter().copied().fold(f64::NEG_INFINITY, f64::max) as f32;
        }

        for _ in 0..n_iter {
            let mut psum = 0.0f64;
            for j in 1..distances.ncols() {
                let d = f64::from(distances[[i, j]]) - f64::from(rho[i]);
                if d > 0.0 {
                    psum += (-d / mid).exp();
                } else {
                    psum += 1.0;
                }
            }

            if (psum - target).abs() < SMOOTH_K_TOLERANCE {
                break;
            }

            if psum > target {
                hi = mid;
                mid = f64::midpoint(lo, hi);
            } else {
                lo = mid;
                if hi >= NPY_FLOATMAX {
                    mid *= 2.0;
                } else {
                    mid = f64::midpoint(lo, hi);
                }
            }
        }

        result[i] = mid as f32;

        if rho[i] > 0.0 {
            let mean_ith = ith.iter().sum::<f64>() / ith.len().max(1) as f64;
            if f64::from(result[i]) < MIN_K_DIST_SCALE * mean_ith {
                result[i] = (MIN_K_DIST_SCALE * mean_ith) as f32;
            }
        } else if f64::from(result[i]) < MIN_K_DIST_SCALE * mean_distances {
            result[i] = (MIN_K_DIST_SCALE * mean_distances) as f32;
        }
    }

    (result, rho)
}

/// Python: `umap_.compute_membership_strengths` — build the COO triplets for
/// the local fuzzy simplicial sets.
///
/// Returns (rows, cols, vals, dists); `dists` is `None` unless
/// `return_dists` is true.
#[must_use]
pub fn compute_membership_strengths(
    knn_indices: &ndarray::Array2<i32>,
    knn_dists: &ndarray::Array2<f32>,
    sigmas: &[f32],
    rhos: &[f32],
    return_dists: bool,
    bipartite: bool,
) -> (Vec<i32>, Vec<i32>, Vec<f32>, Option<Vec<f32>>) {
    let n_samples = knn_indices.nrows();
    let n_neighbors = knn_indices.ncols();

    let mut rows = vec![0i32; n_samples * n_neighbors];
    let mut cols = vec![0i32; n_samples * n_neighbors];
    let mut vals = vec![0.0f32; n_samples * n_neighbors];
    let mut dists = if return_dists {
        Some(vec![0.0f32; n_samples * n_neighbors])
    } else {
        None
    };

    for i in 0..n_samples {
        for j in 0..n_neighbors {
            if knn_indices[[i, j]] == -1 {
                continue; // We didn't get the full knn for i
            }
            // If applied to an adjacency matrix points shouldn't be similar
            // to themselves; for incidence matrices (bipartite) they differ.
            let val = if !bipartite && knn_indices[[i, j]] == i as i32 {
                0.0
            } else if f64::from(knn_dists[[i, j]]) - f64::from(rhos[i]) <= 0.0 || sigmas[i] == 0.0 {
                1.0
            } else {
                (-(f64::from(knn_dists[[i, j]]) - f64::from(rhos[i])) / f64::from(sigmas[i])).exp()
                    as f32
            };

            let idx = i * n_neighbors + j;
            rows[idx] = i as i32;
            cols[idx] = knn_indices[[i, j]];
            vals[idx] = val;
            if let Some(d) = &mut dists {
                d[idx] = knn_dists[[i, j]];
            }
        }
    }

    (rows, cols, vals, dists)
}

/// Python: `umap_.fuzzy_simplicial_set` (with precomputed KNN — the
/// pynndescent branch is replaced by the precomputed-KNN contract).
///
/// `knn_indices`/`knn_dists` must be provided (shape `n_samples` × `n_neighbors`;
/// -1 entries mark pruned/disconnected neighbors).
#[allow(clippy::too_many_arguments)]
pub fn fuzzy_simplicial_set(
    n_samples: usize,
    n_neighbors: usize,
    knn_indices: &ndarray::Array2<i32>,
    knn_dists: &ndarray::Array2<f32>,
    set_op_mix_ratio: f64,
    local_connectivity: f64,
    apply_set_operations: bool,
    return_dists: bool,
) -> Result<(CsrMatrix, Vec<f32>, Vec<f32>, Option<CsrMatrix>)> {
    let (sigmas, rhos) =
        smooth_knn_dist(knn_dists, n_neighbors as f64, 64, local_connectivity, 1.0);

    let (rows, cols, vals, dists) =
        compute_membership_strengths(knn_indices, knn_dists, &sigmas, &rhos, return_dists, false);

    let mut result = CooMatrix::from_triplets(
        (n_samples, n_samples),
        rows.clone(),
        cols.clone(),
        vals.clone(),
    );
    result.eliminate_zeros();

    if apply_set_operations {
        let transpose = result.transpose();
        let prod_matrix = result.tocsr().multiply_elementwise(&transpose.tocsr());

        if (set_op_mix_ratio - 1.0).abs() < f64::EPSILON {
            // Default fuzzy union: result + transpose - result*transpose.
            let t_csr = transpose.tocsr();
            result = result
                .tocsr()
                .add(&t_csr)
                .add(&prod_matrix.scale(-1.0))
                .tocoo();
        } else if set_op_mix_ratio == 0.0 {
            // Pure fuzzy intersection.
            result = prod_matrix.tocoo();
        } else {
            let t_csr = transpose.tocsr();
            let union = result.tocsr().add(&t_csr).add(&prod_matrix.scale(-1.0));
            result = union
                .scale(set_op_mix_ratio as f32)
                .add(&prod_matrix.scale((1.0 - set_op_mix_ratio) as f32))
                .tocoo();
        }
    }

    result.eliminate_zeros();

    if return_dists {
        let dmat = match dists {
            Some(d) => {
                let mut coo = CooMatrix::from_triplets((n_samples, n_samples), rows, cols, d);
                coo.eliminate_zeros();
                coo.tocsr().maximum_with_transpose()
            }
            None => {
                return Err(UmapError::InvalidInput(
                    "return_dists requested but no distances computed".to_string(),
                ))
            }
        };
        Ok((result.tocsr(), sigmas, rhos, Some(dmat)))
    } else {
        Ok((result.tocsr(), sigmas, rhos, None))
    }
}

/// Python: `umap_.fast_intersection` — categorical-distance intersection of
/// a simplicial set with discrete label data (mutates `values`).
pub fn fast_intersection(
    rows: &[i32],
    cols: &[i32],
    values: &mut [f32],
    target: &[f32],
    unknown_dist: f64,
    far_dist: f64,
) {
    // Python uses exact float equality on the label values; the strict
    // comparison lint is suppressed to preserve that semantics.
    #[allow(clippy::float_cmp)]
    for nz in 0..rows.len() {
        let i = rows[nz] as usize;
        let j = cols[nz] as usize;
        if target[i] == -1.0 || target[j] == -1.0 {
            values[nz] *= (-(unknown_dist)).exp() as f32;
        } else if target[i] != target[j] {
            values[nz] *= (-(far_dist)).exp() as f32;
        }
    }
}

/// Python: `umap_.fast_metric_intersection` — metric-distance intersection
/// over discrete space (mutates `values`).
pub fn fast_metric_intersection(
    rows: &[i32],
    cols: &[i32],
    values: &mut [f32],
    discrete_space: &ndarray::Array2<f32>,
    metric: MetricFn,
    metric_args: &[f32],
    scale: f64,
) {
    for nz in 0..rows.len() {
        let i = rows[nz] as usize;
        let j = cols[nz] as usize;
        let x: Vec<f32> = discrete_space.row(i).iter().copied().collect();
        let y: Vec<f32> = discrete_space.row(j).iter().copied().collect();
        let dist = metric(&x, &y, metric_args);
        values[nz] *= (-(scale * f64::from(dist))).exp() as f32;
    }
}

/// Python: `umap_.reprocess_row` — binary search for the exponent that makes
/// the row probabilities sum to log2(k).
fn reprocess_row(probabilities: &[f32], k: f64, n_iters: usize) -> Vec<f32> {
    let target = k.log2();
    let mut lo = 0.0f64;
    let mut hi = f64::INFINITY;
    let mut mid = 1.0f64;

    for _ in 0..n_iters {
        let mut psum = 0.0f64;
        for p in probabilities {
            psum += f64::from(*p).powf(mid);
        }
        if (psum - target).abs() < SMOOTH_K_TOLERANCE {
            break;
        }
        if psum < target {
            hi = mid;
            mid = f64::midpoint(lo, hi);
        } else {
            lo = mid;
            if hi == f64::INFINITY {
                mid *= 2.0;
            } else {
                mid = f64::midpoint(lo, hi);
            }
        }
    }

    probabilities
        .iter()
        .map(|p| f64::from(*p).powf(mid) as f32)
        .collect()
}

/// Python: `umap_.reset_local_metrics` (operates on CSR data in place).
fn reset_local_metrics(indptr: &[usize], data: &mut [f32]) {
    for i in 0..indptr.len().saturating_sub(1) {
        let (s, e) = (indptr[i], indptr[i + 1]);
        if e > s {
            let row = reprocess_row(&data[s..e], 15.0, 32);
            data[s..e].copy_from_slice(&row);
        }
    }
}

/// Python: `umap_.reset_local_connectivity`.
#[must_use]
pub fn reset_local_connectivity(simplicial_set: &CsrMatrix, reset_local_metric: bool) -> CsrMatrix {
    // normalize(simplicial_set, norm="max") — per-row max normalization.
    let mut normalized = simplicial_set.clone();
    for i in 0..normalized.shape.0 {
        let (s, e) = (normalized.indptr[i], normalized.indptr[i + 1]);
        if e > s {
            let max = normalized.data[s..e]
                .iter()
                .copied()
                .fold(f32::NEG_INFINITY, f32::max);
            if max > 0.0 {
                for v in &mut normalized.data[s..e] {
                    *v /= max;
                }
            }
        }
    }
    if reset_local_metric {
        reset_local_metrics(&normalized.indptr, &mut normalized.data);
    }
    let transpose = normalized.transpose();
    let prod_matrix = normalized.multiply_elementwise(&transpose);
    // Python: simplicial_set + transpose - prod_matrix.
    let mut result = normalized.add(&transpose).add(&prod_matrix.scale(-1.0));
    result.eliminate_zeros();
    result
}

/// Python: `umap_.discrete_metric_simplicial_set_intersection`.
///
/// `discrete_space` is (`n_samples`, 1) for categorical labels (the label
/// vector) or (`n_samples`, `n_features`) for metric-based discrete data.
pub fn discrete_metric_simplicial_set_intersection(
    simplicial_set: &CsrMatrix,
    discrete_space: &ndarray::Array2<f32>,
    unknown_dist: f64,
    far_dist: f64,
    metric: Option<MetricFn>,
    metric_args: &[f32],
    metric_scale: f64,
) -> Result<CsrMatrix> {
    let mut coo = simplicial_set.tocoo();
    let mut data = coo.data.clone();

    if let Some(m) = metric {
        fast_metric_intersection(
            &coo.row,
            &coo.col,
            &mut data,
            discrete_space,
            m,
            metric_args,
            metric_scale,
        );
    } else {
        // Categorical labels: column 0 of discrete_space.
        let target: Vec<f32> = (0..discrete_space.nrows())
            .map(|i| discrete_space[[i, 0]])
            .collect();
        fast_intersection(
            &coo.row,
            &coo.col,
            &mut data,
            &target,
            unknown_dist,
            far_dist,
        );
    }
    coo.data = data;
    coo.eliminate_zeros();

    Ok(reset_local_connectivity(&coo.tocsr(), false))
}

/// Python: `umap_.general_simplicial_set_intersection`.
#[must_use]
#[allow(clippy::similar_names)]
pub fn general_simplicial_set_intersection(
    simplicial_set1: &CsrMatrix,
    simplicial_set2: &CsrMatrix,
    weight: f64,
    right_complement: bool,
) -> CsrMatrix {
    let result_coo = if right_complement {
        simplicial_set1.tocoo()
    } else {
        simplicial_set1.add(simplicial_set2).tocoo()
    };

    let row = result_coo.row.clone();
    let col = result_coo.col.clone();
    let mut data = result_coo.data.clone();

    crate::sparse_metrics::general_sset_intersection(
        &simplicial_set1.indptr,
        &simplicial_set1.indices,
        &simplicial_set1.data,
        &simplicial_set2.indptr,
        &simplicial_set2.indices,
        &simplicial_set2.data,
        &row,
        &col,
        &mut data,
        weight as f32,
        right_complement,
    );

    CooMatrix::from_triplets(simplicial_set1.shape, row, col, data).tocsr()
}

/// Python: `umap_.general_simplicial_set_union`.
#[must_use]
pub fn general_simplicial_set_union(
    simplicial_set1: &CsrMatrix,
    simplicial_set2: &CsrMatrix,
) -> CsrMatrix {
    let result_coo = simplicial_set1.add(simplicial_set2).tocoo();
    let mut data = result_coo.data.clone();

    crate::sparse_metrics::general_sset_union(
        &simplicial_set1.indptr,
        &simplicial_set1.indices,
        &simplicial_set1.data,
        &simplicial_set2.indptr,
        &simplicial_set2.indices,
        &simplicial_set2.data,
        &result_coo.row,
        &result_coo.col,
        &mut data,
    );

    CooMatrix::from_triplets(simplicial_set1.shape, result_coo.row, result_coo.col, data).tocsr()
}

/// Python: `umap_.make_epochs_per_sample`.
#[must_use]
pub fn make_epochs_per_sample(weights: &[f32], n_epochs: usize) -> Vec<f64> {
    let mut result = vec![-1.0f64; weights.len()];
    let max = weights.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if max <= 0.0 {
        return result;
    }
    for (i, &w) in weights.iter().enumerate() {
        let n_samples = n_epochs as f64 * (f64::from(w) / f64::from(max));
        if n_samples > 0.0 {
            result[i] = n_epochs as f64 / n_samples;
        }
    }
    result
}
