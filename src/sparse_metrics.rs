//! Ported from Python: umap/sparse.py
//!
//! Sparse distance metrics and sparse set operations — "enough simple sparse
//! operations to enable sparse UMAP" — ported function-by-function from the
//! numba kernels in `umap/sparse.py` (umap-learn 0.5.12).
//!
//! The kernels operate on CSR rows passed as parallel `(indices, data)` slices
//! (the slice view of [`crate::csr::CsrMatrix`]) and on COO triplets (the
//! slice view of [`crate::csr::CooMatrix`]) for the sset kernels.
//!
//! All arithmetic is `f32`, matching numba's behaviour for `float32` inputs.
//! Where the Python kernels mutate numba arrays in place, this port allocates
//! fresh buffers sized so that duplicate/unsorted index input cannot panic
//! (Python raises `IndexError` there); results for well-formed input are
//! identical.

// Exact float comparisons (`!= 0.0`, `== 0.0`, sentinel minimums) are part of
// the ported numba control flow, not precision accidents; likewise the
// if/elif/else index-walking chains mirror the numba kernels exactly.
#![allow(clippy::float_cmp)]
#![allow(clippy::comparison_chain)]

use std::collections::HashSet;

/// A sparse metric computes the distance between two sparse rows given as
/// (indices, data) pairs.
///
/// Python signature: `(idx1, data1, idx2, data2, *kwds)`. The trailing slice
/// carries the extra positional arguments: `n_features` for the discrete
/// metrics (`hamming`, `matching`, `kulsinski`, `rogerstanimoto`,
/// `russellrao`, `sokalmichener`, `correlation`) — the Python
/// `sparse_need_n_features` tuple — and `p` for `minkowski`. Metrics that
/// take no extra arguments ignore it.
pub type SparseMetricFn = fn(&[i32], &[f32], &[i32], &[f32], &[f32]) -> f32;

/// Python default-argument handling for the trailing `*kwds` slice: returns
/// `kwds[0]` when present, otherwise `default`. (Python raises `TypeError`
/// when a required argument such as `n_features` is missing; we fall back to
/// `default` so the kernel stays panic-free.)
fn kwds_arg(kwds: &[f32], default: f32) -> f32 {
    kwds.first().copied().unwrap_or(default)
}

/// Python: `umap.utils.norm` (imported into `umap.sparse`), applied to the
/// data values of a sparse vector. `indices` are unused but kept so the
/// signature matches the sparse (indices, data) calling convention.
pub fn norm(_indices: &[i32], data: &[f32]) -> f32 {
    let mut result = 0.0f32;
    for &v in data {
        result += v * v;
    }
    result.sqrt()
}

// Just reproduce a simpler version of numpy unique (not numba supported yet)
// Python: umap.sparse.arr_unique
fn arr_unique(arr: &[i32]) -> Vec<i32> {
    let mut aux = arr.to_vec();
    aux.sort_unstable();
    // flag = concatenate((ones(1), aux[1:] != aux[:-1])); return aux[flag]
    let mut result = Vec::with_capacity(aux.len());
    for (i, &v) in aux.iter().enumerate() {
        if i == 0 || v != aux[i - 1] {
            result.push(v);
        }
    }
    result
}

// Just reproduce a simpler version of numpy union1d (not numba supported yet)
// Python: umap.sparse.arr_union
pub fn arr_union(i1: &[i32], i2: &[i32]) -> Vec<i32> {
    if i1.is_empty() {
        i2.to_vec()
    } else if i2.is_empty() {
        i1.to_vec()
    } else {
        let mut concatenated = Vec::with_capacity(i1.len() + i2.len());
        concatenated.extend_from_slice(i1);
        concatenated.extend_from_slice(i2);
        arr_unique(&concatenated)
    }
}

// Just reproduce a simpler version of numpy intersect1d (not numba supported
// yet)
// Python: umap.sparse.arr_intersect — note the simplified multiset semantics:
// one output entry per adjacent equal pair in the sorted concatenation.
pub fn arr_intersect(i1: &[i32], i2: &[i32]) -> Vec<i32> {
    let mut aux = Vec::with_capacity(i1.len() + i2.len());
    aux.extend_from_slice(i1);
    aux.extend_from_slice(i2);
    aux.sort_unstable();
    // aux[:-1][aux[1:] == aux[:-1]]
    let mut result = Vec::new();
    for w in aux.windows(2) {
        if w[0] == w[1] {
            result.push(w[0]);
        }
    }
    result
}

// Python: umap.sparse.sparse_sum
fn sparse_sum(ind1: &[i32], data1: &[f32], ind2: &[i32], data2: &[f32]) -> (Vec<i32>, Vec<f32>) {
    // Python writes into the `arr_union` result buffer in place; we allocate a
    // fresh buffer sized for the worst case so duplicate/unsorted indices
    // cannot panic (Python raises IndexError there).
    // indices/data agree in length for well-formed CSR rows.
    let len1 = ind1.len().min(data1.len());
    let len2 = ind2.len().min(data2.len());
    let mut result_ind = vec![0i32; len1 + len2];
    let mut result_data = vec![0.0f32; len1 + len2];

    let mut i1 = 0usize;
    let mut i2 = 0usize;
    let mut nnz = 0usize;

    // pass through both index lists
    while i1 < len1 && i2 < len2 {
        let j1 = ind1[i1];
        let j2 = ind2[i2];

        if j1 == j2 {
            let val = data1[i1] + data2[i2];
            if val != 0.0 {
                result_ind[nnz] = j1;
                result_data[nnz] = val;
                nnz += 1;
            }
            i1 += 1;
            i2 += 1;
        } else if j1 < j2 {
            let val = data1[i1];
            if val != 0.0 {
                result_ind[nnz] = j1;
                result_data[nnz] = val;
                nnz += 1;
            }
            i1 += 1;
        } else {
            let val = data2[i2];
            if val != 0.0 {
                result_ind[nnz] = j2;
                result_data[nnz] = val;
                nnz += 1;
            }
            i2 += 1;
        }
    }

    // pass over the tails
    while i1 < len1 {
        let val = data1[i1];
        if val != 0.0 {
            result_ind[nnz] = ind1[i1];
            result_data[nnz] = val;
            nnz += 1;
        }
        i1 += 1;
    }

    while i2 < len2 {
        let val = data2[i2];
        if val != 0.0 {
            result_ind[nnz] = ind2[i2];
            result_data[nnz] = val;
            nnz += 1;
        }
        i2 += 1;
    }

    // truncate to the correct length in case there were zeros created
    result_ind.truncate(nnz);
    result_data.truncate(nnz);

    (result_ind, result_data)
}

// Python: umap.sparse.sparse_diff
fn sparse_diff(ind1: &[i32], data1: &[f32], ind2: &[i32], data2: &[f32]) -> (Vec<i32>, Vec<f32>) {
    let neg_data2: Vec<f32> = data2.iter().map(|&v| -v).collect();
    sparse_sum(ind1, data1, ind2, &neg_data2)
}

// Python: umap.sparse.sparse_mul
fn sparse_mul(ind1: &[i32], data1: &[f32], ind2: &[i32], data2: &[f32]) -> (Vec<i32>, Vec<f32>) {
    // Python writes into the `arr_intersect` result buffer in place; we
    // allocate a fresh buffer of size min(len1, len2) — the maximum possible
    // nnz, since entries are written only when both indices match — so that
    // duplicate/unsorted inputs cannot panic (Python raises IndexError there).
    let len1 = ind1.len().min(data1.len());
    let len2 = ind2.len().min(data2.len());
    let mut result_ind = vec![0i32; len1.min(len2)];
    let mut result_data = vec![0.0f32; len1.min(len2)];

    let mut i1 = 0usize;
    let mut i2 = 0usize;
    let mut nnz = 0usize;

    // pass through both index lists
    while i1 < len1 && i2 < len2 {
        let j1 = ind1[i1];
        let j2 = ind2[i2];

        if j1 == j2 {
            let val = data1[i1] * data2[i2];
            if val != 0.0 {
                result_ind[nnz] = j1;
                result_data[nnz] = val;
                nnz += 1;
            }
            i1 += 1;
            i2 += 1;
        } else if j1 < j2 {
            i1 += 1;
        } else {
            i2 += 1;
        }
    }

    // truncate to the correct length in case there were zeros created
    result_ind.truncate(nnz);
    result_data.truncate(nnz);

    (result_ind, result_data)
}

/// Safely compute the CSR row slice `[indptr[i], indptr[i+1])` clamped to the
/// available values. Python would raise `IndexError` for out-of-range rows or
/// malformed indptr; we return an empty range instead so the kernel cannot
/// panic.
fn row_range(indptr: &[usize], i: usize, len: usize) -> std::ops::Range<usize> {
    let Some(&start) = indptr.get(i) else {
        return 0..0;
    };
    let Some(&end) = indptr.get(i + 1) else {
        return 0..0;
    };
    let start = start.min(len);
    let end = end.min(len).max(start);
    start..end
}

// Python: umap.sparse.general_sset_intersection — reads the left/right CSR
// arrays and the result COO arrays (result_row/result_col hold the union
// sparsity, result_data is pre-filled by the caller and only overwritten
// where the mixing condition holds). Includes the numba kernel's
// `right_complement` and `mix_weight` parameters.
#[allow(clippy::too_many_lines)]
pub fn general_sset_intersection(
    left_indptr: &[usize],
    left_indices: &[i32],
    left_data: &[f32],
    right_indptr: &[usize],
    right_indices: &[i32],
    right_data: &[f32],
    result_row: &[i32],
    result_col: &[i32],
    result_data: &mut [f32],
    mix_weight: f32,
    right_complement: bool,
) {
    // Python: left_min = max(data1.min() / 2.0, 1.0e-8). numpy raises on an
    // empty data array; the fold below yields +inf for empty input, which
    // simply disables the left side of the mixing condition.
    let left_data_min = left_data.iter().copied().fold(f32::INFINITY, f32::min);
    let left_min = (left_data_min / 2.0).max(1.0e-8);

    let right_min = if right_complement {
        // All right vals may be large!
        let complement_min = right_data
            .iter()
            .copied()
            .map(|v| 1.0 - v)
            .fold(f32::INFINITY, f32::min);
        (complement_min / 2.0).clamp(1.0e-8, 1.0e-4)
    } else {
        // All right vals may be large!
        let right_data_min = right_data.iter().copied().fold(f32::INFINITY, f32::min);
        (right_data_min / 2.0).clamp(1.0e-8, 1.0e-4)
    };

    for (idx, (&i_raw, &j)) in result_row.iter().zip(result_col).enumerate() {
        if idx >= result_data.len() {
            break;
        }
        let i = i_raw as usize;

        let mut left_val = left_min;
        for k in row_range(left_indptr, i, left_indices.len()) {
            if left_indices[k] == j {
                left_val = left_data[k];
            }
        }

        let mut right_val = right_min;
        for k in row_range(right_indptr, i, right_indices.len()) {
            if right_indices[k] == j {
                if right_complement {
                    right_val = 1.0 - right_data[k];
                } else {
                    right_val = right_data[k];
                }
            }
        }

        if left_val > left_min || right_val > right_min {
            if mix_weight < 0.5 {
                result_data[idx] = left_val * right_val.powf(mix_weight / (1.0 - mix_weight));
            } else {
                result_data[idx] = left_val.powf((1.0 - mix_weight) / mix_weight) * right_val;
            }
        }
    }
}

// Python: umap.sparse.general_sset_union — same shape as intersection.
pub fn general_sset_union(
    left_indptr: &[usize],
    left_indices: &[i32],
    left_data: &[f32],
    right_indptr: &[usize],
    right_indices: &[i32],
    right_data: &[f32],
    result_row: &[i32],
    result_col: &[i32],
    result_data: &mut [f32],
) {
    let left_data_min = left_data.iter().copied().fold(f32::INFINITY, f32::min);
    let left_min = (left_data_min / 2.0).max(1.0e-8);
    let right_data_min = right_data.iter().copied().fold(f32::INFINITY, f32::min);
    let right_min = (right_data_min / 2.0).max(1.0e-8);

    for (idx, (&i_raw, &j)) in result_row.iter().zip(result_col).enumerate() {
        if idx >= result_data.len() {
            break;
        }
        let i = i_raw as usize;

        let mut left_val = left_min;
        for k in row_range(left_indptr, i, left_indices.len()) {
            if left_indices[k] == j {
                left_val = left_data[k];
            }
        }

        let mut right_val = right_min;
        for k in row_range(right_indptr, i, right_indices.len()) {
            if right_indices[k] == j {
                right_val = right_data[k];
            }
        }

        result_data[idx] = left_val + right_val - left_val * right_val;
    }
}

// Python: umap.sparse.sparse_euclidean
pub fn sparse_euclidean(
    ind1: &[i32],
    data1: &[f32],
    ind2: &[i32],
    data2: &[f32],
    _kwds: &[f32],
) -> f32 {
    let (_, aux_data) = sparse_diff(ind1, data1, ind2, data2);
    let mut result = 0.0f32;
    for &v in &aux_data {
        result += v * v;
    }
    result.sqrt()
}

// Python: umap.sparse.sparse_manhattan
pub fn sparse_manhattan(
    ind1: &[i32],
    data1: &[f32],
    ind2: &[i32],
    data2: &[f32],
    _kwds: &[f32],
) -> f32 {
    let (_, aux_data) = sparse_diff(ind1, data1, ind2, data2);
    let mut result = 0.0f32;
    for &v in &aux_data {
        result += v.abs();
    }
    result
}

// Python: umap.sparse.sparse_chebyshev
pub fn sparse_chebyshev(
    ind1: &[i32],
    data1: &[f32],
    ind2: &[i32],
    data2: &[f32],
    _kwds: &[f32],
) -> f32 {
    let (_, aux_data) = sparse_diff(ind1, data1, ind2, data2);
    let mut result = 0.0f32;
    for &v in &aux_data {
        result = result.max(v.abs());
    }
    result
}

// Python: umap.sparse.sparse_minkowski (p defaults to 2.0)
pub fn sparse_minkowski(
    ind1: &[i32],
    data1: &[f32],
    ind2: &[i32],
    data2: &[f32],
    kwds: &[f32],
) -> f32 {
    let p = kwds_arg(kwds, 2.0);
    let (_, aux_data) = sparse_diff(ind1, data1, ind2, data2);
    let mut result = 0.0f32;
    for &v in &aux_data {
        result += v.abs().powf(p);
    }
    result.powf(1.0 / p)
}

// Python: umap.sparse.sparse_hamming (requires n_features)
pub fn sparse_hamming(
    ind1: &[i32],
    data1: &[f32],
    ind2: &[i32],
    data2: &[f32],
    kwds: &[f32],
) -> f32 {
    let n_features = kwds_arg(kwds, 0.0);
    let num_not_equal = sparse_diff(ind1, data1, ind2, data2).0.len();
    num_not_equal as f32 / n_features
}

// Python: umap.sparse.sparse_canberra
pub fn sparse_canberra(
    ind1: &[i32],
    data1: &[f32],
    ind2: &[i32],
    data2: &[f32],
    _kwds: &[f32],
) -> f32 {
    let abs_data1: Vec<f32> = data1.iter().map(|&v| v.abs()).collect();
    let abs_data2: Vec<f32> = data2.iter().map(|&v| v.abs()).collect();
    let (denom_inds, mut denom_data) = sparse_sum(ind1, &abs_data1, ind2, &abs_data2);
    for v in &mut denom_data {
        *v = 1.0 / *v;
    }
    let (numer_inds, mut numer_data) = sparse_diff(ind1, data1, ind2, data2);
    for v in &mut numer_data {
        *v = v.abs();
    }

    // Python: sparse_mul(numer_inds, numer_data, denom_inds, denom_data)
    let (_, val_data) = sparse_mul(&numer_inds, &numer_data, &denom_inds, &denom_data);

    val_data.iter().sum()
}

// Python: umap.sparse.sparse_bray_curtis
pub fn sparse_bray_curtis(
    ind1: &[i32],
    data1: &[f32],
    ind2: &[i32],
    data2: &[f32],
    _kwds: &[f32],
) -> f32 {
    let (_, denom_data) = sparse_sum(ind1, data1, ind2, data2);
    let denom_sum: f32 = denom_data.iter().map(|&v| v.abs()).sum();

    if denom_data.is_empty() {
        return 0.0;
    }

    if denom_sum == 0.0 {
        return 0.0;
    }

    let (_, numer_data) = sparse_diff(ind1, data1, ind2, data2);
    let numerator: f32 = numer_data.iter().map(|&v| v.abs()).sum();

    numerator / denom_sum
}

// Python: umap.sparse.sparse_jaccard
pub fn sparse_jaccard(
    ind1: &[i32],
    _data1: &[f32],
    ind2: &[i32],
    _data2: &[f32],
    _kwds: &[f32],
) -> f32 {
    let num_non_zero = arr_union(ind1, ind2).len();
    let num_equal = arr_intersect(ind1, ind2).len();

    if num_non_zero == 0 {
        0.0
    } else {
        (num_non_zero - num_equal) as f32 / num_non_zero as f32
    }
}

// Python: umap.sparse.sparse_matching (requires n_features)
pub fn sparse_matching(
    ind1: &[i32],
    _data1: &[f32],
    ind2: &[i32],
    _data2: &[f32],
    kwds: &[f32],
) -> f32 {
    let n_features = kwds_arg(kwds, 0.0);
    let num_true_true = arr_intersect(ind1, ind2).len();
    let num_non_zero = arr_union(ind1, ind2).len();
    let num_not_equal = num_non_zero - num_true_true;

    num_not_equal as f32 / n_features
}

// Python: umap.sparse.sparse_dice
pub fn sparse_dice(
    ind1: &[i32],
    _data1: &[f32],
    ind2: &[i32],
    _data2: &[f32],
    _kwds: &[f32],
) -> f32 {
    let num_true_true = arr_intersect(ind1, ind2).len();
    let num_non_zero = arr_union(ind1, ind2).len();
    let num_not_equal = num_non_zero - num_true_true;

    if num_not_equal == 0 {
        0.0
    } else {
        num_not_equal as f32 / (2.0 * num_true_true as f32 + num_not_equal as f32)
    }
}

// Python: umap.sparse.sparse_kulsinski (requires n_features)
pub fn sparse_kulsinski(
    ind1: &[i32],
    _data1: &[f32],
    ind2: &[i32],
    _data2: &[f32],
    kwds: &[f32],
) -> f32 {
    let n_features = kwds_arg(kwds, 0.0);
    let num_true_true = arr_intersect(ind1, ind2).len();
    let num_non_zero = arr_union(ind1, ind2).len();
    let num_not_equal = num_non_zero - num_true_true;

    if num_not_equal == 0 {
        0.0
    } else {
        (num_not_equal as f32 - num_true_true as f32 + n_features)
            / (num_not_equal as f32 + n_features)
    }
}

// Python: umap.sparse.sparse_rogers_tanimoto (requires n_features)
pub fn sparse_rogers_tanimoto(
    ind1: &[i32],
    _data1: &[f32],
    ind2: &[i32],
    _data2: &[f32],
    kwds: &[f32],
) -> f32 {
    let n_features = kwds_arg(kwds, 0.0);
    let num_true_true = arr_intersect(ind1, ind2).len();
    let num_non_zero = arr_union(ind1, ind2).len();
    let num_not_equal = num_non_zero - num_true_true;

    (2.0 * num_not_equal as f32) / (n_features + num_not_equal as f32)
}

// Python: umap.sparse.sparse_russellrao (requires n_features)
pub fn sparse_russellrao(
    ind1: &[i32],
    data1: &[f32],
    ind2: &[i32],
    data2: &[f32],
    kwds: &[f32],
) -> f32 {
    let n_features = kwds_arg(kwds, 0.0);
    if ind1.len() == ind2.len() && ind1 == ind2 {
        return 0.0;
    }

    let num_true_true = arr_intersect(ind1, ind2).len();
    let nz1 = data1.iter().filter(|&&v| v != 0.0).count();
    let nz2 = data2.iter().filter(|&&v| v != 0.0).count();

    if num_true_true == nz1 && num_true_true == nz2 {
        0.0
    } else {
        (n_features - num_true_true as f32) / n_features
    }
}

// Python: umap.sparse.sparse_sokal_michener (requires n_features)
pub fn sparse_sokal_michener(
    ind1: &[i32],
    _data1: &[f32],
    ind2: &[i32],
    _data2: &[f32],
    kwds: &[f32],
) -> f32 {
    let n_features = kwds_arg(kwds, 0.0);
    let num_true_true = arr_intersect(ind1, ind2).len();
    let num_non_zero = arr_union(ind1, ind2).len();
    let num_not_equal = num_non_zero - num_true_true;

    (2.0 * num_not_equal as f32) / (n_features + num_not_equal as f32)
}

// Python: umap.sparse.sparse_sokal_sneath
pub fn sparse_sokal_sneath(
    ind1: &[i32],
    _data1: &[f32],
    ind2: &[i32],
    _data2: &[f32],
    _kwds: &[f32],
) -> f32 {
    let num_true_true = arr_intersect(ind1, ind2).len();
    let num_non_zero = arr_union(ind1, ind2).len();
    let num_not_equal = num_non_zero - num_true_true;

    if num_not_equal == 0 {
        0.0
    } else {
        num_not_equal as f32 / (0.5 * num_true_true as f32 + num_not_equal as f32)
    }
}

// Python: umap.sparse.sparse_cosine
pub fn sparse_cosine(
    ind1: &[i32],
    data1: &[f32],
    ind2: &[i32],
    data2: &[f32],
    _kwds: &[f32],
) -> f32 {
    let (_, aux_data) = sparse_mul(ind1, data1, ind2, data2);
    let mut result = 0.0f32;
    let norm1 = norm(ind1, data1);
    let norm2 = norm(ind2, data2);

    for &v in &aux_data {
        result += v;
    }

    if norm1 == 0.0 && norm2 == 0.0 {
        0.0
    } else if norm1 == 0.0 || norm2 == 0.0 {
        1.0
    } else {
        1.0 - result / (norm1 * norm2)
    }
}

// Python: umap.sparse.sparse_hellinger
pub fn sparse_hellinger(
    ind1: &[i32],
    data1: &[f32],
    ind2: &[i32],
    data2: &[f32],
    _kwds: &[f32],
) -> f32 {
    let (_, aux_data) = sparse_mul(ind1, data1, ind2, data2);
    let mut result = 0.0f32;
    let norm1: f32 = data1.iter().sum();
    let norm2: f32 = data2.iter().sum();
    let sqrt_norm_prod = (norm1 * norm2).sqrt();

    for &v in &aux_data {
        result += v.sqrt();
    }

    if norm1 == 0.0 && norm2 == 0.0 {
        0.0
    } else if norm1 == 0.0 || norm2 == 0.0 {
        1.0
    } else if result > sqrt_norm_prod {
        0.0
    } else {
        (1.0 - result / sqrt_norm_prod).sqrt()
    }
}

// Python: umap.sparse.sparse_correlation (requires n_features)
pub fn sparse_correlation(
    ind1: &[i32],
    data1: &[f32],
    ind2: &[i32],
    data2: &[f32],
    kwds: &[f32],
) -> f32 {
    let n_features = kwds_arg(kwds, 0.0);

    if ind1.is_empty() && ind2.is_empty() {
        return 0.0;
    } else if ind1.is_empty() || ind2.is_empty() {
        return 1.0;
    }

    let mut mu_x = 0.0f32;
    let mut mu_y = 0.0f32;

    for &v in data1 {
        mu_x += v;
    }
    for &v in data2 {
        mu_y += v;
    }

    mu_x /= n_features;
    mu_y /= n_features;

    let shifted_data1: Vec<f32> = data1.iter().map(|&v| v - mu_x).collect();
    let shifted_data2: Vec<f32> = data2.iter().map(|&v| v - mu_y).collect();

    let norm1 = (norm(ind1, &shifted_data1) * norm(ind1, &shifted_data1)
        + (n_features - ind1.len() as f32) * mu_x * mu_x)
        .sqrt();
    let norm2 = (norm(ind2, &shifted_data2) * norm(ind2, &shifted_data2)
        + (n_features - ind2.len() as f32) * mu_y * mu_y)
        .sqrt();

    let (dot_prod_inds, dot_prod_data) = sparse_mul(ind1, &shifted_data1, ind2, &shifted_data2);

    // Python: common_indices = set(dot_prod_inds)
    let common_indices: HashSet<i32> = dot_prod_inds.iter().copied().collect();

    let mut dot_product = 0.0f32;
    for &v in &dot_prod_data {
        dot_product += v;
    }

    for (&idx, &shifted) in ind1.iter().zip(&shifted_data1) {
        if !common_indices.contains(&idx) {
            dot_product -= shifted * mu_y;
        }
    }

    for (&idx, &shifted) in ind2.iter().zip(&shifted_data2) {
        if !common_indices.contains(&idx) {
            dot_product -= shifted * mu_x;
        }
    }

    let all_indices = arr_union(ind1, ind2);
    dot_product += mu_x * mu_y * (n_features - all_indices.len() as f32);

    if norm1 == 0.0 && norm2 == 0.0 {
        0.0
    } else if dot_product == 0.0 {
        1.0
    } else {
        1.0 - dot_product / (norm1 * norm2)
    }
}

// Python: umap.sparse.approx_log_Gamma — computed in f64, matching the
// float64 accumulators numba promotes the log computations to.
fn approx_log_gamma(x: f64) -> f64 {
    if x == 1.0 {
        return 0.0;
    }
    x * x.ln() - x + 0.5 * (2.0 * std::f64::consts::PI / x).ln() + 1.0 / (x * 12.0)
}

// Python: umap.sparse.log_beta
fn log_beta(x: f64, y: f64) -> f64 {
    let a = x.min(y);
    let b = x.max(y);
    if b < 5.0 {
        let mut value = -b.ln();
        // Python: for i in range(1, int(a)) — int() truncates toward zero
        for i in 1..(a as i64) {
            value += (i as f64).ln() - (b + i as f64).ln();
        }
        value
    } else {
        approx_log_gamma(x) + approx_log_gamma(y) - approx_log_gamma(x + y)
    }
}

// Python: umap.sparse.log_single_beta
fn log_single_beta(x: f64) -> f64 {
    (2.0f64.ln() * (-2.0 * x + 0.5)) + 0.5 * (2.0 * std::f64::consts::PI / x).ln() + 0.125 / x
}

// Python: umap.sparse.sparse_ll_dirichlet
pub fn sparse_ll_dirichlet(
    ind1: &[i32],
    data1: &[f32],
    ind2: &[i32],
    data2: &[f32],
    _kwds: &[f32],
) -> f32 {
    // The probability of rolling data2 in sum(data2) trials on a die that
    // rolled data1 in sum(data1) trials
    let n1: f32 = data1.iter().sum();
    let n2: f32 = data2.iter().sum();

    if n1 == 0.0 && n2 == 0.0 {
        return 0.0;
    } else if n1 == 0.0 || n2 == 0.0 {
        return 1e8;
    }

    let mut log_b = 0.0f64;
    let mut i1 = 0usize;
    let mut i2 = 0usize;
    // indices/data agree in length for well-formed CSR rows
    let len1 = ind1.len().min(data1.len());
    let len2 = ind2.len().min(data2.len());
    while i1 < len1 && i2 < len2 {
        let j1 = ind1[i1];
        let j2 = ind2[i2];

        if j1 == j2 {
            if data1[i1] * data2[i2] != 0.0 {
                log_b += log_beta(f64::from(data1[i1]), f64::from(data2[i2]));
            }
            i1 += 1;
            i2 += 1;
        } else if j1 < j2 {
            i1 += 1;
        } else {
            i2 += 1;
        }
    }

    let self_denom1: f64 = data1.iter().map(|&d| log_single_beta(f64::from(d))).sum();
    let self_denom2: f64 = data2.iter().map(|&d| log_single_beta(f64::from(d))).sum();

    let n1f = f64::from(n1);
    let n2f = f64::from(n2);
    ((log_b - log_beta(n1f, n2f) - (self_denom2 - log_single_beta(n2f))) / n2f
        + (log_b - log_beta(n2f, n1f) - (self_denom1 - log_single_beta(n1f))) / n1f)
        .sqrt() as f32
}

// Python: umap.sparse.sparse_named_distances — resolve by name. The Python
// dict has exactly these keys (no l2/sqeuclidean/yule entries in the sparse
// dict); aliases included. Returns None for unknown names.
pub fn sparse_named_distances(name: &str) -> Option<SparseMetricFn> {
    Some(match name {
        // general minkowski distances
        "euclidean" => sparse_euclidean,
        "manhattan" | "l1" | "taxicab" => sparse_manhattan,
        "chebyshev" | "linf" | "linfty" | "linfinity" => sparse_chebyshev,
        "minkowski" => sparse_minkowski,
        // Other distances
        "canberra" => sparse_canberra,
        "ll_dirichlet" => sparse_ll_dirichlet,
        "braycurtis" => sparse_bray_curtis,
        // Binary distances
        "hamming" => sparse_hamming,
        "jaccard" => sparse_jaccard,
        "dice" => sparse_dice,
        "matching" => sparse_matching,
        "kulsinski" => sparse_kulsinski,
        "rogerstanimoto" => sparse_rogers_tanimoto,
        "russellrao" => sparse_russellrao,
        "sokalmichener" => sparse_sokal_michener,
        "sokalsneath" => sparse_sokal_sneath,
        "cosine" => sparse_cosine,
        "correlation" => sparse_correlation,
        "hellinger" => sparse_hellinger,
        _ => return None,
    })
}

// Python: umap.sparse.SPARSE_SPECIAL_METRICS — names of the sparse special
// metrics (in this version: hellinger and ll_dirichlet only; there are no
// categorical special metrics in umap-learn 0.5.12).
pub fn sparse_special_metric_names() -> &'static [&'static str] {
    &["hellinger", "ll_dirichlet"]
}
