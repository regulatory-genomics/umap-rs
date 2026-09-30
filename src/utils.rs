//! Ported from Python: umap/utils.py (`fast_knn_indices`, submatrix, norm,
//! `csr_unique`, `average_nn_distance`, ts).
//!
//! Divergence: the repo's `csr_unique` fix (element-by-element object array
//! fill) is ported; `ts` timestamps are omitted (logging only).

use crate::csr::{CooMatrix, CsrMatrix};

/// Python: `umap.utils.fast_knn_indices` — knn indices by row. Ties are
/// resolved by stable sort (Python: `argsort(kind="mergesort")`), i.e. by
/// ascending column index among equal distances.
#[must_use]
pub fn fast_knn_indices(x: &ndarray::Array2<f32>, n_neighbors: usize) -> ndarray::Array2<i32> {
    let mut knn_indices = ndarray::Array2::<i32>::from_elem((x.nrows(), n_neighbors), -1);
    for (row_idx, row) in x.rows().into_iter().enumerate() {
        let mut v: Vec<usize> = (0..row.len()).collect();
        v.sort_by(|&a, &b| {
            row[a]
                .partial_cmp(&row[b])
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.cmp(&b))
        });
        for (j, &idx) in v.iter().take(n_neighbors).enumerate() {
            knn_indices[[row_idx, j]] = idx as i32;
        }
    }
    knn_indices
}

/// Python: `umap.utils.submatrix` — index a submatrix by row indices.
#[must_use]
pub fn submatrix(d: &ndarray::Array2<f32>, row_indices: &[i32]) -> ndarray::Array2<f32> {
    let n = row_indices.len();
    let m = d.ncols();
    let mut out = ndarray::Array2::<f32>::zeros((n, m));
    for (i, &r) in row_indices.iter().enumerate() {
        for j in 0..m {
            out[[i, j]] = d[[r as usize, j]];
        }
    }
    out
}

/// Python: `umap.utils.norm` — Euclidean norm of a dense vector.
#[inline]
#[must_use]
pub fn norm(v: &[f32]) -> f32 {
    v.iter()
        .map(|x| f64::from(*x) * f64::from(*x))
        .sum::<f64>()
        .sqrt() as f32
}

/// Python: `umap.utils.csr_unique` — deduplicate rows of a sparse matrix.
///
/// Divergence: the repo fix is followed — rows are compared as whole tuples
/// (not element-wise) so every row with the same (indices, data) pattern is
/// deduplicated. This is only meaningful for one-hot/discrete input matrices
/// (used for supervised layouts over duplicate samples).
#[must_use]
pub fn csr_unique(matrix: &CsrMatrix) -> CsrMatrix {
    let n = matrix.shape.0;
    let mut keep: Vec<usize> = Vec::with_capacity(n);
    // Rows compared by (indices, bitwise data) — f32 has no Eq; bitwise
    // equality is exact for the one-hot/discrete inputs this is used on.
    let mut seen: std::collections::HashSet<(Vec<i32>, Vec<u32>)> =
        std::collections::HashSet::new();
    for i in 0..n {
        let (idx, data) = matrix.row_slice(i);
        let key = (
            idx.to_vec(),
            data.iter().map(|&v| v.to_bits()).collect::<Vec<u32>>(),
        );
        if seen.insert(key) {
            keep.push(i);
        }
    }

    // Keep rows of the matrix whose index is in `keep`.
    let mut coo = CooMatrix::new((keep.len(), matrix.shape.1));
    let mut keep_iter = keep.iter().copied().peekable();
    let mut new_row = 0usize;
    for i in 0..n {
        if keep_iter.peek() == Some(&{ i }) {
            keep_iter.next();
            let (idx, data) = matrix.row_slice(i);
            for (&c, &v) in idx.iter().zip(data.iter()) {
                coo.row.push(new_row as i32);
                coo.col.push(c);
                coo.data.push(v);
            }
            new_row += 1;
        }
    }
    coo.tocsr()
}

/// Python: `umap.utils.average_nn_distance` — mean of the second-nearest
/// neighbor distances (column 1 of `knn_dists`).
#[must_use]
pub fn average_nn_distance(knn_dists: &ndarray::Array2<f32>) -> f32 {
    if knn_dists.ncols() < 2 {
        return 0.0;
    }
    let sum: f64 = (0..knn_dists.nrows())
        .map(|i| f64::from(knn_dists[[i, 1]]))
        .sum();
    (sum / knn_dists.nrows().max(1) as f64) as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fast_knn_indices_stable_ties() {
        let x = ndarray::arr2(&[[3.0, 1.0, 1.0, 2.0], [5.0, 0.0, 9.0, 0.0]]);
        let idx = fast_knn_indices(&x, 2);
        // Ties (1.0 at cols 1, 2) resolved by ascending column index.
        assert_eq!(idx[[0, 0]], 1);
        assert_eq!(idx[[0, 1]], 2);
        assert_eq!(idx[[1, 0]], 1);
        assert_eq!(idx[[1, 1]], 3);
    }

    #[test]
    fn csr_unique_deduplicates_rows() {
        let coo = CooMatrix::from_triplets(
            (4, 3),
            vec![0, 0, 1, 1, 2, 2, 3],
            vec![0, 1, 0, 1, 0, 1, 2],
            vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        );
        let matrix = coo.tocsr();
        let unique = csr_unique(&matrix);
        // Rows 0, 1, 2 are the identical one-hot pattern (0,1); row 3 differs.
        assert_eq!(unique.shape.0, 2);
    }
}
