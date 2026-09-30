//! Ported from Python: umap/validation.py (`trustworthiness_vector_bulk`,
//! `trustworthiness_vector`). sklearn `KDTree` is replaced by exact brute-force
//! nearest neighbors in the embedding (identical results for exact NN).
// The kernels mirror the numba-compiled Python structure (single-char loop
// variables, index-based loops, many flag parameters matching the Python
// signatures); pedantic lints for those are allowed module-wide.
#![allow(
    clippy::many_single_char_names,
    clippy::needless_range_loop,
    clippy::struct_excessive_bools,
    clippy::type_complexity
)]

use ndarray::Array2;

/// Python: `umap.validation.trustworthiness_vector` — the trustworthiness
/// metric for k = `1..max_k` between the source data and the embedding.
///
/// `source` is the high-dimensional data; embedded neighborhood ranks are
/// computed exactly by brute force over the embedding rows.
#[must_use]
pub fn trustworthiness_vector(
    source: &Array2<f32>,
    embedding: &Array2<f32>,
    max_k: usize,
) -> Vec<f64> {
    let n_samples = embedding.nrows();

    // Exact embedded knn indices (excluding self) via brute force.
    let mut indices_embedded = Array2::<i32>::from_elem((n_samples, max_k), -1);
    for i in 0..n_samples {
        let mut order: Vec<usize> = (0..n_samples).collect();
        order.sort_by(|&a, &b| {
            let da = sq_dist_embed(embedding, i, a);
            let db = sq_dist_embed(embedding, i, b);
            da.partial_cmp(&db)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.cmp(&b))
        });
        for (j, &idx) in order.iter().skip(1).take(max_k).enumerate() {
            indices_embedded[[i, j]] = idx as i32;
        }
    }

    // Source-data ranks: for each i, the rank of every other point in the
    // source distance ordering. Computed via argsort of the distance vector
    // (Python: np.argsort of the metric distances).
    trustworthiness_vector_bulk_with_source(source, &indices_embedded, max_k)
}

/// Python: `trustworthiness_vector_bulk` generalized: source ranks computed
/// from the actual high-dimensional distances (stable ties like mergesort —
/// matches np.argsort's stable default for float ties).
fn trustworthiness_vector_bulk_with_source(
    source: &Array2<f32>,
    indices_embedded: &Array2<i32>,
    max_k: usize,
) -> Vec<f64> {
    let n_samples = source.nrows();
    let mut trustworthiness = vec![0.0f64; max_k + 1];

    for i in 0..n_samples {
        // argsort of source distances from point i (stable: ties by index).
        let mut order: Vec<usize> = (0..n_samples).collect();
        order.sort_by(|&a, &b| {
            let da = sq_dist(source, i, a);
            let db = sq_dist(source, i, b);
            da.partial_cmp(&db)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.cmp(&b))
        });

        // rank[j] = position of point j in the source ordering.
        let mut rank = vec![0usize; n_samples];
        for (pos, &pt) in order.iter().enumerate() {
            rank[pt] = pos;
        }

        for j in 0..max_k {
            let r = rank[indices_embedded[[i, j]] as usize];
            for k in (j + 1)..=max_k {
                if r > k {
                    trustworthiness[k] += f64::from((r - k) as u32);
                }
            }
        }
    }

    for k in 1..=max_k {
        trustworthiness[k] = 1.0
            - trustworthiness[k]
                * (2.0
                    / (n_samples as f64
                        * k as f64
                        * (2.0 * n_samples as f64 - 3.0 * k as f64 - 1.0)));
    }
    trustworthiness[0] = 1.0;
    trustworthiness
}

/// Python: `umap.validation.trustworthiness_vector` final scalar for a given
/// k (the common summary metric).
#[must_use]
pub fn trustworthiness(source: &Array2<f32>, embedding: &Array2<f32>, k: usize) -> f64 {
    trustworthiness_vector(source, embedding, k)[k]
}

#[inline]
fn sq_dist(m: &Array2<f32>, i: usize, j: usize) -> f64 {
    let mut s = 0.0f64;
    for d in 0..m.ncols() {
        let diff = f64::from(m[[i, d]]) - f64::from(m[[j, d]]);
        s += diff * diff;
    }
    s
}

#[inline]
fn sq_dist_embed(m: &Array2<f32>, i: usize, j: usize) -> f64 {
    sq_dist(m, i, j)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn perfect_embedding_has_trustworthiness_one() {
        // Distinct source points (no ties); embedding equals the first two
        // source coordinates: fully trustworthy.
        let source = Array2::from_shape_fn((20, 3), |(i, d)| {
            i as f32 * 1.7 + d as f32 * 3.1 + (d as f32) * (i as f32) * 0.3
        });
        let embedding = Array2::from_shape_fn((20, 2), |(i, d)| source[[i, d]]);
        let t = trustworthiness(&source, &embedding, 5);
        assert!(t > 0.999, "trustworthiness {t}");
    }

    #[test]
    fn scrambled_embedding_has_lower_trustworthiness() {
        let source = Array2::from_shape_fn((20, 3), |(i, d)| {
            i as f32 * 1.7 + d as f32 * 3.1 + (d as f32) * (i as f32) * 0.3
        });
        // Reverse the order of points in the embedding.
        let embedding = Array2::from_shape_fn((20, 2), |(i, d)| {
            let j = 19 - i;
            source[[j, d]]
        });
        let t = trustworthiness(&source, &embedding, 5);
        assert!(t < 1.0);
    }
}
