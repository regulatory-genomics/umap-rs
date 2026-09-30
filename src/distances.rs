//! Ported from Python: umap/distances.py
//!
//! Dense distance metrics, their analytic gradients, and the special
//! (discrete / string) metrics of umap-learn 0.5.12.
//!
//! All arithmetic is performed in `f32` (numba computes in float32 for
//! float32 inputs). Metrics that take extra keyword arguments in Python
//! receive them through the `metric_args: &[f32]` slice; see the individual
//! doc comments for the exact expected ordering (matching Python's
//! `metric_kwds` ordering). Passing an empty slice selects the Python
//! default values (`_mock_ones`, `_mock_identity`, `p=2`, ...).
//!
//! Divergences from the Python original:
//! - `haversine`/`haversine_grad` raise `ValueError` for non-2-dimensional
//!   input in Python; here they return `NaN` (the metric signature cannot
//!   panic). HACK: documented divergence.
//! - `symmetric_kl`/`symmetric_kl_grad` and `gaussian_energy_grad` mutate
//!   their input arrays in place in Python; here inputs are copied because
//!   the API takes shared slices.
//! - `string`/`myers` take Python `str` objects; here strings are packed as
//!   vectors of f32 character codes (see [`levenshtein`]).
//! - `hierarchical_categorical_distance` takes a list of dicts in Python;
//!   here the hierarchy is a flattened f32 encoding (see the doc comment).
//!   HACK: a category missing from a level raises `KeyError` in Python; here
//!   it is treated as "distinct at that level".
//! - `get_discrete_params`, `mahalanobis_f64` and `sinkhorn_distance` are
//!   intentionally omitted: they need Python object arrays / f64 vectors /
//!   cost-matrix kwargs which the f32 metric API cannot express.
//!   FIXME: revisit if a f64 metric API is added.

#![allow(
    // The metrics deliberately compare float32 values exactly, matching the
    // numba implementations (e.g. hamming's `x[i] != y[i]`, binary metrics'
    // `x[i] != 0.0`).
    clippy::float_cmp,
    // Python's dispatch dicts register many aliases that map to the same
    // functions; the match arms mirror those dicts one-to-one.
    clippy::match_same_arms,
    // Arithmetic such as `(kl1 + kl2) / 2` is kept verbatim from the Python
    // source rather than rewritten as `f32::midpoint`.
    clippy::manual_midpoint
)]

use ndarray::{Array2, ArrayView2};
use rayon::prelude::*;

/// A metric computes distance between two f32 vectors. Third slice holds
/// extra metric args (kwds), empty for metrics that take none.
pub type MetricFn = fn(&[f32], &[f32], &[f32]) -> f32;

/// Gradient function: returns (distance, gradient wrt first arg).
pub type GradFn = fn(&[f32], &[f32], &[f32]) -> (f32, Vec<f32>);

// --------------------------------------------------------------------------
// Helpers
// --------------------------------------------------------------------------

/// Python: `sign` (numba version) — returns -1.0 or 1.0, never 0.0.
fn sign(a: f32) -> f32 {
    if a < 0.0 {
        -1.0
    } else {
        1.0
    }
}

/// Python: `np.sign` — 0.0 for 0.0.
fn np_sign(a: f32) -> f32 {
    if a > 0.0 {
        1.0
    } else if a < 0.0 {
        -1.0
    } else {
        0.0
    }
}

/// The Python source accumulates bools into float counts (`num_true_true
/// += x_true and y_true`); numba converts the bools to 0.0/1.0.
fn b2f(b: bool) -> f32 {
    if b {
        1.0
    } else {
        0.0
    }
}

/// Python: `softmax(z)`.
fn softmax(z: &[f32]) -> Vec<f32> {
    let n = z.len();
    let mut out = vec![0.0f32; n];
    if n == 0 {
        return out;
    }

    let mut zmax = z[0];
    for &zi in &z[1..] {
        if zi > zmax {
            zmax = zi;
        }
    }

    let mut s = 0.0f32;
    for (o, &zi) in out.iter_mut().zip(z.iter()) {
        *o = (zi - zmax).exp();
        s += *o;
    }

    if s == 0.0 {
        let uniform = 1.0 / n as f32;
        out.fill(uniform);
    } else {
        let invs = 1.0 / s;
        for o in &mut out {
            *o *= invs;
        }
    }

    out
}

// --------------------------------------------------------------------------
// General minkowski distances
// --------------------------------------------------------------------------

/// Python: `euclidean` — standard euclidean distance.
pub fn euclidean(x: &[f32], y: &[f32], _metric_args: &[f32]) -> f32 {
    let mut result = 0.0f32;
    for i in 0..x.len() {
        let d = x[i] - y[i];
        result += d * d;
    }
    result.sqrt()
}

/// Python: `euclidean_grad` — euclidean distance and its gradient.
pub fn euclidean_grad(x: &[f32], y: &[f32], _metric_args: &[f32]) -> (f32, Vec<f32>) {
    let mut result = 0.0f32;
    for i in 0..x.len() {
        let d = x[i] - y[i];
        result += d * d;
    }
    let d = result.sqrt();
    let grad = x
        .iter()
        .zip(y.iter())
        .map(|(&xi, &yi)| (xi - yi) / (1e-6 + d))
        .collect();
    (d, grad)
}

/// Python: `standardised_euclidean(x, y, sigma)` — euclidean distance
/// standardised against a vector of standard deviations per coordinate.
///
/// `metric_args` holds the per-coordinate `sigma` vector `V` (sklearn
/// `seuclidean` `V` kwarg) in coordinate order; an empty slice selects the
/// Python default of all ones.
pub fn standardised_euclidean(x: &[f32], y: &[f32], metric_args: &[f32]) -> f32 {
    let sigma = |i: usize| -> f32 {
        if metric_args.len() > i {
            metric_args[i]
        } else {
            1.0
        }
    };
    let mut result = 0.0f32;
    for i in 0..x.len() {
        let d = x[i] - y[i];
        result += (d * d) / sigma(i);
    }
    result.sqrt()
}

/// Python: `standardised_euclidean_grad(x, y, sigma)`.
///
/// `metric_args` holds the per-coordinate `sigma` vector (see
/// [`standardised_euclidean`]).
pub fn standardised_euclidean_grad(x: &[f32], y: &[f32], metric_args: &[f32]) -> (f32, Vec<f32>) {
    let sigma = |i: usize| -> f32 {
        if metric_args.len() > i {
            metric_args[i]
        } else {
            1.0
        }
    };
    let mut result = 0.0f32;
    for i in 0..x.len() {
        let d = x[i] - y[i];
        result += (d * d) / sigma(i);
    }
    let d = result.sqrt();
    let grad = x
        .iter()
        .zip(y.iter())
        .enumerate()
        .map(|(i, (&xi, &yi))| (xi - yi) / (1e-6 + d * sigma(i)))
        .collect();
    (d, grad)
}

/// Python: `manhattan` — manhattan, taxicab, or l1 distance.
pub fn manhattan(x: &[f32], y: &[f32], _metric_args: &[f32]) -> f32 {
    let mut result = 0.0f32;
    for i in 0..x.len() {
        result += (x[i] - y[i]).abs();
    }
    result
}

/// Python: `manhattan_grad`.
pub fn manhattan_grad(x: &[f32], y: &[f32], _metric_args: &[f32]) -> (f32, Vec<f32>) {
    let mut result = 0.0f32;
    let mut grad = vec![0.0f32; x.len()];
    for i in 0..x.len() {
        result += (x[i] - y[i]).abs();
        grad[i] = sign(x[i] - y[i]);
    }
    (result, grad)
}

/// Python: `chebyshev` — chebyshev or l-infinity distance.
pub fn chebyshev(x: &[f32], y: &[f32], _metric_args: &[f32]) -> f32 {
    let mut result = 0.0f32;
    for i in 0..x.len() {
        result = result.max((x[i] - y[i]).abs());
    }
    result
}

/// Python: `chebyshev_grad`.
pub fn chebyshev_grad(x: &[f32], y: &[f32], _metric_args: &[f32]) -> (f32, Vec<f32>) {
    let mut result = 0.0f32;
    let mut max_i = 0usize;
    for i in 0..x.len() {
        let v = (x[i] - y[i]).abs();
        if v > result {
            result = v;
            max_i = i;
        }
    }
    let mut grad = vec![0.0f32; x.len()];
    if !grad.is_empty() {
        grad[max_i] = sign(x[max_i] - y[max_i]);
    }
    (result, grad)
}

/// Python: `minkowski(x, y, p=2)` — minkowski distance.
///
/// `metric_args = [p]`; an empty slice selects the Python default `p = 2`.
pub fn minkowski(x: &[f32], y: &[f32], metric_args: &[f32]) -> f32 {
    let p = metric_args.first().copied().unwrap_or(2.0);
    let mut result = 0.0f32;
    for i in 0..x.len() {
        result += (x[i] - y[i]).abs().powf(p);
    }
    result.powf(1.0 / p)
}

/// Python: `minkowski_grad(x, y, p=2.0)`.
///
/// `metric_args = [p]`; an empty slice selects the Python default `p = 2.0`.
pub fn minkowski_grad(x: &[f32], y: &[f32], metric_args: &[f32]) -> (f32, Vec<f32>) {
    let p = metric_args.first().copied().unwrap_or(2.0);
    let mut s = 0.0f32;
    for i in 0..x.len() {
        s += (x[i] - y[i]).abs().powf(p);
    }

    let dist = s.powf(1.0 / p);
    let mut grad = vec![0.0f32; x.len()];

    if s == 0.0 {
        return (dist, grad);
    }

    let inv_denom = s.powf((1.0 - p) / p);
    for (g, i) in grad.iter_mut().zip(0..x.len()) {
        *g = (x[i] - y[i]).abs().powf(p - 1.0) * sign(x[i] - y[i]) * inv_denom;
    }

    (dist, grad)
}

/// Python: `poincare(u, v)` — poincare distance.
pub fn poincare(u: &[f32], v: &[f32], _metric_args: &[f32]) -> f32 {
    let mut sq_u_norm = 0.0f32;
    let mut sq_v_norm = 0.0f32;
    let mut sq_dist = 0.0f32;
    for i in 0..u.len() {
        sq_u_norm += u[i] * u[i];
        sq_v_norm += v[i] * v[i];
        sq_dist += (u[i] - v[i]).powi(2);
    }
    (1.0 + 2.0 * (sq_dist / ((1.0 - sq_u_norm) * (1.0 - sq_v_norm)))).acosh()
}

/// Python: `hyperboloid_grad(x, y)` — hyperboloid distance and gradient
/// (registered under the name `"hyperboloid"`).
#[allow(clippy::many_single_char_names)] // single-char names kept from Python
pub fn hyperboloid_grad(x: &[f32], y: &[f32], _metric_args: &[f32]) -> (f32, Vec<f32>) {
    let mut sx2 = 0.0f32;
    let mut sy2 = 0.0f32;
    for i in 0..x.len() {
        sx2 += x[i] * x[i];
        sy2 += y[i] * y[i];
    }
    let s = (1.0 + sx2).sqrt();
    let t = (1.0 + sy2).sqrt();

    let mut b = s * t;
    for i in 0..x.len() {
        b -= x[i] * y[i];
    }

    if b <= 1.0 {
        b = 1.0 + 1e-8;
    }

    let grad_coeff = 1.0 / ((b - 1.0).sqrt() * (b + 1.0).sqrt());

    let mut grad = vec![0.0f32; x.len()];
    for (g, i) in grad.iter_mut().zip(0..x.len()) {
        *g = grad_coeff * ((x[i] * t) / s - y[i]);
    }

    (b.acosh(), grad)
}

/// Python: `weighted_minkowski(x, y, w, p=2)` — weighted minkowski distance.
///
/// `metric_args = [w_0, ..., w_{n-1}, p]` matching Python's
/// `weighted_minkowski(x, y, w=..., p=...)` kwds ordering (`n = x.len()`).
/// If only one arg is given it is taken as `p` with weights defaulting to
/// ones; an empty slice selects the Python defaults (`w = 1`, `p = 2`).
pub fn weighted_minkowski(x: &[f32], y: &[f32], metric_args: &[f32]) -> f32 {
    let (p, w) = weighted_minkowski_args(x.len(), metric_args);
    let mut result = 0.0f32;
    for i in 0..x.len() {
        result += weight(&w, i) * (x[i] - y[i]).abs().powf(p);
    }
    result.powf(1.0 / p)
}

/// Python: `weighted_minkowski_grad(x, y, w, p=2.0)`.
///
/// `metric_args` ordering as in [`weighted_minkowski`].
#[allow(clippy::many_single_char_names)] // single-char names kept from Python
pub fn weighted_minkowski_grad(x: &[f32], y: &[f32], metric_args: &[f32]) -> (f32, Vec<f32>) {
    let (p, w) = weighted_minkowski_args(x.len(), metric_args);
    let mut s = 0.0f32;
    for i in 0..x.len() {
        s += weight(&w, i) * (x[i] - y[i]).abs().powf(p);
    }

    let dist = s.powf(1.0 / p);
    let mut grad = vec![0.0f32; x.len()];

    if s == 0.0 {
        return (dist, grad);
    }

    let inv_denom = s.powf((1.0 - p) / p);
    for (g, i) in grad.iter_mut().zip(0..x.len()) {
        *g = weight(&w, i) * (x[i] - y[i]).abs().powf(p - 1.0) * sign(x[i] - y[i]) * inv_denom;
    }

    (dist, grad)
}

/// Resolve `w` / `p` for [`weighted_minkowski`] from `metric_args`. Returns
/// `(p, w)` where `w` is empty to signal "all weights are one".
fn weighted_minkowski_args(n: usize, metric_args: &[f32]) -> (f32, Vec<f32>) {
    if metric_args.len() == n + 1 {
        (metric_args[n], metric_args[..n].to_vec())
    } else if metric_args.len() == 1 {
        // Only p given; weights default to ones.
        (metric_args[0], Vec::new())
    } else {
        (2.0, Vec::new())
    }
}

fn weight(w: &[f32], i: usize) -> f32 {
    w.get(i).copied().unwrap_or(1.0)
}

/// Python: `mahalanobis(x, y, vinv)` — mahalanobis distance.
///
/// `metric_args` holds the flattened (row-major) `n x n` inverse covariance
/// matrix `VI` (`n = x.len()`); an empty or short slice selects the Python
/// default of the identity matrix.
pub fn mahalanobis(x: &[f32], y: &[f32], metric_args: &[f32]) -> f32 {
    let n = x.len();
    let vinv = |i: usize, j: usize| -> f32 {
        if metric_args.len() >= n * n {
            metric_args[i * n + j]
        } else if i == j {
            1.0
        } else {
            0.0
        }
    };
    let mut result = 0.0f32;
    for i in 0..n {
        let mut tmp = 0.0f32;
        for j in 0..n {
            tmp += vinv(i, j) * (x[j] - y[j]);
        }
        result += tmp * (x[i] - y[i]);
    }
    result.sqrt()
}

/// Python: `mahalanobis_grad(x, y, vinv)`.
///
/// `metric_args` holds the flattened `n x n` `VI` (see [`mahalanobis`]).
pub fn mahalanobis_grad(x: &[f32], y: &[f32], metric_args: &[f32]) -> (f32, Vec<f32>) {
    let n = x.len();
    let vinv = |i: usize, j: usize| -> f32 {
        if metric_args.len() >= n * n {
            metric_args[i * n + j]
        } else if i == j {
            1.0
        } else {
            0.0
        }
    };
    let mut result = 0.0f32;
    let mut grad_tmp = vec![0.0f32; n];
    for i in 0..n {
        let mut tmp = 0.0f32;
        for j in 0..n {
            tmp += vinv(i, j) * (x[j] - y[j]);
            grad_tmp[i] += vinv(i, j) * (x[j] - y[j]);
        }
        result += tmp * (x[i] - y[i]);
    }
    let dist = result.sqrt();
    let grad = grad_tmp.iter().map(|&g| g / (1e-6 + dist)).collect();
    (dist, grad)
}

// --------------------------------------------------------------------------
// Other distances
// --------------------------------------------------------------------------

/// Python: `hamming`.
pub fn hamming(x: &[f32], y: &[f32], _metric_args: &[f32]) -> f32 {
    let mut result = 0.0f32;
    for i in 0..x.len() {
        if x[i] != y[i] {
            result += 1.0;
        }
    }
    result / x.len() as f32
}

/// Python: `canberra`.
pub fn canberra(x: &[f32], y: &[f32], _metric_args: &[f32]) -> f32 {
    let mut result = 0.0f32;
    for i in 0..x.len() {
        let denominator = x[i].abs() + y[i].abs();
        if denominator > 0.0 {
            result += (x[i] - y[i]).abs() / denominator;
        }
    }
    result
}

/// Python: `canberra_grad`.
pub fn canberra_grad(x: &[f32], y: &[f32], _metric_args: &[f32]) -> (f32, Vec<f32>) {
    let mut result = 0.0f32;
    let mut grad = vec![0.0f32; x.len()];
    for i in 0..x.len() {
        let denominator = x[i].abs() + y[i].abs();
        if denominator > 0.0 {
            result += (x[i] - y[i]).abs() / denominator;
            grad[i] = sign(x[i] - y[i]) / denominator
                - (x[i] - y[i]).abs() * sign(x[i]) / (denominator * denominator);
        }
    }
    (result, grad)
}

/// Python: `bray_curtis` (registered under the name `"braycurtis"`).
pub fn bray_curtis(x: &[f32], y: &[f32], _metric_args: &[f32]) -> f32 {
    let mut numerator = 0.0f32;
    let mut denominator = 0.0f32;
    for i in 0..x.len() {
        numerator += (x[i] - y[i]).abs();
        denominator += (x[i] + y[i]).abs();
    }

    if denominator > 0.0 {
        numerator / denominator
    } else {
        0.0
    }
}

/// Python: `bray_curtis_grad`.
pub fn bray_curtis_grad(x: &[f32], y: &[f32], _metric_args: &[f32]) -> (f32, Vec<f32>) {
    let mut numerator = 0.0f32;
    let mut denominator = 0.0f32;
    for i in 0..x.len() {
        numerator += (x[i] - y[i]).abs();
        denominator += (x[i] + y[i]).abs();
    }

    let (dist, grad) = if denominator > 0.0 {
        let dist = numerator / denominator;
        let grad = x
            .iter()
            .zip(y.iter())
            .map(|(&xi, &yi)| (np_sign(xi - yi) - dist) / denominator)
            .collect();
        (dist, grad)
    } else {
        (0.0, vec![0.0f32; x.len()])
    };

    (dist, grad)
}

/// Python: `jaccard`.
pub fn jaccard(x: &[f32], y: &[f32], _metric_args: &[f32]) -> f32 {
    let mut num_non_zero = 0.0f32;
    let mut num_equal = 0.0f32;
    for i in 0..x.len() {
        let x_true = x[i] != 0.0;
        let y_true = y[i] != 0.0;
        num_non_zero += b2f(x_true || y_true);
        num_equal += b2f(x_true && y_true);
    }

    if num_non_zero == 0.0 {
        0.0
    } else {
        (num_non_zero - num_equal) / num_non_zero
    }
}

/// Python: `matching`.
pub fn matching(x: &[f32], y: &[f32], _metric_args: &[f32]) -> f32 {
    let mut num_not_equal = 0.0f32;
    for i in 0..x.len() {
        let x_true = x[i] != 0.0;
        let y_true = y[i] != 0.0;
        num_not_equal += b2f(x_true != y_true);
    }
    num_not_equal / x.len() as f32
}

/// Python: `dice`.
pub fn dice(x: &[f32], y: &[f32], _metric_args: &[f32]) -> f32 {
    let mut num_true_true = 0.0f32;
    let mut num_not_equal = 0.0f32;
    for i in 0..x.len() {
        let x_true = x[i] != 0.0;
        let y_true = y[i] != 0.0;
        num_true_true += b2f(x_true && y_true);
        num_not_equal += b2f(x_true != y_true);
    }

    if num_not_equal == 0.0 {
        0.0
    } else {
        num_not_equal / (2.0 * num_true_true + num_not_equal)
    }
}

/// Python: `kulsinski`.
pub fn kulsinski(x: &[f32], y: &[f32], _metric_args: &[f32]) -> f32 {
    let mut num_true_true = 0.0f32;
    let mut num_not_equal = 0.0f32;
    for i in 0..x.len() {
        let x_true = x[i] != 0.0;
        let y_true = y[i] != 0.0;
        num_true_true += b2f(x_true && y_true);
        num_not_equal += b2f(x_true != y_true);
    }

    if num_not_equal == 0.0 {
        0.0
    } else {
        (num_not_equal - num_true_true + x.len() as f32) / (num_not_equal + x.len() as f32)
    }
}

/// Python: `rogers_tanimoto` (registered under the name `"rogerstanimoto"`).
pub fn rogers_tanimoto(x: &[f32], y: &[f32], _metric_args: &[f32]) -> f32 {
    let mut num_not_equal = 0.0f32;
    for i in 0..x.len() {
        let x_true = x[i] != 0.0;
        let y_true = y[i] != 0.0;
        num_not_equal += b2f(x_true != y_true);
    }

    (2.0 * num_not_equal) / (x.len() as f32 + num_not_equal)
}

/// Python: `russellrao`.
pub fn russellrao(x: &[f32], y: &[f32], _metric_args: &[f32]) -> f32 {
    let mut num_true_true = 0.0f32;
    let mut num_nonzero_x = 0.0f32;
    let mut num_nonzero_y = 0.0f32;
    for i in 0..x.len() {
        let x_true = x[i] != 0.0;
        let y_true = y[i] != 0.0;
        num_true_true += b2f(x_true && y_true);
        num_nonzero_x += b2f(x_true);
        num_nonzero_y += b2f(y_true);
    }

    if num_true_true == num_nonzero_x && num_true_true == num_nonzero_y {
        0.0
    } else {
        (x.len() as f32 - num_true_true) / x.len() as f32
    }
}

/// Python: `sokal_michener` (registered under the name `"sokalmichener"`).
pub fn sokal_michener(x: &[f32], y: &[f32], _metric_args: &[f32]) -> f32 {
    let mut num_not_equal = 0.0f32;
    for i in 0..x.len() {
        let x_true = x[i] != 0.0;
        let y_true = y[i] != 0.0;
        num_not_equal += b2f(x_true != y_true);
    }

    (2.0 * num_not_equal) / (x.len() as f32 + num_not_equal)
}

/// Python: `sokal_sneath` (registered under the name `"sokalsneath"`).
pub fn sokal_sneath(x: &[f32], y: &[f32], _metric_args: &[f32]) -> f32 {
    let mut num_true_true = 0.0f32;
    let mut num_not_equal = 0.0f32;
    for i in 0..x.len() {
        let x_true = x[i] != 0.0;
        let y_true = y[i] != 0.0;
        num_true_true += b2f(x_true && y_true);
        num_not_equal += b2f(x_true != y_true);
    }

    if num_not_equal == 0.0 {
        0.0
    } else {
        num_not_equal / (0.5 * num_true_true + num_not_equal)
    }
}

/// Python: `haversine` — only defined for 2 dimensional data (latitude,
/// longitude).
///
/// HACK: Python raises `ValueError` for non-2-dimensional input; here `NaN`
/// is returned because the metric signature cannot panic.
pub fn haversine(x: &[f32], y: &[f32], _metric_args: &[f32]) -> f32 {
    if x.len() != 2 {
        return f32::NAN;
    }
    let sin_lat = (0.5 * (x[0] - y[0])).sin();
    let sin_long = (0.5 * (x[1] - y[1])).sin();
    let result = (sin_lat * sin_lat + x[0].cos() * y[0].cos() * sin_long * sin_long).sqrt();
    2.0 * result.asin()
}

/// Python: `haversine_grad`.
///
/// HACK: Python raises `ValueError` for non-2-dimensional input; here
/// `(NaN, [])` is returned because the gradient signature cannot panic.
pub fn haversine_grad(x: &[f32], y: &[f32], _metric_args: &[f32]) -> (f32, Vec<f32>) {
    if x.len() != 2 {
        return (f32::NAN, Vec::new());
    }
    // spectral initialization puts many points near the poles
    // currently, adding pi/2 to the latitude avoids problems
    // TODO: reimplement with quaternions to avoid singularity
    let sin_lat = (0.5 * (x[0] - y[0])).sin();
    let cos_lat = (0.5 * (x[0] - y[0])).cos();
    let sin_long = (0.5 * (x[1] - y[1])).sin();
    let cos_long = (0.5 * (x[1] - y[1])).cos();

    let a_0 = (x[0] + std::f32::consts::FRAC_PI_2).cos()
        * (y[0] + std::f32::consts::FRAC_PI_2).cos()
        * sin_long
        * sin_long;
    let a_1 = a_0 + sin_lat * sin_lat;

    let d = 2.0 * a_1.abs().clamp(0.0, 1.0).sqrt().asin();
    let denom = (a_1 - 1.0).abs().sqrt() * a_1.abs().sqrt();
    let grad = vec![
        (sin_lat * cos_lat
            - (x[0] + std::f32::consts::FRAC_PI_2).cos()
                * (y[0] + std::f32::consts::FRAC_PI_2).cos()
                * sin_long
                * sin_long)
            / (denom + 1e-6),
        ((x[0] + std::f32::consts::FRAC_PI_2).cos()
            * (y[0] + std::f32::consts::FRAC_PI_2).cos()
            * sin_long
            * cos_long)
            / (denom + 1e-6),
    ];
    (d, grad)
}

/// Python: `yule`.
pub fn yule(x: &[f32], y: &[f32], _metric_args: &[f32]) -> f32 {
    let mut num_true_true = 0.0f32;
    let mut num_true_false = 0.0f32;
    let mut num_false_true = 0.0f32;
    for i in 0..x.len() {
        let x_true = x[i] != 0.0;
        let y_true = y[i] != 0.0;
        num_true_true += b2f(x_true && y_true);
        num_true_false += b2f(x_true && !y_true);
        num_false_true += b2f(!x_true && y_true);
    }

    let num_false_false = x.len() as f32 - num_true_true - num_true_false - num_false_true;

    if num_true_false == 0.0 || num_false_true == 0.0 {
        0.0
    } else {
        (2.0 * num_true_false * num_false_true)
            / (num_true_true * num_false_false + num_true_false * num_false_true)
    }
}

/// Python: `cosine`.
pub fn cosine(x: &[f32], y: &[f32], _metric_args: &[f32]) -> f32 {
    let mut result = 0.0f32;
    let mut norm_x = 0.0f32;
    let mut norm_y = 0.0f32;
    for i in 0..x.len() {
        result += x[i] * y[i];
        norm_x += x[i] * x[i];
        norm_y += y[i] * y[i];
    }

    if norm_x == 0.0 && norm_y == 0.0 {
        0.0
    } else if norm_x == 0.0 || norm_y == 0.0 {
        1.0
    } else {
        1.0 - (result / (norm_x * norm_y).sqrt())
    }
}

/// Python: `cosine_grad`.
pub fn cosine_grad(x: &[f32], y: &[f32], _metric_args: &[f32]) -> (f32, Vec<f32>) {
    let mut result = 0.0f32;
    let mut norm_x = 0.0f32;
    let mut norm_y = 0.0f32;

    for i in 0..x.len() {
        result += x[i] * y[i];
        norm_x += x[i] * x[i];
        norm_y += y[i] * y[i];
    }

    if norm_x == 0.0 && norm_y == 0.0 {
        return (0.0, vec![0.0f32; x.len()]);
    }

    if norm_x == 0.0 || norm_y == 0.0 {
        return (1.0, vec![0.0f32; x.len()]);
    }

    let nx = norm_x.sqrt();
    let ny = norm_y.sqrt();

    let dist = 1.0 - result / (nx * ny);

    let inv_nx_ny = 1.0 / (nx * ny);
    let inv_nx3_ny = 1.0 / (norm_x * nx * ny);

    let grad = x
        .iter()
        .zip(y.iter())
        .map(|(&xi, &yi)| xi * result * inv_nx3_ny - yi * inv_nx_ny)
        .collect();

    (dist, grad)
}

/// Python: `correlation`.
pub fn correlation(x: &[f32], y: &[f32], _metric_args: &[f32]) -> f32 {
    let n = x.len() as f32;
    let mut mu_x = 0.0f32;
    let mut mu_y = 0.0f32;
    for i in 0..x.len() {
        mu_x += x[i];
        mu_y += y[i];
    }
    mu_x /= n;
    mu_y /= n;

    let mut norm_x = 0.0f32;
    let mut norm_y = 0.0f32;
    let mut dot_product = 0.0f32;
    for i in 0..x.len() {
        let shifted_x = x[i] - mu_x;
        let shifted_y = y[i] - mu_y;
        norm_x += shifted_x * shifted_x;
        norm_y += shifted_y * shifted_y;
        dot_product += shifted_x * shifted_y;
    }

    if norm_x == 0.0 && norm_y == 0.0 {
        0.0
    } else if dot_product == 0.0 {
        1.0
    } else {
        1.0 - (dot_product / (norm_x * norm_y).sqrt())
    }
}

/// Python: `correlation_grad`.
pub fn correlation_grad(x: &[f32], y: &[f32], _metric_args: &[f32]) -> (f32, Vec<f32>) {
    let n = x.len();
    let n_f = n as f32;

    let mut mu_x = 0.0f32;
    let mut mu_y = 0.0f32;
    for i in 0..n {
        mu_x += x[i];
        mu_y += y[i];
    }
    mu_x /= n_f;
    mu_y /= n_f;

    let mut dot = 0.0f32;
    let mut norm_x = 0.0f32;
    let mut norm_y = 0.0f32;

    for i in 0..n {
        let cx = x[i] - mu_x;
        let cy = y[i] - mu_y;
        dot += cx * cy;
        norm_x += cx * cx;
        norm_y += cy * cy;
    }

    if norm_x == 0.0 && norm_y == 0.0 {
        return (0.0, vec![0.0f32; n]);
    }

    if norm_x == 0.0 || norm_y == 0.0 {
        return (1.0, vec![0.0f32; n]);
    }

    let nx = norm_x.sqrt();
    let ny = norm_y.sqrt();

    let dist = 1.0 - dot / (nx * ny);

    let inv_nx_ny = 1.0 / (nx * ny);
    let inv_nx3_ny = 1.0 / (norm_x * nx * ny);

    let mut grad = vec![0.0f32; n];
    let mut mean_grad = 0.0f32;
    for i in 0..n {
        let cx = x[i] - mu_x;
        grad[i] = cx * dot * inv_nx3_ny - (y[i] - mu_y) * inv_nx_ny;
        mean_grad += grad[i];
    }

    mean_grad /= n_f;
    for g in &mut grad {
        *g -= mean_grad;
    }

    (dist, grad)
}

/// Python: `hellinger`.
pub fn hellinger(x: &[f32], y: &[f32], _metric_args: &[f32]) -> f32 {
    let mut result = 0.0f32;
    let mut l1_norm_x = 0.0f32;
    let mut l1_norm_y = 0.0f32;

    for i in 0..x.len() {
        result += (x[i] * y[i]).sqrt();
        l1_norm_x += x[i];
        l1_norm_y += y[i];
    }

    if l1_norm_x == 0.0 && l1_norm_y == 0.0 {
        0.0
    } else if l1_norm_x == 0.0 || l1_norm_y == 0.0 {
        1.0
    } else {
        (1.0 - result / (l1_norm_x * l1_norm_y).sqrt()).sqrt()
    }
}

/// Python: `hellinger_grad`.
pub fn hellinger_grad(x: &[f32], y: &[f32], _metric_args: &[f32]) -> (f32, Vec<f32>) {
    let mut result = 0.0f32;
    let mut l1_norm_x = 0.0f32;
    let mut l1_norm_y = 0.0f32;

    let mut grad_term = vec![0.0f32; x.len()];

    for i in 0..x.len() {
        grad_term[i] = (x[i] * y[i]).sqrt();
        result += grad_term[i];
        l1_norm_x += x[i];
        l1_norm_y += y[i];
    }

    if l1_norm_x == 0.0 && l1_norm_y == 0.0 {
        return (0.0, vec![0.0f32; x.len()]);
    }

    if l1_norm_x == 0.0 || l1_norm_y == 0.0 {
        return (1.0, vec![0.0f32; x.len()]);
    }

    let dist_denom = (l1_norm_x * l1_norm_y).sqrt();
    let inner = (1.0 - result / dist_denom).max(0.0);
    let dist = inner.sqrt();

    if dist == 0.0 {
        return (dist, vec![0.0f32; x.len()]);
    }

    let mut grad = vec![0.0f32; x.len()];
    let grad_denom = 2.0 * dist;
    let grad_numer_const = (l1_norm_y * result) / (2.0 * dist_denom * dist_denom * dist_denom);

    for (g, i) in grad.iter_mut().zip(0..x.len()) {
        let term = if x[i] > 0.0 && grad_term[i] > 0.0 {
            y[i] / (2.0 * grad_term[i] * dist_denom)
        } else {
            0.0
        };

        *g = (grad_numer_const - term) / grad_denom;
    }

    (dist, grad)
}

/// Python: `softmax_hellinger` — hellinger distance between softmax(x) and
/// softmax(y).
pub fn softmax_hellinger(x: &[f32], y: &[f32], _metric_args: &[f32]) -> f32 {
    let p = softmax(x);
    let q = softmax(y);

    hellinger(&p, &q, &[])
}

/// Python: `softmax_hellinger_grad` — hellinger distance and gradient
/// between softmax(x) and softmax(y).
pub fn softmax_hellinger_grad(x: &[f32], y: &[f32], _metric_args: &[f32]) -> (f32, Vec<f32>) {
    let p = softmax(x);
    let q = softmax(y);

    let (dist, g_p) = hellinger_grad(&p, &q, &[]);

    let mut dot_gp_p = 0.0f32;
    for i in 0..p.len() {
        dot_gp_p += g_p[i] * p[i];
    }

    let grad_x = (0..x.len()).map(|i| p[i] * (g_p[i] - dot_gp_p)).collect();

    (dist, grad_x)
}

// --------------------------------------------------------------------------
// Log-likelihood helpers and ll_dirichlet
// --------------------------------------------------------------------------

/// Python: `approx_log_Gamma`.
fn approx_log_gamma(x: f32) -> f32 {
    if x == 1.0 {
        return 0.0;
    }
    x * x.ln() - x + 0.5 * (2.0 * std::f32::consts::PI / x).ln() + 1.0 / (x * 12.0)
}

/// Python: `log_beta`.
fn log_beta(x: f32, y: f32) -> f32 {
    let a = x.min(y);
    let b = x.max(y);
    if b < 5.0 {
        let mut value = -b.ln();
        for i in 1..a as i64 {
            value += (i as f32).ln() - (b + i as f32).ln();
        }
        value
    } else {
        approx_log_gamma(x) + approx_log_gamma(y) - approx_log_gamma(x + y)
    }
}

/// Python: `log_single_beta`.
fn log_single_beta(x: f32) -> f32 {
    (2.0f32).ln() * (-2.0 * x + 0.5) + 0.5 * (2.0 * std::f32::consts::PI / x).ln() + 0.125 / x
}

/// Python: `ll_dirichlet` — the symmetric relative log likelihood of rolling
/// data2 vs data1 in n trials on a die that rolled data1 in sum(data1)
/// trials.
pub fn ll_dirichlet(data1: &[f32], data2: &[f32], _metric_args: &[f32]) -> f32 {
    let mut n1 = 0.0f32;
    let mut n2 = 0.0f32;
    for &d in data1 {
        n1 += d;
    }
    for &d in data2 {
        n2 += d;
    }

    let mut log_b = 0.0f32;
    let mut self_denom1 = 0.0f32;
    let mut self_denom2 = 0.0f32;

    for i in 0..data1.len() {
        if data1[i] * data2[i] > 0.9 {
            log_b += log_beta(data1[i], data2[i]);
            self_denom1 += log_single_beta(data1[i]);
            self_denom2 += log_single_beta(data2[i]);
        } else {
            if data1[i] > 0.9 {
                self_denom1 += log_single_beta(data1[i]);
            }

            if data2[i] > 0.9 {
                self_denom2 += log_single_beta(data2[i]);
            }
        }
    }

    (1.0 / n2 * (log_b - log_beta(n1, n2) - (self_denom2 - log_single_beta(n2)))
        + 1.0 / n1 * (log_b - log_beta(n2, n1) - (self_denom1 - log_single_beta(n1))))
    .sqrt()
}

/// Python: `symmetric_kl(x, y, z=1e-11)` — symmetrized KL divergence.
///
/// HACK: Python mutates `x` and `y` in place (`x[i] += z`); here the
/// smoothing is applied to local copies because the API takes shared
/// slices.
pub fn symmetric_kl(x: &[f32], y: &[f32], _metric_args: &[f32]) -> f32 {
    let (xv, yv) = symmetric_kl_smoothed(x, y);
    let n = xv.len();

    let mut kl1 = 0.0f32;
    let mut kl2 = 0.0f32;
    for i in 0..n {
        kl1 += xv[i] * (xv[i] / yv[i]).ln();
        kl2 += yv[i] * (yv[i] / xv[i]).ln();
    }

    (kl1 + kl2) / 2.0
}

/// Python: `symmetric_kl_grad(x, y, z=1e-11)`.
///
/// HACK: Python mutates `x` and `y` in place; here the smoothing is applied
/// to local copies (see [`symmetric_kl`]).
pub fn symmetric_kl_grad(x: &[f32], y: &[f32], _metric_args: &[f32]) -> (f32, Vec<f32>) {
    let (xv, yv) = symmetric_kl_smoothed(x, y);
    let n = xv.len();

    let mut kl1 = 0.0f32;
    let mut kl2 = 0.0f32;
    for i in 0..n {
        kl1 += xv[i] * (xv[i] / yv[i]).ln();
        kl2 += yv[i] * (yv[i] / xv[i]).ln();
    }

    let dist = (kl1 + kl2) / 2.0;
    let grad = (0..n)
        .map(|i| ((yv[i] / xv[i]).ln() - (xv[i] / yv[i]) + 1.0) / 2.0)
        .collect();

    (dist, grad)
}

/// The `x[i] += z; y[i] += z; x[i] /= x_sum; y[i] /= y_sum` smoothing phase
/// shared by `symmetric_kl` and `symmetric_kl_grad` (Python `z = 1e-11`).
fn symmetric_kl_smoothed(x: &[f32], y: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let z = 1e-11f32;
    let mut xv = x.to_vec();
    let mut yv = y.to_vec();

    let mut x_sum = 0.0f32;
    let mut y_sum = 0.0f32;
    for i in 0..xv.len() {
        xv[i] += z;
        x_sum += xv[i];
        yv[i] += z;
        y_sum += yv[i];
    }

    for i in 0..xv.len() {
        xv[i] /= x_sum;
        yv[i] /= y_sum;
    }

    (xv, yv)
}

// --------------------------------------------------------------------------
// Gaussian energy gradients (registered in named_distances_with_gradients)
// --------------------------------------------------------------------------

/// Python: `spherical_gaussian_energy_grad(x, y)` (registered under the name
/// `"spherical_gaussian_energy"`).
pub fn spherical_gaussian_energy_grad(
    x: &[f32],
    y: &[f32],
    _metric_args: &[f32],
) -> (f32, Vec<f32>) {
    let mu_1 = x[0] - y[0];
    let mu_2 = x[1] - y[1];

    let sigma = x[2].abs() + y[2].abs();
    let sign_sigma = np_sign(x[2]);

    let dist = (mu_1 * mu_1 + mu_2 * mu_2) / (2.0 * sigma)
        + sigma.ln()
        + (2.0 * std::f32::consts::PI).ln();
    let mut grad = vec![0.0f32; 3];

    grad[0] = mu_1 / sigma;
    grad[1] = mu_2 / sigma;
    grad[2] = sign_sigma * (1.0 / sigma - (mu_1 * mu_1 + mu_2 * mu_2) / (2.0 * sigma * sigma));

    (dist, grad)
}

/// Python: `diagonal_gaussian_energy_grad(x, y)` (registered under the name
/// `"diagonal_gaussian_energy"`).
pub fn diagonal_gaussian_energy_grad(
    x: &[f32],
    y: &[f32],
    _metric_args: &[f32],
) -> (f32, Vec<f32>) {
    let mu_1 = x[0] - y[0];
    let mu_2 = x[1] - y[1];

    let sigma_11 = x[2].abs() + y[2].abs();
    let sigma_12 = 0.0f32;
    let sigma_22 = x[3].abs() + y[3].abs();

    let det = sigma_11 * sigma_22;
    let sign_s1 = np_sign(x[2]);
    let sign_s2 = np_sign(x[3]);

    if det == 0.0 {
        // TODO: figure out the right thing to do here
        return (mu_1 * mu_1 + mu_2 * mu_2, vec![0.0, 0.0, 1.0, 1.0]);
    }

    let cross_term = 2.0 * sigma_12;
    let m_dist =
        sigma_22.abs() * (mu_1 * mu_1) - cross_term * mu_1 * mu_2 + sigma_11.abs() * (mu_2 * mu_2);

    let dist = (m_dist / det + det.abs().ln()) / 2.0 + (2.0 * std::f32::consts::PI).ln();
    let mut grad = vec![0.0f32; 6];

    grad[0] = (2.0 * sigma_22 * mu_1 - cross_term * mu_2) / (2.0 * det);
    grad[1] = (2.0 * sigma_11 * mu_2 - cross_term * mu_1) / (2.0 * det);
    grad[2] = sign_s1 * (sigma_22 * (det - m_dist) + det * mu_2 * mu_2) / (2.0 * det * det);
    grad[3] = sign_s2 * (sigma_11 * (det - m_dist) + det * mu_1 * mu_1) / (2.0 * det * det);

    (dist, grad)
}

/// Python: `gaussian_energy_grad(x, y)` (registered under the name
/// `"gaussian_energy"`).
///
/// HACK: Python mutates `x` and `y` in place (widths/heights clamped to be
/// positive, angle folded into [-pi, pi]); here the folded values are kept
/// in local copies because the API takes shared slices.
#[allow(clippy::many_single_char_names)] // single-char names kept from Python
pub fn gaussian_energy_grad(x: &[f32], y: &[f32], _metric_args: &[f32]) -> (f32, Vec<f32>) {
    let mut xv = x.to_vec();
    let mut yv = y.to_vec();

    let mu_1 = xv[0] - yv[0];
    let mu_2 = xv[1] - yv[1];

    // Ensure width are positive
    xv[2] = xv[2].abs();
    yv[2] = yv[2].abs();

    // Ensure heights are positive
    xv[3] = xv[3].abs();
    yv[3] = yv[3].abs();

    // Ensure angle is in range -pi,pi
    xv[4] = xv[4].sin().asin();
    yv[4] = yv[4].sin().asin();

    // Covariance entries for y
    let a = yv[2] * yv[4].cos() * yv[4].cos() + yv[3] * yv[4].sin() * yv[4].sin();
    let b = (yv[2] - yv[3]) * yv[4].sin() * yv[4].cos();
    let c = yv[3] * yv[4].cos() * yv[4].cos() + yv[2] * yv[4].sin() * yv[4].sin();

    // Sum of covariance matrices
    let sigma_11 = xv[2] * xv[4].cos() * xv[4].cos() + xv[3] * xv[4].sin() * xv[4].sin() + a;
    let sigma_12 = (xv[2] - xv[3]) * xv[4].sin() * xv[4].cos() + b;
    let sigma_22 = xv[2] * xv[4].sin() * xv[4].sin() + xv[3] * xv[4].cos() * xv[4].cos() + c;

    // Determinant of the sum of covariances
    let det_sigma = (sigma_11 * sigma_22 - sigma_12 * sigma_12).abs();
    let x_inv_sigma_y_numerator =
        sigma_22 * mu_1 * mu_1 - 2.0 * sigma_12 * mu_1 * mu_2 + sigma_11 * mu_2 * mu_2;

    if det_sigma < 1e-32 {
        return (mu_1 * mu_1 + mu_2 * mu_2, vec![0.0, 0.0, 1.0, 1.0, 0.0]);
    }

    let dist =
        x_inv_sigma_y_numerator / det_sigma + det_sigma.ln() + (2.0 * std::f32::consts::PI).ln();

    let mut grad = vec![0.0f32; 5];
    grad[0] = (2.0 * sigma_22 * mu_1 - 2.0 * sigma_12 * mu_2) / det_sigma;
    grad[1] = (2.0 * sigma_11 * mu_2 - 2.0 * sigma_12 * mu_1) / det_sigma;

    grad[2] = mu_2 * (mu_2 * xv[4].cos() * xv[4].cos() - mu_1 * xv[4].cos() * xv[4].sin());
    grad[2] += mu_1 * (mu_1 * xv[4].sin() * xv[4].sin() - mu_2 * xv[4].cos() * xv[4].sin());
    grad[2] *= det_sigma;
    grad[2] -= x_inv_sigma_y_numerator * xv[4].cos() * xv[4].cos() * sigma_22;
    grad[2] -= x_inv_sigma_y_numerator * xv[4].sin() * xv[4].sin() * sigma_11;
    grad[2] += x_inv_sigma_y_numerator * 2.0 * sigma_12 * xv[4].sin() * xv[4].cos();
    grad[2] /= det_sigma * det_sigma + 1e-8;

    grad[3] = mu_1 * (mu_1 * xv[4].cos() * xv[4].cos() - mu_2 * xv[4].cos() * xv[4].sin());
    grad[3] += mu_2 * (mu_2 * xv[4].sin() * xv[4].sin() - mu_1 * xv[4].cos() * xv[4].sin());
    grad[3] *= det_sigma;
    grad[3] -= x_inv_sigma_y_numerator * xv[4].sin() * xv[4].sin() * sigma_22;
    grad[3] -= x_inv_sigma_y_numerator * xv[4].cos() * xv[4].cos() * sigma_11;
    grad[3] -= x_inv_sigma_y_numerator * 2.0 * sigma_12 * xv[4].sin() * xv[4].cos();
    grad[3] /= det_sigma * det_sigma + 1e-8;

    grad[4] = (xv[3] - xv[2])
        * (2.0 * mu_1 * mu_2 * (2.0 * xv[4]).cos()
            - (mu_1 * mu_1 - mu_2 * mu_2) * (2.0 * xv[4]).sin());
    grad[4] *= det_sigma;
    grad[4] -= x_inv_sigma_y_numerator * (xv[3] - xv[2]) * (2.0 * xv[4]).sin() * sigma_22;
    grad[4] -= x_inv_sigma_y_numerator * (xv[2] - xv[3]) * (2.0 * xv[4]).sin() * sigma_11;
    grad[4] -= x_inv_sigma_y_numerator * 2.0 * sigma_12 * (xv[2] - xv[3]) * (2.0 * xv[4]).cos();
    grad[4] /= det_sigma * det_sigma + 1e-8;

    (dist, grad)
}

/// Python: `spherical_gaussian_grad(x, y)` — defined but not registered in
/// either Python dispatch dict; ported for completeness.
pub fn spherical_gaussian_grad(x: &[f32], y: &[f32], _metric_args: &[f32]) -> (f32, Vec<f32>) {
    let mu_1 = x[0] - y[0];
    let mu_2 = x[1] - y[1];

    let sigma = x[2] + y[2];
    let sigma_sign = np_sign(sigma);

    if sigma == 0.0 {
        return (10.0, vec![0.0, 0.0, -1.0]);
    }

    let dist = (mu_1 * mu_1 + mu_2 * mu_2) / sigma.abs()
        + 2.0 * sigma.abs().ln()
        + (2.0 * std::f32::consts::PI).ln();
    let mut grad = vec![0.0f32; 3];

    grad[0] = (2.0 * mu_1) / sigma.abs();
    grad[1] = (2.0 * mu_2) / sigma.abs();
    grad[2] = sigma_sign * (-(mu_1 * mu_1 + mu_2 * mu_2) / (sigma * sigma) + (2.0 / sigma.abs()));

    (dist, grad)
}

/// Distance-only wrapper around [`spherical_gaussian_energy_grad`].
///
/// NOTE: Python's `named_distances` does not register these embedding
/// "metrics"; the wrappers exist so [`named_distances_with_gradients`] can
/// return a `(MetricFn, GradFn)` pair for every name in the Python dict.
pub fn spherical_gaussian_energy(x: &[f32], y: &[f32], metric_args: &[f32]) -> f32 {
    spherical_gaussian_energy_grad(x, y, metric_args).0
}

/// Distance-only wrapper around [`diagonal_gaussian_energy_grad`]. See the
/// note on [`spherical_gaussian_energy`].
pub fn diagonal_gaussian_energy(x: &[f32], y: &[f32], metric_args: &[f32]) -> f32 {
    diagonal_gaussian_energy_grad(x, y, metric_args).0
}

/// Distance-only wrapper around [`gaussian_energy_grad`]. See the note on
/// [`spherical_gaussian_energy`].
pub fn gaussian_energy(x: &[f32], y: &[f32], metric_args: &[f32]) -> f32 {
    gaussian_energy_grad(x, y, metric_args).0
}

/// Distance-only wrapper around [`hyperboloid_grad`]. See the note on
/// [`spherical_gaussian_energy`].
pub fn hyperboloid(x: &[f32], y: &[f32], metric_args: &[f32]) -> f32 {
    hyperboloid_grad(x, y, metric_args).0
}

// --------------------------------------------------------------------------
// Special discrete distances -- where x and y are objects, not vectors
// --------------------------------------------------------------------------

/// Python: `categorical_distance(x, y)` — scalar categories packed as
/// 1-element f32 slices; 0.0 if equal, 1.0 otherwise. Empty slices compare
/// as equal (0.0).
pub fn categorical_distance(x: &[f32], y: &[f32], _metric_args: &[f32]) -> f32 {
    match (x.first(), y.first()) {
        (Some(&a), Some(&b)) if a == b => 0.0,
        (Some(_), Some(_)) => 1.0,
        _ => 0.0,
    }
}

/// Python: `hierarchical_categorical_distance(x, y, cat_hierarchy)`.
///
/// Scalar categories are packed as 1-element f32 slices. The
/// `cat_hierarchy` (Python: a list of dicts mapping category value to
/// cluster id) is passed as a flattened f32 encoding:
///
/// ```text
/// metric_args = [
///     n_levels,
///     size_0, cat_0, id_0, cat_1, id_1, ..., cat_{size_0-1}, id_{size_0-1},
///     size_1, cat_0, id_0, ...,
///     ...
/// ]
/// ```
///
/// i.e. the number of levels, then for each level the number of entries
/// followed by `(category_value, cluster_id)` pairs. Returns
/// `level / n_levels` at the first level where both categories map to the
/// same cluster id, else 1.0. HACK: a category missing from a level raises
/// `KeyError` in Python; here it is treated as "distinct at that level".
pub fn hierarchical_categorical_distance(x: &[f32], y: &[f32], metric_args: &[f32]) -> f32 {
    let Some(&n_levels) = metric_args.first() else {
        return 1.0;
    };
    let n_levels = n_levels as usize;

    let cat_x = x.first().copied();
    let cat_y = y.first().copied();

    let mut pos = 1usize;
    for level in 0..n_levels {
        if pos >= metric_args.len() {
            return 1.0;
        }
        let size = metric_args[pos] as usize;
        pos += 1;
        let end = pos.saturating_add(size.saturating_mul(2));
        if end > metric_args.len() {
            return 1.0;
        }

        let mut id_x = None;
        let mut id_y = None;
        for k in 0..size {
            let cat = metric_args[pos + 2 * k];
            let id = metric_args[pos + 2 * k + 1];
            if Some(cat) == cat_x {
                id_x = Some(id);
            }
            if Some(cat) == cat_y {
                id_y = Some(id);
            }
        }

        if let (Some(a), Some(b)) = (id_x, id_y) {
            if a == b {
                return level as f32 / n_levels as f32;
            }
        }

        pos = end;
    }

    1.0
}

/// Python: `ordinal_distance(x, y, support_size=1.0)`.
///
/// Scalar values packed as 1-element f32 slices;
/// `metric_args = [support_size]` (empty slice selects the Python default
/// of 1.0).
pub fn ordinal_distance(x: &[f32], y: &[f32], metric_args: &[f32]) -> f32 {
    let support_size = metric_args.first().copied().unwrap_or(1.0);
    (x[0] - y[0]).abs() / support_size
}

/// Python: `count_distance(x, y, poisson_lambda=1.0, normalisation=1.0)`.
///
/// Scalar counts packed as 1-element f32 slices;
/// `metric_args = [poisson_lambda, normalisation]` matching Python's kwds
/// ordering (empty slice selects the Python defaults of 1.0, 1.0).
pub fn count_distance(x: &[f32], y: &[f32], metric_args: &[f32]) -> f32 {
    let poisson_lambda = metric_args.first().copied().unwrap_or(1.0);
    let normalisation = metric_args.get(1).copied().unwrap_or(1.0);

    let lo = x[0].min(y[0]) as i64;
    let hi = x[0].max(y[0]) as i64;

    let log_lambda = poisson_lambda.ln();

    let mut log_k_factorial = 0.0f32;
    if (2..10).contains(&lo) {
        for k in 2..lo {
            log_k_factorial += (k as f32).ln();
        }
    } else if lo >= 10 {
        log_k_factorial = approx_log_gamma((lo + 1) as f32);
    }

    let mut result = 0.0f32;

    for k in lo..hi {
        result += k as f32 * log_lambda - poisson_lambda - log_k_factorial;
        log_k_factorial += (k as f32).ln();
    }

    result / normalisation
}

/// Python: `levenshtein(x, y, normalisation=1.0, max_distance=20)` — the
/// Levenshtein (edit) distance between two strings using dynamic
/// programming.
///
/// Strings are packed as vectors of f32 character codes (`x[i]` is the code
/// point of the i-th character);
/// `metric_args = [normalisation, max_distance]` (empty slice selects the
/// Python defaults of 1.0 and 20.0).
///
/// NOTE: the Python version normalises via scipy levenshtein tables; this
/// is a pure-Rust dynamic-programming Levenshtein with identical results
/// for integer-valued distances (the normalised values coincide).
pub fn levenshtein(x: &[f32], y: &[f32], metric_args: &[f32]) -> f32 {
    let normalisation = metric_args.first().copied().unwrap_or(1.0);
    let max_distance = metric_args.get(1).copied().unwrap_or(20.0);

    let x_len = x.len();
    let y_len = y.len();

    if (x_len as f32 - y_len as f32).abs() > max_distance {
        return max_distance / normalisation;
    }

    if x_len == 0 {
        return y_len as f32 / normalisation;
    }
    if y_len == 0 {
        return x_len as f32 / normalisation;
    }

    let mut v0: Vec<f32> = (0..=y_len).map(|i| i as f32).collect();
    let mut v1 = vec![0.0f32; y_len + 1];

    for (i, &xi) in x.iter().enumerate() {
        // First column: cost of deleting all chars up to i
        v1[0] = (i + 1) as f32;

        for (j, &yi) in y.iter().enumerate() {
            let deletion_cost = v0[j + 1] + 1.0;
            let insertion_cost = v1[j] + 1.0;
            let substitution_cost = v0[j] + if xi == yi { 0.0 } else { 1.0 };

            v1[j + 1] = deletion_cost.min(insertion_cost).min(substitution_cost);
        }

        std::mem::swap(&mut v0, &mut v1);

        if v0.iter().copied().fold(f32::INFINITY, f32::min) > max_distance {
            return max_distance / normalisation;
        }
    }

    v0[y_len] / normalisation
}

/// Python: `levenshtein_myers_ascii(x, y, normalisation=1.0, max_distance=20)`
/// (registered under the name `"myers"`) — the Levenshtein (edit) distance
/// between two ASCII strings using Myers' bit-parallel algorithm.
///
/// Strings are packed as vectors of f32 character codes;
/// `metric_args = [normalisation, max_distance]` as in [`levenshtein`].
/// HACK: Python computes `1 << (x_len - 1)` with a negative shift when
/// `x_len == 0` (numba does not raise); here an explicit early return
/// produces the same value Python ends up with (`y_len / normalisation`).
pub fn levenshtein_myers_ascii(x: &[f32], y: &[f32], metric_args: &[f32]) -> f32 {
    let normalisation = metric_args.first().copied().unwrap_or(1.0);
    let max_distance = metric_args.get(1).copied().unwrap_or(20.0);

    let x_len = x.len();
    let y_len = y.len();

    if (x_len as f32 - y_len as f32).abs() > max_distance {
        return max_distance / normalisation;
    }

    if x_len == 0 {
        return y_len as f32 / normalisation;
    }

    // Myers' bit-parallel algorithm is limited to word size
    // fall back to levenshtein if words are large
    if x_len > 63 || y_len > 63 {
        return levenshtein(x, y, metric_args);
    }

    // Peq[c]: bitmask with bit i set where x[i] == character c
    let mut peq = [0u64; 128];
    for (i, &c) in x.iter().enumerate() {
        let c = c as u32 as usize;
        if c < 128 {
            peq[c] |= 1 << i;
        }
    }

    // Pv: positive vertical differences (initially all 1s)
    let mut pv: u64 = (1 << x_len) - 1;

    // Mv: negative vertical differences (initially all 0s)
    let mut mv: u64 = 0;

    // Initial edit distance: deleting all characters from x
    let mut score: i64 = x_len as i64;

    // Mask for the highest bit (row x_len - 1)
    let top_bit: u64 = 1 << (x_len - 1);

    for &c in y {
        let c = c as u32 as usize;
        let eq: u64 = if c < 128 { peq[c] } else { 0 };

        let xv = eq | mv;
        let xh = (((xv & pv).wrapping_add(pv)) ^ pv) | xv;

        let mut ph = mv | !(xh | pv);
        let mut mh = pv & xh;

        // Update score using the highest bit
        if ph & top_bit != 0 {
            score += 1;
        } else if mh & top_bit != 0 {
            score -= 1;
        }

        // Prepare for next column
        ph = (ph << 1) | 1;
        mh <<= 1;

        pv = mh | !(xh | ph);
        mv = ph & xh;
    }

    if score as f32 > max_distance {
        return max_distance / normalisation;
    }

    score as f32 / normalisation
}

// --------------------------------------------------------------------------
// Name dispatch
// --------------------------------------------------------------------------

/// Python: `umap.distances.named_distances` — resolve a metric by name.
/// Returns `None` for unknown names.
pub fn named_distances(name: &str) -> Option<MetricFn> {
    Some(match name {
        // general minkowski distances
        "euclidean" | "l2" => euclidean,
        // NOTE: "sqeuclidean" is a historical alias (present in older
        // umap-learn releases, absent from the 0.5.12 dict) required by the
        // port contract; it resolves to the euclidean metric as before.
        "sqeuclidean" => euclidean,
        "manhattan" | "taxicab" | "l1" => manhattan,
        "chebyshev" | "linfinity" | "linfty" | "linf" => chebyshev,
        "minkowski" => minkowski,
        "poincare" => poincare,
        // Standardised/weighted distances
        "seuclidean" | "standardised_euclidean" => standardised_euclidean,
        "wminkowski" | "weighted_minkowski" => weighted_minkowski,
        "mahalanobis" => mahalanobis,
        // Other distances
        "canberra" => canberra,
        "cosine" => cosine,
        "correlation" => correlation,
        "hellinger" => hellinger,
        "softmax_hellinger" => softmax_hellinger,
        "haversine" => haversine,
        "braycurtis" => bray_curtis,
        "ll_dirichlet" => ll_dirichlet,
        "symmetric_kl" => symmetric_kl,
        // Binary distances
        "hamming" => hamming,
        "jaccard" => jaccard,
        "dice" => dice,
        "matching" => matching,
        "kulsinski" => kulsinski,
        "rogerstanimoto" => rogers_tanimoto,
        "russellrao" => russellrao,
        "sokalsneath" => sokal_sneath,
        "sokalmichener" => sokal_michener,
        "yule" => yule,
        // Special discrete distances
        "categorical" => categorical_distance,
        "ordinal" => ordinal_distance,
        "hierarchical_categorical" => hierarchical_categorical_distance,
        "count" => count_distance,
        "string" => levenshtein,
        "myers" => levenshtein_myers_ascii,
        _ => return None,
    })
}

/// Python: `umap.distances.named_distances_with_gradients` — returns the
/// metric and its analytic gradient pair. Returns `None` for unknown names.
///
/// NOTE: for the embedding metrics (`spherical_gaussian_energy`,
/// `diagonal_gaussian_energy`, `gaussian_energy`, `hyperboloid`) the Python
/// dict only registers gradient functions; the metric half here is a
/// distance-only wrapper around the gradient.
pub fn named_distances_with_gradients(name: &str) -> Option<(MetricFn, GradFn)> {
    Some(match name {
        // general minkowski distances
        "euclidean" | "l2" => (euclidean, euclidean_grad),
        "manhattan" | "taxicab" | "l1" => (manhattan, manhattan_grad),
        "chebyshev" | "linfinity" | "linfty" | "linf" => (chebyshev, chebyshev_grad),
        "minkowski" => (minkowski, minkowski_grad),
        // Standardised/weighted distances
        "seuclidean" | "standardised_euclidean" => {
            (standardised_euclidean, standardised_euclidean_grad)
        }
        "wminkowski" | "weighted_minkowski" => (weighted_minkowski, weighted_minkowski_grad),
        "mahalanobis" => (mahalanobis, mahalanobis_grad),
        // Other distances
        "canberra" => (canberra, canberra_grad),
        "cosine" => (cosine, cosine_grad),
        "correlation" => (correlation, correlation_grad),
        "hellinger" => (hellinger, hellinger_grad),
        "softmax_hellinger" => (softmax_hellinger, softmax_hellinger_grad),
        "haversine" => (haversine, haversine_grad),
        "braycurtis" => (bray_curtis, bray_curtis_grad),
        "symmetric_kl" => (symmetric_kl, symmetric_kl_grad),
        // Special embeddings
        "spherical_gaussian_energy" => (spherical_gaussian_energy, spherical_gaussian_energy_grad),
        "diagonal_gaussian_energy" => (diagonal_gaussian_energy, diagonal_gaussian_energy_grad),
        "gaussian_energy" => (gaussian_energy, gaussian_energy_grad),
        "hyperboloid" => (hyperboloid, hyperboloid_grad),
        _ => return None,
    })
}

/// Python: `umap.distances.DISCRETE_METRICS` — the list of discrete metric
/// names.
pub fn discrete_metrics() -> &'static [&'static str] {
    &[
        "categorical",
        "hierarchical_categorical",
        "ordinal",
        "count",
        "string",
        "myers",
    ]
}

/// Python: `umap.distances.SPECIAL_METRICS` (name entries only; the Python
/// tuple also lists the corresponding function objects).
pub fn special_metrics() -> &'static [&'static str] {
    &["hellinger", "ll_dirichlet", "symmetric_kl", "poincare"]
}

// --------------------------------------------------------------------------
// Pairwise distances
// --------------------------------------------------------------------------

/// Materialise all rows of a matrix as owned contiguous slices (rows of an
/// `Array2` are contiguous; non-contiguous views are gathered).
fn rows_as_slices(x: ArrayView2<'_, f32>) -> Vec<Vec<f32>> {
    x.rows().into_iter().map(|r| r.to_vec()).collect()
}

/// A simple dense pairwise distance matrix over rows of `x` and `y` using
/// `metric` (replacement for sklearn `pairwise_distances` + numba
/// dispatch). Rows are processed in parallel with rayon.
pub fn pairwise_distances(
    x: ArrayView2<'_, f32>,
    y: ArrayView2<'_, f32>,
    metric: MetricFn,
    metric_args: &[f32],
) -> Array2<f32> {
    let n = x.nrows();
    let m = y.nrows();

    let xs = rows_as_slices(x);
    let ys = rows_as_slices(y);
    let ys_ref: &[Vec<f32>] = &ys;

    let rows: Vec<f32> = (0..n)
        .into_par_iter()
        .flat_map_iter(|i| {
            let xi = &xs[i];
            (0..m).map(move |j| metric(xi, &ys_ref[j], metric_args))
        })
        .collect();

    Array2::from_shape_vec((n, m), rows).unwrap_or_else(|_| Array2::zeros((n, m)))
}

/// Python: `pairwise_special_metric(X, metric=...)` (the
/// `parallel_special_metric` symmetric path): fills `i < j` and mirrors;
/// the diagonal is left at 0.
pub fn pairwise_special_metric(
    x: ArrayView2<'_, f32>,
    metric: MetricFn,
    metric_args: &[f32],
) -> Array2<f32> {
    let n = x.nrows();
    let mut result = Array2::<f32>::zeros((n, n));

    for i in 0..n {
        let row_i = x.row(i);
        let xi = row_i.as_slice().unwrap_or(&[]);
        for j in (i + 1)..n {
            let row_j = x.row(j);
            let xj = row_j.as_slice().unwrap_or(&[]);
            let d = metric(xi, xj, metric_args);
            result[(i, j)] = d;
            result[(j, i)] = d;
        }
    }

    result
}
