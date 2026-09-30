//! Ported from Python: umap/layouts.py.
//!
//! SGD layout optimizers. The numba `parallel=True` Hogwild! semantics are
//! reproduced with raw-pointer shared mutation under rayon: parallel runs
//! have nondeterministic float accumulation (same as Python), while serial
//! runs are fully deterministic. The per-sample RNG-state replication trick
//! (`rng_state_per_sample`) is ported exactly, including the f64-bit-pattern
//! seeding from the first embedding coordinate (layouts.py:366-368).
//!
//! Divergence: the aligned kernel (`optimize_layout_aligned_euclidean`) is
//! out of scope (`AlignedUMAP` not ported).
// The kernels mirror the numba-compiled Python structure (single-char loop
// variables, index-based loops, many flag parameters matching the Python
// signatures); pedantic lints for those are allowed module-wide.
#![allow(
    clippy::many_single_char_names,
    clippy::needless_range_loop,
    clippy::struct_excessive_bools,
    clippy::type_complexity
)]

use crate::rng::tau_rand_int;

/// A f64 value clamped into [-4.0, 4.0]. Python: `layouts.clip`.
#[inline]
fn clip(val: f64) -> f64 {
    val.clamp(-4.0, 4.0)
}

/// Reduced (squared) Euclidean distance between two rows, computed in f32.
/// Python: `layouts.rdist` (numba fastmath f32 kernel).
#[inline]
fn rdist(x_ptr: *const f32, y_ptr: *const f32, dim: usize) -> f32 {
    // SAFETY: caller guarantees both pointers are valid for `dim` reads.
    let mut result = 0.0f32;
    for i in 0..dim {
        // SAFETY: within bounds per caller contract.
        let diff = unsafe { *x_ptr.add(i) } - unsafe { *y_ptr.add(i) };
        result += diff * diff;
    }
    result
}

/// Read one element from a raw row buffer.
#[inline]
#[allow(unsafe_code)]
unsafe fn fget(ptr: *const f32, idx: usize) -> f32 {
    unsafe { *ptr.add(idx) }
}

/// Add a delta to one element of a raw row buffer.
#[inline]
#[allow(unsafe_code)]
unsafe fn fadd(ptr: *mut f32, idx: usize, delta: f32) {
    unsafe {
        *ptr.add(idx) += delta;
    }
}

/// `tau_rand_int` over the per-vertex state stored flat in an i64 buffer.
#[inline]
#[allow(unsafe_code)]
unsafe fn tau_rand_int_at(ptr: *mut i64, vertex: usize) -> i32 {
    let state = unsafe { &mut *ptr.add(3 * vertex).cast::<[i64; 3]>() };
    tau_rand_int(state)
}

/// Python modulo semantics for the negative-sampling index.
#[inline]
fn python_mod(value: i32, n: usize) -> usize {
    value.rem_euclid(n as i32) as usize
}

/// The tail embedding: either the same buffer as the head (the `fit` case,
/// Python passes the same array twice) or a distinct mutable buffer
/// (the `transform` case with `move_other=True`).
pub enum TailEmbedding<'a> {
    /// `tail_embedding` is `head_embedding` (aliased buffer, Python semantics).
    Same,
    /// A separate array (may be mutated when `move_other` is true).
    Distinct(&'a mut ndarray::Array2<f32>),
}

/// Auxiliary data for the densMAP objective.
/// Python: `densmap_kwds` dict entries used by `optimize_layout_euclidean`.
#[derive(Debug, Clone)]
pub struct DensmapKwds {
    /// Python densMAP dictionary key `lambda`
    pub lambda: f64,
    /// Python densMAP dictionary key `frac`
    pub frac: f64,
    /// Python densMAP dictionary key `var_shift`
    pub var_shift: f64,
    /// Python densMAP dictionary key `R` — per-vertex standardized radii.
    pub r: Vec<f32>,
    /// Python densMAP dictionary key `mu` — per-edge membership weights (graph.data).
    pub mu: Vec<f32>,
    /// Python densMAP dictionary key `mu_sum` — per-vertex sums of membership weights.
    pub mu_sum: Vec<f32>,
}

/// A raw pointer wrapper asserting Send+Sync for Hogwild!-style shared
/// mutation. SAFETY: sharing `*mut T` across threads is intentionally
/// unsynchronized (matching numba parallel=True semantics in the Python
/// original); the data races on f32 words are the documented Hogwild!
/// behavior of the algorithm.
#[allow(unsafe_code)]
struct SendPtr<T>(*mut T);
// SAFETY: see struct doc — deliberate Hogwild! shared mutation.
unsafe impl<T> Send for SendPtr<T> {}
// SAFETY: see struct doc — deliberate Hogwild! shared mutation.
unsafe impl<T> Sync for SendPtr<T> {}

impl<T> SendPtr<T> {
    #[inline]
    fn get(&self) -> *mut T {
        self.0
    }
}

impl<T> Clone for SendPtr<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Copy for SendPtr<T> {}

/// Immutable parameters shared by every edge in the euclidean kernel.
#[allow(unsafe_code)]
struct EuclideanParams<'a> {
    head_ptr: SendPtr<f32>,
    tail_ptr: SendPtr<f32>,
    head: &'a [i32],
    tail: &'a [i32],
    n_vertices: usize,
    epochs_per_sample: &'a [f64],
    a: f64,
    b: f64,
    rng_state_per_sample: SendPtr<i64>,
    gamma: f64,
    dim: usize,
    move_other: bool,
    alpha: f64,
    epochs_per_negative_sample: &'a [f64],
    n: usize,
    densmap_flag: bool,
    dens_phi_sum: &'a [f32],
    dens_re_sum: &'a [f32],
    dens_re_cov: f64,
    dens_re_std: f64,
    dens_re_mean: f64,
    dens_lambda: f64,
    dens_r: &'a [f32],
    dens_mu: &'a [f32],
    dens_mu_tot: f64,
}

/// Python: `_optimize_layout_euclidean_single_epoch` inner loop body for one
/// edge. `epoch_of_next_sample[i]` and `epoch_of_next_negative_sample[i]` are
/// passed separately so the parallel driver can split them per edge.
#[allow(unsafe_code)]
unsafe fn euclidean_process_edge(
    p: &EuclideanParams<'_>,
    i: usize,
    epoch_of_next_sample: &mut f64,
    epoch_of_next_negative_sample: &mut f64,
) {
    if *epoch_of_next_sample > p.n as f64 {
        return;
    }
    let j = p.head[i] as usize;
    let k = p.tail[i] as usize;

    let cur = p.head_ptr.get().add(j * p.dim);
    let oth = p.tail_ptr.get().add(k * p.dim);

    let dist_squared = f64::from(rdist(cur, oth, p.dim));

    let mut grad_cor_coeff = 0.0f64;
    if p.densmap_flag {
        // Python: densMAP density-correlation term (layouts.py:102-134).
        let phi = 1.0 / (1.0 + p.a * dist_squared.powf(p.b));
        let dphi_term =
            p.a * p.b * dist_squared.powf(p.b - 1.0) / (1.0 + p.a * dist_squared.powf(p.b));

        let q_jk = phi / f64::from(p.dens_phi_sum[k]);
        let q_kj = phi / f64::from(p.dens_phi_sum[j]);

        let drk =
            q_jk * ((1.0 - p.b * (1.0 - phi)) / f64::from(p.dens_re_sum[k]).exp() + dphi_term);
        let drj =
            q_kj * ((1.0 - p.b * (1.0 - phi)) / f64::from(p.dens_re_sum[j]).exp() + dphi_term);

        let re_std_sq = p.dens_re_std * p.dens_re_std;
        let weight_k = f64::from(p.dens_r[k])
            - p.dens_re_cov * (f64::from(p.dens_re_sum[k]) - p.dens_re_mean) / re_std_sq;
        let weight_j = f64::from(p.dens_r[j])
            - p.dens_re_cov * (f64::from(p.dens_re_sum[j]) - p.dens_re_mean) / re_std_sq;

        grad_cor_coeff = p.dens_lambda * p.dens_mu_tot * (weight_k * drk + weight_j * drj)
            / (f64::from(p.dens_mu[i]) * p.dens_re_std)
            / p.n_vertices as f64;
    }

    let grad_coeff = if dist_squared > 0.0 {
        let mut gc = -2.0 * p.a * p.b * dist_squared.powf(p.b - 1.0);
        gc /= p.a * dist_squared.powf(p.b) + 1.0;
        gc
    } else {
        0.0
    };

    for d in 0..p.dim {
        let mut grad_d = clip(grad_coeff * (f64::from(fget(cur, d)) - f64::from(fget(oth, d))));
        if p.densmap_flag {
            grad_d +=
                clip(2.0 * grad_cor_coeff * (f64::from(fget(cur, d)) - f64::from(fget(oth, d))));
        }
        fadd(cur, d, (grad_d * p.alpha) as f32);
        if p.move_other {
            fadd(oth, d, (-grad_d * p.alpha) as f32);
        }
    }

    *epoch_of_next_sample += p.epochs_per_sample[i];

    let n_neg_samples =
        ((p.n as f64 - *epoch_of_next_negative_sample) / p.epochs_per_negative_sample[i]) as i32;

    for _ in 0..n_neg_samples {
        let k = python_mod(
            tau_rand_int_at(p.rng_state_per_sample.get(), j),
            p.n_vertices,
        );
        let oth = p.tail_ptr.get().add(k * p.dim);
        let dist_squared = f64::from(rdist(cur, oth, p.dim));

        let grad_coeff = if dist_squared > 0.0 {
            2.0 * p.gamma * p.b / ((0.001 + dist_squared) * (p.a * dist_squared.powf(p.b) + 1.0))
        } else if j == k {
            continue;
        } else {
            0.0
        };

        for d in 0..p.dim {
            let grad_d = if grad_coeff > 0.0 {
                clip(grad_coeff * (f64::from(fget(cur, d)) - f64::from(fget(oth, d))))
            } else {
                0.0
            };
            fadd(cur, d, (grad_d * p.alpha) as f32);
        }
    }

    *epoch_of_next_negative_sample += f64::from(n_neg_samples) * p.epochs_per_negative_sample[i];
}

/// Python: `_optimize_layout_euclidean_densmap_epoch_init` — recompute the
/// per-vertex density statistics at the start of each densMAP epoch.
/// Ported serially (the Python version is numba-prange with shared-array
/// races; serial is deterministic and statistically equivalent).
fn densmap_epoch_init<S: ndarray::Data<Elem = f32>>(
    head_embedding: &ndarray::ArrayBase<S, ndarray::Ix2>,
    head: &[i32],
    tail: &[i32],
    a: f64,
    b: f64,
    re_sum: &mut [f32],
    phi_sum: &mut [f32],
) {
    let dim = head_embedding.ncols();
    re_sum.fill(0.0);
    phi_sum.fill(0.0);

    for (idx, (&j, &k)) in head.iter().zip(tail.iter()).enumerate() {
        let _ = idx;
        let (j, k) = (j as usize, k as usize);
        let cur = head_embedding.row(j);
        let oth = head_embedding.row(k);
        let dist_squared = f64::from(rdist(cur.as_ptr(), oth.as_ptr(), dim));
        let phi = 1.0 / (1.0 + a * dist_squared.powf(b));
        re_sum[j] += (phi * dist_squared) as f32;
        re_sum[k] += (phi * dist_squared) as f32;
        phi_sum[j] += phi as f32;
        phi_sum[k] += phi as f32;
    }

    let epsilon = 1e-8;
    for i in 0..re_sum.len() {
        re_sum[i] = (epsilon + (f64::from(re_sum[i]) / f64::from(phi_sum[i])).ln()) as f32;
    }
}

/// Python: `optimize_layout_euclidean`.
///
/// Improve an embedding using stochastic gradient descent to minimize the
/// fuzzy set cross entropy between the 1-skeletons of the high and low
/// dimensional fuzzy simplicial sets.
///
/// `checkpoints` mirrors Python's `n_epochs` being a list: when given,
/// snapshots at those epochs (plus the final embedding) are returned in
/// increasing epoch order.
#[allow(unsafe_code)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::fn_params_excessive_bools)]
pub fn optimize_layout_euclidean(
    head_embedding: &mut ndarray::Array2<f32>,
    tail_embedding: TailEmbedding<'_>,
    head: &[i32],
    tail: &[i32],
    n_epochs: usize,
    n_vertices: usize,
    epochs_per_sample: &[f64],
    a: f64,
    b: f64,
    rng_state: &mut [i64; 3],
    gamma: f64,
    initial_alpha: f64,
    negative_sample_rate: f64,
    parallel: bool,
    verbose: bool,
    densmap: bool,
    densmap_kwds: Option<&DensmapKwds>,
    checkpoints: Option<&[usize]>,
    move_other: bool,
) -> Vec<ndarray::Array2<f32>> {
    let dim = head_embedding.ncols();
    let mut alpha = initial_alpha;

    let epochs_per_negative_sample: Vec<f64> = epochs_per_sample
        .iter()
        .map(|&e| e / negative_sample_rate)
        .collect();
    let mut epoch_of_next_negative_sample = epochs_per_negative_sample.clone();
    let mut epoch_of_next_sample = epochs_per_sample.to_vec();

    let (dens_mu_tot, dens_lambda, dens_r, dens_mu, dens_frac, dens_var_shift) = match densmap_kwds
    {
        Some(kw) => (
            f64::from(kw.mu_sum.iter().sum::<f32>()) / 2.0,
            kw.lambda,
            kw.r.clone(),
            kw.mu.clone(),
            kw.frac,
            kw.var_shift,
        ),
        None => (0.0, 0.0, vec![0.0f32], vec![0.0f32], 0.0, 0.0),
    };
    let mut dens_phi_sum = vec![0.0f32; n_vertices];
    let mut dens_re_sum = vec![0.0f32; n_vertices];

    let mut embedding_list: Vec<ndarray::Array2<f32>> = Vec::new();

    // Python layouts.py:366-368: replicate the rng_state per sample and add
    // the int64 bit-pattern of the f64-cast first embedding coordinate.
    let mut rng_state_per_sample: Vec<[i64; 3]> = (0..head_embedding.nrows())
        .map(|j| {
            let coord_bits = f64::from(head_embedding[[j, 0]]).to_bits() as i64;
            [
                rng_state[0].wrapping_add(coord_bits),
                rng_state[1].wrapping_add(coord_bits),
                rng_state[2].wrapping_add(coord_bits),
            ]
        })
        .collect();

    // move_other is purely the explicit flag (Python default false); the fit
    // path passes true explicitly even for the aliased same-array case.
    let (head_ptr, tail_ptr) = match tail_embedding {
        TailEmbedding::Same => (
            SendPtr(head_embedding.as_mut_ptr()),
            SendPtr(head_embedding.as_mut_ptr()),
        ),
        TailEmbedding::Distinct(t) => {
            assert_eq!(t.ncols(), dim, "tail embedding width mismatch");
            (
                SendPtr(head_embedding.as_mut_ptr()),
                SendPtr(t.as_mut_ptr()),
            )
        }
    };
    let rng_ptr = SendPtr(rng_state_per_sample.as_mut_ptr().cast::<i64>());
    let dens_mu_tot_base = dens_mu_tot;

    for n in 0..n_epochs {
        let densmap_flag =
            densmap && dens_lambda > 0.0 && ((n + 1) as f64 / n_epochs as f64) > (1.0 - dens_frac);

        let (dens_re_std, dens_re_mean, dens_re_cov) = if densmap_flag {
            // Reconstruct the head embedding view for the serial init pass.
            // SAFETY: head_ptr stays valid for the duration of the call;
            // embeddings are not reallocated during optimization.
            let head_len = head_embedding.nrows() * dim;
            let head_view = unsafe {
                ndarray::ArrayView2::from_shape(
                    (head_embedding.nrows(), dim),
                    std::slice::from_raw_parts(head_ptr.get(), head_len),
                )
                .expect("head embedding shape")
            };
            densmap_epoch_init(
                &head_view,
                head,
                tail,
                a,
                b,
                &mut dens_re_sum,
                &mut dens_phi_sum,
            );
            let var = variance(
                &dens_re_sum
                    .iter()
                    .map(|&v| f64::from(v))
                    .collect::<Vec<_>>(),
            );
            let std = (var + dens_var_shift).sqrt();
            let mean = dens_re_sum.iter().map(|&v| f64::from(v)).sum::<f64>() / n_vertices as f64;
            let cov = dens_re_sum
                .iter()
                .zip(dens_r.iter())
                .map(|(&a_, &b_)| f64::from(a_) * f64::from(b_))
                .sum::<f64>()
                / (n_vertices - 1) as f64;
            (std, mean, cov)
        } else {
            (0.0, 0.0, 0.0)
        };

        let p = EuclideanParams {
            head_ptr,
            tail_ptr,
            head,
            tail,
            n_vertices,
            epochs_per_sample,
            a,
            b,
            rng_state_per_sample: rng_ptr,
            gamma,
            dim,
            move_other,
            alpha,
            epochs_per_negative_sample: &epochs_per_negative_sample,
            n,
            densmap_flag,
            dens_phi_sum: &dens_phi_sum,
            dens_re_sum: &dens_re_sum,
            dens_re_cov,
            dens_re_std,
            dens_re_mean,
            dens_lambda,
            dens_r: &dens_r,
            dens_mu: &dens_mu,
            dens_mu_tot: dens_mu_tot_base,
        };

        if parallel {
            // Hogwild!-style: edges processed in parallel with shared
            // mutation of the embedding buffers (nondeterministic floats,
            // deterministic RNG — mirrors numba parallel=True).
            use rayon::prelude::*;
            epoch_of_next_sample
                .par_iter_mut()
                .enumerate()
                .zip(epoch_of_next_negative_sample.par_iter_mut())
                .for_each(|((i, eos), eons)| {
                    // SAFETY: each closure has exclusive access to its edge
                    // epoch slots; embedding-buffer aliasing across threads
                    // is the intended Hogwild! behavior (see module docs).
                    unsafe { euclidean_process_edge(&p, i, eos, eons) };
                });
        } else {
            for i in 0..epochs_per_sample.len() {
                // SAFETY: raw pointers valid for the call duration.
                unsafe {
                    euclidean_process_edge(
                        &p,
                        i,
                        &mut epoch_of_next_sample[i],
                        &mut epoch_of_next_negative_sample[i],
                    );
                }
            }
        }

        alpha = initial_alpha * (1.0 - (n as f64 / n_epochs as f64));

        if verbose && n % (n_epochs / 10).max(1) == 0 {
            eprintln!("\tcompleted {n} / {n_epochs} epochs");
        }

        if let Some(cps) = checkpoints {
            if cps.contains(&n) {
                embedding_list.push(head_embedding.clone());
            }
        }
    }

    if checkpoints.is_some() {
        embedding_list.push(head_embedding.clone());
    }

    embedding_list
}

/// Output metric for the generic/inverse layout kernels: returns
/// (distance, gradient wrt the first argument). Same contract as
/// `crate::distances::GradFn`.
pub type OutputMetricFn = fn(&[f32], &[f32], &[f32]) -> (f32, Vec<f32>);

/// Resolve a named metric (with gradients) to an output-metric function for
/// the generic/inverse SGD kernels (Python `named_distances_with_gradients`).
#[must_use]
pub fn named_output_metric(name: &str) -> Option<OutputMetricFn> {
    crate::distances::named_distances_with_gradients(name).map(|(_, grad)| grad)
}

/// Python: `_optimize_layout_generic_single_epoch` — serial kernel for
/// non-euclidean output metrics.
#[allow(unsafe_code)]
fn generic_process_edge(
    head_ptr: *mut f32,
    tail_ptr: *mut f32,
    head: &[i32],
    tail: &[i32],
    i: usize,
    dim: usize,
    alpha: f64,
    move_other: bool,
    n: usize,
    epochs_per_sample: &[f64],
    epoch_of_next_sample: &mut f64,
    epoch_of_next_negative_sample: &mut f64,
    epochs_per_negative_sample: &[f64],
    rng_state_per_sample: *mut i64,
    n_vertices: usize,
    a: f64,
    b: f64,
    gamma: f64,
    output_metric: OutputMetricFn,
    output_metric_kwds: &[f32],
) {
    if *epoch_of_next_sample > n as f64 {
        return;
    }
    let j = head[i] as usize;
    let k = tail[i] as usize;

    // Copy rows for the output metric calls (view semantics: fresh copy per
    // iteration matches Python reading the live arrays).
    let cur_row: Vec<f32> = (0..dim)
        .map(|d| unsafe { *head_ptr.add(j * dim + d) })
        .collect();
    let oth_row: Vec<f32> = (0..dim)
        .map(|d| unsafe { *tail_ptr.add(k * dim + d) })
        .collect();

    let (dist_output, grad_dist_output) = output_metric(&cur_row, &oth_row, output_metric_kwds);
    let (_, rev_grad) = output_metric(&oth_row, &cur_row, output_metric_kwds);

    let dist_output = f64::from(dist_output);
    let w_l = if dist_output > 0.0 {
        (1.0 + a * dist_output.powf(2.0 * b)).powi(-1)
    } else {
        1.0
    };
    let grad_coeff = 2.0 * b * (w_l - 1.0) / (dist_output + 1e-6);

    for d in 0..dim {
        let grad_d = clip(grad_coeff * f64::from(grad_dist_output[d]));
        unsafe {
            *head_ptr.add(j * dim + d) += (grad_d * alpha) as f32;
        }
        if move_other {
            let grad_d = clip(grad_coeff * f64::from(rev_grad[d]));
            unsafe {
                *tail_ptr.add(k * dim + d) += (grad_d * alpha) as f32;
            }
        }
    }

    *epoch_of_next_sample += epochs_per_sample[i];

    let n_neg_samples =
        ((n as f64 - *epoch_of_next_negative_sample) / epochs_per_negative_sample[i]) as i32;

    for _ in 0..n_neg_samples {
        let k = python_mod(
            unsafe { tau_rand_int_at(rng_state_per_sample, j) },
            n_vertices,
        );
        let oth_row: Vec<f32> = (0..dim)
            .map(|d| unsafe { *tail_ptr.add(k * dim + d) })
            .collect();
        // Re-read the current row (it may have been updated this epoch).
        let cur_row: Vec<f32> = (0..dim)
            .map(|d| unsafe { *head_ptr.add(j * dim + d) })
            .collect();

        let (dist_output, grad_dist_output) = output_metric(&cur_row, &oth_row, output_metric_kwds);
        let dist_output = f64::from(dist_output);

        let w_l = if dist_output > 0.0 {
            (1.0 + a * dist_output.powf(2.0 * b)).powi(-1)
        } else if j == k {
            continue;
        } else {
            1.0
        };

        let grad_coeff = gamma * 2.0 * b * w_l / (dist_output + 1e-6);

        for d in 0..dim {
            let grad_d = clip(grad_coeff * f64::from(grad_dist_output[d]));
            unsafe {
                *head_ptr.add(j * dim + d) += (grad_d * alpha) as f32;
            }
        }
    }

    *epoch_of_next_negative_sample += f64::from(n_neg_samples) * epochs_per_negative_sample[i];
}

/// Python: `optimize_layout_generic` — layout optimization with an arbitrary
/// differentiable output metric.
#[allow(unsafe_code)]
#[allow(clippy::too_many_arguments)]
pub fn optimize_layout_generic(
    head_embedding: &mut ndarray::Array2<f32>,
    tail_embedding: TailEmbedding<'_>,
    head: &[i32],
    tail: &[i32],
    n_epochs: usize,
    n_vertices: usize,
    epochs_per_sample: &[f64],
    a: f64,
    b: f64,
    rng_state: &mut [i64; 3],
    gamma: f64,
    initial_alpha: f64,
    negative_sample_rate: f64,
    output_metric: OutputMetricFn,
    output_metric_kwds: &[f32],
    move_other: bool,
) {
    let dim = head_embedding.ncols();
    let mut alpha = initial_alpha;

    let epochs_per_negative_sample: Vec<f64> = epochs_per_sample
        .iter()
        .map(|&e| e / negative_sample_rate)
        .collect();
    let mut epoch_of_next_negative_sample = epochs_per_negative_sample.clone();
    let mut epoch_of_next_sample = epochs_per_sample.to_vec();

    // Python layouts.py:630-632 — same per-sample RNG replication.
    let mut rng_state_per_sample: Vec<[i64; 3]> = (0..head_embedding.nrows())
        .map(|j| {
            let coord_bits = f64::from(head_embedding[[j, 0]]).to_bits() as i64;
            [
                rng_state[0].wrapping_add(coord_bits),
                rng_state[1].wrapping_add(coord_bits),
                rng_state[2].wrapping_add(coord_bits),
            ]
        })
        .collect();

    // move_other is purely the explicit flag (Python default false); the fit
    // path passes true explicitly even for the aliased same-array case.
    let (head_ptr, tail_ptr) = match tail_embedding {
        TailEmbedding::Same => (head_embedding.as_mut_ptr(), head_embedding.as_mut_ptr()),
        TailEmbedding::Distinct(t) => (head_embedding.as_mut_ptr(), t.as_mut_ptr()),
    };
    let rng_ptr = rng_state_per_sample.as_mut_ptr().cast::<i64>();

    for n in 0..n_epochs {
        for i in 0..epochs_per_sample.len() {
            generic_process_edge(
                head_ptr,
                tail_ptr,
                head,
                tail,
                i,
                dim,
                alpha,
                move_other,
                n,
                epochs_per_sample,
                &mut epoch_of_next_sample[i],
                &mut epoch_of_next_negative_sample[i],
                &epochs_per_negative_sample,
                rng_ptr,
                n_vertices,
                a,
                b,
                gamma,
                output_metric,
                output_metric_kwds,
            );
        }
        alpha = initial_alpha * (1.0 - (n as f64 / n_epochs as f64));
    }
}

/// Python: `_optimize_layout_inverse_single_epoch` — serial kernel.
#[allow(unsafe_code)]
fn inverse_process_edge(
    head_ptr: *mut f32,
    tail_ptr: *mut f32,
    head: &[i32],
    tail: &[i32],
    i: usize,
    dim: usize,
    alpha: f64,
    move_other: bool,
    n: usize,
    epochs_per_sample: &[f64],
    epoch_of_next_sample: &mut f64,
    epoch_of_next_negative_sample: &mut f64,
    epochs_per_negative_sample: &[f64],
    rng_state: &mut [i64; 3],
    n_vertices: usize,
    weight: &[f32],
    sigmas: &[f32],
    rhos: &[f32],
    gamma: f64,
    output_metric: OutputMetricFn,
    output_metric_kwds: &[f32],
) {
    if *epoch_of_next_sample > n as f64 {
        return;
    }
    let j = head[i] as usize;
    let k = tail[i] as usize;

    let cur_row: Vec<f32> = (0..dim)
        .map(|d| unsafe { *head_ptr.add(j * dim + d) })
        .collect();
    let oth_row: Vec<f32> = (0..dim)
        .map(|d| unsafe { *tail_ptr.add(k * dim + d) })
        .collect();

    let (_dist_output, grad_dist_output) = output_metric(&cur_row, &oth_row, output_metric_kwds);

    let w_l = f64::from(weight[i]);
    let grad_coeff = -(1.0 / (w_l * f64::from(sigmas[k]) + 1e-6));

    for d in 0..dim {
        let grad_d = clip(grad_coeff * f64::from(grad_dist_output[d]));
        unsafe {
            *head_ptr.add(j * dim + d) += (grad_d * alpha) as f32;
        }
        if move_other {
            unsafe {
                *tail_ptr.add(k * dim + d) += (-grad_d * alpha) as f32;
            }
        }
    }

    *epoch_of_next_sample += epochs_per_sample[i];

    let n_neg_samples =
        ((n as f64 - *epoch_of_next_negative_sample) / epochs_per_negative_sample[i]) as i32;

    for _ in 0..n_neg_samples {
        let k = python_mod(tau_rand_int(rng_state), n_vertices);
        let oth_row: Vec<f32> = (0..dim)
            .map(|d| unsafe { *tail_ptr.add(k * dim + d) })
            .collect();
        let cur_row: Vec<f32> = (0..dim)
            .map(|d| unsafe { *head_ptr.add(j * dim + d) })
            .collect();

        let (dist_output, grad_dist_output) = output_metric(&cur_row, &oth_row, output_metric_kwds);

        let dist_output = f64::from(dist_output);
        let w_h =
            (-((dist_output - f64::from(rhos[k])).max(1e-6) / (f64::from(sigmas[k]) + 1e-6))).exp();
        let grad_coeff = -gamma * ((0.0 - w_h) / ((1.0 - w_h) * f64::from(sigmas[k]) + 1e-6));

        for d in 0..dim {
            let grad_d = clip(grad_coeff * f64::from(grad_dist_output[d]));
            unsafe {
                *head_ptr.add(j * dim + d) += (grad_d * alpha) as f32;
            }
        }
    }

    *epoch_of_next_negative_sample += f64::from(n_neg_samples) * epochs_per_negative_sample[i];
}

/// Python: `optimize_layout_inverse` — layout optimization for the inverse
/// transform objective.
#[allow(unsafe_code)]
#[allow(clippy::too_many_arguments)]
pub fn optimize_layout_inverse(
    head_embedding: &mut ndarray::Array2<f32>,
    tail_embedding: TailEmbedding<'_>,
    head: &[i32],
    tail: &[i32],
    weight: &[f32],
    sigmas: &[f32],
    rhos: &[f32],
    n_epochs: usize,
    n_vertices: usize,
    epochs_per_sample: &[f64],
    _a: f64,
    _b: f64,
    rng_state: &mut [i64; 3],
    gamma: f64,
    initial_alpha: f64,
    negative_sample_rate: f64,
    output_metric: OutputMetricFn,
    output_metric_kwds: &[f32],
    move_other: bool,
) {
    let dim = head_embedding.ncols();
    let mut alpha = initial_alpha;

    let epochs_per_negative_sample: Vec<f64> = epochs_per_sample
        .iter()
        .map(|&e| e / negative_sample_rate)
        .collect();
    let mut epoch_of_next_negative_sample = epochs_per_negative_sample.clone();
    let mut epoch_of_next_sample = epochs_per_sample.to_vec();

    // move_other is purely the explicit flag (Python default false); the fit
    // path passes true explicitly even for the aliased same-array case.
    let (head_ptr, tail_ptr) = match tail_embedding {
        TailEmbedding::Same => (head_embedding.as_mut_ptr(), head_embedding.as_mut_ptr()),
        TailEmbedding::Distinct(t) => (head_embedding.as_mut_ptr(), t.as_mut_ptr()),
    };

    for n in 0..n_epochs {
        for i in 0..epochs_per_sample.len() {
            inverse_process_edge(
                head_ptr,
                tail_ptr,
                head,
                tail,
                i,
                dim,
                alpha,
                move_other,
                n,
                epochs_per_sample,
                &mut epoch_of_next_sample[i],
                &mut epoch_of_next_negative_sample[i],
                &epochs_per_negative_sample,
                rng_state,
                n_vertices,
                weight,
                sigmas,
                rhos,
                gamma,
                output_metric,
                output_metric_kwds,
            );
        }
        alpha = initial_alpha * (1.0 - (n as f64 / n_epochs as f64));
    }
}

/// Population variance (Python: np.var).
fn variance(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        return 0.0;
    }
    let mean = xs.iter().sum::<f64>() / xs.len() as f64;
    xs.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / xs.len() as f64
}
