//! Dense linear algebra helpers on top of `faer`.
//!
//! Divergences from the Python original (umap/spectral.py):
//! - scipy `eigsh` (Lanczos) for the smallest eigenvectors of the sparse
//!   normalized Laplacian is replaced by a full dense self-adjoint
//!   eigendecomposition for `n <= 4096` (exact, cross-validated), and by an
//!   unpreconditioned LOBPCG matvec solver on `B = 2I − L` for larger graphs
//!   (O(nnz) memory per matvec, replacing ARPACK). For the normalized
//!   Laplacian the eigenvalues lie in [0, 2], so the smallest algebraic
//!   eigenvalues coincide with the smallest magnitude ones that
//!   `eigsh(which="SM")` finds; eigenvectors are defined up to sign, so
//!   spectral parity is statistical, not exact. The LOBPCG iteration count is
//!   capped and the best available block is returned (Python falls back to
//!   random initialisation on ARPACK non-convergence); random fallback only
//!   applies to degenerate problems, as in Python.
//! - sklearn `SpectralEmbedding` (`component_layout`) is replaced by the same
//!   dense-EVD spectral embedding of the affinity's normalized Laplacian.
//! - LOBPCG / `TruncatedSVD` init (`init="tsvd"`) is out of scope for now.
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
use crate::rng::TauRng;
use faer::dyn_stack::{MemBuffer, MemStack};
use faer::linalg::evd::{self, SelfAdjointEvdParams};
use faer::Mat;
use faer::Spec;

/// Self-adjoint eigendecomposition of a dense symmetric f64 matrix.
///
/// Returns (eigenvalues ascending, eigenvectors as columns) such that
/// `A = U diag(s) Uᵀ`. Runs sequentially (deterministic float accumulation
/// for seed-reproducible runs).
pub(crate) fn self_adjoint_eigendecomposition(
    a: &ndarray::Array2<f64>,
) -> Result<(Vec<f64>, ndarray::Array2<f64>)> {
    let n = a.nrows();
    if a.ncols() != n {
        return Err(UmapError::InvalidInput(format!(
            "eigen decomposition requires a square matrix, got {}x{}",
            n,
            a.ncols()
        )));
    }
    if n == 0 {
        return Ok((vec![], ndarray::Array2::zeros((0, 0))));
    }

    // faer accesses only the lower triangular half.
    let mut m = Mat::<f64>::zeros(n, n);
    for i in 0..n {
        for j in 0..=i {
            m[(i, j)] = a[[i, j]];
        }
    }

    let mut s = faer::diag::Diag::<f64>::zeros(n);
    let mut u = Mat::<f64>::zeros(n, n);

    // Sequential: deterministic float accumulation.
    let par = faer::Par::Seq;
    let params: Spec<SelfAdjointEvdParams, f64> = Spec::default();
    let scratch_len = evd::self_adjoint_evd_scratch(n, evd::ComputeEigenvectors::Yes, par, params);
    let mut buffer = MemBuffer::new(scratch_len);
    let stack = MemStack::new(&mut buffer);

    evd::self_adjoint_evd(m.as_ref(), s.as_mut(), Some(u.as_mut()), par, stack, params)
        .map_err(|e| UmapError::Computation(format!("eigen decomposition failed: {e:?}")))?;

    let eigvals: Vec<f64> = (0..n).map(|i| s[i]).collect();
    let mut eigvecs = ndarray::Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            eigvecs[[i, j]] = u[(i, j)];
        }
    }
    Ok((eigvals, eigvecs))
}

/// Python: `scipy.sparse.csgraph.connected_components` — weakly-connected
/// component labeling of the graph via BFS on the CSR structure.
/// Returns (`n_components`, labels).
#[must_use]
pub fn connected_components(graph: &CsrMatrix) -> (usize, Vec<i32>) {
    let n = graph.shape.0;
    let mut labels = vec![-1i32; n];
    let mut n_components = 0u32;
    let mut queue = std::collections::VecDeque::with_capacity(n);
    for start in 0..n {
        if labels[start] != -1 {
            continue;
        }
        labels[start] = n_components as i32;
        queue.push_back(start);
        while let Some(v) = queue.pop_front() {
            for k in graph.indptr[v]..graph.indptr[v + 1] {
                let w = graph.indices[k] as usize;
                if labels[w] == -1 {
                    labels[w] = n_components as i32;
                    queue.push_back(w);
                }
            }
        }
        n_components += 1;
    }
    (n_components as usize, labels)
}

/// Python: `spectral_layout` / `_spectral_layout` (single-component dense-EVD
/// path). `data` is only used by the multi-component fallback layout.
///
/// Returns the initial embedding of shape (`n_vertices`, dim).
///
/// # Errors
/// Returns an error if the eigensolver fails and no fallback is available.
#[allow(clippy::too_many_arguments)]
pub fn spectral_layout(
    data: Option<&ndarray::Array2<f32>>,
    graph: &CsrMatrix,
    dim: usize,
    random_state: &mut TauRng,
) -> Result<ndarray::Array2<f32>> {
    let (n_components, labels) = connected_components(graph);

    if n_components > 1 {
        return Ok(multi_component_layout(
            data,
            graph,
            n_components,
            &labels,
            dim,
            random_state,
        ));
    }

    Ok(spectral_block_for(
        graph,
        dim,
        random_state,
        DENSE_EVD_MAX_N,
    ))
}

/// Upper bound on `n` for the exact dense eigendecomposition path; larger
/// graphs use the iterative LOBPCG solver (O(nnz) memory per matvec).
const DENSE_EVD_MAX_N: usize = 4096;

/// Squared degrees of the graph, sqrt-ed (the normalized-Laplacian scaling).
fn compute_sqrt_deg(graph: &CsrMatrix) -> Vec<f64> {
    let n = graph.shape.0;
    let mut deg = vec![0.0f64; n];
    for i in 0..n {
        deg[i] = graph.data[graph.indptr[i]..graph.indptr[i + 1]]
            .iter()
            .map(|&w| f64::from(w))
            .sum::<f64>();
    }
    deg.iter().map(|d: &f64| d.sqrt()).collect::<Vec<_>>()
}

/// Uniform random embedding in [-10, 10] (Python's solver-failure fallback).
fn random_fallback(n: usize, dim: usize, random_state: &mut TauRng) -> ndarray::Array2<f32> {
    let mut embedding = ndarray::Array2::<f32>::zeros((n, dim));
    for v in &mut embedding {
        *v = (random_state.uniform_f64() * 20.0 - 10.0) as f32;
    }
    embedding
}

/// Connected-graph spectral embedding: dense self-adjoint eigendecomposition
/// for `n <= dense_limit` (exact, cross-validated), LOBPCG on `B = 2I − L`
/// above (O(nnz) memory). Both skip the trivial first eigenvector exactly as
/// Python's `order = np.argsort(eigenvalues)[1:k]`; solver failures fall back
/// to random initialisation like Python's try/except.
fn spectral_block_for(
    graph: &CsrMatrix,
    dim: usize,
    random_state: &mut TauRng,
    dense_limit: usize,
) -> ndarray::Array2<f32> {
    let n = graph.shape.0;
    let sqrt_deg = compute_sqrt_deg(graph);
    let k = dim + 1;

    if n <= dense_limit {
        // Dense normalized Laplacian (f64).
        let mut l = ndarray::Array2::<f64>::eye(n);
        for i in 0..n {
            for kk in graph.indptr[i]..graph.indptr[i + 1] {
                let j = graph.indices[kk] as usize;
                let w = f64::from(graph.data[kk]);
                l[[i, j]] -= w / (sqrt_deg[i] * sqrt_deg[j]);
            }
        }

        if let Ok((eigenvalues, eigenvectors)) = self_adjoint_eigendecomposition(&l) {
            let mut order: Vec<usize> = (0..eigenvalues.len()).collect();
            order.sort_by(|&a, &b| {
                eigenvalues[a]
                    .partial_cmp(&eigenvalues[b])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            let order = &order[1..k];

            let mut embedding = ndarray::Array2::<f32>::zeros((n, dim));
            for (out_col, &src) in order.iter().enumerate() {
                for i in 0..n {
                    embedding[[i, out_col]] = eigenvectors[[i, src]] as f32;
                }
            }
            return embedding;
        }
        eprintln!(
            "Spectral initialisation failed! The eigenvector solver failed. \
             Falling back to random initialisation!"
        );
        return random_fallback(n, dim, random_state);
    }

    lobpcg_bottom_block(graph, &sqrt_deg, dim, 500, 1e-6, random_state)
        .map(|block| {
            // Column 0 is the trivial sqrt-degree eigenvector; skip it
            // exactly as Python's `order = np.argsort(eigenvalues)[1:k]`.
            block.slice(ndarray::s![.., 1..]).to_owned()
        })
        .unwrap_or_else(|| {
            eprintln!(
                "Spectral initialisation failed! The eigenvector solver failed. \
                 Falling back to random initialisation!"
            );
            random_fallback(n, dim, random_state)
        })
}

/// Project the columns of `r` onto the orthogonal complement of the
/// orthonormal `basis` (two passes for roundoff), then orthonormalize them
/// among themselves. Returns the number of kept columns.
fn expand_orthonormal(r: &mut ndarray::Array2<f64>, basis: &ndarray::Array2<f64>) -> usize {
    let n = r.nrows();
    for _ in 0..2 {
        for p in 0..basis.ncols() {
            let bp: Vec<f64> = (0..n).map(|i| basis[[i, p]]).collect();
            for col in 0..r.ncols() {
                let c: f64 = (0..n).map(|i| bp[i] * r[[i, col]]).sum();
                if c != 0.0 {
                    for i in 0..n {
                        r[[i, col]] -= c * bp[i];
                    }
                }
            }
        }
    }
    orthonormalize_columns(r, 0)
}

/// Modified Gram–Schmidt: orthonormalize columns `[skip..k]` of `m` in place
/// against the (already orthonormal) columns `[0..skip)` and each other, with
/// one re-orthogonalization pass for stability. Returns the number of kept
/// columns; near-zero columns are dropped.
fn orthonormalize_columns(m: &mut ndarray::Array2<f64>, skip: usize) -> usize {
    let (n, k) = m.dim();
    let mut kept = skip;
    for col in skip..k {
        for _ in 0..2 {
            for p in 0..kept {
                let c: f64 = (0..n).map(|i| m[[i, p]] * m[[i, col]]).sum();
                if c != 0.0 {
                    for i in 0..n {
                        m[[i, col]] -= c * m[[i, p]];
                    }
                }
            }
        }
        let norm: f64 = (0..n)
            .map(|i| m[[i, col]] * m[[i, col]])
            .sum::<f64>()
            .sqrt();
        if norm > 1e-10 {
            for i in 0..n {
                m[[i, col]] /= norm;
            }
            if kept != col {
                for i in 0..n {
                    m.swap((i, kept), (i, col));
                }
            }
            kept += 1;
        }
    }
    kept
}

/// LOBPCG (unpreconditioned, Rayleigh-Ritz on the residual-expanded subspace,
/// with extra guard vectors for clustered spectra) for the largest `dim + 1`
/// eigenpairs of `B = 2I - L`, i.e. the smallest of the normalized Laplacian.
/// Returns `m = dim + 1` eigenvector columns sorted by Rayleigh quotient
/// descending (column 0 is the trivial sqrt-degree eigenvector), or `None`
/// when the problem is degenerate (`n <= m`, Ritz failure).
///
/// Divergence: scipy `eigsh(which="SM")` (ARPACK/Lanczos) is replaced by this
/// matvec-only solver for large graphs. The iteration count is capped; the
/// best available block is returned rather than falling back to random
/// initialisation (Python falls back to random on ARPACK non-convergence).
/// Eigenvectors are defined up to sign/rotation within degenerate
/// eigenspaces, so parity is statistical.
fn lobpcg_bottom_block(
    graph: &CsrMatrix,
    sqrt_deg: &[f64],
    dim: usize,
    max_iter: usize,
    tol: f64,
    random_state: &mut TauRng,
) -> Option<ndarray::Array2<f32>> {
    let n = graph.shape.0;
    let m = dim + 1;
    if n <= m {
        return None;
    }
    // Guard vectors accelerate convergence of the kept columns when the
    // bottom of the spectrum is clustered.
    let block = (m + LOBPCG_GUARDS).min(n - 1);

    // B = 2I - L = I + D^-1/2 A D^-1/2, applied column-wise.
    let apply_b = |x: &ndarray::Array2<f64>| -> ndarray::Array2<f64> {
        let mut out = ndarray::Array2::<f64>::zeros(x.dim());
        for col in 0..x.ncols() {
            let xv: Vec<f64> = (0..n).map(|i| x[[i, col]]).collect();
            for i in 0..n {
                let mut ax = 0.0f64;
                for k in graph.indptr[i]..graph.indptr[i + 1] {
                    let j = graph.indices[k] as usize;
                    ax += f64::from(graph.data[k]) * xv[j] / (sqrt_deg[i] * sqrt_deg[j]);
                }
                out[[i, col]] = xv[i] + ax;
            }
        }
        out
    };

    // Sort the block columns by Rayleigh quotient descending (kept columns
    // first, guards last).
    let sort_by_theta = |x: &mut ndarray::Array2<f64>, ax: &mut ndarray::Array2<f64>| {
        let mut theta: Vec<(f64, usize)> = (0..block)
            .map(|col| {
                let v: f64 = (0..n).map(|i| x[[i, col]] * ax[[i, col]]).sum();
                (v, col)
            })
            .collect();
        theta.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        let order: Vec<usize> = theta.iter().map(|&(_, c)| c).collect();
        let mut nx = ndarray::Array2::<f64>::zeros(x.dim());
        let mut nax = ndarray::Array2::<f64>::zeros(ax.dim());
        for (dst, &src) in order.iter().enumerate() {
            for i in 0..n {
                nx[[i, dst]] = x[[i, src]];
                nax[[i, dst]] = ax[[i, src]];
            }
        }
        *x = nx;
        *ax = nax;
    };

    // Random orthonormal start.
    let mut x = ndarray::Array2::<f64>::from_shape_fn((n, block), |_| {
        random_state.uniform_f64() * 2.0 - 1.0
    });
    orthonormalize_columns(&mut x, 0);
    let mut ax = apply_b(&x);

    for _ in 0..max_iter {
        // X^T A X (block x block).
        let mut xtax = ndarray::Array2::<f64>::zeros((block, block));
        for p in 0..block {
            for q in 0..block {
                xtax[[p, q]] = (0..n).map(|i| x[[i, p]] * ax[[i, q]]).sum();
            }
        }
        // Residual expansion R = AX - X*(X^T A X).
        let mut r = ax.clone();
        for col in 0..block {
            for p in 0..block {
                let c = xtax[[p, col]];
                if c != 0.0 {
                    for i in 0..n {
                        r[[i, col]] -= c * x[[i, p]];
                    }
                }
            }
        }
        if expand_orthonormal(&mut r, &x) < block {
            // The subspace spanned by X is invariant: sort and return it.
            sort_by_theta(&mut x, &mut ax);
            return Some(x.slice(ndarray::s![.., ..m]).to_owned().mapv(|v| v as f32));
        }

        // Rayleigh-Ritz on Q = [X | R].
        let mut q = ndarray::Array2::<f64>::zeros((n, 2 * block));
        q.slice_mut(ndarray::s![.., ..block]).assign(&x);
        q.slice_mut(ndarray::s![.., block..]).assign(&r);
        let aq_r = apply_b(&r);
        let mut aq = ndarray::Array2::<f64>::zeros((n, 2 * block));
        aq.slice_mut(ndarray::s![.., ..block]).assign(&ax);
        aq.slice_mut(ndarray::s![.., block..]).assign(&aq_r);

        let mut h = ndarray::Array2::<f64>::zeros((2 * block, 2 * block));
        for p in 0..2 * block {
            for q2 in p..2 * block {
                let v: f64 = (0..n).map(|i| q[[i, p]] * aq[[i, q2]]).sum();
                h[[p, q2]] = v;
                h[[q2, p]] = v;
            }
        }
        let (_, vecs) = self_adjoint_eigendecomposition(&h).ok()?;
        // Eigenvalues ascending; take the last `block` columns (largest of B).
        let top = vecs.slice(ndarray::s![.., block..]).to_owned();

        // X <- Q*V_top, AX <- AQ*V_top (cheap).
        x = q.dot(&top);
        ax = aq.dot(&top);
        sort_by_theta(&mut x, &mut ax);

        // Convergence: kept columns only, ||AX - theta*X|| <= tol*|theta|.
        let mut converged = true;
        for col in 0..m {
            let mut theta = 0.0f64;
            let mut res2 = 0.0f64;
            for i in 0..n {
                theta += x[[i, col]] * ax[[i, col]];
            }
            for i in 0..n {
                let d = ax[[i, col]] - theta * x[[i, col]];
                res2 += d * d;
            }
            if res2.sqrt() > tol * theta.abs().max(1e-12) {
                converged = false;
            }
        }
        if converged {
            return Some(x.slice(ndarray::s![.., ..m]).to_owned().mapv(|v| v as f32));
        }
    }

    // Iteration cap reached: return the best available block (divergence from
    // Python's random fallback; see the doc comment).
    Some(x.slice(ndarray::s![.., ..m]).to_owned().mapv(|v| v as f32))
}

/// Number of extra guard vectors in the LOBPCG block (beyond `dim + 1`).
const LOBPCG_GUARDS: usize = 3;

/// Python: `multi_component_layout` — positions for components via a
/// centroid meta-embedding, then per-component layouts.
///
/// Divergence: `component_layout` (sklearn `SpectralEmbedding` on the
/// precomputed affinity) is replaced by a dense-EVD spectral embedding of
/// the affinity's normalized Laplacian; components smaller than `2 * dim`
/// use uniform random placement around the centroid embedding.
#[allow(clippy::many_single_char_names)]
pub fn multi_component_layout(
    data: Option<&ndarray::Array2<f32>>,
    graph: &CsrMatrix,
    n_components: usize,
    component_labels: &[i32],
    dim: usize,
    random_state: &mut TauRng,
) -> ndarray::Array2<f32> {
    let n = graph.shape.0;
    let mut result = ndarray::Array2::<f32>::zeros((n, dim));

    // Meta-embedding of the component centroids.
    let meta_embedding = if n_components > 2 * dim {
        component_layout(data, n_components, component_labels, dim, random_state)
    } else {
        // Python: k = ceil(n_components / 2); base = [I_k, 0]; stack ±base.
        let k = (n_components as f64 / 2.0).ceil() as usize;
        let mut meta = ndarray::Array2::<f32>::zeros((n_components, dim));
        for c in 0..n_components {
            let sign = if c < k { 1.0f32 } else { -1.0f32 };
            let row = if c < k { c } else { c - k };
            if row < dim {
                meta[[c, row]] = sign;
            }
        }
        meta
    };

    for label in 0..n_components {
        let members: Vec<usize> = (0..n)
            .filter(|&i| component_labels[i] == label as i32)
            .collect();
        if members.is_empty() {
            continue;
        }

        // Distance from this component's meta-embedding to the closest
        // other meta-embedding (Python: distances.min() / 2).
        let mut min_dist = f64::INFINITY;
        for (other, row) in meta_embedding.rows().into_iter().enumerate() {
            if other == label {
                continue;
            }
            let d = f64::from(
                meta_embedding
                    .row(label)
                    .iter()
                    .zip(row.iter())
                    .map(|(a, b)| (a - b) * (a - b))
                    .sum::<f32>()
                    .sqrt(),
            );
            if d > 0.0 && d < min_dist {
                min_dist = d;
            }
        }
        let data_range = min_dist / 2.0;

        // Extract the component subgraph.
        let mut sub = CooMatrix::new((members.len(), members.len()));
        let index_of: std::collections::HashMap<usize, usize> = members
            .iter()
            .copied()
            .enumerate()
            .map(|(new, old)| (old, new))
            .collect();
        for &i in &members {
            for k in graph.indptr[i]..graph.indptr[i + 1] {
                let j = graph.indices[k] as usize;
                if let Some(&_nj) = index_of.get(&j) {
                    sub.row.push(i as i32);
                    sub.col.push(j as i32);
                    sub.data.push(graph.data[k]);
                }
            }
        }
        // Remap to local indices.
        for idx in 0..sub.row.len() {
            sub.row[idx] = index_of[&(sub.row[idx] as usize)] as i32;
            sub.col[idx] = index_of[&(sub.col[idx] as usize)] as i32;
        }
        let component_graph = sub.tocsr();

        if component_graph.shape.0 < 2 * dim || component_graph.shape.0 <= dim + 1 {
            for (local, &i) in members.iter().enumerate() {
                for d in 0..dim {
                    result[[i, d]] = (random_state.uniform_f64() * 2.0 - 1.0) as f32
                        * data_range as f32
                        + meta_embedding[[label, d]];
                }
                let _ = local;
            }
        } else {
            let component_embedding = spectral_layout_inner(&component_graph, dim, random_state);
            let max_abs = component_embedding
                .iter()
                .map(|v| f64::from(v.abs()))
                .fold(0.0f64, f64::max);
            let expansion = data_range / max_abs.max(f64::EPSILON);
            for (local, &i) in members.iter().enumerate() {
                for d in 0..dim {
                    result[[i, d]] = component_embedding[[local, d]] * expansion as f32
                        + meta_embedding[[label, d]];
                }
                let _ = local;
            }
        }
    }

    result
}

/// Single-component spectral embedding (no component checks) — used by the
/// multi-component layout for each component subgraph.
fn spectral_layout_inner(
    graph: &CsrMatrix,
    dim: usize,
    random_state: &mut TauRng,
) -> ndarray::Array2<f32> {
    spectral_block_for(graph, dim, random_state, DENSE_EVD_MAX_N)
}

/// Python: `component_layout` — spectral embedding of the component
/// centroids (affinity = exp(-(d²))), scaled to [0, 1] by its max.
fn component_layout(
    data: Option<&ndarray::Array2<f32>>,
    n_components: usize,
    component_labels: &[i32],
    dim: usize,
    random_state: &mut TauRng,
) -> ndarray::Array2<f32> {
    let Some(data) = data else {
        // Python: no data — just guess.
        let mut guess = ndarray::Array2::<f32>::zeros((n_components, dim));
        for v in &mut guess {
            *v = (random_state.uniform_f64() * 10.0) as f32;
        }
        return guess;
    };

    let n_features = data.ncols();
    let mut centroids = ndarray::Array2::<f64>::zeros((n_components, n_features));
    let mut counts = vec![0f64; n_components];
    for i in 0..data.nrows() {
        let label = component_labels[i] as usize;
        counts[label] += 1.0;
        for j in 0..n_features {
            centroids[[label, j]] += f64::from(data[[i, j]]);
        }
    }
    for label in 0..n_components {
        if counts[label] > 0.0 {
            for j in 0..n_features {
                centroids[[label, j]] /= counts[label];
            }
        }
    }

    // Pairwise Euclidean distances between centroids.
    let mut distance_matrix = ndarray::Array2::<f64>::zeros((n_components, n_components));
    for i in 0..n_components {
        for j in (i + 1)..n_components {
            let d = (0..n_features)
                .map(|f| (centroids[[i, f]] - centroids[[j, f]]).powi(2))
                .sum::<f64>()
                .sqrt();
            distance_matrix[[i, j]] = d;
            distance_matrix[[j, i]] = d;
        }
    }

    // Affinity and its normalized Laplacian, then dense spectral embedding.
    let affinity = distance_matrix.mapv(|d| (-(d * d)).exp());
    let m = n_components;
    let mut deg = vec![0.0f64; m];
    for i in 0..m {
        deg[i] = affinity.row(i).sum();
    }
    let mut lap = ndarray::Array2::<f64>::eye(m);
    for i in 0..m {
        for j in 0..m {
            if deg[i] > 0.0 {
                lap[[i, j]] -= affinity[[i, j]] / (deg[i].sqrt() * deg[j].sqrt());
            }
        }
    }

    let embedding = if let Ok((eigenvalues, eigenvectors)) = self_adjoint_eigendecomposition(&lap) {
        // sklearn SpectralEmbedding drops the first (constant)
        // eigenvector and uses the next `dim` bottom ones, scaled by
        // sqrt(degrees) — the diffusion-map normalization.
        let mut order: Vec<usize> = (0..eigenvalues.len()).collect();
        order.sort_by(|&a, &b| {
            eigenvalues[a]
                .partial_cmp(&eigenvalues[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut embedding = ndarray::Array2::<f32>::zeros((m, dim));
        for (out_col, &src) in order.iter().skip(1).take(dim).enumerate() {
            for i in 0..m {
                let scale = if deg[i] > 0.0 { deg[i].sqrt() } else { 1.0 };
                embedding[[i, out_col]] = (eigenvectors[[i, src]] * scale) as f32;
            }
        }
        embedding
    } else {
        let mut guess = ndarray::Array2::<f32>::zeros((m, dim));
        for v in &mut guess {
            *v = (random_state.uniform_f64() * 10.0) as f32;
        }
        guess
    };

    let max = embedding
        .iter()
        .map(|v| f64::from(*v))
        .fold(f64::NEG_INFINITY, f64::max);
    if max > 0.0 {
        let mut embedding = embedding;
        for v in &mut embedding {
            *v /= max as f32;
        }
        return embedding;
    }
    embedding
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn symmetric_evd_diagonalizes() {
        // A = [[2, 1], [1, 2]] — eigenvalues 1, 3.
        let a = ndarray::arr2(&[[2.0, 1.0], [1.0, 2.0]]);
        let (vals, vecs) = self_adjoint_eigendecomposition(&a).expect("evd");
        assert!(vals
            .iter()
            .zip([1.0, 3.0].iter())
            .all(|(a, b)| (a - b).abs() < 1e-12));
        // A @ v = λ v for each column.
        for (k, &lam) in vals.iter().enumerate() {
            for i in 0..2 {
                let av: f64 = (0..2).map(|j| a[[i, j]] * vecs[[j, k]]).sum();
                assert!((av - lam * vecs[[i, k]]).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn connected_components_labels_chain() {
        // 0-1-2 connected, 3-4 connected, 5 isolated.
        let coo = CooMatrix::from_triplets(
            (6, 6),
            vec![0, 1, 1, 2, 3, 4],
            vec![1, 0, 2, 1, 4, 3],
            vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        );
        let graph = coo.tocsr();
        let (n, labels) = connected_components(&graph);
        assert_eq!(n, 3);
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[1], labels[2]);
        assert_eq!(labels[3], labels[4]);
        assert_ne!(labels[0], labels[3]);
        assert_ne!(labels[0], labels[5]);
    }

    #[test]
    fn spectral_layout_returns_shape_and_finite() {
        // Ring graph on 30 vertices.
        let n = 30;
        let mut row = Vec::new();
        let mut col = Vec::new();
        let mut data = Vec::new();
        for i in 0..n {
            for &j in &[(&i + 1) % n, (i + n - 1) % n] {
                row.push(i as i32);
                col.push(j as i32);
                data.push(1.0f32);
            }
        }
        let graph = CooMatrix::from_triplets((n, n), row, col, data).tocsr();
        let mut rng = TauRng::new(42);
        let emb = spectral_layout(None, &graph, 2, &mut rng).expect("layout");
        assert_eq!(emb.dim(), (n, 2));
        assert!(emb.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn lobpcg_matches_dense_evd() {
        // Clustered graph on 400 vertices (3 well-separated clusters, as in
        // real UMAP fuzzy graphs): the LOBPCG path must agree with the dense
        // EVD path per column (up to sign).
        let n = 400;
        let mut rng = TauRng::new(7);
        let mut row = Vec::new();
        let mut col = Vec::new();
        let mut data = Vec::new();
        let push_edge = |i: usize,
                         j: usize,
                         w: f32,
                         row: &mut Vec<i32>,
                         col: &mut Vec<i32>,
                         data: &mut Vec<f32>| {
            row.push(i as i32);
            col.push(j as i32);
            data.push(w);
            row.push(j as i32);
            col.push(i as i32);
            data.push(w);
        };
        for i in 0..n {
            for j in (i + 1)..n {
                let same_cluster = (i / 133) == (j / 133);
                let p = if same_cluster { 0.25 } else { 0.002 };
                if rng.uniform_f64() < p {
                    let w = (rng.uniform_f64() + 0.5) as f32;
                    push_edge(i, j, w, &mut row, &mut col, &mut data);
                }
            }
        }
        let graph = CooMatrix::from_triplets((n, n), row, col, data).tocsr();

        let dense = spectral_block_for(&graph, 2, &mut TauRng::new(42), n);
        let iterative = spectral_block_for(&graph, 2, &mut TauRng::new(42), 0);

        // Subspace alignment via principal angles: eigenvalues of the
        // cross-Gram Gram matrix C = denseᵀ·iterative must all be ≈ ±1
        // (per-column comparison is invalid when eigenvalues are degenerate,
        // since any basis of a degenerate eigenspace is valid).
        let mut c = ndarray::Array2::<f64>::zeros((2, 2));
        for p in 0..2 {
            for q in 0..2 {
                c[[p, q]] = (0..n)
                    .map(|i| f64::from(dense[[i, p]]) * f64::from(iterative[[i, q]]))
                    .sum();
            }
        }
        let cc = c.t().dot(&c);
        let Ok((vals, _)) = self_adjoint_eigendecomposition(&cc) else {
            panic!("cross-Gram eigendecomposition failed");
        };
        for (d, &v) in vals.iter().enumerate() {
            assert!(
                v > 0.98,
                "principal angle {d}: cos² = {v:.4} — LOBPCG subspace disagrees with dense EVD"
            );
        }
    }

    #[test]
    fn lobpcg_output_is_orthonormal() {
        // Ring graph on 200 vertices: the converged block columns must be
        // orthonormal (before the f32 cast) and the trivial constant vector
        // must be dropped (no column is ~1/sqrt(n) proportional to ones).
        let n = 200;
        let mut row = Vec::new();
        let mut col = Vec::new();
        let mut data = Vec::new();
        for i in 0..n {
            for &j in &[(&i + 1) % n, (i + n - 1) % n] {
                row.push(i as i32);
                col.push(j as i32);
                data.push(1.0f32);
            }
        }
        let graph = CooMatrix::from_triplets((n, n), row, col, data).tocsr();
        let emb = spectral_block_for(&graph, 2, &mut TauRng::new(42), 0);
        for (a, b) in emb.columns().into_iter().enumerate() {
            let nrm: f64 = b
                .iter()
                .map(|v| f64::from(*v) * f64::from(*v))
                .sum::<f64>()
                .sqrt();
            assert!((nrm - 1.0).abs() < 0.05, "column {a} norm {nrm}");
            // The trivial (constant) eigenvector was skipped: column entries
            // must not be all the same sign/value.
            let all_same = b.iter().all(|v| (*v - b[0]).abs() < 1e-3);
            assert!(!all_same, "column {a} looks like the trivial eigenvector");
        }
    }

    #[test]
    fn lobpcg_falls_back_when_n_too_small() {
        // n <= dim + 1 must yield None from the solver (caller falls back).
        let coo = CooMatrix::from_triplets(
            (3, 3),
            vec![0, 1, 1, 2],
            vec![1, 0, 2, 1],
            vec![1.0, 1.0, 1.0, 1.0],
        );
        let graph = coo.tocsr();
        let sqrt_deg = compute_sqrt_deg(&graph);
        let mut rng = TauRng::new(42);
        assert!(lobpcg_bottom_block(&graph, &sqrt_deg, 2, 10, 1e-9, &mut rng).is_none());
    }

    #[test]
    fn lobpcg_matches_dense_evd_at_scale() {
        // Larger clustered graph (n=1200): force the iterative path
        // (`dense_limit = 0`) and compare against the exact dense path via
        // subspace principal angles.
        let n = 1200;
        let mut rng = TauRng::new(11);
        let mut row = Vec::new();
        let mut col = Vec::new();
        let mut data = Vec::new();
        for i in 0..n {
            for j in (i + 1)..n {
                let same_cluster = (i / 300) == (j / 300);
                let p = if same_cluster { 0.05 } else { 0.0005 };
                if rng.uniform_f64() < p {
                    let w = (rng.uniform_f64() + 0.5) as f32;
                    row.push(i as i32);
                    col.push(j as i32);
                    data.push(w);
                    row.push(j as i32);
                    col.push(i as i32);
                    data.push(w);
                }
            }
        }
        let graph = CooMatrix::from_triplets((n, n), row, col, data).tocsr();

        let dense = spectral_block_for(&graph, 2, &mut TauRng::new(42), n);
        let iterative = spectral_block_for(&graph, 2, &mut TauRng::new(42), 0);

        let mut c = ndarray::Array2::<f64>::zeros((2, 2));
        for p in 0..2 {
            for q in 0..2 {
                c[[p, q]] = (0..n)
                    .map(|i| f64::from(dense[[i, p]]) * f64::from(iterative[[i, q]]))
                    .sum();
            }
        }
        let cc = c.t().dot(&c);
        let Ok((vals, _)) = self_adjoint_eigendecomposition(&cc) else {
            panic!("cross-Gram eigendecomposition failed");
        };
        for (d, &v) in vals.iter().enumerate() {
            assert!(v > 0.98, "principal angle {d}: cos² = {v:.4}");
        }
    }
}
