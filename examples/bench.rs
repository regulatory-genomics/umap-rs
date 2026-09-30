//! Benchmark harness for umap-rs.
//!
//! Times the fit phases (kNN, fuzzy graph, spectral init, SGD kernel),
//! end-to-end fit, `transform`, and `inverse_transform` on the deterministic
//! LCG dataset used by the cross-validation tests, and reports
//! trustworthiness as the quality column.
//!
//! Run with `cargo run --release --example bench`. The matching Python
//! baseline (umap-learn, JIT-warmed, seeded and fully-threaded variants) is
//! cross-validation script `bench_python.py`; see the comparison table in the
//! benchmark results. Phase times use the crate's public APIs with the same
//! hyperparameters as the Python baseline (`n_neighbors = 15`,
//! `min_dist = 0.1`, `n_epochs = 50`, euclidean).
//!
//! Notes:
//! - `sgd` is timed with random initialisation to isolate the SGD kernel from
//!   the spectral init (which is timed separately).
//! - `fit` uses the crate's standard entry points: `Umap::fit` for
//!   `n < 4096`, `fit_with_knn` above (the kNN cost is timed separately).

#![allow(clippy::unreadable_literal, clippy::many_single_char_names)]
#![allow(clippy::items_after_statements)]

use std::io::Write;
use std::time::Instant;

use umap_rs::distances::named_distances;
use umap_rs::embedding::{find_ab_params, simplicial_set_embedding, Init, SsetParams};
use umap_rs::fuzzy::fuzzy_simplicial_set;
use umap_rs::rng::TauRng;
use umap_rs::spectral::spectral_layout;
use umap_rs::umap::{brute_force_knn, Umap, UmapConfig};
use umap_rs::validation::trustworthiness;

/// Scales benchmarked (same as the Python baseline script).
const SCALES: [usize; 4] = [500, 2000, 5000, 10000];
const K: usize = 15;
const N_EPOCHS: usize = 50;
const N_TRANSFORM: usize = 1000;
const DATASET_SEED: u64 = 2024;

/// Deterministic LCG dataset (identical to the cross-validation tests and the
/// Python baseline script): 4 clusters offset in the first two dims.
fn lcg_dataset(n: usize, d: usize, seed: u64) -> ndarray::Array2<f32> {
    let mut state = seed;
    let mut out = ndarray::Array2::<f32>::zeros((n, d));
    for i in 0..n {
        for j in 0..d {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            out[[i, j]] = ((state >> 11) as f64 / (1u64 << 53) as f64) as f32 * 2.0 - 1.0;
        }
    }
    for i in 0..n {
        let c = (i / (n / 4)) as f32;
        out[[i, 0]] += c * 8.0;
        out[[i, 1]] += c * 5.0;
    }
    out
}

/// Phase times for one dataset scale (seconds).
struct PhaseTimes {
    knn: f64,
    fuzzy: f64,
    spectral: f64,
    sgd: f64,
    fit: f64,
    transform: f64,
    inverse: f64,
    trust: f64,
}

fn bench_scale(n: usize) -> PhaseTimes {
    let x = lcg_dataset(n, 10, DATASET_SEED);
    let metric = named_distances("euclidean").expect("euclidean");

    // kNN (brute force; the crate's fit does the same for n < 4096).
    let t = Instant::now();
    let (knn_indices, knn_dists) = brute_force_knn(&x, &x, K, metric, &[]);
    let knn = t.elapsed().as_secs_f64();

    // Fuzzy simplicial set.
    let t = Instant::now();
    let (graph, _sigmas, _rhos, _dists) =
        fuzzy_simplicial_set(n, K, &knn_indices, &knn_dists, 1.0, 1.0, true, false).expect("fuzzy");
    let fuzzy = t.elapsed().as_secs_f64();

    // Spectral init (n > 4096 takes the LOBPCG path).
    let t = Instant::now();
    let spectral_emb =
        spectral_layout(Some(&x), &graph, 2, &mut TauRng::new(42)).expect("spectral layout");
    let spectral = t.elapsed().as_secs_f64();
    assert_eq!(spectral_emb.dim(), (n, 2));

    // SGD kernel with random init (isolates the kernel from the spectral
    // init); single-threaded, matching the seeded configuration.
    let (a, b) = find_ab_params(1.0, 0.1);
    let params = SsetParams {
        n_components: 2,
        initial_alpha: 1.0,
        a,
        b,
        gamma: 1.0,
        negative_sample_rate: 5.0,
        n_epochs: N_EPOCHS,
        init: Init::Random,
        metric,
        metric_args: vec![],
        parallel: false,
        verbose: false,
        densmap: false,
        densmap_kwds: None,
        output_dens: false,
        output_metric: None,
        output_metric_kwds: vec![],
        embedding_checkpoints: None,
    };
    let t = Instant::now();
    let (emb_random, _aux) =
        simplicial_set_embedding(Some(&x), &graph, &params, &mut TauRng::new(42))
            .expect("sgd embedding");
    let sgd = t.elapsed().as_secs_f64();
    assert_eq!(emb_random.dim(), (n, 2));

    // End-to-end fit with the default (spectral) init.
    let cfg = UmapConfig {
        n_neighbors: K,
        min_dist: 0.1,
        n_components: 2,
        n_epochs: Some(N_EPOCHS),
        random_state: Some(42),
        ..UmapConfig::default()
    };
    let t = Instant::now();
    let model = if n < 4096 {
        Umap::with_config(cfg.clone()).fit(&x).expect("fit")
    } else {
        Umap::with_config(cfg.clone())
            .fit_with_knn(Some(&x), &knn_indices, &knn_dists)
            .expect("fit")
    };
    let fit = t.elapsed().as_secs_f64();

    // Held-out transform / inverse_transform.
    let new_data = lcg_dataset(N_TRANSFORM, 10, 9999);
    let t = Instant::now();
    let transformed = model.transform(&new_data).expect("transform");
    let transform = t.elapsed().as_secs_f64();

    let t = Instant::now();
    let inverted = model.inverse_transform(&transformed).expect("inverse");
    let inverse = t.elapsed().as_secs_f64();
    assert_eq!(inverted.nrows(), N_TRANSFORM);

    let trust = trustworthiness(&x, &model.embedding, K);

    PhaseTimes {
        knn,
        fuzzy,
        spectral,
        sgd,
        fit,
        transform,
        inverse,
        trust,
    }
}

fn main() {
    println!();
    println!("umap-rs benchmark (release build, single-threaded, n_epochs = {N_EPOCHS}, k = {K})");
    println!();
    println!(
        "| n | knn (s) | fuzzy (s) | spectral (s) | sgd (s) | fit (s) | transform (s) | inverse (s) | trust k={K} |"
    );
    println!("|---:|---:|---:|---:|---:|---:|---:|---:|---:|");
    for &n in &SCALES {
        let pt = bench_scale(n);
        let secs = |v: f64| format!("{v:.3}");
        println!(
            "| {n} | {} | {} | {} | {} | {} | {} | {} | {:.4} |",
            secs(pt.knn),
            secs(pt.fuzzy),
            secs(pt.spectral),
            secs(pt.sgd),
            secs(pt.fit),
            secs(pt.transform),
            secs(pt.inverse),
            pt.trust
        );
        // Flush per row so partial results survive interruption.
        let _ = std::io::stdout().flush();
    }
    println!();
}
