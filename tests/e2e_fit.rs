//! End-to-end cross-validation of the fit path against the Python repo
//! baseline (see /`tmp/opencode/cv/gen_e2e_ref.py` for the generator).
//!
//! Parity is statistical: the spectral initialization's eigenvectors are only
//! defined up to sign and the SGD trajectory is chaotic w.r.t. init, so the
//! test compares the trustworthiness metric (Python: 0.96750 seed 42,
//! 0.97060 seed 7) within ±0.02, plus same-seed reproducibility.
#![allow(clippy::unreadable_literal)]

use umap_rs::umap::{Umap, UmapConfig};
use umap_rs::validation::trustworthiness;

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
    let cluster = |i: usize| i / (n / 4);
    for i in 0..n {
        let c = cluster(i) as f32;
        out[[i, 0]] += c * 8.0;
        out[[i, 1]] += c * 5.0;
    }
    out
}

#[test]
fn e2e_trustworthiness_matches_python_baseline() {
    let x = lcg_dataset(150, 5, 12345);

    for (seed, py_t10) in [(42u64, 0.9675043370508054), (7u64, 0.9705972738537795)] {
        let umap = Umap::with_config(UmapConfig {
            n_neighbors: 10,
            min_dist: 0.1,
            n_components: 2,
            n_epochs: Some(50),
            random_state: Some(seed),
            ..UmapConfig::default()
        });
        let model = umap.fit(&x).expect("fit");
        let t10 = trustworthiness(&x, &model.embedding, 10);
        assert!(
            (t10 - py_t10).abs() < 0.02,
            "seed {seed}: rust t10 {t10} vs python {py_t10}"
        );
        assert!(model.embedding.iter().all(|v| v.is_finite()));
    }
}

#[test]
fn e2e_same_seed_is_reproducible() {
    let x = lcg_dataset(150, 5, 12345);
    let fit_once = || {
        Umap::with_config(UmapConfig {
            n_neighbors: 10,
            min_dist: 0.1,
            n_components: 2,
            n_epochs: Some(50),
            random_state: Some(42),
            ..UmapConfig::default()
        })
        .fit(&x)
        .expect("fit")
    };
    let m1 = fit_once();
    let m2 = fit_once();
    assert!(m1
        .embedding
        .iter()
        .zip(m2.embedding.iter())
        .all(|(a, b)| (a - b).abs() < 1e-6));
}

#[test]
fn e2e_transform_and_inverse_are_sane() {
    let x = lcg_dataset(150, 5, 12345);
    let umap = Umap::with_config(UmapConfig {
        n_neighbors: 10,
        min_dist: 0.1,
        n_components: 2,
        n_epochs: Some(50),
        random_state: Some(42),
        ..UmapConfig::default()
    });
    let model = umap.fit(&x).expect("fit");

    // New points from the same distribution (cluster 0).
    let new = x.slice(ndarray::s![0..10, ..]).to_owned();
    let transformed = model.transform(&new).expect("transform");
    assert_eq!(transformed.dim(), (10, 2));
    assert!(transformed.iter().all(|v| v.is_finite()));

    // Transformed points should land near their training images (cluster 0
    // occupies the region around the first points' embedding).
    let d = (0..10)
        .map(|i| {
            (0..2)
                .map(|d| transformed[[i, d]] - model.embedding[[i, d]])
                .map(|v| v * v)
                .sum::<f32>()
                .sqrt()
        })
        .fold(0.0f32, f32::max);
    assert!(d < 25.0, "transformed too far from training images: {d}");

    let inv = model.inverse_transform(&transformed).expect("inverse");
    assert_eq!(inv.dim(), (10, 5));
}
