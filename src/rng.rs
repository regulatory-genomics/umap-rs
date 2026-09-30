//! Ported from Python: umap/utils.py (`tau_rand_int`, `tau_rand`).
//!
//! The tau-leap integer RNG is ported *exactly* (integer-exact): the same
//! 3-element int64 state sequence yields the same values as the numba
//! implementation. Seeding differs from Python (Python draws the seed state
//! from numpy's Mersenne Twister); see [`TauRng`].

/// A fast (pseudo)-random number generator returning an int32.
///
/// Python: `umap.utils.tau_rand_int(state)` where `state` is an int64 array
/// of shape (3,). The state is mutated in place, matching the Python
/// semantics; each call advances all three state words.
///
/// Note: Python applies `% n` to the result with Python modulo semantics
/// (result has the sign of the divisor). In Rust use `rem_euclid` to match.
#[inline]
pub fn tau_rand_int(state: &mut [i64; 3]) -> i32 {
    // All arithmetic is int64 with wrapping semantics (numba matches Python
    // int64 overflow behavior for @njit code via NumPy int64 semantics).
    state[0] = (((state[0] & 0xFFFF_FFFE) << 12) & 0xFFFF_FFFF)
        ^ ((((state[0] << 13) & 0xFFFF_FFFF) ^ state[0]) >> 19);
    state[1] = (((state[1] & 0xFFFF_FFF8) << 4) & 0xFFFF_FFFF)
        ^ ((((state[1] << 2) & 0xFFFF_FFFF) ^ state[1]) >> 25);
    state[2] = (((state[2] & 0xFFFF_FFF0) << 17) & 0xFFFF_FFFF)
        ^ ((((state[2] << 3) & 0xFFFF_FFFF) ^ state[2]) >> 11);

    (state[0] ^ state[1] ^ state[2]) as i32
}

/// A fast (pseudo)-random number generator for floats in the range [0, 1).
///
/// Python: `umap.utils.tau_rand(state)`.
#[inline]
pub fn tau_rand(state: &mut [i64; 3]) -> f32 {
    let integer = tau_rand_int(state);
    (f64::from(integer) / f64::from(0x7FFF_FFFF)).abs() as f32
}

/// Seed-state handling for the tau-leap RNG.
///
/// Divergence from Python: umap-learn draws the 3-element int64 `rng_state`
/// from `np.random.RandomState.randint(INT32_MIN, INT32_MAX, 3)`. This port
/// generates the state from a `SplitMix64` stream seeded by a u64, giving
/// different (but statistically equivalent) streams. For cross-validation
/// the exact state can be supplied via [`TauRng::from_state`].
#[derive(Debug, Clone)]
#[allow(dead_code)] // `state` readable via from_state/tau_rand free functions
pub struct TauRng {
    state: [i64; 3],
    splitmix: u64,
}

/// Python: `INT32_MIN = np.iinfo(np.int32).min + 1`.
pub(crate) const INT32_MIN: i64 = i32::MIN as i64 + 1;
/// Python: `INT32_MAX = np.iinfo(np.int32).max - 1`.
pub(crate) const INT32_MAX: i64 = i32::MAX as i64 - 1;

impl TauRng {
    /// Create a new RNG from a u64 seed (analogue of
    /// `check_random_state(seed)` in Python).
    #[must_use]
    pub fn new(seed: u64) -> Self {
        Self {
            state: [0; 3],
            splitmix: seed,
        }
    }

    /// Create a RNG with an exact internal state (for cross-validation).
    #[must_use]
    pub fn from_state(state: [i64; 3]) -> Self {
        Self { state, splitmix: 0 }
    }

    /// Python: `random_state.randint(INT32_MIN, INT32_MAX, 3).astype(np.int64)`
    /// — draws a fresh 3-element `rng_state` in [`INT32_MIN`, `INT32_MAX`].
    pub fn draw_rng_state(&mut self) -> [i64; 3] {
        let mut out = [0i64; 3];
        for v in &mut out {
            *v = self.next_in_range(INT32_MIN, INT32_MAX);
        }
        out
    }

    /// Uniform integer in [lo, hi] (inclusive), mirroring numpy's randint.
    fn next_in_range(&mut self, lo: i64, hi: i64) -> i64 {
        // SplitMix64 to get 64 random bits, then reduce into range.
        self.splitmix = self.splitmix.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.splitmix;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        let span = (hi - lo + 1) as u64;
        lo + ((z % span) as i64)
    }

    /// Draw a uniform f64 in [0, 1) — analogue of `random_state.uniform`.
    pub fn uniform_f64(&mut self) -> f64 {
        self.splitmix = self.splitmix.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.splitmix;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        (z >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Draw a standard normal f64 — analogue of `random_state.normal()`.
    /// Uses the Box–Muller transform. Divergence: numpy uses the
    /// polar (ziggurat-like) method; values are statistically equivalent.
    #[must_use]
    pub fn normal_f64(&mut self) -> f64 {
        let u1 = self.uniform_f64().max(f64::EPSILON);
        let u2 = self.uniform_f64();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tau_rand_int_matches_numba_reference() {
        // Reference values produced by umap-learn's numba tau_rand_int:
        //   import numpy as np, numba
        //   from umap.utils import tau_rand_int
        //   state = np.array([12345, 67890, 13579], dtype=np.int64)
        //   [int(tau_rand_int(state)) for _ in range(3)]
        let mut state: [i64; 3] = [12345, 67890, 13579];
        let vals: Vec<i32> = (0..3).map(|_| tau_rand_int(&mut state)).collect();
        // Python/numba int64 wrapping arithmetic must match exactly.
        assert_eq!(vals, [1_762_857_971i32, 962_756_195i32, 1_349_868_690i32]);
    }

    #[test]
    fn tau_rand_in_range() {
        let mut state: [i64; 3] = [1, 2, 3];
        for _ in 0..1000 {
            let v = tau_rand(&mut state);
            assert!((0.0..=1.0).contains(&v));
        }
    }
}
