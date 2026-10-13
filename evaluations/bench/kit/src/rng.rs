//! A seeded SplitMix64 generator.
//!
//! This is a copy of the algorithm in `bins/ptr-bench/src/experiments/rng.rs`
//! (copied, not depended on, so the kit stays free of product crates). A seed
//! reproduces its stream on every machine and toolchain.

/// SplitMix64 state.
#[derive(Clone, Debug)]
pub struct Rng(u64);

impl Rng {
    /// Start a stream from `seed`.
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }

    /// An independent stream for one case of a run.
    pub fn fork(&mut self, label: u64) -> Self {
        Self(self.next_u64() ^ label.wrapping_mul(0xA24B_AED4_963E_E407))
    }

    /// Next 64 random bits.
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `0..bound`; a zero bound returns 0.
    pub fn below(&mut self, bound: u64) -> u64 {
        // Lemire's multiply-shift; the bias is below 2^-64 * bound.
        ((u128::from(self.next_u64()) * u128::from(bound)) >> 64) as u64
    }

    /// Uniform index into a collection of `len` items; `len == 0` returns 0.
    pub fn index(&mut self, len: usize) -> usize {
        self.below(len as u64) as usize
    }

    /// Uniform in `[0, 1)`.
    pub fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// `len` random bytes.
    pub fn bytes(&mut self, len: usize) -> Vec<u8> {
        (0..len).map(|_| self.next_u64() as u8).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splitmix64_matches_the_reference_stream_for_seed_zero() {
        // Reference outputs of SplitMix64 seeded with 0 (Vigna's reference code).
        let mut rng = Rng::new(0);
        assert_eq!(rng.next_u64(), 0xE220_A839_7B1D_CDAF);
        assert_eq!(rng.next_u64(), 0x6E78_9E6A_A1B9_65F4);
        assert_eq!(rng.next_u64(), 0x06C4_5D18_8009_454F);
    }

    #[test]
    fn a_seed_reproduces_its_stream_and_distinct_seeds_differ() {
        let mut a = Rng::new(17);
        let mut b = Rng::new(17);
        let mut c = Rng::new(18);
        let mut same = true;
        for _ in 0..1000 {
            let x = a.next_u64();
            assert_eq!(x, b.next_u64());
            same &= x == c.next_u64();
        }
        assert!(!same);
    }

    #[test]
    fn below_and_index_stay_in_bounds_and_zero_bound_is_zero() {
        let mut rng = Rng::new(29);
        for _ in 0..10_000 {
            assert!(rng.below(7) < 7);
            assert!(rng.index(3) < 3);
        }
        assert_eq!(rng.below(0), 0);
        assert_eq!(rng.index(0), 0);
        assert_eq!(rng.below(1), 0);
    }

    #[test]
    fn unit_is_half_open_and_roughly_uniform() {
        let mut rng = Rng::new(43);
        let mut sum = 0.0;
        for _ in 0..20_000 {
            let u = rng.unit();
            assert!((0.0..1.0).contains(&u));
            sum += u;
        }
        let mean = sum / 20_000.0;
        assert!((mean - 0.5).abs() < 0.01, "mean {mean}");
    }

    #[test]
    fn fork_is_deterministic_and_label_sensitive() {
        let first = Rng::new(5).fork(1).next_u64();
        assert_eq!(first, Rng::new(5).fork(1).next_u64());
        assert_ne!(first, Rng::new(5).fork(2).next_u64());
    }

    #[test]
    fn bytes_have_the_requested_length_and_are_reproducible() {
        assert_eq!(Rng::new(1).bytes(33).len(), 33);
        assert_eq!(Rng::new(1).bytes(33), Rng::new(1).bytes(33));
        assert!(Rng::new(1).bytes(0).is_empty());
    }
}
