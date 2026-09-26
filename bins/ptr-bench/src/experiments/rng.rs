//! A seeded SplitMix64 generator. Experiments name their seeds in their
//! manifests, so every case must be reproducible from the seed alone and must
//! not depend on a library's versioned stream.

pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }

    /// An independent stream for one case of a run.
    pub fn fork(&mut self, label: u64) -> Self {
        Self(self.next_u64() ^ label.wrapping_mul(0xA24B_AED4_963E_E407))
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `0..bound`; `bound` must be positive.
    pub fn below(&mut self, bound: u64) -> u64 {
        // Lemire's multiply-shift; the bias is below 2^-64 * bound.
        ((u128::from(self.next_u64()) * u128::from(bound)) >> 64) as u64
    }

    pub fn index(&mut self, len: usize) -> usize {
        self.below(len as u64) as usize
    }

    pub fn range(&mut self, low: u64, high_inclusive: u64) -> u64 {
        low + self.below(high_inclusive - low + 1)
    }

    /// Uniform in `[0, 1)`.
    pub fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    pub fn chance(&mut self, probability: f64) -> bool {
        self.unit() < probability
    }

    pub fn f32_between(&mut self, low: f32, high: f32) -> f32 {
        low + (high - low) * self.unit() as f32
    }

    pub fn bytes(&mut self, len: usize) -> Vec<u8> {
        (0..len).map(|_| self.next_u64() as u8).collect()
    }

    pub fn digest(&mut self) -> [u8; 32] {
        let mut out = [0u8; 32];
        for chunk in out.chunks_mut(8) {
            chunk.copy_from_slice(&self.next_u64().to_le_bytes());
        }
        out
    }

    pub fn pick<'a, T>(&mut self, items: &'a [T]) -> &'a T {
        &items[self.index(items.len())]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_seed_reproduces_its_stream_and_bounds_hold() {
        let mut first = Rng::new(17);
        let mut second = Rng::new(17);
        for _ in 0..1000 {
            assert_eq!(first.next_u64(), second.next_u64());
        }
        let mut rng = Rng::new(29);
        for _ in 0..10_000 {
            assert!(rng.below(7) < 7);
            let unit = rng.unit();
            assert!((0.0..1.0).contains(&unit));
            let value = rng.range(3, 5);
            assert!((3..=5).contains(&value));
        }
    }
}
