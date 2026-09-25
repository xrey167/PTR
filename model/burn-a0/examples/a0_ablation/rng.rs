//! The study's generators: the same splitmix64 and FNV-1a-64 the benchmark
//! generator uses (benchmarks/operator-routing/generator.py), so orders and
//! digests computed here and there agree bit for bit.

pub const GOLDEN_GAMMA: u64 = 0x9E37_79B9_7F4A_7C15;
const FNV_OFFSET: u64 = 0xCBF2_9CE4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01B3;

/// The splitmix64 finalizer of crates/ptr-types/src/slot_encoding.rs.
pub fn mix64(z: u64) -> u64 {
    let z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    let z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// One output of a splitmix64 generator whose state starts at `x`.
pub fn splitmix64(x: u64) -> u64 {
    mix64(x.wrapping_add(GOLDEN_GAMMA))
}

/// Classic splitmix64: state += gamma, output = mix64(state).
pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(GOLDEN_GAMMA);
        mix64(self.state)
    }

    /// Uniform in [0, 1): the top 53 bits over 2^53, exact in f64.
    pub fn u01(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / 9_007_199_254_740_992.0
    }

    /// floor(u01 * n), always below n for the sizes used here.
    pub fn randbelow(&mut self, n: usize) -> usize {
        (self.u01() * n as f64) as usize
    }
}

/// Fisher-Yates over 0..n with `rng`, from the last position down.
pub fn permutation(n: usize, rng: &mut SplitMix64) -> Vec<usize> {
    let mut order: Vec<usize> = (0..n).collect();
    for i in (1..n).rev() {
        let j = rng.randbelow(i + 1);
        order.swap(i, j);
    }
    order
}

/// FNV-1a-64, continuing from `state` (start with [`fnv1a64_start`]).
pub fn fnv1a64(bytes: &[u8], mut state: u64) -> u64 {
    for byte in bytes {
        state ^= u64::from(*byte);
        state = state.wrapping_mul(FNV_PRIME);
    }
    state
}

pub fn fnv1a64_start() -> u64 {
    FNV_OFFSET
}
