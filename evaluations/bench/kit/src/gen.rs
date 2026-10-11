//! Seeded workload generators. Every function is a pure function of its seed
//! (through [`Rng`]), so a benchmark names a seed and every machine runs the
//! same workload.

use crate::rng::Rng;

/// Reasons a generator refuses its parameters.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GenError {
    /// A size, count or dimension that must be positive was zero.
    Zero(&'static str),
}

impl std::fmt::Display for GenError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Zero(what) => write!(f, "{what} must be positive"),
        }
    }
}

impl std::error::Error for GenError {}

/// `count` keys of exactly `len` bytes drawn from `a-z0-9`.
pub fn keys(seed: u64, count: usize, len: usize) -> Vec<String> {
    const ALPHABET: &[u8] = b"abcdefghijklmnopqrstuvwxyz0123456789";
    let mut rng = Rng::new(seed);
    (0..count)
        .map(|_| {
            (0..len)
                .map(|_| ALPHABET[rng.index(ALPHABET.len())] as char)
                .collect()
        })
        .collect()
}

/// `count` payloads whose lengths are uniform in `min_len..=max_len`.
pub fn payloads(seed: u64, count: usize, min_len: usize, max_len: usize) -> Vec<Vec<u8>> {
    let mut rng = Rng::new(seed);
    let (low, high) = if min_len <= max_len {
        (min_len, max_len)
    } else {
        (max_len, min_len)
    };
    (0..count)
        .map(|_| {
            let len = low + rng.index(high - low + 1);
            rng.bytes(len)
        })
        .collect()
}

/// A Fisher-Yates permutation of `0..n`.
pub fn permutation(seed: u64, n: usize) -> Vec<usize> {
    let mut rng = Rng::new(seed);
    let mut items: Vec<usize> = (0..n).collect();
    for i in (1..n).rev() {
        items.swap(i, rng.index(i + 1));
    }
    items
}

/// One standard-normal draw (Box-Muller, the cosine branch only).
pub fn gaussian(rng: &mut Rng) -> f64 {
    let u1 = 1.0 - rng.unit(); // (0, 1], so ln is finite
    let u2 = rng.unit();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

/// A Zipf(s) sampler over ranks `0..n` (rank 0 is the most frequent), by
/// inverse CDF with a binary search.
#[derive(Clone, Debug)]
pub struct Zipf {
    cdf: Vec<f64>,
}

impl Zipf {
    /// Build the table for `n` ranks and exponent `exponent` (typically ~1.0).
    pub fn new(n: usize, exponent: f64) -> Result<Self, GenError> {
        if n == 0 {
            return Err(GenError::Zero("zipf rank count"));
        }
        let mut total = 0.0;
        let mut cdf = Vec::with_capacity(n);
        for rank in 0..n {
            total += 1.0 / ((rank + 1) as f64).powf(exponent);
            cdf.push(total);
        }
        for value in &mut cdf {
            *value /= total;
        }
        Ok(Self { cdf })
    }

    /// Number of ranks.
    pub fn len(&self) -> usize {
        self.cdf.len()
    }

    /// Never true: construction refuses zero ranks.
    pub fn is_empty(&self) -> bool {
        self.cdf.is_empty()
    }

    /// Draw one rank.
    pub fn sample(&self, rng: &mut Rng) -> usize {
        let u = rng.unit();
        self.cdf.partition_point(|&c| c < u).min(self.cdf.len() - 1)
    }
}

/// A seeded Zipf corpus: `docs` documents of `doc_len` token ids from a
/// `vocab`-sized vocabulary.
pub fn zipf_corpus(
    seed: u64,
    vocab: usize,
    docs: usize,
    doc_len: usize,
    exponent: f64,
) -> Result<Vec<Vec<u32>>, GenError> {
    let zipf = Zipf::new(vocab, exponent)?;
    let mut rng = Rng::new(seed);
    Ok((0..docs)
        .map(|_| (0..doc_len).map(|_| zipf.sample(&mut rng) as u32).collect())
        .collect())
}

/// A Gaussian mixture in `dim` dimensions.
#[derive(Clone, Debug, PartialEq)]
pub struct Mixture {
    /// Dimension of every vector.
    pub dim: usize,
    /// Cluster centres, `clusters * dim` values, row-major.
    pub centres: Vec<f32>,
    /// Row-major vectors, `labels.len() * dim` values.
    pub vectors: Vec<f32>,
    /// The cluster each vector was drawn from.
    pub labels: Vec<u32>,
}

/// Draw `n` vectors from `clusters` Gaussian clusters. Centres are standard
/// normal; each vector is its centre plus `spread`-scaled standard normal noise.
pub fn gaussian_mixture(
    seed: u64,
    dim: usize,
    clusters: usize,
    n: usize,
    spread: f32,
) -> Result<Mixture, GenError> {
    if dim == 0 {
        return Err(GenError::Zero("dimension"));
    }
    if clusters == 0 {
        return Err(GenError::Zero("cluster count"));
    }
    let mut rng = Rng::new(seed);
    let centres: Vec<f32> = (0..clusters * dim)
        .map(|_| gaussian(&mut rng) as f32)
        .collect();
    let mut vectors = Vec::with_capacity(n * dim);
    let mut labels = Vec::with_capacity(n);
    for _ in 0..n {
        let cluster = rng.index(clusters);
        labels.push(cluster as u32);
        for d in 0..dim {
            vectors.push(centres[cluster * dim + d] + spread * gaussian(&mut rng) as f32);
        }
    }
    Ok(Mixture {
        dim,
        centres,
        vectors,
        labels,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keys_are_deterministic_sized_and_in_alphabet() {
        let a = keys(7, 50, 12);
        assert_eq!(a, keys(7, 50, 12));
        assert_ne!(a, keys(8, 50, 12));
        assert_eq!(a.len(), 50);
        assert!(a.iter().all(|k| k.len() == 12
            && k.bytes()
                .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit())));
        assert!(keys(7, 0, 12).is_empty());
    }

    #[test]
    fn payload_lengths_respect_bounds_even_when_swapped() {
        let a = payloads(3, 200, 4, 40);
        assert_eq!(a, payloads(3, 200, 4, 40));
        assert!(a.iter().all(|p| (4..=40).contains(&p.len())));
        let swapped = payloads(3, 200, 40, 4);
        assert!(swapped.iter().all(|p| (4..=40).contains(&p.len())));
        assert!(payloads(3, 5, 0, 0).iter().all(Vec::is_empty));
    }

    #[test]
    fn permutation_is_a_deterministic_bijection() {
        let p = permutation(11, 1000);
        assert_eq!(p, permutation(11, 1000));
        assert_ne!(p, permutation(12, 1000));
        let mut sorted = p.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..1000).collect::<Vec<_>>());
        assert!(permutation(1, 0).is_empty());
        assert_eq!(permutation(1, 1), vec![0]);
    }

    #[test]
    fn gaussian_has_mean_near_zero_and_variance_near_one() {
        let mut rng = Rng::new(99);
        let n = 50_000;
        let xs: Vec<f64> = (0..n).map(|_| gaussian(&mut rng)).collect();
        let mean = xs.iter().sum::<f64>() / n as f64;
        let var = xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
        assert!(xs.iter().all(|x| x.is_finite()));
        assert!(mean.abs() < 0.03, "mean {mean}");
        assert!((var - 1.0).abs() < 0.05, "var {var}");
    }

    #[test]
    fn zipf_refuses_zero_ranks_and_ranks_are_ordered_by_frequency() {
        assert_eq!(
            Zipf::new(0, 1.0).unwrap_err(),
            GenError::Zero("zipf rank count")
        );
        let zipf = Zipf::new(100, 1.0).unwrap();
        assert_eq!(zipf.len(), 100);
        assert!(!zipf.is_empty());
        let mut rng = Rng::new(5);
        let mut counts = [0usize; 100];
        for _ in 0..100_000 {
            counts[zipf.sample(&mut rng)] += 1;
        }
        assert!(counts[0] > counts[1] && counts[1] > counts[9] && counts[9] > counts[99]);
        // P(rank 0) = 1 / H_100 ~= 0.1928 for exponent 1.
        let p0 = counts[0] as f64 / 100_000.0;
        assert!((p0 - 0.1928).abs() < 0.01, "p0 {p0}");
    }

    #[test]
    fn zipf_single_rank_always_samples_zero() {
        let zipf = Zipf::new(1, 1.0).unwrap();
        let mut rng = Rng::new(2);
        assert!((0..100).all(|_| zipf.sample(&mut rng) == 0));
    }

    #[test]
    fn zipf_corpus_is_deterministic_and_in_vocabulary() {
        let a = zipf_corpus(21, 500, 30, 40, 1.0).unwrap();
        assert_eq!(a, zipf_corpus(21, 500, 30, 40, 1.0).unwrap());
        assert_eq!(a.len(), 30);
        assert!(a
            .iter()
            .all(|d| d.len() == 40 && d.iter().all(|&t| t < 500)));
        assert_eq!(
            zipf_corpus(21, 0, 3, 3, 1.0).unwrap_err(),
            GenError::Zero("zipf rank count")
        );
    }

    #[test]
    fn mixture_is_deterministic_shaped_and_clustered() {
        let m = gaussian_mixture(31, 8, 4, 400, 0.05).unwrap();
        assert_eq!(m, gaussian_mixture(31, 8, 4, 400, 0.05).unwrap());
        assert_eq!(m.vectors.len(), 400 * 8);
        assert_eq!(m.centres.len(), 4 * 8);
        assert_eq!(m.labels.len(), 400);
        assert!(m.labels.iter().all(|&l| l < 4));
        // With a small spread every vector is close to its own centre.
        for (i, &label) in m.labels.iter().enumerate() {
            let mut dist2 = 0.0f32;
            for d in 0..8 {
                let delta = m.vectors[i * 8 + d] - m.centres[label as usize * 8 + d];
                dist2 += delta * delta;
            }
            assert!(dist2 < 1.0, "vector {i} is {dist2} from its centre");
        }
    }

    #[test]
    fn mixture_refuses_zero_dimension_or_clusters() {
        assert_eq!(
            gaussian_mixture(1, 0, 2, 5, 1.0).unwrap_err(),
            GenError::Zero("dimension")
        );
        assert_eq!(
            gaussian_mixture(1, 2, 0, 5, 1.0).unwrap_err(),
            GenError::Zero("cluster count")
        );
        let empty = gaussian_mixture(1, 2, 2, 0, 1.0).unwrap();
        assert!(empty.vectors.is_empty() && empty.labels.is_empty());
    }

    #[test]
    fn gen_error_displays_the_parameter() {
        assert_eq!(
            GenError::Zero("dimension").to_string(),
            "dimension must be positive"
        );
    }
}
