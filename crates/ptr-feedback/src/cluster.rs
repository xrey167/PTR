//! Near-duplicate clustering, de-duplication and a diversity measure for a [`Population`].
//!
//! A candidate's text is normalized (lower-cased, split on non-alphanumeric characters) and
//! turned into word shingles. Two candidates are similar when the Jaccard index of their shingle
//! sets reaches the threshold; clusters are the connected components of that relation, so the
//! result does not depend on candidate order. A seeded MinHash signature is kept as a compact
//! fingerprint for estimating similarity; clustering itself uses the exact Jaccard index.
//! Similarity is lexical only: it says nothing about meaning.

use std::collections::BTreeSet;
use std::error::Error;
use std::fmt;

use ptr_types::{CandidateId, VerificationLevel};

use crate::candidate::{Candidate, Population};
use crate::text::CandidateText;

/// Most candidates one clustering call accepts (the comparison is quadratic).
pub const MAX_CLUSTER_CANDIDATES: usize = 2048;
/// Longest candidate text, in bytes, that is fingerprinted.
pub const MAX_FINGERPRINT_TEXT_BYTES: usize = 1 << 20;
/// Most MinHash functions in a signature.
pub const MAX_MINHASH_FUNCTIONS: usize = 256;

/// Why clustering could not run.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ClusterError {
    /// The threshold is not a finite number in `(0, 1]`.
    InvalidThreshold,
    /// The shingle size is zero or above 16 words.
    InvalidShingleSize,
    /// The MinHash function count is zero or above [`MAX_MINHASH_FUNCTIONS`].
    InvalidSignatureSize,
    /// More than [`MAX_CLUSTER_CANDIDATES`] candidates.
    TooManyCandidates,
    /// A candidate text exceeds [`MAX_FINGERPRINT_TEXT_BYTES`].
    TextTooLong,
}

impl fmt::Display for ClusterError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::InvalidThreshold => "the similarity threshold must be in (0, 1]",
            Self::InvalidShingleSize => "the shingle size must be between 1 and 16 words",
            Self::InvalidSignatureSize => "the MinHash signature size is out of range",
            Self::TooManyCandidates => "too many candidates to cluster",
            Self::TextTooLong => "a candidate text is too long to fingerprint",
        })
    }
}

impl Error for ClusterError {}

/// Similarity at or above which two candidates are the same cluster.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SimilarityThreshold(f64);

impl SimilarityThreshold {
    /// # Errors
    /// [`ClusterError::InvalidThreshold`] unless `value` is finite and in `(0, 1]`.
    pub fn new(value: f64) -> Result<Self, ClusterError> {
        if value.is_finite() && value > 0.0 && value <= 1.0 {
            Ok(Self(value))
        } else {
            Err(ClusterError::InvalidThreshold)
        }
    }

    pub fn get(self) -> f64 {
        self.0
    }
}

/// How candidates are fingerprinted and grouped.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ClusterOptions {
    pub threshold: SimilarityThreshold,
    /// Words per shingle (1 to 16).
    pub shingle_size: usize,
    /// Number of seeded MinHash functions in each signature.
    pub signature_size: usize,
    /// Seed of the MinHash function family.
    pub seed: u64,
    /// A candidate that passes at this level is preferred as a cluster representative.
    pub required_level: VerificationLevel,
}

impl ClusterOptions {
    /// Word trigrams, 64 hash functions, seed 0.
    pub fn new(threshold: SimilarityThreshold, required_level: VerificationLevel) -> Self {
        Self {
            threshold,
            shingle_size: 3,
            signature_size: 64,
            seed: 0,
            required_level,
        }
    }

    fn validate(&self) -> Result<(), ClusterError> {
        if self.shingle_size == 0 || self.shingle_size > 16 {
            return Err(ClusterError::InvalidShingleSize);
        }
        if self.signature_size == 0 || self.signature_size > MAX_MINHASH_FUNCTIONS {
            return Err(ClusterError::InvalidSignatureSize);
        }
        Ok(())
    }
}

/// A deterministic fingerprint of one text.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Fingerprint {
    digest: u64,
    shingles: Vec<u64>,
    signature: Vec<u64>,
    /// The options that define the hash family: `(seed, shingle_size)`. The signature length is
    /// `signature.len()`. Signatures are comparable slot by slot only within one family.
    family: (u64, usize),
}

impl Fingerprint {
    /// # Errors
    /// Returns [`ClusterError`] for invalid options or a text above the byte bound.
    pub fn of(text: &str, options: &ClusterOptions) -> Result<Self, ClusterError> {
        options.validate()?;
        Self::build(text, options)
    }

    fn build(text: &str, options: &ClusterOptions) -> Result<Self, ClusterError> {
        if text.len() > MAX_FINGERPRINT_TEXT_BYTES {
            return Err(ClusterError::TextTooLong);
        }
        let tokens = tokenize(text);
        let mut digest = FNV_OFFSET;
        for token in &tokens {
            digest = fnv_extend(digest, token.as_bytes());
            digest = fnv_extend(digest, &[0x1f]);
        }
        let shingles = shingle_set(&tokens, options.shingle_size);
        let signature = (0..options.signature_size)
            .map(|index| {
                let key = mix(options.seed ^ mix(index as u64 + 1));
                shingles
                    .iter()
                    .map(|shingle| mix(shingle ^ key))
                    .min()
                    .unwrap_or(u64::MAX)
            })
            .collect();
        Ok(Self {
            digest,
            shingles,
            signature,
            family: (options.seed, options.shingle_size),
        })
    }

    /// Digest of the normalized token sequence: equal exactly for texts that differ only in
    /// case, punctuation or spacing.
    pub fn digest(&self) -> u64 {
        self.digest
    }

    /// Exact Jaccard index of the two shingle sets. Texts with equal digests are 1.0; two
    /// texts without any token are equal.
    pub fn jaccard(&self, other: &Self) -> f64 {
        if self.digest == other.digest {
            return 1.0;
        }
        let (mut i, mut j, mut shared) = (0, 0, 0usize);
        while i < self.shingles.len() && j < other.shingles.len() {
            match self.shingles[i].cmp(&other.shingles[j]) {
                std::cmp::Ordering::Less => i += 1,
                std::cmp::Ordering::Greater => j += 1,
                std::cmp::Ordering::Equal => {
                    shared += 1;
                    i += 1;
                    j += 1;
                }
            }
        }
        let union = self.shingles.len() + other.shingles.len() - shared;
        if union == 0 {
            1.0
        } else {
            shared as f64 / union as f64
        }
    }

    /// MinHash estimate of the Jaccard index: the fraction of equal signature slots. `None`
    /// when the signatures come from different families (different size, seed or shingle
    /// size), because their slots are then different hash functions or shingle sets.
    pub fn estimated_jaccard(&self, other: &Self) -> Option<f64> {
        if self.family != other.family
            || self.signature.len() != other.signature.len()
            || self.signature.is_empty()
        {
            return None;
        }
        let equal = self
            .signature
            .iter()
            .zip(&other.signature)
            .filter(|(a, b)| a == b)
            .count();
        Some(equal as f64 / self.signature.len() as f64)
    }
}

/// A group of near-duplicate candidates, as indices into the population.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Cluster {
    /// The best-verified member, then the lowest id.
    pub representative: usize,
    /// Every member including the representative, ordered by (id, index).
    pub members: Vec<usize>,
    pub representative_id: CandidateId,
}

/// Group `population` into near-duplicate clusters. Clusters are ordered by their first member
/// (by id, then index) and do not depend on the order candidates were pushed. A cluster's
/// representative is a candidate that passes at the required level if any does, then the
/// highest verifier score among passing ones, then the lowest id; a score never lifts a
/// candidate that does not pass above one that does.
///
/// # Errors
/// Returns [`ClusterError`] for invalid options, too many candidates or a text above the bound.
pub fn cluster_candidates<T: CandidateText>(
    population: &Population<T>,
    options: &ClusterOptions,
) -> Result<Vec<Cluster>, ClusterError> {
    options.validate()?;
    let candidates = &population.candidates;
    if candidates.len() > MAX_CLUSTER_CANDIDATES {
        return Err(ClusterError::TooManyCandidates);
    }
    let prints = fingerprints(candidates, options)?;
    let mut parent: Vec<usize> = (0..candidates.len()).collect();
    for i in 0..prints.len() {
        for j in (i + 1)..prints.len() {
            if prints[i].jaccard(&prints[j]) >= options.threshold.get() {
                let (a, b) = (find(&mut parent, i), find(&mut parent, j));
                if a != b {
                    parent[a.max(b)] = a.min(b);
                }
            }
        }
    }
    let mut groups: Vec<Vec<usize>> = Vec::new();
    let mut slot_of_root: Vec<Option<usize>> = vec![None; candidates.len()];
    for index in 0..candidates.len() {
        let root = find(&mut parent, index);
        match slot_of_root[root] {
            Some(slot) => groups[slot].push(index),
            None => {
                slot_of_root[root] = Some(groups.len());
                groups.push(vec![index]);
            }
        }
    }
    let by_id = |a: &usize, b: &usize| {
        candidates[*a]
            .id
            .cmp(&candidates[*b].id)
            .then_with(|| a.cmp(b))
    };
    let mut clusters: Vec<Cluster> = groups
        .into_iter()
        .map(|mut members| {
            members.sort_by(by_id);
            let mut representative = members[0];
            for next in members.iter().copied().skip(1) {
                if better_representative(&candidates[next], &candidates[representative], options) {
                    representative = next;
                }
            }
            Cluster {
                representative,
                representative_id: candidates[representative].id.clone(),
                members,
            }
        })
        .collect();
    clusters.sort_by(|a, b| by_id(&a.members[0], &b.members[0]));
    Ok(clusters)
}

/// `population` with only each cluster's representative kept, in the original order.
///
/// # Errors
/// As [`cluster_candidates`].
pub fn dedup_population<T: CandidateText + Clone>(
    population: &Population<T>,
    options: &ClusterOptions,
) -> Result<Population<T>, ClusterError> {
    let clusters = cluster_candidates(population, options)?;
    let mut keep: Vec<usize> = clusters
        .iter()
        .map(|cluster| cluster.representative)
        .collect();
    keep.sort_unstable();
    Ok(Population {
        candidates: keep
            .into_iter()
            .map(|index| population.candidates[index].clone())
            .collect(),
    })
}

/// Diversity of a population: the mean pairwise Jaccard distance (`1 - similarity`), in
/// `[0, 1]`. Fewer than two candidates have no pair and score 0.
///
/// # Errors
/// As [`cluster_candidates`].
pub fn population_diversity<T: CandidateText>(
    population: &Population<T>,
    options: &ClusterOptions,
) -> Result<f64, ClusterError> {
    options.validate()?;
    if population.candidates.len() > MAX_CLUSTER_CANDIDATES {
        return Err(ClusterError::TooManyCandidates);
    }
    let prints = fingerprints(&population.candidates, options)?;
    if prints.len() < 2 {
        return Ok(0.0);
    }
    let mut total = 0.0;
    let mut pairs = 0u64;
    for i in 0..prints.len() {
        for j in (i + 1)..prints.len() {
            total += 1.0 - prints[i].jaccard(&prints[j]);
            pairs += 1;
        }
    }
    Ok(total / pairs as f64)
}

fn fingerprints<T: CandidateText>(
    candidates: &[Candidate<T>],
    options: &ClusterOptions,
) -> Result<Vec<Fingerprint>, ClusterError> {
    candidates
        .iter()
        .map(|candidate| Fingerprint::build(&candidate.value.candidate_text(), options))
        .collect()
}

fn better_representative<T>(
    next: &Candidate<T>,
    best: &Candidate<T>,
    options: &ClusterOptions,
) -> bool {
    let rank = |candidate: &Candidate<T>| {
        let passes = candidate.passes(options.required_level);
        let score = candidate
            .verification
            .as_ref()
            .map_or(0.0, |report| f64::from(report.score.get()));
        (passes, score)
    };
    let (next_passes, next_score) = rank(next);
    let (best_passes, best_score) = rank(best);
    if next_passes != best_passes {
        return next_passes;
    }
    // Both pass and the next scores strictly higher; otherwise the incumbent, which has the
    // lowest id because members arrive ordered by id, keeps its place.
    next_passes && next_score > best_score
}

fn find(parent: &mut [usize], mut node: usize) -> usize {
    while parent[node] != node {
        parent[node] = parent[parent[node]];
        node = parent[node];
    }
    node
}

fn tokenize(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|word| !word.is_empty())
        .map(str::to_lowercase)
        .collect()
}

fn shingle_set(tokens: &[String], size: usize) -> Vec<u64> {
    if tokens.is_empty() {
        return Vec::new();
    }
    let windows = if tokens.len() < size {
        1
    } else {
        tokens.len() - size + 1
    };
    let mut set: BTreeSet<u64> = BTreeSet::new();
    for start in 0..windows {
        let end = (start + size).min(tokens.len());
        let mut hash = FNV_OFFSET;
        for token in &tokens[start..end] {
            hash = fnv_extend(hash, token.as_bytes());
            hash = fnv_extend(hash, &[0x1f]);
        }
        set.insert(mix(hash));
    }
    set.into_iter().collect()
}

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

fn fnv_extend(mut hash: u64, bytes: &[u8]) -> u64 {
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

/// splitmix64 finalizer.
pub(crate) fn mix(value: u64) -> u64 {
    let mut z = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ptr_types::Probability;
    use ptr_verifier::{VerificationReport, VerificationStatus};

    fn options(threshold: f64) -> ClusterOptions {
        ClusterOptions::new(
            SimilarityThreshold::new(threshold).unwrap(),
            VerificationLevel::Deterministic,
        )
    }

    fn report(status: VerificationStatus, score: f32) -> VerificationReport {
        VerificationReport {
            status,
            level: VerificationLevel::Deterministic,
            score: Probability::new(score).unwrap(),
            findings: vec![],
        }
    }

    fn cand(id: &str, text: &str, report: Option<VerificationReport>) -> Candidate<String> {
        Candidate {
            id: CandidateId::from(id),
            value: text.to_owned(),
            verification: report,
        }
    }

    fn pop(items: Vec<Candidate<String>>) -> Population<String> {
        Population { candidates: items }
    }

    #[test]
    fn threshold_rejects_values_outside_zero_one() {
        for bad in [0.0, -0.1, 1.5, f64::NAN, f64::INFINITY] {
            assert_eq!(
                SimilarityThreshold::new(bad),
                Err(ClusterError::InvalidThreshold)
            );
        }
        assert!(SimilarityThreshold::new(1.0).is_ok());
    }

    #[test]
    fn case_and_punctuation_do_not_change_the_digest() {
        let o = options(0.5);
        let a = Fingerprint::of("Move the file, then verify.", &o).unwrap();
        let b = Fingerprint::of("move  the FILE then verify", &o).unwrap();
        assert_eq!(a.digest(), b.digest());
        assert_eq!(a.jaccard(&b), 1.0);
    }

    #[test]
    fn jaccard_and_minhash_agree_on_disjoint_and_identical_sets() {
        let o = options(0.5);
        let a = Fingerprint::of("alpha beta gamma delta epsilon", &o).unwrap();
        let c = Fingerprint::of("one two three four five six", &o).unwrap();
        assert_eq!(a.jaccard(&c), 0.0);
        assert_eq!(a.estimated_jaccard(&a), Some(1.0));
        assert!(a.estimated_jaccard(&c).unwrap() < 0.2);
        let small_options = ClusterOptions {
            signature_size: 8,
            ..o
        };
        let small = Fingerprint::of("x", &small_options).unwrap();
        assert_eq!(a.estimated_jaccard(&small), None);
    }

    #[test]
    fn estimates_are_only_defined_within_one_minhash_family() {
        let base = options(0.5);
        let text = "alpha beta gamma delta epsilon zeta";
        let a = Fingerprint::of(text, &base).unwrap();
        let other_seed = ClusterOptions { seed: 7, ..base };
        let other_shingle = ClusterOptions {
            shingle_size: 2,
            ..base
        };
        let same_family = Fingerprint::of(text, &base).unwrap();
        assert_eq!(a.estimated_jaccard(&same_family), Some(1.0));
        let seeded = Fingerprint::of(text, &other_seed).unwrap();
        assert_eq!(a.signature.len(), seeded.signature.len());
        assert_eq!(a.estimated_jaccard(&seeded), None);
        let shingled = Fingerprint::of(text, &other_shingle).unwrap();
        assert_eq!(a.estimated_jaccard(&shingled), None);
    }

    #[test]
    fn short_texts_use_one_whole_text_shingle() {
        let o = options(0.5);
        let a = Fingerprint::of("hi there", &o).unwrap();
        let b = Fingerprint::of("hi there", &o).unwrap();
        let c = Fingerprint::of("bye there", &o).unwrap();
        assert_eq!(a.jaccard(&b), 1.0);
        assert_eq!(a.jaccard(&c), 0.0);
    }

    #[test]
    fn near_duplicates_cluster_and_distinct_texts_do_not() {
        let base = "open the config file and set the retry count to three then restart";
        let near = "open the config file and set the retry count to three then restart now";
        let p = pop(vec![
            cand("a", base, None),
            cand(
                "b",
                "write a poem about the sea and the stars tonight",
                None,
            ),
            cand("c", near, None),
        ]);
        let clusters = cluster_candidates(&p, &options(0.7)).unwrap();
        assert_eq!(clusters.len(), 2);
        assert_eq!(clusters[0].members, [0, 2]);
        assert_eq!(clusters[1].members, [1]);
    }

    #[test]
    fn representative_is_best_verified_then_lowest_id() {
        let text = "the same candidate text repeated for every member of the cluster";
        let p = pop(vec![
            cand("a", text, Some(report(VerificationStatus::Fail, 0.99))),
            cand("b", text, Some(report(VerificationStatus::Pass, 0.60))),
            cand("c", text, Some(report(VerificationStatus::Pass, 0.60))),
            cand("d", text, Some(report(VerificationStatus::Pass, 0.40))),
        ]);
        let clusters = cluster_candidates(&p, &options(0.9)).unwrap();
        assert_eq!(clusters.len(), 1);
        assert_eq!(clusters[0].representative_id.to_string(), "b");
        let unverified = pop(vec![cand("z", text, None), cand("m", text, None)]);
        let clusters = cluster_candidates(&unverified, &options(0.9)).unwrap();
        assert_eq!(clusters[0].representative_id.to_string(), "m");
    }

    #[test]
    fn clustering_does_not_depend_on_push_order() {
        let t1 = "alpha beta gamma delta epsilon zeta eta theta";
        let t2 = "completely different words appear in this other sentence here";
        let forward = pop(vec![
            cand("a", t1, None),
            cand("b", t2, None),
            cand("c", t1, None),
        ]);
        let backward = pop(vec![
            cand("c", t1, None),
            cand("b", t2, None),
            cand("a", t1, None),
        ]);
        let ids = |p: &Population<String>| -> Vec<Vec<String>> {
            cluster_candidates(p, &options(0.8))
                .unwrap()
                .iter()
                .map(|c| {
                    c.members
                        .iter()
                        .map(|i| p.candidates[*i].id.to_string())
                        .collect()
                })
                .collect()
        };
        assert_eq!(ids(&forward), ids(&backward));
    }

    #[test]
    fn dedup_keeps_one_per_cluster_in_original_order() {
        let t = "keep exactly one of these identical candidates please";
        let p = pop(vec![
            cand("a", t, None),
            cand("b", "something else entirely different in wording", None),
            cand("c", t, Some(report(VerificationStatus::Pass, 0.5))),
        ]);
        let kept = dedup_population(&p, &options(0.9)).unwrap();
        let ids: Vec<String> = kept.candidates.iter().map(|c| c.id.to_string()).collect();
        assert_eq!(ids, ["b", "c"]);
    }

    #[test]
    fn diversity_is_zero_for_clones_and_high_for_distinct_texts() {
        let o = options(0.5);
        let t = "same text same text same text";
        let clones = pop(vec![cand("a", t, None), cand("b", t, None)]);
        assert_eq!(population_diversity(&clones, &o).unwrap(), 0.0);
        let distinct = pop(vec![
            cand("a", "red green blue yellow orange purple", None),
            cand("b", "one two three four five six seven", None),
        ]);
        assert_eq!(population_diversity(&distinct, &o).unwrap(), 1.0);
        assert_eq!(population_diversity(&pop(vec![]), &o).unwrap(), 0.0);
    }

    #[test]
    fn bounds_and_invalid_options_are_typed_errors() {
        let o = options(0.5);
        let many = pop((0..=MAX_CLUSTER_CANDIDATES)
            .map(|i| cand(&i.to_string(), "x", None))
            .collect());
        assert_eq!(
            cluster_candidates(&many, &o).unwrap_err(),
            ClusterError::TooManyCandidates
        );
        let long = "a".repeat(MAX_FINGERPRINT_TEXT_BYTES + 1);
        assert_eq!(
            Fingerprint::of(&long, &o).unwrap_err(),
            ClusterError::TextTooLong
        );
        let bad_shingle = ClusterOptions {
            shingle_size: 0,
            ..o
        };
        assert_eq!(
            Fingerprint::of("a", &bad_shingle).unwrap_err(),
            ClusterError::InvalidShingleSize
        );
        let bad_sig = ClusterOptions {
            signature_size: 0,
            ..o
        };
        assert_eq!(
            Fingerprint::of("a", &bad_sig).unwrap_err(),
            ClusterError::InvalidSignatureSize
        );
    }

    #[test]
    fn the_seed_changes_the_signature_but_not_the_exact_similarity() {
        let a = options(0.5);
        let b = ClusterOptions { seed: 7, ..a };
        let text = "a few words to hash into a signature here";
        let fa = Fingerprint::of(text, &a).unwrap();
        let fb = Fingerprint::of(text, &b).unwrap();
        assert_ne!(fa.signature, fb.signature);
        assert_eq!(fa.digest(), fb.digest());
    }
}
