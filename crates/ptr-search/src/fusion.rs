use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use ptr_types::{CapsuleId, Generation, Validity};

use crate::SearchHit;

/// The largest total of list weights a fusion accepts: `f32::MAX`.
///
/// Every fused score is at most the total weight (a list adds at most its
/// weight to a hit, since it names each capsule generation once and its
/// normalised contribution is at most one), so within this bound no fused
/// score overflows `f32`, whatever the rank constant, ranks or raw scores.
pub const MAX_TOTAL_WEIGHT: f32 = f32::MAX;

/// Why a fusion refused its input. Every refusal is decided from the
/// parameters and lists before any score is accumulated.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FusionError {
    /// The weight of list `list` (zero-based) is negative or not finite.
    InvalidWeight { list: usize },
    /// The list weights sum beyond [`MAX_TOTAL_WEIGHT`], where a hit ranked
    /// first everywhere could fuse to an infinite score and outrank every
    /// finite one.
    WeightTotal,
    /// The rank constant `k` is negative or not finite.
    InvalidRankConstant,
    /// Hit `index` (zero-based) of list `list` names a capsule generation an
    /// earlier hit of the same list already names. A ranked list holds each
    /// candidate once; a repeat would add its weight again.
    DuplicateHit { list: usize, index: usize },
    /// Hit `index` of list `list` has a NaN or infinite score, which score
    /// fusion cannot normalise. Rank fusion never reads scores.
    InvalidScore { list: usize, index: usize },
}

impl FusionError {
    /// A stable code for each refusal.
    pub fn code(&self) -> &'static str {
        match self {
            Self::InvalidWeight { .. } => "PTR_SEARCH_INVALID_WEIGHT",
            Self::WeightTotal => "PTR_SEARCH_WEIGHT_TOTAL",
            Self::InvalidRankConstant => "PTR_SEARCH_INVALID_RANK_CONSTANT",
            Self::DuplicateHit { .. } => "PTR_SEARCH_DUPLICATE_HIT",
            Self::InvalidScore { .. } => "PTR_SEARCH_INVALID_SCORE",
        }
    }
}

impl fmt::Display for FusionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidWeight { list } => {
                write!(
                    formatter,
                    "the weight of list {list} is negative or not finite"
                )
            }
            Self::WeightTotal => write!(
                formatter,
                "the list weights sum beyond {MAX_TOTAL_WEIGHT}, where a fused score can overflow"
            ),
            Self::InvalidRankConstant => {
                write!(formatter, "the rank constant is negative or not finite")
            }
            Self::DuplicateHit { list, index } => write!(
                formatter,
                "hit {index} of list {list} repeats a capsule generation of the same list"
            ),
            Self::InvalidScore { list, index } => {
                write!(
                    formatter,
                    "hit {index} of list {list} has a nonfinite score"
                )
            }
        }
    }
}

impl std::error::Error for FusionError {}

/// One ranked list entering fusion, with the weight its ranks carry.
#[derive(Clone, Copy, Debug)]
pub struct WeightedList<'a> {
    pub weight: f32,
    pub hits: &'a [SearchHit],
}

/// A fused result, keyed by capsule *and* generation.
#[derive(Clone, Debug, PartialEq)]
pub struct FusedHit {
    pub capsule: CapsuleId,
    pub generation: Generation,
    /// Finite, nonnegative and at most the total list weight.
    pub score: f32,
    /// Backends that returned this capsule generation, in first-seen order.
    pub backends: Vec<String>,
}

/// Check the parameters of a weighted rank fusion, as
/// [`weighted_rank_fusion`] does before it reads a hit: `k` finite and
/// nonnegative, then every weight finite and nonnegative, in list order, then
/// the weights summing to at most [`MAX_TOTAL_WEIGHT`].
///
/// A caller that must refuse bad parameters before doing the work its lists
/// come from (running the queries, say) applies this first; fusing afterwards
/// with the same parameters then refuses only a malformed list.
///
/// # Errors
/// [`FusionError::InvalidRankConstant`], [`FusionError::InvalidWeight`] for the
/// first bad weight, or [`FusionError::WeightTotal`].
pub fn check_rank_fusion<I>(weights: I, k: f32) -> Result<(), FusionError>
where
    I: IntoIterator<Item = f32>,
{
    if !(k.is_finite() && k >= 0.0) {
        return Err(FusionError::InvalidRankConstant);
    }
    check_weights(weights)
}

fn check_weights<I>(weights: I) -> Result<(), FusionError>
where
    I: IntoIterator<Item = f32>,
{
    // Summed in f64, which holds any number of f32 weights without
    // overflowing, so the total is compared rather than itself overflowing.
    let mut total = 0.0_f64;
    for (list, weight) in weights.into_iter().enumerate() {
        if !(weight.is_finite() && weight >= 0.0) {
            return Err(FusionError::InvalidWeight { list });
        }
        total += f64::from(weight);
    }
    if total > f64::from(MAX_TOTAL_WEIGHT) {
        return Err(FusionError::WeightTotal);
    }
    Ok(())
}

/// Refuse a list that names a capsule generation twice, and with `scores`, a
/// hit whose score is not finite.
fn check_lists(lists: &[WeightedList<'_>], scores: bool) -> Result<(), FusionError> {
    for (list, weighted) in lists.iter().enumerate() {
        let mut seen = BTreeSet::new();
        for (index, hit) in weighted.hits.iter().enumerate() {
            if scores && !hit.score.is_finite() {
                return Err(FusionError::InvalidScore { list, index });
            }
            if !seen.insert((&hit.capsule, hit.generation)) {
                return Err(FusionError::DuplicateHit { list, index });
            }
        }
    }
    Ok(())
}

/// Scores accumulated in `f64` per capsule generation, then rounded once.
#[derive(Default)]
struct Accumulator {
    fused: BTreeMap<(CapsuleId, Generation), (f64, Vec<String>)>,
}

impl Accumulator {
    fn add(&mut self, hit: &SearchHit, contribution: f64) {
        let (score, backends) = self
            .fused
            .entry((hit.capsule.clone(), hit.generation))
            .or_default();
        *score += contribution;
        if !backends.contains(&hit.backend) {
            backends.push(hit.backend.clone());
        }
    }

    /// Round each score to `f32` and order by score, then capsule, then
    /// generation. Rounding comes first, so hits reported with equal scores
    /// are exactly the ones ordered by capsule and generation.
    fn finish(self) -> Vec<FusedHit> {
        let mut out: Vec<FusedHit> = self
            .fused
            .into_iter()
            .map(|((capsule, generation), (score, backends))| FusedHit {
                capsule,
                generation,
                score: score as f32,
                backends,
            })
            .collect();
        out.sort_by(|a, b| {
            b.score
                .total_cmp(&a.score)
                .then_with(|| a.capsule.cmp(&b.capsule))
                .then_with(|| a.generation.cmp(&b.generation))
        });
        out
    }
}

/// Weighted reciprocal rank fusion: `score = sum_l weight_l / (k + rank_l)`,
/// with ranks starting at 1.
///
/// Results are keyed by capsule and generation, as in
/// [`crate::reciprocal_rank_fusion`], but retain both fields in each hit.
/// Two generations of one capsule are different evidence and are
/// never merged into one score; a stale generation that one backend still
/// returns cannot borrow the rank of the live one. Ties are broken by capsule,
/// then generation, so the order is deterministic.
///
/// Scores are accumulated in `f64` and rounded to `f32` once. Every fused
/// score is finite, nonnegative and at most the total weight: a list names a
/// hit once and adds at most its weight to it, since `k + rank >= 1`.
///
/// # Errors
/// Refuses, before accumulating anything, the parameters
/// [`check_rank_fusion`] refuses, then a list that names a capsule generation
/// twice ([`FusionError::DuplicateHit`]). Raw hit scores are never read.
pub fn weighted_rank_fusion(
    lists: &[WeightedList<'_>],
    k: f32,
) -> Result<Vec<FusedHit>, FusionError> {
    check_rank_fusion(lists.iter().map(|list| list.weight), k)?;
    check_lists(lists, false)?;
    let k = f64::from(k);
    let mut fused = Accumulator::default();
    for list in lists {
        let weight = f64::from(list.weight);
        for (rank, hit) in list.hits.iter().enumerate() {
            fused.add(hit, weight / (k + rank as f64 + 1.0));
        }
    }
    Ok(fused.finish())
}

/// Convex combination of per-list min-max normalised scores:
/// `score = sum_l weight_l * (s - min_l) / (max_l - min_l)`, where a result
/// absent from a list contributes `0` for it and a list whose scores are all
/// equal normalises them to `1`.
///
/// With weights tuned on a small set of labelled queries this beats rank
/// fusion in controlled comparisons (Bruch, Gai and Ingber, "An Analysis of
/// Fusion Functions for Hybrid Retrieval", TOIS 2023); without labelled
/// queries, [`weighted_rank_fusion`] is the safer default. Keyed by capsule and
/// generation, like rank fusion.
///
/// The combination is convex when the weights sum to one; the weights are not
/// rescaled. The range `max_l - min_l` and every normalised score are computed
/// in `f64`, where the difference of any two finite `f32` scores is finite, so
/// scores spanning the whole `f32` range normalise into `[0, 1]` rather than
/// overflowing. Every fused score is finite, nonnegative and at most the total
/// weight.
///
/// # Errors
/// Refuses, before accumulating anything, a weight that is negative or not
/// finite ([`FusionError::InvalidWeight`]), weights summing beyond
/// [`MAX_TOTAL_WEIGHT`] ([`FusionError::WeightTotal`]), a hit whose score is
/// not finite ([`FusionError::InvalidScore`]) and a list that names a capsule
/// generation twice ([`FusionError::DuplicateHit`]).
pub fn convex_score_fusion(lists: &[WeightedList<'_>]) -> Result<Vec<FusedHit>, FusionError> {
    check_weights(lists.iter().map(|list| list.weight))?;
    check_lists(lists, true)?;
    let mut fused = Accumulator::default();
    for list in lists {
        let (min, max) = list
            .hits
            .iter()
            .map(|hit| f64::from(hit.score))
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), score| {
                (lo.min(score), hi.max(score))
            });
        let range = max - min;
        let weight = f64::from(list.weight);
        for hit in list.hits {
            let normalised = if range > 0.0 {
                (f64::from(hit.score) - min) / range
            } else {
                1.0
            };
            fused.add(hit, weight * normalised);
        }
    }
    Ok(fused.finish())
}

/// Keep only hits the authoritative lifecycle answers [`Validity::Live`] for.
///
/// `validity` is asked about each hit's capsule *and* generation, and answers
/// as `PtrRuntime::generation_validity` does: `None` for an unknown capsule or
/// a generation ahead of it. Only `Some(Validity::Live)` keeps a hit, so a
/// superseded, revoked, disputed or unknown generation is dropped. Asking for
/// validity rather than the live generation is what drops a revoked one: a
/// revocation leaves the capsule's live generation in place and adds a
/// tombstone, so a revoked generation still equals the live one.
///
/// Filtering here is not a substitute for the promotion checks on
/// [`SearchHit`]; it keeps stale and revoked candidates from taking fusion
/// slots a live candidate would otherwise have had. Order, scores and
/// candidate metadata of the kept hits are unchanged.
pub fn retain_live<F>(hits: Vec<SearchHit>, mut validity: F) -> Vec<SearchHit>
where
    F: FnMut(&CapsuleId, Generation) -> Option<Validity>,
{
    hits.into_iter()
        .filter(|hit| validity(&hit.capsule, hit.generation) == Some(Validity::Live))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hit(capsule: &str, generation: u64, backend: &str) -> SearchHit {
        SearchHit::new(
            CapsuleId::from(capsule),
            Generation(generation),
            1.0,
            backend,
        )
    }

    #[test]
    fn two_generations_of_one_capsule_are_never_merged() {
        let lexical = [hit("a", 1, "lexical"), hit("b", 1, "lexical")];
        let dense = [hit("a", 2, "dense")];
        let fused = weighted_rank_fusion(
            &[
                WeightedList {
                    weight: 1.0,
                    hits: &lexical,
                },
                WeightedList {
                    weight: 1.0,
                    hits: &dense,
                },
            ],
            60.0,
        )
        .unwrap();
        assert_eq!(fused.len(), 3);
        assert!(fused
            .iter()
            .any(|f| f.capsule == CapsuleId::from("a") && f.generation == Generation(2)));
    }

    #[test]
    fn agreement_across_backends_outranks_a_single_first_place() {
        let lexical = [hit("solo", 1, "lexical"), hit("both", 1, "lexical")];
        let dense = [hit("both", 1, "dense")];
        let fused = weighted_rank_fusion(
            &[
                WeightedList {
                    weight: 1.0,
                    hits: &lexical,
                },
                WeightedList {
                    weight: 1.0,
                    hits: &dense,
                },
            ],
            60.0,
        )
        .unwrap();
        assert_eq!(fused[0].capsule, CapsuleId::from("both"));
        assert_eq!(fused[0].backends, vec!["lexical", "dense"]);
    }

    #[test]
    fn a_zero_weight_list_contributes_nothing() {
        let lexical = [hit("a", 1, "lexical")];
        let dense = [hit("b", 1, "dense")];
        let fused = weighted_rank_fusion(
            &[
                WeightedList {
                    weight: 1.0,
                    hits: &lexical,
                },
                WeightedList {
                    weight: 0.0,
                    hits: &dense,
                },
            ],
            60.0,
        )
        .unwrap();
        assert_eq!(fused[0].capsule, CapsuleId::from("a"));
        assert_eq!(fused[1].score, 0.0);
    }

    #[test]
    fn convex_fusion_normalises_each_list_before_weighting() {
        let scored = |capsule: &str, score: f32, backend: &str| {
            SearchHit::new(CapsuleId::from(capsule), Generation(1), score, backend)
        };
        // Lexical scores are unbounded, dense scores are cosines; after
        // normalisation "b" wins by being strong in both.
        let lexical = [
            scored("a", 40.0, "lexical"),
            scored("b", 30.0, "lexical"),
            scored("c", 10.0, "lexical"),
        ];
        let dense = [
            scored("b", 0.9, "dense"),
            scored("a", 0.2, "dense"),
            scored("c", 0.1, "dense"),
        ];
        let fused = convex_score_fusion(&[
            WeightedList {
                weight: 0.5,
                hits: &lexical,
            },
            WeightedList {
                weight: 0.5,
                hits: &dense,
            },
        ])
        .unwrap();
        assert_eq!(fused[0].capsule, CapsuleId::from("b"));
        assert!((fused[2].score - 0.0).abs() < 1e-6);
    }

    #[test]
    fn stale_revoked_and_unknown_generations_are_dropped() {
        let hits = vec![
            hit("a", 1, "x"),
            hit("a", 2, "x"),
            hit("revoked", 4, "x"),
            hit("gone", 1, "x"),
        ];
        let kept = retain_live(hits, |capsule, generation| match capsule.0.as_str() {
            "a" if generation == Generation(1) => Some(Validity::Superseded),
            "a" if generation == Generation(2) => Some(Validity::Live),
            // Revoked while still the capsule's live generation.
            "revoked" => Some(Validity::Revoked),
            _ => None,
        });
        assert_eq!(kept.len(), 1);
        assert_eq!(kept[0].generation, Generation(2));
    }
}
