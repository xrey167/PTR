use std::collections::BTreeMap;

use ptr_types::{CapsuleId, Generation};

use crate::SearchHit;

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
    pub score: f32,
    /// Backends that returned this capsule generation, in first-seen order.
    pub backends: Vec<String>,
}

/// Weighted reciprocal rank fusion: `score = sum_l weight_l / (k + rank_l)`,
/// with ranks starting at 1.
///
/// Unlike [`crate::reciprocal_rank_fusion`], results are keyed by capsule and
/// generation. Two generations of one capsule are different evidence and are
/// never merged into one score; a stale generation that one backend still
/// returns cannot borrow the rank of the live one. Ties are broken by capsule,
/// then generation, so the order is deterministic.
pub fn weighted_rank_fusion(lists: &[WeightedList<'_>], k: f32) -> Vec<FusedHit> {
    let mut fused: BTreeMap<(CapsuleId, Generation), FusedHit> = BTreeMap::new();
    for list in lists {
        for (rank, hit) in list.hits.iter().enumerate() {
            let entry = fused
                .entry((hit.capsule.clone(), hit.generation))
                .or_insert_with(|| FusedHit {
                    capsule: hit.capsule.clone(),
                    generation: hit.generation,
                    score: 0.0,
                    backends: Vec::new(),
                });
            entry.score += list.weight / (k + rank as f32 + 1.0);
            if !entry.backends.contains(&hit.backend) {
                entry.backends.push(hit.backend.clone());
            }
        }
    }
    let mut out: Vec<FusedHit> = fused.into_values().collect();
    out.sort_by(|a, b| {
        b.score
            .total_cmp(&a.score)
            .then_with(|| a.capsule.cmp(&b.capsule))
            .then_with(|| a.generation.cmp(&b.generation))
    });
    out
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
pub fn convex_score_fusion(lists: &[WeightedList<'_>]) -> Vec<FusedHit> {
    let mut fused: BTreeMap<(CapsuleId, Generation), FusedHit> = BTreeMap::new();
    for list in lists {
        let (min, max) = list
            .hits
            .iter()
            .fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), hit| {
                (lo.min(hit.score), hi.max(hit.score))
            });
        for hit in list.hits {
            let normalised = if max > min {
                (hit.score - min) / (max - min)
            } else {
                1.0
            };
            let entry = fused
                .entry((hit.capsule.clone(), hit.generation))
                .or_insert_with(|| FusedHit {
                    capsule: hit.capsule.clone(),
                    generation: hit.generation,
                    score: 0.0,
                    backends: Vec::new(),
                });
            entry.score += list.weight * normalised;
            if !entry.backends.contains(&hit.backend) {
                entry.backends.push(hit.backend.clone());
            }
        }
    }
    let mut out: Vec<FusedHit> = fused.into_values().collect();
    out.sort_by(|a, b| {
        b.score
            .total_cmp(&a.score)
            .then_with(|| a.capsule.cmp(&b.capsule))
            .then_with(|| a.generation.cmp(&b.generation))
    });
    out
}

/// Keep only hits whose generation is the capsule's live generation.
///
/// `live` answers from the authoritative lifecycle (a revoked or unknown
/// capsule answers `None`). Filtering here is not a substitute for the
/// promotion checks on [`SearchHit`]; it keeps stale candidates from taking
/// fusion slots a live candidate would otherwise have had.
pub fn retain_live<F>(hits: Vec<SearchHit>, live: F) -> Vec<SearchHit>
where
    F: Fn(&CapsuleId) -> Option<Generation>,
{
    hits.into_iter()
        .filter(|hit| live(&hit.capsule) == Some(hit.generation))
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
        );
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
        );
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
        );
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
        ]);
        assert_eq!(fused[0].capsule, CapsuleId::from("b"));
        assert!((fused[2].score - 0.0).abs() < 1e-6);
    }

    #[test]
    fn stale_and_unknown_generations_are_dropped() {
        let hits = vec![hit("a", 1, "x"), hit("a", 2, "x"), hit("gone", 1, "x")];
        let kept = retain_live(hits, |capsule| {
            (capsule == &CapsuleId::from("a")).then_some(Generation(2))
        });
        assert_eq!(kept.len(), 1);
        assert_eq!(kept[0].generation, Generation(2));
    }
}
