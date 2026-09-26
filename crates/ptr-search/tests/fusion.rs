use ptr_search::{
    check_rank_fusion, convex_score_fusion, reciprocal_rank_fusion, retain_live,
    weighted_rank_fusion, FusionError, SearchHit, WeightedList, MAX_TOTAL_WEIGHT,
};
use ptr_types::{CapsuleId, Generation, Validity};

fn hit(id: &str, generation: u64, score: f32, backend: &str) -> SearchHit {
    SearchHit::new(CapsuleId::from(id), Generation(generation), score, backend)
}

#[test]
fn rank_fusion_uses_weights_and_ranks_instead_of_raw_backend_scores() {
    let lexical = [hit("a", 1, -10.0, "lexical"), hit("b", 1, 500.0, "lexical")];
    let dense = [hit("b", 1, 0.1, "dense")];
    let fused = weighted_rank_fusion(
        &[
            WeightedList {
                weight: 2.0,
                hits: &lexical,
            },
            WeightedList {
                weight: 1.0,
                hits: &dense,
            },
        ],
        1.0,
    )
    .unwrap();
    assert_eq!(fused.len(), 2);
    assert_eq!(fused[0].capsule, CapsuleId::from("b"));
    assert!((fused[0].score - (2.0 / 3.0 + 0.5)).abs() < 1e-6);
    assert_eq!(fused[1].score, 1.0);
}

#[test]
fn equal_scores_sort_by_capsule_then_generation_under_both_fusions() {
    let b = [hit("b", 1, 7.0, "dense")];
    let a2 = [hit("a", 2, 7.0, "dense")];
    let a1 = [hit("a", 1, 7.0, "lexical")];
    let lists = [
        WeightedList {
            weight: 1.0,
            hits: &b,
        },
        WeightedList {
            weight: 1.0,
            hits: &a2,
        },
        WeightedList {
            weight: 1.0,
            hits: &a1,
        },
    ];
    for fused in [
        weighted_rank_fusion(&lists, 60.0).unwrap(),
        convex_score_fusion(&lists).unwrap(),
    ] {
        let keys: Vec<_> = fused
            .iter()
            .map(|h| (h.capsule.0.as_str(), h.generation.0))
            .collect();
        assert_eq!(keys, vec![("a", 1), ("a", 2), ("b", 1)]);
    }
}

#[test]
fn constant_and_empty_lists_have_defined_convex_scores() {
    let constant = [hit("a", 1, -3.0, "lexical"), hit("b", 1, -3.0, "lexical")];
    let dense = [hit("a", 2, 0.9, "dense")];
    let fused = convex_score_fusion(&[
        WeightedList {
            weight: 0.25,
            hits: &constant,
        },
        WeightedList {
            weight: 0.75,
            hits: &dense,
        },
        WeightedList {
            weight: 100.0,
            hits: &[],
        },
    ])
    .unwrap();
    assert_eq!(fused.len(), 3);
    assert_eq!(
        (fused[0].capsule.0.as_str(), fused[0].generation),
        ("a", Generation(2))
    );
    assert_eq!(
        fused.iter().map(|h| h.score).collect::<Vec<_>>(),
        vec![0.75, 0.25, 0.25]
    );
    assert!(convex_score_fusion(&[]).unwrap().is_empty());
    assert!(weighted_rank_fusion(&[], 60.0).unwrap().is_empty());
}

#[test]
fn backend_provenance_is_unique_and_keeps_first_seen_order() {
    let dense = [hit("a", 1, 1.0, "dense")];
    let lexical = [hit("a", 1, 1.0, "lexical")];
    let lists = [
        WeightedList {
            weight: 0.25,
            hits: &dense,
        },
        WeightedList {
            weight: 0.5,
            hits: &lexical,
        },
        WeightedList {
            weight: 0.25,
            hits: &dense,
        },
    ];
    for fused in [
        weighted_rank_fusion(&lists, 1.0).unwrap(),
        convex_score_fusion(&lists).unwrap(),
    ] {
        assert_eq!(fused.len(), 1);
        assert_eq!(fused[0].backends, vec!["dense", "lexical"]);
    }
}

#[test]
fn filtering_live_hits_preserves_order_scores_and_candidate_metadata() {
    let first = hit("b", 3, 0.2, "dense");
    let second = hit("a", 2, 9.0, "lexical");
    let kept = retain_live(
        vec![
            hit("a", 1, 100.0, "stale"),
            first.clone(),
            hit("unknown", 1, 99.0, "dense"),
            second.clone(),
            hit("a", 3, 98.0, "future"),
        ],
        |id, generation| match (id.0.as_str(), generation.0) {
            ("a", 1) => Some(Validity::Superseded),
            ("a", 2) | ("b", 3) => Some(Validity::Live),
            _ => None,
        },
    );
    assert_eq!(kept, vec![first, second]);
}

/// The lifecycle rule `PtrRuntime::generation_validity` applies: a revocation
/// adds a tombstone and leaves the live generation in place.
struct Lifecycle {
    live: Vec<(&'static str, u64)>,
    tombstones: Vec<(&'static str, u64)>,
}

impl Lifecycle {
    fn live_generation(&self, capsule: &CapsuleId) -> Option<Generation> {
        self.live
            .iter()
            .find(|(id, _)| *id == capsule.0)
            .map(|(_, generation)| Generation(*generation))
    }

    fn generation_validity(&self, capsule: &CapsuleId, generation: Generation) -> Option<Validity> {
        if self
            .tombstones
            .contains(&(capsule.0.as_str(), generation.0))
        {
            return Some(Validity::Revoked);
        }
        match self.live_generation(capsule) {
            Some(live) if live == generation => Some(Validity::Live),
            Some(live) if live > generation => Some(Validity::Superseded),
            _ => None,
        }
    }
}

#[test]
fn a_revoked_generation_is_dropped_although_it_is_still_the_live_one() {
    let lifecycle = Lifecycle {
        live: vec![("revoked", 1), ("kept", 1)],
        tombstones: vec![("revoked", 1)],
    };
    let revoked = CapsuleId::from("revoked");
    // What an equality filter over the live generation saw: still live.
    assert_eq!(lifecycle.live_generation(&revoked), Some(Generation(1)));
    let kept = retain_live(
        vec![
            hit("revoked", 1, 9.0, "dense"),
            hit("kept", 1, 1.0, "dense"),
        ],
        |capsule, generation| lifecycle.generation_validity(capsule, generation),
    );
    assert_eq!(kept, vec![hit("kept", 1, 1.0, "dense")]);
}

#[test]
fn weights_whose_total_could_overflow_a_fused_score_are_refused() {
    let lexical = [hit("a", 1, 1.0, "lexical"), hit("b", 1, 1.0, "lexical")];
    let dense = [hit("a", 1, 1.0, "dense"), hit("b", 1, 1.0, "dense")];
    let lists = |weight: f32| {
        [
            WeightedList {
                weight,
                hits: &lexical,
            },
            WeightedList {
                weight,
                hits: &dense,
            },
        ]
    };
    // With k = 0 a hit first in both lists fused to f32::MAX + f32::MAX, an
    // infinite score that outranked every finite one.
    assert_eq!(
        weighted_rank_fusion(&lists(f32::MAX), 0.0),
        Err(FusionError::WeightTotal)
    );
    assert_eq!(
        convex_score_fusion(&lists(f32::MAX)),
        Err(FusionError::WeightTotal)
    );
    assert_eq!(
        check_rank_fusion([f32::MAX, f32::MAX], 0.0),
        Err(FusionError::WeightTotal)
    );
    // At the bound the largest score is the total weight, and finite.
    let half = MAX_TOTAL_WEIGHT / 2.0;
    for fused in [
        weighted_rank_fusion(&lists(half), 0.0).unwrap(),
        convex_score_fusion(&lists(half)).unwrap(),
    ] {
        assert_eq!(fused[0].score, MAX_TOTAL_WEIGHT);
        assert!(fused.iter().all(|hit| hit.score.is_finite()));
    }
    // Unit weights, as the unweighted form uses, are far inside the bound.
    assert!(reciprocal_rank_fusion(&[lexical.to_vec(), dense.to_vec()], 0.0).is_ok());
}

#[test]
fn invalid_weights_and_rank_constants_are_refused_before_any_hit_is_read() {
    let duplicated = [hit("a", 1, 1.0, "x"), hit("a", 1, 1.0, "x")];
    for bad in [-1.0, -f32::MIN_POSITIVE, f32::NAN, f32::INFINITY] {
        let lists = [
            WeightedList {
                weight: 1.0,
                hits: &duplicated,
            },
            WeightedList {
                weight: bad,
                hits: &duplicated,
            },
        ];
        assert_eq!(
            weighted_rank_fusion(&lists, 60.0),
            Err(FusionError::InvalidWeight { list: 1 })
        );
        assert_eq!(
            convex_score_fusion(&lists),
            Err(FusionError::InvalidWeight { list: 1 })
        );
        let lists = [WeightedList {
            weight: 1.0,
            hits: &duplicated,
        }];
        assert_eq!(
            weighted_rank_fusion(&lists, bad),
            Err(FusionError::InvalidRankConstant)
        );
    }
}

#[test]
fn a_list_naming_one_capsule_generation_twice_is_refused() {
    let repeated = [
        hit("a", 1, 3.0, "x"),
        hit("a", 2, 2.0, "x"),
        hit("a", 1, 1.0, "x"),
    ];
    let lists = [
        WeightedList {
            weight: 0.5,
            hits: &[],
        },
        WeightedList {
            weight: 0.5,
            hits: &repeated,
        },
    ];
    // Summed, each repeat added the list's weight again, past the total.
    assert_eq!(
        weighted_rank_fusion(&lists, 0.0),
        Err(FusionError::DuplicateHit { list: 1, index: 2 })
    );
    assert_eq!(
        convex_score_fusion(&lists),
        Err(FusionError::DuplicateHit { list: 1, index: 2 })
    );
    assert_eq!(
        reciprocal_rank_fusion(&[repeated.to_vec()], 60.0),
        Err(FusionError::DuplicateHit { list: 0, index: 2 })
    );
}

#[test]
fn scores_spanning_the_whole_f32_range_normalise_into_zero_to_one() {
    let wide = [
        hit("top", 1, f32::MAX, "lexical"),
        hit("middle", 1, 0.0, "lexical"),
        hit("bottom", 1, -f32::MAX, "lexical"),
    ];
    let fused = convex_score_fusion(&[WeightedList {
        weight: 1.0,
        hits: &wide,
    }])
    .unwrap();
    // max - min overflowed f32 to infinity, which made the top hit NaN.
    let scores: Vec<(&str, f32)> = fused
        .iter()
        .map(|hit| (hit.capsule.0.as_str(), hit.score))
        .collect();
    assert_eq!(scores, vec![("top", 1.0), ("middle", 0.5), ("bottom", 0.0)]);
}

#[test]
fn score_fusion_refuses_a_nonfinite_score_and_rank_fusion_never_reads_it() {
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let hits = [hit("a", 1, 1.0, "x"), hit("b", 1, bad, "x")];
        let lists = [WeightedList {
            weight: 1.0,
            hits: &hits,
        }];
        assert_eq!(
            convex_score_fusion(&lists),
            Err(FusionError::InvalidScore { list: 0, index: 1 })
        );
        let fused = weighted_rank_fusion(&lists, 60.0).unwrap();
        assert_eq!(fused[0].capsule, CapsuleId::from("a"));
    }
}

#[test]
fn every_fusion_refusal_has_a_stable_code() {
    let codes = [
        (
            FusionError::InvalidWeight { list: 0 },
            "PTR_SEARCH_INVALID_WEIGHT",
        ),
        (FusionError::WeightTotal, "PTR_SEARCH_WEIGHT_TOTAL"),
        (
            FusionError::InvalidRankConstant,
            "PTR_SEARCH_INVALID_RANK_CONSTANT",
        ),
        (
            FusionError::DuplicateHit { list: 0, index: 1 },
            "PTR_SEARCH_DUPLICATE_HIT",
        ),
        (
            FusionError::InvalidScore { list: 0, index: 1 },
            "PTR_SEARCH_INVALID_SCORE",
        ),
    ];
    for (error, code) in codes {
        assert_eq!(error.code(), code);
        assert!(!error.to_string().is_empty());
    }
}
