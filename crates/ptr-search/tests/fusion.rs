use ptr_search::{convex_score_fusion, retain_live, weighted_rank_fusion, SearchHit, WeightedList};
use ptr_types::{CapsuleId, Generation};

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
    );
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
        weighted_rank_fusion(&lists, 60.0),
        convex_score_fusion(&lists),
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
    ]);
    assert_eq!(fused.len(), 3);
    assert_eq!(
        (fused[0].capsule.0.as_str(), fused[0].generation),
        ("a", Generation(2))
    );
    assert_eq!(
        fused.iter().map(|h| h.score).collect::<Vec<_>>(),
        vec![0.75, 0.25, 0.25]
    );
    assert!(convex_score_fusion(&[]).is_empty());
    assert!(weighted_rank_fusion(&[], 60.0).is_empty());
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
        weighted_rank_fusion(&lists, 1.0),
        convex_score_fusion(&lists),
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
            hit("revoked", 1, 99.0, "dense"),
            second.clone(),
            hit("a", 3, 98.0, "future"),
        ],
        |id| match id.0.as_str() {
            "a" => Some(Generation(2)),
            "b" => Some(Generation(3)),
            _ => None,
        },
    );
    assert_eq!(kept, vec![first, second]);
}
