use ptr_lineage::{aimer_score, rank_experts_for_pruning, ExpertWeights, LineageError};

fn close(left: f64, right: f64) {
    assert!((left - right).abs() < 1e-12, "{left} != {right}");
}

#[test]
fn uniform_magnitudes_score_one() {
    close(aimer_score(&[2.0; 16]).unwrap(), 1.0);
    let alternating: Vec<f64> = (0..16)
        .map(|i| if i % 2 == 0 { 2.0 } else { -2.0 })
        .collect();
    close(aimer_score(&alternating).unwrap(), 1.0);
}

#[test]
fn one_hot_scores_inverse_root_n() {
    for n in [1_usize, 4, 25, 1000] {
        let mut weights = vec![0.0; n];
        weights[n / 2] = -7.0;
        close(aimer_score(&weights).unwrap(), 1.0 / (n as f64).sqrt());
    }
}

#[test]
fn hand_computed_example() {
    close(
        aimer_score(&[3.0, 4.0]).unwrap(),
        7.0 / (2.0_f64.sqrt() * 5.0),
    );
}

#[test]
fn score_is_scale_invariant_without_overflow() {
    let base = [3.0, -4.0, 0.5, 1.25];
    let expected = aimer_score(&base).unwrap();
    for scale in [1e300, 1e-300] {
        let scaled: Vec<f64> = base.iter().map(|value| value * scale).collect();
        let score = aimer_score(&scaled).unwrap();
        assert!(score.is_finite());
        close(score, expected);
    }
}

#[test]
fn unscorable_vectors_are_refused() {
    assert_eq!(
        aimer_score(&[]),
        Err(LineageError::Empty { field: "weights" })
    );
    assert_eq!(
        aimer_score(&[0.0; 4]),
        Err(LineageError::ZeroNorm { field: "weights" })
    );
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert_eq!(
            aimer_score(&[1.0, bad]),
            Err(LineageError::NonFinite { field: "weights" })
        );
    }
}

#[test]
fn highest_score_is_pruned_first_and_ties_keep_input_order() {
    let spread = [1.0, 1.0, 1.0, 1.0];
    let peaked = [10.0, 0.0, 0.0, 0.0];
    let middle = [3.0, 4.0, 0.0, 0.0];
    let experts = [
        ExpertWeights {
            id: "peaked",
            weights: &peaked,
        },
        ExpertWeights {
            id: "spread-a",
            weights: &spread,
        },
        ExpertWeights {
            id: "middle",
            weights: &middle,
        },
        ExpertWeights {
            id: "spread-b",
            weights: &spread,
        },
    ];
    assert_eq!(
        rank_experts_for_pruning(&experts).unwrap(),
        vec!["spread-a", "spread-b", "middle", "peaked"]
    );
}

#[test]
fn ranking_refuses_bad_input() {
    let empty: [ExpertWeights<'_, u32>; 0] = [];
    assert_eq!(
        rank_experts_for_pruning(&empty),
        Err(LineageError::Empty { field: "experts" })
    );

    let short = [1.0, 2.0];
    let long = [1.0, 2.0, 3.0];
    let mismatched = [
        ExpertWeights {
            id: 0_u32,
            weights: &long,
        },
        ExpertWeights {
            id: 1,
            weights: &short,
        },
    ];
    assert_eq!(
        rank_experts_for_pruning(&mismatched),
        Err(LineageError::ShapeMismatch {
            field: "expert weights",
            expected: 3,
            actual: 2,
        })
    );

    let zeros = [0.0, 0.0, 0.0];
    let unscorable = [
        ExpertWeights {
            id: 0_u32,
            weights: &long,
        },
        ExpertWeights {
            id: 7,
            weights: &zeros,
        },
    ];
    assert_eq!(
        rank_experts_for_pruning(&unscorable),
        Err(LineageError::ExpertRejected {
            expert: "7".to_string(),
            source: Box::new(LineageError::ZeroNorm { field: "weights" }),
        })
    );
}
