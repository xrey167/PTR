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

#[test]
fn rounding_prone_vector_scores_identically_in_any_order() {
    let tiny = 2.0_f64.powi(-53);
    let forward = [1.0, tiny, tiny];
    let reversed = [tiny, tiny, 1.0];
    let middle = [tiny, 1.0, tiny];
    let expected = aimer_score(&forward).unwrap().to_bits();
    assert_eq!(aimer_score(&reversed).unwrap().to_bits(), expected);
    assert_eq!(aimer_score(&middle).unwrap().to_bits(), expected);
}

fn lcg_weights(count: usize) -> Vec<f64> {
    let mut state: u64 = 0x2545_F491_4F6C_DD1D;
    (0..count)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let mantissa = (state >> 11) as f64 / (1_u64 << 53) as f64;
            let exponent = i32::try_from((state >> 3) % 40).unwrap() - 20;
            let sign = if state & 1 == 0 { 1.0 } else { -1.0 };
            sign * (mantissa + 0.01) * 2.0_f64.powi(exponent)
        })
        .collect()
}

#[test]
fn mixed_magnitude_vector_scores_identically_under_permutations() {
    let base = lcg_weights(64);
    let expected = aimer_score(&base).unwrap().to_bits();

    let mut reversed = base.clone();
    reversed.reverse();
    let mut rotated = base.clone();
    rotated.rotate_left(17);
    let (evens, odds): (Vec<f64>, Vec<f64>) = (
        base.iter().copied().step_by(2).collect(),
        base.iter().copied().skip(1).step_by(2).collect(),
    );
    let interleaved: Vec<f64> = odds.iter().chain(evens.iter()).copied().collect();
    let mut sorted = base.clone();
    sorted.sort_by(f64::total_cmp);

    for permuted in [reversed, rotated, interleaved, sorted] {
        assert_eq!(permuted.len(), base.len());
        assert_eq!(aimer_score(&permuted).unwrap().to_bits(), expected);
    }
}

#[test]
fn permuted_experts_tie_and_keep_input_order() {
    let tiny = 2.0_f64.powi(-53);
    let forward = [1.0, tiny, tiny];
    let reversed = [tiny, tiny, 1.0];
    let a = ExpertWeights {
        id: "a",
        weights: &forward,
    };
    let b = ExpertWeights {
        id: "b",
        weights: &reversed,
    };
    assert_eq!(rank_experts_for_pruning(&[a, b]).unwrap(), vec!["a", "b"]);
    assert_eq!(rank_experts_for_pruning(&[b, a]).unwrap(), vec!["b", "a"]);
}
