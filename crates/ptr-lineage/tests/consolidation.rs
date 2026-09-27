use ptr_lineage::{
    ties_merge, AccuracyMatrix, ForgettingGate, GateViolation, LineageError, PublicSuite,
};

#[test]
fn ties_elects_by_magnitude_and_averages_only_nonzero_agreeing_updates() {
    let result = ties_merge(
        &[
            vec![2.0, 2.0, 0.0],
            vec![2.0, -2.0, 6.0],
            vec![-5.0, 0.0, 0.0],
        ],
        1.0,
    )
    .unwrap();
    // A single large negative wins; an exact sign tie cancels; zeros do not dilute 6.
    assert_eq!(result, vec![-5.0, 0.0, 6.0]);
}

#[test]
fn trimming_rounds_up_and_breaks_equal_magnitudes_by_position() {
    assert_eq!(
        ties_merge(&[vec![4.0, -4.0, 4.0]], 0.1).unwrap(),
        vec![4.0, 0.0, 0.0]
    );
    assert_eq!(
        ties_merge(&[vec![4.0, -4.0, 4.0]], 0.5).unwrap(),
        vec![4.0, -4.0, 0.0]
    );
}

#[test]
fn ties_merges_entries_near_the_largest_finite_value_without_overflow() {
    // Summing the agreeing entries before dividing would overflow to
    // infinity; their mean is the entry itself.
    assert_eq!(
        ties_merge(&[vec![f64::MAX, 1.0], vec![f64::MAX, 1.0]], 1.0).unwrap(),
        vec![f64::MAX, 1.0]
    );
    assert_eq!(
        ties_merge(&[vec![1e308], vec![1e308]], 1.0).unwrap(),
        vec![1e308]
    );
    assert_eq!(
        ties_merge(&[vec![-f64::MAX], vec![-f64::MAX], vec![-f64::MAX]], 1.0).unwrap(),
        vec![-f64::MAX]
    );
    // Every merged entry lies between the smallest and largest entry it
    // averages, so a merge of finite vectors is finite.
    let merged = ties_merge(&[vec![f64::MAX], vec![1e308], vec![f64::MAX / 3.0]], 1.0).unwrap();
    assert!(merged[0].is_finite() && merged[0] >= f64::MAX / 3.0 && merged[0] <= f64::MAX);
}

#[test]
fn the_elected_sign_is_that_of_the_true_sum_when_a_running_sum_would_overflow() {
    // The running sum reaches +inf after two entries and stays there, although
    // the column sums to -1e308: the sign is minus and the merged entry the
    // mean of the three negative entries.
    assert_eq!(
        ties_merge(
            &[
                vec![1e308],
                vec![1e308],
                vec![-1e308],
                vec![-1e308],
                vec![-1e308]
            ],
            1.0
        )
        .unwrap(),
        vec![-1e308]
    );
    // Equal magnitudes of both signs still cancel exactly.
    assert_eq!(
        ties_merge(
            &[
                vec![f64::MAX],
                vec![f64::MAX],
                vec![-f64::MAX],
                vec![-f64::MAX]
            ],
            1.0
        )
        .unwrap(),
        vec![0.0]
    );
}

#[test]
fn an_entry_too_small_to_survive_rescaling_still_decides_the_sign_when_the_large_ones_cancel() {
    // The running sum overflows at MAX + MAX; with an unbounded exponent range
    // the large entries then cancel exactly and 1e-20 makes the sum positive.
    // Dividing the column by 2^1023 would flush 1e-20 to zero and elect no
    // sign. The merged entry is the mean of MAX, MAX and 1e-20.
    assert_eq!(
        ties_merge(
            &[
                vec![f64::MAX],
                vec![f64::MAX],
                vec![-f64::MAX],
                vec![-f64::MAX],
                vec![1e-20]
            ],
            1.0
        )
        .unwrap(),
        vec![f64::MAX / 3.0 * 2.0]
    );
    assert_eq!(
        ties_merge(
            &[
                vec![-1e308],
                vec![-1e308],
                vec![1e308],
                vec![1e308],
                vec![-1e-300]
            ],
            1.0
        )
        .unwrap(),
        vec![-1e308 / 3.0 * 2.0]
    );
}

#[test]
fn summaries_of_extreme_finite_scores_do_not_overflow() {
    // The mean of two scores of 1e308 is 1e308, not their overflowing sum.
    let matrix = AccuracyMatrix::new(vec![vec![1e308, 0.0], vec![1e308, 1e308]]).unwrap();
    assert_eq!(matrix.average_accuracy(), 1e308);
    // Changes of +2e308 and -2e308 overflow to +inf and -inf, whose sum is
    // NaN; the exact backward transfer is (2e308 - 2e308 - 1) / 3.
    let matrix = AccuracyMatrix::new(vec![
        vec![-1e308, 0.0, 0.0, 0.0],
        vec![0.0, 1e308, 0.0, 0.0],
        vec![0.0, 0.0, 1.0, 0.0],
        vec![1e308, -1e308, 0.0, 0.0],
    ])
    .unwrap();
    assert!(
        (matrix.backward_transfer() + 1.0 / 3.0).abs() < 1e-12,
        "{}",
        matrix.backward_transfer()
    );
    // Task 0 is forgotten by 2e308, beyond f64::MAX, and task 1 not at all:
    // the average forgetting is 1e308 and the backward transfer -1e308.
    let matrix = AccuracyMatrix::new(vec![
        vec![1e308, 0.0, 0.0],
        vec![0.0, 0.0, 0.0],
        vec![-1e308, 0.0, 0.0],
    ])
    .unwrap();
    assert_eq!(matrix.task_forgetting(), vec![f64::INFINITY, 0.0]);
    assert_eq!(matrix.average_forgetting(), 1e308);
    assert_eq!(matrix.backward_transfer(), -1e308);
}

#[test]
fn summaries_of_subnormal_scores_lose_nothing_before_they_are_rounded_once() {
    // 2^-1074, the smallest positive subnormal: half of it rounds to zero.
    let unit = 5e-324;
    // Task 0 is forgotten by one unit, which halving the scores would lose.
    let matrix = AccuracyMatrix::new(vec![vec![unit, 0.0], vec![0.0, 0.0]]).unwrap();
    assert_eq!(matrix.task_forgetting(), vec![unit]);
    assert_eq!(matrix.average_forgetting(), unit);
    assert_eq!(matrix.backward_transfer(), -unit);
    // And a forgetting of one unit, from two units to one, is not doubled.
    let matrix = AccuracyMatrix::new(vec![vec![2.0 * unit, 0.0], vec![unit, 0.0]]).unwrap();
    assert_eq!(matrix.task_forgetting(), vec![unit]);
    assert_eq!(matrix.average_forgetting(), unit);
    assert_eq!(matrix.backward_transfer(), -unit);
    // A gate that allows no forgetting at all sees it.
    let strict = ForgettingGate {
        max_average_forgetting: 0.0,
        max_task_forgetting: 0.0,
        min_backward_transfer: 0.0,
        max_public_regression: 0.0,
    };
    let matrix = AccuracyMatrix::new(vec![vec![unit, 0.0], vec![0.0, 0.0]]).unwrap();
    let report = strict
        .evaluate(
            &ptr_lineage::AdapterId::from("candidate"),
            &matrix,
            PublicSuite {
                serving: 0.5,
                candidate: 0.5,
            },
        )
        .unwrap();
    assert_eq!(
        report.violations(),
        &[
            GateViolation::AverageForgetting {
                observed: unit,
                limit: 0.0
            },
            GateViolation::TaskForgetting {
                task: 0,
                observed: unit,
                limit: 0.0
            },
            GateViolation::BackwardTransfer {
                observed: -unit,
                limit: 0.0
            },
        ]
    );
    // A mean is rounded once, so it is zero exactly when its exact value is
    // at most half a unit: one unit of forgetting over three tasks is a
    // third of a unit, which rounds to zero, and so is the backward transfer
    // (minus a third), while two units over three round to one. Each task's
    // own forgetting is exact, so the strict gate still sees the forgetting
    // through its per-task limit.
    let mut values = vec![vec![0.0; 4]; 4];
    values[0][0] = unit;
    let matrix = AccuracyMatrix::new(values.clone()).unwrap();
    assert_eq!(matrix.task_forgetting(), vec![unit, 0.0, 0.0]);
    assert_eq!(matrix.average_forgetting(), 0.0);
    assert_eq!(matrix.backward_transfer(), 0.0);
    let report = strict
        .evaluate(
            &ptr_lineage::AdapterId::from("candidate"),
            &matrix,
            PublicSuite {
                serving: 0.5,
                candidate: 0.5,
            },
        )
        .unwrap();
    assert_eq!(
        report.violations(),
        &[GateViolation::TaskForgetting {
            task: 0,
            observed: unit,
            limit: 0.0
        }]
    );
    values[0][0] = 2.0 * unit;
    let matrix = AccuracyMatrix::new(values).unwrap();
    assert_eq!(matrix.average_forgetting(), unit);
    assert_eq!(matrix.backward_transfer(), -unit);
    // The mean of scores is their sum divided once: (5 * 2^50 + 7) units
    // over five is 2^50 + 1.4 units, which rounds to 2^50 + 1. Dividing a
    // significand first and rounding again to the subnormal grid would give
    // 2^50 + 1.5 and then 2^50 + 2.
    let total = (5.0 * 2f64.powi(50) + 7.0) * unit;
    let last = vec![total - 4.0 * unit, unit, unit, unit, unit];
    assert_eq!(last.iter().sum::<f64>(), total);
    let mut values = vec![vec![0.0; 5]; 4];
    values.push(last);
    let matrix = AccuracyMatrix::new(values).unwrap();
    assert_eq!(matrix.average_accuracy(), total / 5.0);
    assert_eq!(matrix.average_accuracy(), (2f64.powi(50) + 1.0) * unit);
    // So is the mean TIES takes of the agreeing entries.
    let merged = ties_merge(
        &[
            vec![total - 4.0 * unit],
            vec![unit],
            vec![unit],
            vec![unit],
            vec![unit],
        ],
        1.0,
    )
    .unwrap();
    assert_eq!(merged, vec![total / 5.0]);
}

#[test]
fn merging_refuses_nonfinite_updates_and_invalid_densities() {
    assert_eq!(
        ties_merge(&[], 1.0),
        Err(LineageError::Empty {
            field: "task vectors"
        })
    );
    for invalid in [0.0, -1.0, 1.01, f64::NAN, f64::INFINITY] {
        assert!(matches!(
            ties_merge(&[vec![1.0]], invalid),
            Err(LineageError::InvalidParameter {
                field: "density",
                ..
            })
        ));
    }
    for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert_eq!(
            ties_merge(&[vec![1.0, invalid]], 1.0),
            Err(LineageError::NonFinite {
                field: "task vector"
            })
        );
    }
}

#[test]
fn forgetting_gate_accepts_equality_at_every_threshold() {
    let matrix = AccuracyMatrix::new(vec![vec![0.75, 0.0], vec![0.5, 1.0]]).unwrap();
    let gate = ForgettingGate {
        max_average_forgetting: 0.25,
        max_task_forgetting: 0.25,
        min_backward_transfer: -0.25,
        max_public_regression: 0.125,
    };
    assert!(gate
        .evaluate(
            &ptr_lineage::AdapterId::from("candidate"),
            &matrix,
            PublicSuite {
                serving: 0.625,
                candidate: 0.5
            }
        )
        .unwrap()
        .passed());
    let report = gate
        .evaluate(
            &ptr_lineage::AdapterId::from("candidate"),
            &matrix,
            PublicSuite {
                serving: 0.75,
                candidate: 0.5,
            },
        )
        .unwrap();
    assert_eq!(
        report.violations(),
        &[GateViolation::PublicRegression {
            observed: 0.25,
            limit: 0.125
        }]
    );
}

#[test]
fn a_single_task_has_no_forgetting_or_backward_transfer() {
    let matrix = AccuracyMatrix::new(vec![vec![0.75]]).unwrap();
    assert_eq!(matrix.average_accuracy(), 0.75);
    assert_eq!(matrix.backward_transfer(), 0.0);
    assert_eq!(matrix.average_forgetting(), 0.0);
    assert!(matrix.task_forgetting().is_empty());
}
