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
            &matrix,
            PublicSuite {
                serving: 0.625,
                candidate: 0.5
            }
        )
        .passed());
    let report = gate.evaluate(
        &matrix,
        PublicSuite {
            serving: 0.75,
            candidate: 0.5,
        },
    );
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
