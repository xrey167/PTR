use ptr_lineage::{
    AccuracyMatrix, AdapterId, BoundGateReport, ForgettingGate, GateSubject, PublicSuite,
};

fn report(public: PublicSuite) -> BoundGateReport {
    let gate = ForgettingGate {
        max_average_forgetting: 0.5,
        max_task_forgetting: 0.5,
        min_backward_transfer: -0.5,
        max_public_regression: 0.02,
    };
    let matrix = AccuracyMatrix::new(vec![vec![0.9, 0.0], vec![0.1, 0.9]]).unwrap();
    gate.evaluate(&AdapterId::from("cand"), &matrix, public)
        .unwrap()
        .bind(GateSubject::new(
            [1; 32], [2; 32], [3; 32], [4; 32], [5; 32],
        ))
}

const OK: PublicSuite = PublicSuite {
    serving: 0.8,
    candidate: 0.8,
};
const BAD: PublicSuite = PublicSuite {
    serving: 0.9,
    candidate: 0.1,
};

#[test]
fn round_trips_passing_and_failing_reports() {
    for r in [report(OK), report(BAD)] {
        let bytes = r.encode_canonical();
        let back = BoundGateReport::decode_canonical(&bytes).unwrap();
        assert_eq!(back, r);
        assert_eq!(back.digest(), r.digest());
    }
    assert!(!report(BAD).passed());
}

#[test]
fn rejects_malformed_and_non_canonical_bytes() {
    let bytes = report(BAD).encode_canonical();
    for n in 0..bytes.len() {
        assert!(
            BoundGateReport::decode_canonical(&bytes[..n]).is_err(),
            "prefix {n}"
        );
    }
    let mut trailing = bytes.clone();
    trailing.push(0);
    assert!(BoundGateReport::decode_canonical(&trailing).is_err());
    for i in 0..bytes.len() {
        let mut flipped = bytes.clone();
        flipped[i] ^= 1;
        assert!(
            BoundGateReport::decode_canonical(&flipped).is_err(),
            "byte {i}"
        );
    }
    let huge = vec![0u8; (1 << 20) + 1];
    assert!(BoundGateReport::decode_canonical(&huge).is_err());
}
