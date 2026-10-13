//! Public-API tests of the big-integer arithmetic verifier.

use ptr_verifier::{ArithmeticVerifier, VerificationStatus, Verifier};

fn check(task: &str, candidate: &str) -> (VerificationStatus, Vec<String>) {
    let report = ArithmeticVerifier::new(task).verify(&candidate.to_owned());
    (
        report.status,
        report.findings.into_iter().map(|f| f.code).collect(),
    )
}

const E20: &str = "100000000000000000000";

#[test]
fn ten_to_the_20_times_ten_to_the_20_over_ten_to_the_20_passes_and_fails_exactly() {
    let task = format!("{E20} * {E20} / {E20}");
    assert_eq!(check(&task, E20), (VerificationStatus::Pass, vec![]));
    assert_eq!(
        check(&task, "100000000000000000001"),
        (VerificationStatus::Fail, vec!["arith.value".into()])
    );
}

#[test]
fn tasks_beyond_the_bounds_stay_unknown() {
    // Final value above 10^30.
    assert_eq!(
        check(&format!("{E20} * {E20}"), "1"),
        (VerificationStatus::Unknown, vec!["arith.task".into()])
    );
    // A literal that does not fit the integer kernel.
    assert_eq!(
        check(&"9".repeat(600), "1"),
        (VerificationStatus::Unknown, vec!["arith.task".into()])
    );
}

#[test]
fn previously_decidable_inputs_are_unchanged() {
    for (task, candidate, status) in [
        ("1 + 2 * 3", "7", VerificationStatus::Pass),
        ("7 / 3", "7/3", VerificationStatus::Pass),
        ("1/2 + 1/3", "5/6", VerificationStatus::Pass),
        ("1 / 0", "1", VerificationStatus::Unknown),
        ("1 + 2 * 3", "9", VerificationStatus::Fail),
    ] {
        assert_eq!(check(task, candidate).0, status, "{task}");
    }
}
