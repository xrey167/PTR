use ptr_feedback::{Candidate, Population};
use ptr_types::{CandidateId, Probability, VerificationLevel};
use ptr_verifier::{VerificationReport, VerificationStatus};

fn report(status: VerificationStatus, score: f32) -> VerificationReport {
    VerificationReport {
        status,
        level: VerificationLevel::Deterministic,
        score: Probability::new(score).unwrap(),
        findings: vec![],
    }
}

#[test]
fn failed_high_score_cannot_beat_passing_candidate() {
    let mut population = Population::default();
    population.push(Candidate {
        id: CandidateId::from("failed"),
        value: "bad",
        verification: Some(report(VerificationStatus::Fail, 0.99)),
    });
    population.push(Candidate {
        id: CandidateId::from("passed"),
        value: "good",
        verification: Some(report(VerificationStatus::Pass, 0.70)),
    });

    assert_eq!(population.best_verified().unwrap().id.to_string(), "passed");
}
