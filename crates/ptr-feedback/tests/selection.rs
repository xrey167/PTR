use ptr_feedback::{Candidate, Population};
use ptr_types::{CandidateId, Probability, VerificationLevel};
use ptr_verifier::{Finding, VerificationReport, VerificationStatus};

fn report(status: VerificationStatus, score: f32) -> VerificationReport {
    VerificationReport {
        status,
        level: VerificationLevel::Deterministic,
        score: Probability::new(score).unwrap(),
        findings: vec![],
    }
}

fn hard(code: &str) -> Finding {
    Finding {
        code: code.into(),
        message: String::new(),
        hard: true,
    }
}

fn candidate(id: &str, report: VerificationReport) -> Candidate<&'static str> {
    Candidate {
        id: CandidateId::from(id),
        value: "v",
        verification: Some(report),
    }
}

#[test]
fn failed_high_score_cannot_beat_passing_candidate() {
    let mut population = Population::default();
    population.push(candidate("failed", report(VerificationStatus::Fail, 0.99)));
    population.push(candidate("passed", report(VerificationStatus::Pass, 0.70)));

    let best = population.best_verified(VerificationLevel::Deterministic);
    assert_eq!(best.unwrap().id.to_string(), "passed");
}

/// Regression: a `Pass` that carries a hard finding used to be selected on its status alone.
#[test]
fn a_forged_pass_with_a_hard_finding_is_never_selected() {
    let mut forged = report(VerificationStatus::Pass, 1.0);
    forged.findings.push(hard("target"));
    let mut population = Population::default();
    population.push(candidate("forged", forged));
    assert!(population
        .best_verified(VerificationLevel::Unverified)
        .is_none());

    population.push(candidate("honest", report(VerificationStatus::Pass, 0.2)));
    let best = population.best_verified(VerificationLevel::Deterministic);
    assert_eq!(best.unwrap().id.to_string(), "honest");
}

#[test]
fn a_pass_below_the_required_level_is_not_selected() {
    let mut shallow = report(VerificationStatus::Pass, 1.0);
    shallow.level = VerificationLevel::LatentAgreement;
    let mut population = Population::default();
    population.push(candidate("shallow", shallow));
    assert!(population
        .best_verified(VerificationLevel::Deterministic)
        .is_none());
    assert_eq!(
        population
            .best_verified(VerificationLevel::LatentAgreement)
            .unwrap()
            .id
            .to_string(),
        "shallow"
    );
}

#[test]
fn unverified_unknown_and_disputed_candidates_are_never_selected() {
    let mut population = Population::default();
    population.push(Candidate {
        id: CandidateId::from("unverified"),
        value: "v",
        verification: None,
    });
    population.push(candidate(
        "unknown",
        report(VerificationStatus::Unknown, 1.0),
    ));
    population.push(candidate(
        "disputed",
        report(VerificationStatus::Disputed, 1.0),
    ));
    assert!(population
        .best_verified(VerificationLevel::Unverified)
        .is_none());
}
