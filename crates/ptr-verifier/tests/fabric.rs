use ptr_types::{Probability, VerificationLevel};
use ptr_verifier::{
    Finding, NamedVerifier, VerificationReport, VerificationStatus, Verifier, VerifierFabric,
};

struct Pass;
impl Verifier<u32> for Pass {
    fn verify(&self, _: &u32) -> VerificationReport {
        VerificationReport {
            status: VerificationStatus::Pass,
            level: VerificationLevel::SampleVerified,
            score: Probability::new(0.9).unwrap(),
            findings: vec![],
        }
    }
}
impl NamedVerifier<u32> for Pass {
    fn name(&self) -> &'static str {
        "pass"
    }
}

struct HardFail;
impl Verifier<u32> for HardFail {
    fn verify(&self, _: &u32) -> VerificationReport {
        VerificationReport {
            status: VerificationStatus::Fail,
            level: VerificationLevel::Deterministic,
            score: Probability::new(0.1).unwrap(),
            findings: vec![Finding {
                code: "hard".into(),
                message: "deterministic contradiction".into(),
                hard: true,
            }],
        }
    }
}
impl NamedVerifier<u32> for HardFail {
    fn name(&self) -> &'static str {
        "hard-fail"
    }
}

#[test]
fn deterministic_failure_dominates_pass() {
    let mut fabric = VerifierFabric::default();
    fabric.push(Pass);
    fabric.push(HardFail);
    let report = fabric.verify(&42);
    assert_eq!(report.status, VerificationStatus::Fail);
    assert_eq!(report.level, VerificationLevel::Deterministic);
    assert!(report.findings.iter().any(|finding| finding.hard));
}
