//! Public-API tests of the evidence promotion verifier.

use ptr_types::{CommitIndex, Probability, VerificationLevel};
use ptr_verifier::{
    code_spec, Evidence, EvidenceOrigin, Finding, NamedVerifier, PromotionClaim, PromotionPolicy,
    PromotionTarget, PromotionVerifier, VerificationReport, VerificationStatus, Verifier,
    VerifierFabric,
};

fn ev(id: &str, origin: EvidenceOrigin, source: &str, committed: bool) -> Evidence {
    Evidence {
        id: id.into(),
        origin,
        source: source.into(),
        commit: committed.then_some(CommitIndex(1)),
    }
}

fn claim(target: PromotionTarget, evidence: Vec<Evidence>) -> PromotionClaim {
    PromotionClaim {
        claim: "the build is reproducible".into(),
        target,
        evidence,
        deterministic_findings: vec![],
    }
}

fn codes(report: &VerificationReport) -> Vec<&str> {
    report.findings.iter().map(|f| f.code.as_str()).collect()
}

#[test]
fn every_promotion_code_is_in_the_table_and_hard() {
    let verifier = PromotionVerifier::new(
        PromotionPolicy::new(2)
            .unwrap()
            .with_max_evidence(3)
            .unwrap(),
    );
    let mut seen = std::collections::BTreeSet::new();
    let mut run = |c: PromotionClaim| {
        for code in codes(&verifier.verify(&c)) {
            let spec = code_spec(code).unwrap_or_else(|| panic!("{code} not in the table"));
            assert!(spec.hard, "{code}");
            seen.insert(code.to_owned());
        }
    };
    let good = ev("a", EvidenceOrigin::Ledger, "l", true);
    run(claim(PromotionTarget::Verified, vec![]));
    run(claim(
        PromotionTarget::Verified,
        vec![ev("s", EvidenceOrigin::Search, "x", true)],
    ));
    run(claim(
        PromotionTarget::Verified,
        vec![ev("u", EvidenceOrigin::Raw, "r", false)],
    ));
    run(claim(
        PromotionTarget::Authoritative,
        vec![ev("r", EvidenceOrigin::Raw, "r", true)],
    ));
    run(claim(PromotionTarget::Verified, vec![good.clone(); 4]));
    run(claim(
        PromotionTarget::Verified,
        vec![ev("", EvidenceOrigin::Raw, "r", true)],
    ));
    let mut contradicted = claim(PromotionTarget::Verified, vec![good]);
    contradicted.deterministic_findings = vec![Finding {
        code: "fact.contradicted".into(),
        message: "m".into(),
        hard: true,
    }];
    run(contradicted);
    for expected in [
        "promotion.no_evidence",
        "promotion.derived_only",
        "promotion.uncommitted",
        "promotion.insufficient_sources",
        "promotion.no_ledger_evidence",
        "promotion.oversize",
        "promotion.malformed",
        "promotion.contradicted",
    ] {
        assert!(seen.contains(expected), "{expected} was never emitted");
    }
}

#[test]
fn two_independent_committed_sources_promote_to_authoritative() {
    let verifier = PromotionVerifier::new(PromotionPolicy::new(2).unwrap());
    let c = claim(
        PromotionTarget::Authoritative,
        vec![
            ev("e1", EvidenceOrigin::Ledger, "ledger", true),
            ev("e2", EvidenceOrigin::Raw, "vendor-feed", true),
            ev("e3", EvidenceOrigin::Search, "engine", true),
        ],
    );
    let report = verifier.verify(&c);
    assert_eq!(report.status, VerificationStatus::Pass);
    assert_eq!(report.level, VerificationLevel::Deterministic);
    assert_eq!(report.score.get(), 1.0);
    assert_eq!(
        NamedVerifier::<PromotionClaim>::name(&verifier),
        "promotion"
    );
}

struct ConfidentLearned;
impl Verifier<PromotionClaim> for ConfidentLearned {
    fn verify(&self, _: &PromotionClaim) -> VerificationReport {
        VerificationReport {
            status: VerificationStatus::Pass,
            level: VerificationLevel::LatentAgreement,
            score: Probability::new(0.999).unwrap(),
            findings: vec![],
        }
    }
}
impl NamedVerifier<PromotionClaim> for ConfidentLearned {
    fn name(&self) -> &'static str {
        "learned"
    }
}

#[test]
fn a_learned_score_cannot_rescue_derived_only_or_contradicted_promotions() {
    let mut fabric = VerifierFabric::<PromotionClaim>::default();
    fabric.push(ConfidentLearned);
    fabric.push(PromotionVerifier::default());

    let derived = claim(
        PromotionTarget::Authoritative,
        vec![ev("s", EvidenceOrigin::Learned, "model", true)],
    );
    let report = fabric.verify(&derived);
    assert_eq!(report.status, VerificationStatus::Fail);
    assert!(report
        .findings
        .iter()
        .any(|f| f.code == "promotion.derived_only" && f.hard));

    let mut contradicted = claim(
        PromotionTarget::Verified,
        vec![ev("e", EvidenceOrigin::Ledger, "ledger", true)],
    );
    contradicted.deterministic_findings = vec![Finding {
        code: "arith.value".into(),
        message: "wrong".into(),
        hard: true,
    }];
    let report = fabric.verify(&contradicted);
    assert_eq!(report.status, VerificationStatus::Fail);
    assert_eq!(report.score.get(), 0.0);
    assert!(report
        .findings
        .iter()
        .any(|f| f.code == "promotion.contradicted"));

    let sound = claim(
        PromotionTarget::Verified,
        vec![ev("e", EvidenceOrigin::Ledger, "ledger", true)],
    );
    let report = fabric.verify(&sound);
    assert_eq!(report.status, VerificationStatus::Pass);
    assert_eq!(report.level, VerificationLevel::LatentAgreement);
}
