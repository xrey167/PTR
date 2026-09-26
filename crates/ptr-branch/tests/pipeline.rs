use ptr_branch::{
    calibration_draw, certify, AutoThreshold, Branch, BranchId, TriageDecision, TriagePolicy,
};
use ptr_config::PtrConfig;
use ptr_ledger::LedgerEvent;
use ptr_runtime::execution::RequiredVerification;
use ptr_runtime::{PtrRuntime, RuntimeError};
use ptr_semdb::SemanticDelta;
use ptr_types::{CapsuleId, Generation, PrincipalId, Probability, ProjectId, VerificationLevel};
use ptr_verifier::{VerificationReport, VerificationStatus};

fn passing() -> VerificationReport {
    VerificationReport {
        status: VerificationStatus::Pass,
        level: VerificationLevel::Deterministic,
        score: Probability::new(0.95).unwrap(),
        findings: vec![],
    }
}

fn runtime_with_price() -> PtrRuntime {
    let mut runtime = PtrRuntime::new(PtrConfig::default()).unwrap();
    runtime
        .commit(LedgerEvent::CapsuleCommitted {
            project: ProjectId::from("shop"),
            capsule: CapsuleId::from("pricing-policy"),
            generation: Generation(1),
        })
        .unwrap();
    let mut delta = SemanticDelta::default();
    delta.upserts.insert("price:sku-1".into(), "10".into());
    let revision = runtime.revision();
    runtime.apply_semantic_delta(revision, delta).unwrap();
    runtime
}

#[test]
fn a_certified_and_verified_branch_reaches_semantic_state_only_through_the_runtime() {
    let mut runtime = runtime_with_price();
    let mut work = Branch::open(
        BranchId::from("b1"),
        PrincipalId::from("pricing-agent"),
        runtime.snapshot(),
    );
    work.rely_on("pricing-policy", Generation(1)).unwrap();
    work.read("price:sku-1").unwrap();
    work.put("price:sku-1", "11".into()).unwrap();
    let sealed = work.seal().unwrap();

    let certification = certify(&sealed, &runtime.snapshot(), |target, generation| {
        runtime.generation_validity(target, generation)
    })
    .unwrap();

    let policy = TriagePolicy::new(AutoThreshold::AtLeast(0.8), 0.0).unwrap();
    let triage = policy
        .triage(
            &passing(),
            Probability::new(0.9).unwrap(),
            calibration_draw(sealed.id(), 1),
        )
        .unwrap();
    assert_eq!(triage.decision, TriageDecision::AutoPropose);

    let (expected, delta) = certification.plan().clone().into_parts();
    runtime
        .apply_verified_semantic_delta(
            expected,
            delta,
            RequiredVerification::Deterministic,
            |view| {
                assert_eq!(view.get("price:sku-1"), Some("11"));
                passing()
            },
        )
        .unwrap();
    assert_eq!(runtime.snapshot().get("price:sku-1"), Some("11"));
}

#[test]
fn a_revocation_between_sealing_and_merging_stops_the_branch() {
    let mut runtime = runtime_with_price();
    let mut work = Branch::open(
        BranchId::from("b1"),
        PrincipalId::from("pricing-agent"),
        runtime.snapshot(),
    );
    work.rely_on("pricing-policy", Generation(1)).unwrap();
    work.read("price:sku-1").unwrap();
    work.put("price:sku-1", "11".into()).unwrap();
    let sealed = work.seal().unwrap();

    runtime
        .commit(LedgerEvent::Revoked {
            subject: "pricing-policy".into(),
            generation: Generation(1),
        })
        .unwrap();

    assert!(certify(&sealed, &runtime.snapshot(), |target, generation| {
        runtime.generation_validity(target, generation)
    })
    .is_err());
}

#[test]
fn a_plan_certified_before_another_commit_is_refused_by_the_runtime() {
    let mut runtime = runtime_with_price();
    let mut work = Branch::open(
        BranchId::from("b1"),
        PrincipalId::from("pricing-agent"),
        runtime.snapshot(),
    );
    work.read("price:sku-1").unwrap();
    work.put("price:sku-1", "11".into()).unwrap();
    let plan = certify(&work.seal().unwrap(), &runtime.snapshot(), |t, g| {
        runtime.generation_validity(t, g)
    })
    .unwrap()
    .plan()
    .clone();

    let mut other = SemanticDelta::default();
    other.upserts.insert("unrelated".into(), "x".into());
    let revision = runtime.revision();
    runtime.apply_semantic_delta(revision, other).unwrap();

    let (expected, delta) = plan.into_parts();
    assert!(matches!(
        runtime.apply_verified_semantic_delta(
            expected,
            delta,
            RequiredVerification::Deterministic,
            |_| passing()
        ),
        Err(RuntimeError::Semantic(_))
    ));
}
