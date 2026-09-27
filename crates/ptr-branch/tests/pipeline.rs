use std::collections::BTreeMap;

use ptr_branch::{
    calibration_draw, certify, AutoThreshold, Branch, BranchId, MergePlan, TriageDecision,
    TriagePolicy,
};
use ptr_config::PtrConfig;
use ptr_ledger::LedgerEvent;
use ptr_runtime::execution::RequiredVerification;
use ptr_runtime::semantic::{StaleReliance, StaleTarget};
use ptr_runtime::{PtrRuntime, RuntimeError};
use ptr_semdb::SemanticDelta;
use ptr_types::{
    CapabilityId, CapsuleId, Effect, Generation, PrincipalId, Probability, ProjectId, Validity,
    VerificationLevel,
};
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

    let (expected, delta, relied) = certification.plan().clone().into_parts();
    runtime
        .apply_certified_semantic_delta(
            expected,
            delta,
            &relied,
            RequiredVerification::Deterministic,
            |view| {
                assert_eq!(view.get("price:sku-1"), Some("11"));
                passing()
            },
        )
        .unwrap();
    assert_eq!(runtime.snapshot().get("price:sku-1"), Some("11"));
}

/// A branch that relies on `pricing-policy` at generation 1 and reprices,
/// sealed and certified against `runtime` as it is now.
fn certified_repricing(runtime: &PtrRuntime) -> MergePlan {
    let mut work = Branch::open(
        BranchId::from("b1"),
        PrincipalId::from("pricing-agent"),
        runtime.snapshot(),
    );
    work.rely_on("pricing-policy", Generation(1)).unwrap();
    work.read("price:sku-1").unwrap();
    work.put("price:sku-1", "11".into()).unwrap();
    certify(
        &work.seal().unwrap(),
        &runtime.snapshot(),
        |target, generation| runtime.generation_validity(target, generation),
    )
    .unwrap()
    .plan()
    .clone()
}

#[test]
fn a_revocation_or_supersession_after_certification_refuses_the_commit_and_appends_nothing() {
    let revoke = LedgerEvent::Revoked {
        subject: "pricing-policy".into(),
        generation: Generation(1),
    };
    let supersede = LedgerEvent::CapsuleSuperseded {
        capsule: CapsuleId::from("pricing-policy"),
        old: Generation(1),
        new: Generation(2),
    };
    for (change, validity) in [
        (revoke, Validity::Revoked),
        (supersede, Validity::Superseded),
    ] {
        let mut runtime = runtime_with_price();
        let plan = certified_repricing(&runtime);
        // The lifecycle changes after certification; the semantic revision
        // the plan is bound to does not move.
        runtime.commit(change).unwrap();
        assert_eq!(plan.expected(), runtime.revision());
        let events = runtime.committed_events().len();

        let (expected, delta, relied) = plan.into_parts();
        let refused = runtime
            .apply_certified_semantic_delta(
                expected,
                delta,
                &relied,
                RequiredVerification::Deterministic,
                |_| passing(),
            )
            .unwrap_err();
        let RuntimeError::StaleReliance(stale) = refused else {
            panic!("{refused:?}");
        };
        assert_eq!(
            stale,
            StaleReliance {
                targets: BTreeMap::from([(
                    "pricing-policy".to_owned(),
                    StaleTarget {
                        relied: Generation(1),
                        validity: Some(validity),
                    },
                )]),
            }
        );
        assert_eq!(stale.code(), "PTR_RUNTIME_STALE_RELIANCE");
        assert_eq!(runtime.committed_events().len(), events);
        assert_eq!(runtime.revision(), expected);
        assert_eq!(runtime.snapshot().get("price:sku-1"), Some("10"));
    }

    // With the lifecycle unchanged, the same plan commits.
    let mut runtime = runtime_with_price();
    let plan = certified_repricing(&runtime);
    let events = runtime.committed_events().len();
    let (expected, delta, relied) = plan.into_parts();
    let commit = runtime
        .apply_certified_semantic_delta(
            expected,
            delta,
            &relied,
            RequiredVerification::Deterministic,
            |_| passing(),
        )
        .unwrap();
    assert!(commit.commit_index.is_some());
    assert_eq!(runtime.committed_events().len(), events + 1);
    assert_eq!(runtime.snapshot().get("price:sku-1"), Some("11"));
}

#[test]
fn a_commit_that_moves_neither_the_revision_nor_a_relied_generation_does_not_refuse_the_plan() {
    // The plan's only freshness fences are the semantic revision and the
    // generations it relied on. A new hard constraint, a capsule the branch
    // did not rely on and a verifier attestation move neither, so the plan
    // certified before them still commits: the documented contract, not an
    // accident of this test.
    let mut runtime = runtime_with_price();
    let plan = certified_repricing(&runtime);
    for unrelated in [
        LedgerEvent::HardConstraintCommitted {
            key: "max-price".into(),
            generation: Generation(1),
        },
        LedgerEvent::CapsuleCommitted {
            project: ProjectId::from("shop"),
            capsule: CapsuleId::from("shipping-policy"),
            generation: Generation(1),
        },
        LedgerEvent::VerifierAttested {
            subject: "price:sku-1".into(),
            passed: false,
        },
    ] {
        runtime.commit(unrelated).unwrap();
        assert_eq!(plan.expected(), runtime.revision());
    }
    assert!(!plan.relied().contains_key("constraint:max-price"));
    let (expected, delta, relied) = plan.into_parts();
    let commit = runtime
        .apply_certified_semantic_delta(
            expected,
            delta,
            &relied,
            RequiredVerification::Deterministic,
            |_| passing(),
        )
        .unwrap();
    assert!(commit.commit_index.is_some());
    assert_eq!(runtime.snapshot().get("price:sku-1"), Some("11"));
}

#[test]
fn an_unsettled_effect_attempt_fences_the_plan_until_it_is_reconciled() {
    // An effect record moves neither the revision nor a relied generation, so
    // it is no freshness check, and the plan's documentation listed it among
    // the records that refuse nothing. But an attempt that is neither settled
    // nor reconciled fences every commit of the runtime, this plan's
    // included, until an operator or the dispatch settles it.
    let mut runtime = runtime_with_price();
    let plan = certified_repricing(&runtime);
    let attempt = runtime
        .commit(LedgerEvent::EffectAttempted {
            key: None,
            project: ProjectId::from("shop"),
            principal: "pricing-agent".into(),
            target: "pricing-policy".into(),
            operation: "notify".into(),
            capability: CapabilityId::from("mail.send"),
            effect: Effect::External,
            generation: Generation(1),
            revision: runtime.revision(),
            verification: VerificationLevel::Deterministic,
            action_digest: [0; 32],
        })
        .unwrap();
    assert_eq!(plan.expected(), runtime.revision());
    assert_eq!(
        runtime.generation_validity("pricing-policy", Generation(1)),
        Some(Validity::Live)
    );
    let (expected, delta, relied) = plan.into_parts();
    assert_eq!(
        runtime
            .apply_certified_semantic_delta(
                expected,
                delta.clone(),
                &relied,
                RequiredVerification::Deterministic,
                |_| unreachable!("verified while fenced"),
            )
            .unwrap_err(),
        RuntimeError::ExecutionFenced
    );
    assert_eq!(runtime.snapshot().get("price:sku-1"), Some("10"));

    // Reconciling the attempt lifts the fence without moving the revision,
    // and the same plan then commits.
    runtime
        .reconcile_effect(attempt, false, "operator-confirmed-not-sent")
        .unwrap();
    assert_eq!(expected, runtime.revision());
    let commit = runtime
        .apply_certified_semantic_delta(
            expected,
            delta,
            &relied,
            RequiredVerification::Deterministic,
            |_| passing(),
        )
        .unwrap();
    assert!(commit.commit_index.is_some());
    assert_eq!(runtime.snapshot().get("price:sku-1"), Some("11"));
}

#[test]
fn a_plan_committed_without_its_relied_generations_is_not_stopped_by_the_runtime() {
    // Why the commit path is `apply_certified_semantic_delta` with the plan's
    // `relied`, and why using it is the caller's obligation: the revocation
    // does not move the revision, and `apply_verified_semantic_delta` checks
    // no relied generation, so a caller that drops `relied` appends work
    // built on a revoked generation.
    let mut runtime = runtime_with_price();
    let plan = certified_repricing(&runtime);
    runtime
        .commit(LedgerEvent::Revoked {
            subject: "pricing-policy".into(),
            generation: Generation(1),
        })
        .unwrap();
    let (expected, delta, relied) = plan.into_parts();
    assert_eq!(
        relied,
        BTreeMap::from([("pricing-policy".to_owned(), Generation(1))])
    );
    assert!(matches!(
        runtime.apply_certified_semantic_delta(
            expected,
            delta.clone(),
            &relied,
            RequiredVerification::Deterministic,
            |_| passing(),
        ),
        Err(RuntimeError::StaleReliance(_))
    ));
    let commit = runtime
        .apply_verified_semantic_delta(expected, delta, RequiredVerification::Deterministic, |_| {
            passing()
        })
        .unwrap();
    assert!(commit.commit_index.is_some());
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

    let (expected, delta, relied) = plan.into_parts();
    assert!(matches!(
        runtime.apply_certified_semantic_delta(
            expected,
            delta,
            &relied,
            RequiredVerification::Deterministic,
            |_| passing()
        ),
        Err(RuntimeError::Semantic(_))
    ));
}
