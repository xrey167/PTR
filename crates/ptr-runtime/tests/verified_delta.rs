use ptr_config::PtrConfig;
use ptr_ledger::LedgerEvent;
use ptr_runtime::execution::RequiredVerification;
use ptr_runtime::{PtrRuntime, RuntimeError};
use ptr_semdb::{SemanticDelta, SemanticError};
use ptr_types::{CapsuleId, Generation, Probability, ProjectId, Validity, VerificationLevel};
use ptr_verifier::{Finding, VerificationReport, VerificationStatus};

fn report(status: VerificationStatus, level: VerificationLevel, hard: bool) -> VerificationReport {
    VerificationReport {
        status,
        level,
        score: Probability::new(0.9).unwrap(),
        findings: if hard {
            vec![Finding {
                code: "constraint".into(),
                message: "hard constraint violated".into(),
                hard: true,
            }]
        } else {
            vec![]
        },
    }
}

#[test]
fn noops_still_require_verification_and_never_append_a_record() {
    for status in [VerificationStatus::Pass, VerificationStatus::Fail] {
        let mut runtime = PtrRuntime::new(PtrConfig::default()).unwrap();
        let revision = runtime.revision();
        let events = runtime.committed_events().len();
        let mut calls = 0;
        let result = runtime.apply_verified_semantic_delta(
            revision,
            SemanticDelta::default(),
            RequiredVerification::Deterministic,
            |view| {
                calls += 1;
                assert_eq!(view.keys().count(), 0);
                report(status, VerificationLevel::Deterministic, false)
            },
        );
        assert_eq!(calls, 1);
        match status {
            VerificationStatus::Pass => {
                let commit = result.unwrap();
                assert_eq!(commit.revision, revision);
                assert_eq!(commit.commit_index, None);
                assert!(commit.affected.is_empty());
            }
            _ => assert!(matches!(
                result,
                Err(RuntimeError::DeltaVerificationRejected {
                    status: VerificationStatus::Fail,
                    level: VerificationLevel::Deterministic,
                    hard_findings: 0,
                })
            )),
        }
        assert_eq!(runtime.revision(), revision);
        assert_eq!(runtime.committed_events().len(), events);
    }
}

#[test]
fn malformed_deltas_are_rejected_before_the_verifier_runs() {
    let mut runtime = PtrRuntime::new(PtrConfig::default()).unwrap();
    let revision = runtime.revision();
    let events = runtime.committed_events().len();
    let result = runtime.apply_verified_semantic_delta(
        revision,
        delta("", "invalid"),
        RequiredVerification::Deterministic,
        |_| panic!("invalid input must not reach verification"),
    );
    assert!(matches!(
        result,
        Err(RuntimeError::Semantic(SemanticError::InvalidKey))
    ));
    assert_eq!(runtime.revision(), revision);
    assert_eq!(runtime.committed_events().len(), events);
}

#[test]
fn rejecting_an_overwrite_preserves_values_and_allows_a_verified_retry() {
    let mut runtime = PtrRuntime::new(PtrConfig::default()).unwrap();
    runtime
        .apply_semantic_delta(runtime.revision(), delta("price", "10"))
        .unwrap();
    let revision = runtime.revision();
    let events = runtime.committed_events().len();
    let mut refused = report(
        VerificationStatus::Pass,
        VerificationLevel::Deterministic,
        true,
    );
    refused.findings.push(Finding {
        code: "second".into(),
        message: "another failure".into(),
        hard: true,
    });
    refused.findings.push(Finding {
        code: "advice".into(),
        message: "soft finding".into(),
        hard: false,
    });
    let result = runtime.apply_verified_semantic_delta(
        revision,
        delta("price", "12"),
        RequiredVerification::Deterministic,
        |view| {
            assert_eq!(view.get("price"), Some("12"));
            refused
        },
    );
    assert!(matches!(
        result,
        Err(RuntimeError::DeltaVerificationRejected {
            status: VerificationStatus::Pass,
            level: VerificationLevel::Deterministic,
            hard_findings: 2,
        })
    ));
    assert_eq!(runtime.snapshot().get("price"), Some("10"));
    assert_eq!(runtime.revision(), revision);
    assert_eq!(runtime.committed_events().len(), events);
    let commit = runtime
        .apply_verified_semantic_delta(
            revision,
            delta("price", "12"),
            RequiredVerification::Deterministic,
            |_| {
                report(
                    VerificationStatus::Pass,
                    VerificationLevel::Deterministic,
                    false,
                )
            },
        )
        .unwrap();
    assert!(commit.commit_index.is_some());
    assert_eq!(runtime.committed_events().len(), events + 1);
    assert_eq!(runtime.snapshot().get("price"), Some("12"));
}

#[test]
fn full_semantic_verification_does_not_satisfy_a_deterministic_requirement() {
    let mut runtime = PtrRuntime::new(PtrConfig::default()).unwrap();
    let revision = runtime.revision();
    let events = runtime.committed_events().len();
    let result = runtime.apply_verified_semantic_delta(
        revision,
        delta("price", "12"),
        RequiredVerification::Deterministic,
        |_| {
            report(
                VerificationStatus::Pass,
                VerificationLevel::FullSemantic,
                false,
            )
        },
    );
    assert!(matches!(
        result,
        Err(RuntimeError::DeltaVerificationRejected {
            status: VerificationStatus::Pass,
            level: VerificationLevel::FullSemantic,
            hard_findings: 0,
        })
    ));
    assert_eq!(runtime.revision(), revision);
    assert_eq!(runtime.committed_events().len(), events);
    assert_eq!(runtime.snapshot().get("price"), None);
}

fn delta(key: &str, value: &str) -> SemanticDelta {
    let mut delta = SemanticDelta::default();
    delta.upserts.insert(key.into(), value.into());
    delta
}

#[test]
fn a_revoked_generation_is_revoked_although_it_is_still_the_live_generation() {
    let mut runtime = PtrRuntime::new(PtrConfig::default()).unwrap();
    runtime
        .commit(LedgerEvent::CapsuleCommitted {
            project: ProjectId::from("p"),
            capsule: CapsuleId::from("fact:a"),
            generation: Generation(1),
        })
        .unwrap();
    assert_eq!(
        runtime.generation_validity("fact:a", Generation(1)),
        Some(Validity::Live)
    );
    runtime
        .commit(LedgerEvent::Revoked {
            subject: "fact:a".into(),
            generation: Generation(1),
        })
        .unwrap();
    // The live-generation map alone still names generation 1.
    assert_eq!(runtime.live_generation("fact:a"), Some(Generation(1)));
    assert_eq!(
        runtime.generation_validity("fact:a", Generation(1)),
        Some(Validity::Revoked)
    );
}

#[test]
fn superseded_and_unknown_generations_are_not_live() {
    let mut runtime = PtrRuntime::new(PtrConfig::default()).unwrap();
    runtime
        .commit(LedgerEvent::CapsuleCommitted {
            project: ProjectId::from("p"),
            capsule: CapsuleId::from("fact:b"),
            generation: Generation(1),
        })
        .unwrap();
    runtime
        .commit(LedgerEvent::CapsuleSuperseded {
            capsule: CapsuleId::from("fact:b"),
            old: Generation(1),
            new: Generation(2),
        })
        .unwrap();
    assert_eq!(
        runtime.generation_validity("fact:b", Generation(1)),
        Some(Validity::Superseded)
    );
    assert_eq!(
        runtime.generation_validity("fact:b", Generation(2)),
        Some(Validity::Live)
    );
    assert_eq!(runtime.generation_validity("fact:b", Generation(3)), None);
    assert_eq!(
        runtime.generation_validity("fact:none", Generation(1)),
        None
    );
}

#[test]
fn a_verified_delta_commits_the_state_its_verifier_saw() {
    let mut runtime = PtrRuntime::new(PtrConfig::default()).unwrap();
    let before = runtime.revision();
    let commit = runtime
        .apply_verified_semantic_delta(
            before,
            delta("price:sku-1", "12"),
            RequiredVerification::FullSemantic,
            |view| {
                assert_eq!(view.get("price:sku-1"), Some("12"));
                report(
                    VerificationStatus::Pass,
                    VerificationLevel::FullSemantic,
                    false,
                )
            },
        )
        .unwrap();
    assert!(commit.commit_index.is_some());
    assert_eq!(runtime.snapshot().get("price:sku-1"), Some("12"));
}

#[test]
fn no_score_or_shallow_level_or_hard_finding_gets_a_delta_past_verification() {
    let refusals = [
        report(
            VerificationStatus::Pass,
            VerificationLevel::SampleVerified,
            false,
        ),
        report(
            VerificationStatus::Unknown,
            VerificationLevel::Deterministic,
            false,
        ),
        report(
            VerificationStatus::Disputed,
            VerificationLevel::Deterministic,
            false,
        ),
        report(
            VerificationStatus::Pass,
            VerificationLevel::Deterministic,
            true,
        ),
    ];
    for refused in refusals {
        let mut runtime = PtrRuntime::new(PtrConfig::default()).unwrap();
        let before = runtime.revision();
        let committed_before = runtime.committed_events().len();
        let result = runtime.apply_verified_semantic_delta(
            before,
            delta("price:sku-1", "1"),
            RequiredVerification::FullSemantic,
            |_| refused.clone(),
        );
        assert!(matches!(
            result,
            Err(RuntimeError::DeltaVerificationRejected { .. })
        ));
        assert_eq!(runtime.revision(), before);
        assert_eq!(runtime.committed_events().len(), committed_before);
    }
}

#[test]
fn a_verified_delta_against_a_moved_revision_is_refused_before_verification() {
    let mut runtime = PtrRuntime::new(PtrConfig::default()).unwrap();
    let stale = runtime.revision();
    runtime
        .apply_semantic_delta(stale, delta("other", "x"))
        .unwrap();
    let mut verified = false;
    let result = runtime.apply_verified_semantic_delta(
        stale,
        delta("price:sku-1", "1"),
        RequiredVerification::Deterministic,
        |_| {
            verified = true;
            report(
                VerificationStatus::Pass,
                VerificationLevel::Deterministic,
                false,
            )
        },
    );
    assert!(matches!(result, Err(RuntimeError::Semantic(_))));
    assert!(!verified);
}
