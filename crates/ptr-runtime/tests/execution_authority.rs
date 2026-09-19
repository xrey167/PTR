mod common;

use common::*;
use ptr_config::PtrConfig;
use ptr_ledger::LedgerEvent;
use ptr_runtime::{execution::*, PtrRuntime, RuntimeError};
use ptr_security::AuthorizationDenial;
use ptr_types::{
    CapabilityId, Effect, Generation, ProjectId, RequestId, TypeId, VerificationLevel,
};
use ptr_verifier::VerificationStatus;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::time::Duration;

#[test]
fn exact_verified_action_reaches_only_registered_executor_with_bound_identity() {
    let (mut runtime, action) = fixture();
    let probe = Probe::default();
    let session = session(&mut runtime, &action, &probe, "alice");
    let permit = runtime
        .prepare_execution(&session, &ProjectId::from("p"), &action, TTL)
        .unwrap();
    assert_eq!(
        probe.verifications(),
        0,
        "no stale preparation-time verification report"
    );
    assert_eq!(format!("{permit:?}"), "ExecutionPermit(<opaque>)");
    assert_eq!(
        runtime.execute_prepared(&session, permit).unwrap(),
        b"executed"
    );
    assert_eq!(probe.verifications(), 1);
    assert_eq!(
        *probe.calls.lock().unwrap(),
        vec![("alice".into(), ProjectId::from("p"), action)]
    );
}

#[test]
fn mutating_original_action_cannot_substitute_the_permitted_payload() {
    let (mut runtime, mut action) = fixture();
    let original = action.clone();
    let probe = Probe::default();
    let session = session(&mut runtime, &action, &probe, "alice");
    let permit = runtime
        .prepare_execution(&session, &ProjectId::from("p"), &action, TTL)
        .unwrap();
    action.payload = b"unverified replacement".to_vec();
    action.target = "other".into();
    action.effect = Effect::Irreversible;
    runtime.execute_prepared(&session, permit).unwrap();
    assert_eq!(probe.calls.lock().unwrap()[0].2, original);
}

#[test]
fn every_scope_coordinate_is_matched_exactly_before_verification() {
    let (mut runtime, action) = fixture();
    let probe = Probe::default();
    let session = session(&mut runtime, &action, &probe, "alice");
    assert!(matches!(
        runtime.prepare_execution(&session, &ProjectId::from("other"), &action, TTL),
        Err(ExecutionError::ScopeDenied)
    ));
    for field in 0..5 {
        let mut changed = action.clone();
        match field {
            0 => changed.target.push_str("/child"),
            1 => changed.operation = "delete".into(),
            2 => changed.capability = CapabilityId::from("other"),
            3 => changed.input_type = TypeId::from("other"),
            _ => changed.effect = Effect::External,
        }
        assert!(matches!(
            runtime.prepare_execution(&session, &ProjectId::from("p"), &changed, TTL),
            Err(ExecutionError::ScopeDenied)
        ));
    }
    assert_eq!(probe.verifications(), 0);
    assert_eq!(probe.executions(), 0);
}

#[test]
fn grant_cannot_reassign_actual_capsule_project() {
    let (mut runtime, action) = fixture();
    let probe = Probe::default();
    let mut wrong = scope(&action);
    wrong.project = ProjectId::from("other");
    let session = runtime
        .register_execution_session(
            "alice",
            vec![grant(
                wrong,
                &probe,
                RequiredVerification::FullSemantic,
                ExecutorMode::Success,
            )],
            TTL,
        )
        .unwrap();
    assert!(matches!(
        runtime.prepare_execution(&session, &ProjectId::from("other"), &action, TTL),
        Err(ExecutionError::ProjectMismatch)
    ));
    assert_eq!(probe.executions(), 0);
}

#[test]
fn cross_principal_and_same_principal_new_session_cannot_reuse_permit() {
    for principal in ["bob", "alice"] {
        let (mut runtime, action) = fixture();
        let probe = Probe::default();
        let alice = session(&mut runtime, &action, &probe, "alice");
        let other = session(&mut runtime, &action, &probe, principal);
        let permit = runtime
            .prepare_execution(&alice, &ProjectId::from("p"), &action, TTL)
            .unwrap();
        assert_eq!(
            runtime.execute_prepared(&other, permit),
            Err(ExecutionError::SessionMismatch)
        );
        assert_eq!(probe.executions(), 0);
    }
}

#[test]
fn revoked_session_and_clones_never_reactivate_on_reregistration() {
    let (mut runtime, action) = fixture();
    let probe = Probe::default();
    let old = session(&mut runtime, &action, &probe, "alice");
    let cloned = old.clone();
    let permit = runtime
        .prepare_execution(&old, &ProjectId::from("p"), &action, TTL)
        .unwrap();
    runtime.revoke_execution_session(&old).unwrap();
    let _new = session(&mut runtime, &action, &probe, "alice");
    assert_eq!(
        runtime.execute_prepared(&cloned, permit),
        Err(ExecutionError::SessionClosed)
    );
    assert_eq!(probe.executions(), 0);
}

#[test]
fn runtime_replay_does_not_restore_session_or_execution_authority() {
    let (mut runtime, action) = fixture();
    let probe = Probe::default();
    let old = session(&mut runtime, &action, &probe, "alice");
    let permit = runtime
        .prepare_execution(&old, &ProjectId::from("p"), &action, TTL)
        .unwrap();
    let mut restarted =
        PtrRuntime::replay(PtrConfig::default(), runtime.committed_events()).unwrap();
    assert_eq!(
        restarted.execute_prepared(&old, permit),
        Err(ExecutionError::ForeignRuntime)
    );
    assert_eq!(
        restarted.revoke_execution_session(&old),
        Err(ExecutionError::ForeignRuntime)
    );
    assert_eq!(probe.executions(), 0);
}

#[test]
fn semantic_change_rejects_prepared_action_before_verifier_or_executor() {
    let (mut runtime, action) = fixture();
    let probe = Probe::default();
    let session = session(&mut runtime, &action, &probe, "alice");
    let permit = runtime
        .prepare_execution(&session, &ProjectId::from("p"), &action, TTL)
        .unwrap();
    runtime
        .ingest_text(RequestId::from("request"), "changed ground state")
        .unwrap();
    assert_eq!(
        runtime.execute_prepared(&session, permit),
        Err(ExecutionError::StalePermit)
    );
    assert!(matches!(
        runtime.prepare_execution(&session, &ProjectId::from("p"), &action, TTL),
        Err(ExecutionError::AuthorizationDenied(
            AuthorizationDenial::StaleRevision { .. }
        ))
    ));
    assert_eq!(probe.verifications(), 0);
    assert_eq!(probe.executions(), 0);
}

#[test]
fn revoke_between_prepare_and_execute_invalidates_permit_and_fresh_requests() {
    let (mut runtime, action) = fixture();
    let probe = Probe::default();
    let session = session(&mut runtime, &action, &probe, "alice");
    let permit = runtime
        .prepare_execution(&session, &ProjectId::from("p"), &action, TTL)
        .unwrap();
    runtime
        .commit(LedgerEvent::Revoked {
            subject: action.target.clone(),
            generation: action.generation,
        })
        .unwrap();
    assert_eq!(
        runtime.execute_prepared(&session, permit),
        Err(ExecutionError::StalePermit)
    );
    assert!(matches!(
        runtime.prepare_execution(&session, &ProjectId::from("p"), &action, TTL),
        Err(ExecutionError::AuthorizationDenied(
            AuthorizationDenial::StaleGeneration { .. }
        ))
    ));
    assert_eq!(probe.executions(), 0);
}

#[test]
fn permission_revoke_then_restore_cannot_resurrect_old_permit() {
    let (mut runtime, action) = fixture();
    let probe = Probe::default();
    let session = session(&mut runtime, &action, &probe, "alice");
    let permit = runtime
        .prepare_execution(&session, &ProjectId::from("p"), &action, TTL)
        .unwrap();
    runtime.permissions_mut().capabilities.clear();
    runtime
        .permissions_mut()
        .capabilities
        .insert(action.capability.clone());
    assert_eq!(
        runtime.execute_prepared(&session, permit),
        Err(ExecutionError::StalePermit)
    );
    assert_eq!(probe.executions(), 0);
}

#[test]
fn research_flags_never_disable_scoped_gateway_checks_even_for_reads() {
    let (mut runtime, mut action) = fixture();
    runtime.config.action_boundary.require_capability = false;
    runtime.config.action_boundary.require_current_revision = false;
    runtime.config.action_boundary.require_live_generation = false;
    runtime.permissions_mut().capabilities.clear();
    action.effect = Effect::Read;
    let probe = Probe::default();
    let session = session(&mut runtime, &action, &probe, "alice");
    assert!(matches!(
        runtime.prepare_execution(&session, &ProjectId::from("p"), &action, TTL),
        Err(ExecutionError::AuthorizationDenied(
            AuthorizationDenial::MissingCapability { .. }
        ))
    ));
    assert_eq!(probe.executions(), 0);
}

#[test]
fn all_nonpass_or_insufficient_verifications_reject_at_dispatch() {
    for (status, level) in [
        (VerificationStatus::Fail, VerificationLevel::Deterministic),
        (
            VerificationStatus::Disputed,
            VerificationLevel::Deterministic,
        ),
        (
            VerificationStatus::Unknown,
            VerificationLevel::Deterministic,
        ),
        (VerificationStatus::Pass, VerificationLevel::Unverified),
        (VerificationStatus::Pass, VerificationLevel::LatentAgreement),
        (VerificationStatus::Pass, VerificationLevel::SampleVerified),
    ] {
        let (mut runtime, action) = fixture();
        let probe = Probe::default();
        let session = session(&mut runtime, &action, &probe, "alice");
        let permit = runtime
            .prepare_execution(&session, &ProjectId::from("p"), &action, TTL)
            .unwrap();
        probe.report.lock().unwrap().status = status;
        probe.report.lock().unwrap().level = level;
        assert_eq!(
            runtime.execute_prepared(&session, permit),
            Err(ExecutionError::VerificationRejected { status, level })
        );
        assert_eq!(probe.verifications(), 1);
        assert_eq!(probe.executions(), 0);
    }
}

#[test]
fn pass_report_with_hard_finding_cannot_authorize_effect() {
    let (mut runtime, action) = fixture();
    let probe = Probe::default();
    let session = session(&mut runtime, &action, &probe, "alice");
    let permit = runtime
        .prepare_execution(&session, &ProjectId::from("p"), &action, TTL)
        .unwrap();
    probe.hard_finding();
    assert_eq!(
        runtime.execute_prepared(&session, permit),
        Err(ExecutionError::HardFinding)
    );
    assert_eq!(probe.executions(), 0);
}

#[test]
fn required_deterministic_verification_cannot_be_replaced_with_full_semantic() {
    let (mut runtime, action) = fixture();
    let probe = Probe::default();
    let session = runtime
        .register_execution_session(
            "alice",
            vec![grant(
                scope(&action),
                &probe,
                RequiredVerification::Deterministic,
                ExecutorMode::Success,
            )],
            TTL,
        )
        .unwrap();
    let permit = runtime
        .prepare_execution(&session, &ProjectId::from("p"), &action, TTL)
        .unwrap();
    probe.report.lock().unwrap().level = VerificationLevel::FullSemantic;
    assert!(matches!(
        runtime.execute_prepared(&session, permit),
        Err(ExecutionError::VerificationRejected { .. })
    ));
    assert_eq!(probe.executions(), 0);
}

#[test]
fn ambiguous_executor_error_or_panic_fences_new_permits_and_commits() {
    for mode in [ExecutorMode::Error, ExecutorMode::Panic] {
        let (mut runtime, action) = fixture();
        let probe = Probe::default();
        let session = runtime
            .register_execution_session(
                "alice",
                vec![grant(
                    scope(&action),
                    &probe,
                    RequiredVerification::FullSemantic,
                    mode,
                )],
                TTL,
            )
            .unwrap();
        let permit = runtime
            .prepare_execution(&session, &ProjectId::from("p"), &action, TTL)
            .unwrap();
        let outcome = catch_unwind(AssertUnwindSafe(|| {
            runtime.execute_prepared(&session, permit)
        }));
        assert!(matches!(
            outcome,
            Err(_) | Ok(Err(ExecutionError::Executor(_)))
        ));
        assert_eq!(probe.executions(), 1);
        assert!(matches!(
            runtime.prepare_execution(&session, &ProjectId::from("p"), &action, TTL),
            Err(ExecutionError::RuntimeFenced)
        ));
        assert_eq!(
            runtime.commit(LedgerEvent::Revoked {
                subject: action.target.clone(),
                generation: action.generation
            }),
            Err(RuntimeError::ExecutionFenced)
        );
        assert_eq!(probe.executions(), 1);
    }
}

#[test]
fn one_successful_effect_invalidates_other_prepared_permits() {
    let (mut runtime, action) = fixture();
    let probe = Probe::default();
    let session = session(&mut runtime, &action, &probe, "alice");
    let first = runtime
        .prepare_execution(&session, &ProjectId::from("p"), &action, TTL)
        .unwrap();
    let second = runtime
        .prepare_execution(&session, &ProjectId::from("p"), &action, TTL)
        .unwrap();
    runtime.execute_prepared(&session, first).unwrap();
    assert_eq!(
        runtime.execute_prepared(&session, second),
        Err(ExecutionError::StalePermit)
    );
    assert_eq!(probe.executions(), 1);
}

#[test]
fn rejected_lifecycle_event_does_not_poison_valid_execution_authority() {
    let (mut runtime, action) = fixture();
    let probe = Probe::default();
    let session = session(&mut runtime, &action, &probe, "alice");
    let permit = runtime
        .prepare_execution(&session, &ProjectId::from("p"), &action, TTL)
        .unwrap();
    assert!(runtime
        .commit(LedgerEvent::CapsuleCommitted {
            project: ProjectId::from("p"),
            capsule: action.target.as_str().into(),
            generation: Generation(0)
        })
        .is_err());
    runtime.execute_prepared(&session, permit).unwrap();
    assert_eq!(probe.executions(), 1);
}

#[test]
fn invalid_and_ambiguous_host_grants_are_rejected() {
    let (mut runtime, action) = fixture();
    let probe = Probe::default();
    let mk = || {
        grant(
            scope(&action),
            &probe,
            RequiredVerification::FullSemantic,
            ExecutorMode::Success,
        )
    };
    assert!(matches!(
        runtime.register_execution_session("alice", vec![mk(), mk()], TTL),
        Err(ExecutionError::DuplicateScope)
    ));
    assert!(matches!(
        runtime.register_execution_session("alice", vec![mk()], Duration::ZERO),
        Err(ExecutionError::InvalidTtl)
    ));
    assert!(matches!(
        runtime.register_execution_session(" alice", vec![mk()], TTL),
        Err(ExecutionError::InvalidSession)
    ));
    assert!(matches!(
        runtime.register_execution_session("alice", vec![], TTL),
        Err(ExecutionError::InvalidGrant)
    ));
    let session = session(&mut runtime, &action, &probe, "alice");
    assert!(matches!(
        runtime.prepare_execution(&session, &ProjectId::from("p"), &action, Duration::ZERO),
        Err(ExecutionError::InvalidTtl)
    ));
    assert_eq!(probe.executions(), 0);
}

#[test]
fn idempotent_semantic_ingestion_does_not_invalidate_valid_permits() {
    let (mut runtime, action) = fixture();
    let probe = Probe::default();
    let session = session(&mut runtime, &action, &probe, "alice");
    let permit = runtime
        .prepare_execution(&session, &ProjectId::from("p"), &action, TTL)
        .unwrap();
    let before = runtime.committed_events().len();
    runtime
        .ingest_text(RequestId::from("request"), "ground state")
        .unwrap();
    assert_eq!(runtime.committed_events().len(), before);
    runtime.execute_prepared(&session, permit).unwrap();
    assert_eq!(probe.executions(), 1);
}

#[test]
fn recovery_snapshot_never_restores_execution_sessions_permissions_or_permits() {
    let (mut runtime, action) = fixture();
    let probe = Probe::default();
    let old = session(&mut runtime, &action, &probe, "alice");
    let permit = runtime
        .prepare_execution(&old, &ProjectId::from("p"), &action, TTL)
        .unwrap();
    let snapshot = runtime.export_recovery_snapshot().unwrap();
    let mut restored = PtrRuntime::restore_recovery_snapshot(
        PtrConfig::default(),
        snapshot.bytes(),
        snapshot.anchor(),
    )
    .unwrap();
    assert_eq!(
        restored.execute_prepared(&old, permit),
        Err(ExecutionError::ForeignRuntime)
    );
    let new = session(&mut restored, &action, &probe, "alice");
    assert!(restored
        .prepare_execution(&new, &ProjectId::from("p"), &action, TTL)
        .is_err());
    assert_eq!(probe.verifications(), 0);
    assert_eq!(probe.executions(), 0);
}

#[test]
fn ambiguous_execution_cannot_export_a_snapshot_or_trusted_journal_anchor() {
    let (mut runtime, action) = fixture();
    let probe = Probe::default();
    let session = runtime
        .register_execution_session(
            "alice",
            vec![grant(
                scope(&action),
                &probe,
                RequiredVerification::Deterministic,
                ExecutorMode::Error,
            )],
            TTL,
        )
        .unwrap();
    let permit = runtime
        .prepare_execution(&session, &ProjectId::from("p"), &action, TTL)
        .unwrap();
    assert!(runtime.execute_prepared(&session, permit).is_err());
    assert!(matches!(
        runtime.export_recovery_snapshot(),
        Err(RuntimeError::ExecutionFenced)
    ));
    assert!(matches!(
        runtime.journal_anchor(),
        Err(RuntimeError::ExecutionFenced)
    ));
}
