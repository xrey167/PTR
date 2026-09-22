//! Durable execution audit, restart-surviving fences, reconciliation and
//! at-most-once execution.
//!
//! The property under test is the one a flag cannot provide: after a crash
//! inside the effect window, the runtime must reopen *knowing* that an external
//! effect may already have applied. Every assertion here is about committed
//! history, because that is the only thing a restart preserves.

mod common;

use common::*;
use ptr_config::PtrConfig;
use ptr_core::action_head::ActionIr;
use ptr_ledger::{CommittedEvent, LedgerEvent, MAX_RETAINED_RESPONSE};
use ptr_runtime::{execution::*, PtrRuntime, RuntimeError};
use ptr_types::Probability;
use ptr_types::{
    CapabilityId, CapsuleId, CommitIndex, Effect, Generation, ProjectId, RequestId, Revision,
    TypeId, VerificationLevel,
};
use ptr_verifier::{VerificationReport, VerificationStatus, Verifier};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

struct Temp(PathBuf);
impl Temp {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "ptr-execution-audit-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&path).unwrap();
        Self(path)
    }
    fn log(&self) -> PathBuf {
        self.0.join("journal.log")
    }
}
impl Drop for Temp {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// The same ground state as the in-memory fixture, on a durable ledger, so a
/// test can actually drop the runtime and open the log again.
fn durable_fixture(path: &PathBuf) -> (PtrRuntime, ActionIr) {
    let mut runtime = PtrRuntime::open_durable(PtrConfig::default(), path).unwrap();
    let revision = runtime
        .ingest_text(RequestId::from("request"), "ground state")
        .unwrap();
    runtime
        .commit(LedgerEvent::CapsuleCommitted {
            project: ProjectId::from("p"),
            capsule: CapsuleId::from("capsule:a"),
            generation: Generation(1),
        })
        .unwrap();
    let action = action_at(revision);
    authorize(&mut runtime, &action);
    (runtime, action)
}

/// Reopen the same log. Process-local authority is deliberately not restored by
/// P0.3, so a caller has to install permissions again — the fence, in contrast,
/// must come back by itself.
fn reopen(path: &PathBuf, action: &ActionIr) -> PtrRuntime {
    let mut runtime = PtrRuntime::open_durable(PtrConfig::default(), path).unwrap();
    authorize(&mut runtime, action);
    runtime
}

fn action_at(revision: Revision) -> ActionIr {
    ActionIr {
        operation: "write".into(),
        target: "capsule:a".into(),
        capability: CapabilityId::from("file.write"),
        effect: Effect::Mutation,
        input_type: TypeId::from("Bytes"),
        generation: Generation(1),
        revision,
        payload: b"verified payload".to_vec(),
    }
}

fn authorize(runtime: &mut PtrRuntime, action: &ActionIr) {
    runtime
        .permissions_mut()
        .capabilities
        .insert(action.capability.clone());
    runtime.permissions_mut().allow_mutation = true;
}

/// An adapter that answers one byte past the retained bound. It lives here
/// rather than in the shared fixtures because only this file needs it.
struct OversizeAdapter(Probe);

impl Verifier<ActionIr> for OversizeAdapter {
    fn verify(&self, action: &ActionIr) -> VerificationReport {
        self.0.verified.fetch_add(1, Ordering::SeqCst);
        VerificationReport {
            status: if action.payload == b"verified payload" {
                VerificationStatus::Pass
            } else {
                VerificationStatus::Fail
            },
            level: VerificationLevel::Deterministic,
            score: Probability::new(1.0).unwrap(),
            findings: vec![],
        }
    }
}

impl ActionExecutor for OversizeAdapter {
    fn execute(&self, dispatch: VerifiedDispatch<'_>) -> Result<Vec<u8>, String> {
        self.0.calls.lock().unwrap().push((
            dispatch.principal().to_owned(),
            dispatch.project().clone(),
            dispatch.action().clone(),
        ));
        Ok(vec![7u8; MAX_RETAINED_RESPONSE + 1])
    }
}

fn attempt_with(key: Option<&str>) -> LedgerEvent {
    LedgerEvent::EffectAttempted {
        key: key.map(str::to_owned),
        project: ProjectId::from("p"),
        principal: "alice".into(),
        target: "capsule:a".into(),
        operation: "write".into(),
        capability: CapabilityId::from("file.write"),
        effect: Effect::Mutation,
        generation: Generation(1),
        revision: Revision(0),
        verification: VerificationLevel::Deterministic,
        action_digest: [0; 32],
    }
}

/// Number the events from 1 so a replay's index check accepts them.
fn history(events: Vec<LedgerEvent>) -> Vec<CommittedEvent> {
    events
        .into_iter()
        .enumerate()
        .map(|(offset, event)| CommittedEvent {
            index: CommitIndex(offset as u64 + 1),
            event,
        })
        .collect()
}

fn replayed(events: Vec<LedgerEvent>) -> Result<PtrRuntime, RuntimeError> {
    PtrRuntime::replay(PtrConfig::default(), &history(events))
}

fn attempts(runtime: &PtrRuntime) -> Vec<(CommitIndex, LedgerEvent)> {
    runtime
        .committed_events()
        .iter()
        .filter(|committed| {
            matches!(
                committed.event,
                LedgerEvent::EffectAttempted { .. }
                    | LedgerEvent::EffectSettled { .. }
                    | LedgerEvent::EffectReconciled { .. }
            )
        })
        .map(|committed| (committed.index, committed.event.clone()))
        .collect()
}

#[test]
fn an_applied_effect_commits_its_attempt_before_and_its_settlement_after() {
    let (mut runtime, action) = fixture();
    let probe = Probe::default();
    let session = session(&mut runtime, &action, &probe, "alice");
    let permit = runtime
        .prepare_execution(&session, &ProjectId::from("p"), &action, TTL)
        .unwrap();
    let output = runtime.execute_prepared(&session, permit).unwrap();
    assert_eq!(output, b"executed");

    // Two records, in this order. The attempt precedes the effect, which is the
    // whole point: a record written afterwards cannot describe a crash before it.
    let records = attempts(&runtime);
    assert_eq!(records.len(), 2);
    let (attempt_index, attempt) = &records[0];
    let LedgerEvent::EffectAttempted {
        key,
        project,
        principal,
        target,
        operation,
        capability,
        effect,
        generation,
        revision,
        verification,
        action_digest,
    } = attempt
    else {
        panic!("first effect record is not an attempt: {attempt:?}");
    };
    assert_eq!(*key, None);
    assert_eq!(*project, ProjectId::from("p"));
    assert_eq!(principal, "alice");
    assert_eq!(*target, action.target);
    assert_eq!(*operation, action.operation);
    assert_eq!(*capability, action.capability);
    assert_eq!(*effect, action.effect);
    assert_eq!(*generation, action.generation);
    assert_eq!(*revision, action.revision);
    // The level that actually admitted it, not the level the grant required.
    assert_eq!(*verification, VerificationLevel::Deterministic);
    assert_eq!(*action_digest, action_digest_of(&action));

    let (_, settled) = &records[1];
    assert_eq!(
        settled,
        &LedgerEvent::EffectSettled {
            attempt: *attempt_index,
            response: Some(b"executed".to_vec()),
            response_digest: response_digest_of(b"executed"),
        }
    );

    // One verification, and the level recorded above is the one it returned
    // rather than a level copied from the grant's requirement.
    assert_eq!(probe.verifications(), 1);

    // Settled means unfenced: ordinary work continues.
    assert!(runtime.unsettled_effects().is_empty());
    assert_eq!(
        runtime
            .materialized_state()
            .values
            .get(&format!("effect:{}:state", attempt_index.0)),
        Some(&"settled".to_owned())
    );
    assert!(runtime
        .commit(LedgerEvent::Revoked {
            subject: "unrelated".into(),
            generation: Generation(1),
        })
        .is_ok());
}

#[test]
fn a_reopened_runtime_is_still_fenced_by_an_unsettled_attempt() {
    for mode in [ExecutorMode::Error, ExecutorMode::Panic] {
        let temp = Temp::new();
        let (mut runtime, action) = durable_fixture(&temp.log());
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
        let attempt = runtime.unsettled_effects()[0].attempt;
        drop(runtime);

        // Nothing carried the fence across but the journal.
        let mut reopened = reopen(&temp.log(), &action);
        let unsettled = reopened.unsettled_effects();
        assert_eq!(unsettled.len(), 1);
        assert_eq!(unsettled[0].attempt, attempt);
        assert_eq!(unsettled[0].target, action.target);
        assert_eq!(unsettled[0].operation, action.operation);
        assert_eq!(unsettled[0].effect, action.effect);

        // A fenced runtime issues no new authority and commits nothing, so a
        // possibly-applied effect cannot be followed by more of them.
        let probe = Probe::default();
        assert!(matches!(
            reopened.register_execution_session(
                "alice",
                vec![grant(
                    scope(&action),
                    &probe,
                    RequiredVerification::FullSemantic,
                    ExecutorMode::Success,
                )],
                TTL,
            ),
            Err(ExecutionError::AmbiguousOutcome { attempt: named }) if named == attempt
        ));
        assert_eq!(
            reopened.commit(LedgerEvent::Revoked {
                subject: "unrelated".into(),
                generation: Generation(1),
            }),
            Err(RuntimeError::ExecutionFenced)
        );
        assert_eq!(probe.executions(), 0);
    }
}

#[test]
fn a_fenced_runtime_vouches_for_no_journal_position_so_no_floor_can_pass_it() {
    let (mut runtime, action) = fixture();
    let probe = Probe::default();
    let session = runtime
        .register_execution_session(
            "alice",
            vec![grant(
                scope(&action),
                &probe,
                RequiredVerification::FullSemantic,
                ExecutorMode::Error,
            )],
            TTL,
        )
        .unwrap();
    let permit = runtime
        .prepare_execution(&session, &ProjectId::from("p"), &action, TTL)
        .unwrap();
    assert!(runtime.execute_prepared(&session, permit).is_err());

    // A compacted snapshot is what lets a floor rise, and it needs an anchor.
    // Refusing the anchor is what keeps a floor from discarding the very record
    // that says an effect may have applied.
    assert_eq!(runtime.journal_anchor(), Err(RuntimeError::ExecutionFenced));
    assert_eq!(
        runtime.export_compacted_snapshot().err(),
        Some(RuntimeError::ExecutionFenced)
    );
}

#[test]
fn reconciliation_records_what_an_operator_established_and_lifts_the_fence() {
    for applied in [true, false] {
        let (mut runtime, action) = fixture();
        let probe = Probe::default();
        let session = runtime
            .register_execution_session(
                "alice",
                vec![grant(
                    scope(&action),
                    &probe,
                    RequiredVerification::FullSemantic,
                    ExecutorMode::Error,
                )],
                TTL,
            )
            .unwrap();
        let permit = runtime
            .prepare_execution(&session, &ProjectId::from("p"), &action, TTL)
            .unwrap();
        assert!(runtime.execute_prepared(&session, permit).is_err());
        let attempt = runtime.unsettled_effects()[0].attempt;

        let index = runtime
            .reconcile_effect(attempt, applied, "operator checked the receiving system")
            .unwrap();
        assert!(runtime.unsettled_effects().is_empty());
        assert_eq!(
            runtime.committed_events()[index.0 as usize - 1].event,
            LedgerEvent::EffectReconciled {
                attempt,
                applied,
                evidence: "operator checked the receiving system".into(),
            }
        );
        assert_eq!(
            runtime
                .materialized_state()
                .values
                .get(&format!("effect:{}:applied", attempt.0)),
            Some(&applied.to_string())
        );

        // The runtime is usable again, and it says so the same way for both
        // outcomes: reconciliation ends the ambiguity, it does not judge it.
        assert!(runtime
            .commit(LedgerEvent::Revoked {
                subject: "unrelated".into(),
                generation: Generation(1),
            })
            .is_ok());
    }
}

#[test]
fn reconciling_an_attempt_that_awaits_nothing_is_refused() {
    let (mut runtime, _) = fixture();
    assert_eq!(
        runtime.reconcile_effect(CommitIndex(1), true, "no such attempt"),
        Err(ExecutionError::UnknownAttempt {
            attempt: CommitIndex(1)
        })
    );
    // Control characters would make an audit line unreadable at best and
    // forgeable at worst.
    assert_eq!(
        runtime.reconcile_effect(CommitIndex(1), true, "bad\nevidence"),
        Err(ExecutionError::InvalidEvidence)
    );
    assert!(attempts(&runtime).is_empty());
}

#[test]
fn a_retry_under_the_same_key_is_answered_from_history_without_executing_again() {
    let (mut runtime, action) = fixture();
    let probe = Probe::default();
    let session = session(&mut runtime, &action, &probe, "alice");
    let first = runtime
        .prepare_execution_once(&session, &ProjectId::from("p"), &action, TTL, "invoice-7")
        .unwrap();
    assert_eq!(
        runtime.execute_prepared(&session, first).unwrap(),
        b"executed"
    );
    assert_eq!(probe.executions(), 1);

    let retry = runtime
        .prepare_execution_once(&session, &ProjectId::from("p"), &action, TTL, "invoice-7")
        .unwrap();
    assert_eq!(
        runtime.execute_prepared(&session, retry).unwrap(),
        b"executed"
    );
    // The point of the key: the effect applied once, and the caller still got an
    // answer rather than an error.
    assert_eq!(probe.executions(), 1);
    assert_eq!(attempts(&runtime).len(), 2);

    // A different key is a different request and does execute.
    let other = runtime
        .prepare_execution_once(&session, &ProjectId::from("p"), &action, TTL, "invoice-8")
        .unwrap();
    assert!(runtime.execute_prepared(&session, other).is_ok());
    assert_eq!(probe.executions(), 2);
}

#[test]
fn an_at_most_once_key_still_holds_after_a_restart() {
    let temp = Temp::new();
    let (mut runtime, action) = durable_fixture(&temp.log());
    let probe = Probe::default();
    let session = session(&mut runtime, &action, &probe, "alice");
    let permit = runtime
        .prepare_execution_once(&session, &ProjectId::from("p"), &action, TTL, "invoice-7")
        .unwrap();
    assert_eq!(
        runtime.execute_prepared(&session, permit).unwrap(),
        b"executed"
    );
    assert_eq!(probe.executions(), 1);
    drop(runtime);

    let mut reopened = reopen(&temp.log(), &action);
    let probe = Probe::default();
    let session = common::session(&mut reopened, &action, &probe, "alice");
    let retry = reopened
        .prepare_execution_once(&session, &ProjectId::from("p"), &action, TTL, "invoice-7")
        .unwrap();
    assert_eq!(
        reopened.execute_prepared(&session, retry).unwrap(),
        b"executed"
    );
    // A fresh process, a fresh executor, and still no second effect.
    assert_eq!(probe.executions(), 0);
}

#[test]
fn a_key_reconciled_as_unapplied_may_be_attempted_again() {
    let (mut runtime, action) = fixture();
    let probe = Probe::default();
    let session = runtime
        .register_execution_session(
            "alice",
            vec![grant(
                scope(&action),
                &probe,
                RequiredVerification::FullSemantic,
                ExecutorMode::Error,
            )],
            TTL,
        )
        .unwrap();
    let permit = runtime
        .prepare_execution_once(&session, &ProjectId::from("p"), &action, TTL, "invoice-7")
        .unwrap();
    assert!(runtime.execute_prepared(&session, permit).is_err());
    let attempt = runtime.unsettled_effects()[0].attempt;
    runtime
        .reconcile_effect(attempt, false, "receiving system has no record")
        .unwrap();

    // Nothing applied, so the at-most-once budget was never spent.
    let probe = Probe::default();
    let session = common::session(&mut runtime, &action, &probe, "alice");
    let again = runtime
        .prepare_execution_once(&session, &ProjectId::from("p"), &action, TTL, "invoice-7")
        .unwrap();
    assert_eq!(
        runtime.execute_prepared(&session, again).unwrap(),
        b"executed"
    );
    assert_eq!(probe.executions(), 1);
}

#[test]
fn an_oversize_response_is_settled_by_digest_and_a_retry_is_refused_rather_than_repeated() {
    let (mut runtime, action) = fixture();
    let probe = Probe::default();
    let session = runtime
        .register_execution_session(
            "alice",
            vec![ExecutionGrant::new(
                scope(&action),
                RequiredVerification::Deterministic,
                OversizeAdapter(probe.clone()),
                OversizeAdapter(probe.clone()),
            )],
            TTL,
        )
        .unwrap();
    let permit = runtime
        .prepare_execution_once(&session, &ProjectId::from("p"), &action, TTL, "bulk-1")
        .unwrap();
    let output = runtime.execute_prepared(&session, permit).unwrap();
    assert_eq!(output.len(), MAX_RETAINED_RESPONSE + 1);
    assert!(runtime.unsettled_effects().is_empty());

    let records = attempts(&runtime);
    let attempt = records[0].0;
    assert_eq!(
        records[1].1,
        LedgerEvent::EffectSettled {
            attempt,
            // Not retained, but the digest is recorded either way: "not kept" is
            // never allowed to become "not known".
            response: None,
            response_digest: response_digest_of(&output),
        }
    );

    let retry = runtime
        .prepare_execution_once(&session, &ProjectId::from("p"), &action, TTL, "bulk-1")
        .unwrap();
    assert_eq!(
        runtime.execute_prepared(&session, retry),
        Err(ExecutionError::ResponseNotRetained { attempt })
    );
    // Refusing is the honest answer. Re-executing would apply the effect twice
    // and inventing a response would be worse than either.
    assert_eq!(probe.executions(), 1);
}

#[test]
fn a_malformed_key_is_refused_at_preparation_and_in_committed_history() {
    let (mut runtime, action) = fixture();
    let probe = Probe::default();
    let session = session(&mut runtime, &action, &probe, "alice");
    for malformed in ["", " leading", "trailing ", "line\nbreak"] {
        assert!(matches!(
            runtime.prepare_execution_once(
                &session,
                &ProjectId::from("p"),
                &action,
                TTL,
                malformed
            ),
            Err(ExecutionError::InvalidKey)
        ));
    }
    assert!(attempts(&runtime).is_empty());

    // The same rule applies to a history that was not produced here.
    assert_eq!(
        replayed(vec![attempt_with(Some("line\nbreak"))]).err(),
        Some(RuntimeError::InvalidEffectKey {
            key: "line\nbreak".into()
        })
    );
}

#[test]
fn a_second_live_attempt_under_one_key_is_refused_during_replay() {
    assert_eq!(
        replayed(vec![
            attempt_with(Some("invoice-7")),
            attempt_with(Some("invoice-7"))
        ])
        .err(),
        Some(RuntimeError::EffectKeyInFlight {
            key: "invoice-7".into()
        })
    );

    // Settling the first one first makes the second attempt ordinary: the rule
    // is about two *live* attempts, not about reusing a key ever again.
    let settled = replayed(vec![
        attempt_with(Some("invoice-7")),
        LedgerEvent::EffectSettled {
            attempt: CommitIndex(1),
            response: Some(b"executed".to_vec()),
            response_digest: response_digest_of(b"executed"),
        },
        attempt_with(Some("invoice-7")),
    ])
    .unwrap();
    assert_eq!(settled.unsettled_effects().len(), 1);
}

#[test]
fn a_settlement_naming_no_live_attempt_is_refused_during_replay() {
    for event in [
        LedgerEvent::EffectSettled {
            attempt: CommitIndex(9),
            response: None,
            response_digest: [0; 32],
        },
        LedgerEvent::EffectReconciled {
            attempt: CommitIndex(9),
            applied: true,
            evidence: "checked".into(),
        },
    ] {
        assert_eq!(
            replayed(vec![attempt_with(None), event]).err(),
            Some(RuntimeError::UnknownEffectAttempt {
                attempt: CommitIndex(9)
            })
        );
    }

    // Settling the same attempt twice is the same fault: the second settlement
    // names an attempt that is no longer awaiting one.
    assert_eq!(
        replayed(vec![
            attempt_with(None),
            LedgerEvent::EffectSettled {
                attempt: CommitIndex(1),
                response: None,
                response_digest: [0; 32],
            },
            LedgerEvent::EffectSettled {
                attempt: CommitIndex(1),
                response: None,
                response_digest: [0; 32],
            },
        ])
        .err(),
        Some(RuntimeError::UnknownEffectAttempt {
            attempt: CommitIndex(1)
        })
    );
}

#[test]
fn a_retained_response_that_contradicts_its_own_digest_is_refused() {
    assert_eq!(
        replayed(vec![
            attempt_with(None),
            LedgerEvent::EffectSettled {
                attempt: CommitIndex(1),
                response: Some(b"executed".to_vec()),
                response_digest: response_digest_of(b"something else"),
            },
        ])
        .err(),
        Some(RuntimeError::InconsistentEffectResponse {
            attempt: CommitIndex(1)
        })
    );
}

#[test]
fn a_retained_response_past_the_bound_is_refused_before_it_is_adopted() {
    let oversize = vec![7u8; MAX_RETAINED_RESPONSE + 1];
    // The digest agrees, so the size check is the only thing that can reject it.
    assert_eq!(
        replayed(vec![
            attempt_with(None),
            LedgerEvent::EffectSettled {
                attempt: CommitIndex(1),
                response: Some(oversize.clone()),
                response_digest: response_digest_of(&oversize),
            },
        ])
        .err(),
        Some(RuntimeError::InconsistentEffectResponse {
            attempt: CommitIndex(1)
        })
    );

    // Exactly at the bound is accepted, so the limit is the limit and not an
    // off-by-one that happens to be safe.
    let exact = vec![7u8; MAX_RETAINED_RESPONSE];
    assert!(replayed(vec![
        attempt_with(None),
        LedgerEvent::EffectSettled {
            attempt: CommitIndex(1),
            response: Some(exact.clone()),
            response_digest: response_digest_of(&exact),
        },
    ])
    .is_ok());
}

#[test]
fn an_unsettled_attempt_survives_replay_of_the_history_that_created_it() {
    let runtime = replayed(vec![attempt_with(Some("invoice-7"))]).unwrap();
    let unsettled = runtime.unsettled_effects();
    assert_eq!(unsettled.len(), 1);
    assert_eq!(unsettled[0].attempt, CommitIndex(1));
    assert_eq!(unsettled[0].key.as_deref(), Some("invoice-7"));
    assert_eq!(unsettled[0].effect, Effect::Mutation);
}

#[test]
fn an_action_that_never_reaches_an_executor_leaves_no_effect_record() {
    // The audit describes attempted effects, not refused preparations. A denial
    // leaves nothing outside the runtime to reconcile, so a record for it would
    // be a fence with nothing behind it.
    for prepare_denial in [Denial::HardFinding, Denial::FailedVerification] {
        let (mut runtime, action) = fixture();
        let probe = Probe::default();
        let mut action = action;
        match prepare_denial {
            Denial::HardFinding => probe.hard_finding(),
            // The fixture verifier fails any payload it does not recognise.
            Denial::FailedVerification => action.payload = b"unverified payload".to_vec(),
        }
        let session = common::session(&mut runtime, &action, &probe, "alice");
        let permit = runtime
            .prepare_execution_once(&session, &ProjectId::from("p"), &action, TTL, "invoice-7")
            .unwrap();
        assert!(runtime.execute_prepared(&session, permit).is_err());

        assert!(attempts(&runtime).is_empty());
        assert!(runtime.unsettled_effects().is_empty());
        assert_eq!(probe.executions(), 0);
        // Nothing was spent, so the same key is still free.
        let probe = Probe::default();
        let (mut runtime, action) = fixture();
        let session = common::session(&mut runtime, &action, &probe, "alice");
        let permit = runtime
            .prepare_execution_once(&session, &ProjectId::from("p"), &action, TTL, "invoice-7")
            .unwrap();
        assert!(runtime.execute_prepared(&session, permit).is_ok());
    }
}

enum Denial {
    HardFinding,
    FailedVerification,
}

fn action_digest_of(action: &ActionIr) -> [u8; 32] {
    action_digest(action)
}

fn response_digest_of(response: &[u8]) -> [u8; 32] {
    ptr_ledger::integrity::sha256(response)
}
