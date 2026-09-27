//! An at-most-once key names one request: the action its applied attempt
//! recorded, the project and principal it was recorded for, and the attempt
//! that settled it.
//!
//! A retry under a spent key is answered from history. That answer is right
//! only for the request the key was spent on, so a permit under the key for
//! another action, project or principal is refused rather than handed the
//! first request's receipt — which reported an effect that never ran for it and
//! gave one principal another's response. And a retry names the attempt that
//! settled its own key, read from the key's entry rather than guessed from
//! response bytes that another key may share, or from records that compaction
//! has already removed.
//!
//! The request is the action without the position it was issued at. Admission
//! requires a retry to carry the current revision and generation, so the retry
//! is compared at the ones its attempt recorded; compared as recorded, no retry
//! could be answered once either had moved. And a principal too long for a
//! snapshot to carry is refused before it can spend a key.
//!
//! Every property is checked where the entry is built: live, after an ordinary
//! restart that replays the log, and after a compacted round trip.

mod common;

use common::{fixture, scope, TTL};
use ptr_config::PtrConfig;
use ptr_core::action_head::ActionIr;
use ptr_ledger::{integrity, FileLedger, LedgerEvent};
use ptr_runtime::compacted::{CompactedAnchor, CompactedError};
use ptr_runtime::{execution::*, PtrRuntime, RuntimeError};
use ptr_types::{
    CapabilityId, CapsuleId, CommitIndex, Effect, Generation, NodeId, Probability, ProjectId,
    RequestId, TypeId, VerificationLevel,
};
use ptr_verifier::{VerificationReport, VerificationStatus, Verifier};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

struct Temp(PathBuf);

impl Temp {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "ptr-at-most-once-{}-{}",
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

/// An adapter that verifies any payload, so two actions that differ only in
/// their payload are both admissible. It answers `applied <payload>`, fails
/// without saying whether it applied on the payload `uncertain`, and counts
/// every call it receives, synchronous or detached.
#[derive(Clone, Default)]
struct Echo {
    calls: Arc<Mutex<Vec<Vec<u8>>>>,
}

impl Echo {
    fn calls(&self) -> usize {
        self.calls.lock().unwrap().len()
    }
}

impl Verifier<ActionIr> for Echo {
    fn verify(&self, _: &ActionIr) -> VerificationReport {
        VerificationReport {
            status: VerificationStatus::Pass,
            level: VerificationLevel::Deterministic,
            score: Probability::new(1.0).unwrap(),
            findings: vec![],
        }
    }
}

impl ActionExecutor for Echo {
    fn execute(&self, dispatch: VerifiedDispatch<'_>) -> Result<Vec<u8>, String> {
        let payload = dispatch.action().payload.clone();
        self.calls.lock().unwrap().push(payload.clone());
        if payload == b"uncertain" {
            return Err("outcome unknown".into());
        }
        Ok([b"applied ".as_slice(), &payload].concat())
    }
}

impl DetachedExecutor for Echo {
    fn start(&self, dispatch: VerifiedDispatch<'_>) -> Result<(), String> {
        self.calls
            .lock()
            .unwrap()
            .push(dispatch.action().payload.clone());
        Ok(())
    }
}

fn with_payload(action: &ActionIr, payload: &[u8]) -> ActionIr {
    ActionIr {
        payload: payload.to_vec(),
        ..action.clone()
    }
}

fn synchronous(
    runtime: &mut PtrRuntime,
    principal: &str,
    action: &ActionIr,
    echo: &Echo,
) -> ExecutionSession {
    let grant = ExecutionGrant::new(
        scope(action),
        RequiredVerification::Deterministic,
        echo.clone(),
        echo.clone(),
    );
    runtime
        .register_execution_session(principal, vec![grant], TTL)
        .unwrap()
}

fn detached(
    runtime: &mut PtrRuntime,
    principal: &str,
    action: &ActionIr,
    echo: &Echo,
) -> ExecutionSession {
    let grant = ExecutionGrant::detached(
        scope(action),
        RequiredVerification::Deterministic,
        echo.clone(),
        echo.clone(),
    );
    runtime
        .register_execution_session(principal, vec![grant], TTL)
        .unwrap()
}

fn once(
    runtime: &PtrRuntime,
    session: &ExecutionSession,
    action: &ActionIr,
    key: &str,
) -> ExecutionPermit {
    runtime
        .prepare_execution_once(session, &ProjectId::from("p"), action, TTL, key)
        .unwrap()
}

/// The index of every attempt record, in commit order.
fn attempts(runtime: &PtrRuntime) -> Vec<CommitIndex> {
    runtime
        .committed_events()
        .iter()
        .filter(|committed| matches!(committed.event, LedgerEvent::EffectAttempted { .. }))
        .map(|committed| committed.index)
        .collect()
}

/// Permissions are this process's rather than committed history's, so a
/// reopened or restored runtime starts with none.
fn regrant(runtime: &mut PtrRuntime, action: &ActionIr) {
    runtime
        .permissions_mut()
        .capabilities
        .insert(action.capability.clone());
    runtime.permissions_mut().allow_mutation = true;
}

/// Export a compacted snapshot and restore it with nothing above its floor, so
/// every record the snapshot describes is gone.
fn round_trip(runtime: &PtrRuntime, action: &ActionIr) -> PtrRuntime {
    let snapshot = runtime.export_compacted_snapshot().unwrap();
    let mut restored = PtrRuntime::restore_compacted(
        PtrConfig::default(),
        snapshot.bytes(),
        snapshot.anchor(),
        &[],
    )
    .unwrap();
    regrant(&mut restored, action);
    restored
}

/// The in-memory fixture's ground state on a durable ledger, so a test can
/// drop the runtime and open the log again.
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
    let action = ActionIr {
        operation: "write".into(),
        target: "capsule:a".into(),
        capability: CapabilityId::from("file.write"),
        effect: Effect::Mutation,
        input_type: TypeId::from("Bytes"),
        generation: Generation(1),
        revision,
        payload: b"verified payload".to_vec(),
    };
    regrant(&mut runtime, &action);
    (runtime, action)
}

fn reopen(path: &PathBuf, action: &ActionIr) -> PtrRuntime {
    let mut runtime = PtrRuntime::open_durable(PtrConfig::default(), path).unwrap();
    regrant(&mut runtime, action);
    runtime
}

#[test]
fn a_spent_key_is_refused_for_another_action_rather_than_answered_with_the_first_response() {
    let (mut runtime, action) = fixture();
    let echo = Echo::default();
    let alice = synchronous(&mut runtime, "alice", &action, &echo);
    let permit = once(&runtime, &alice, &action, "invoice-7");
    assert_eq!(
        runtime.execute_prepared(&alice, permit).unwrap(),
        b"applied verified payload"
    );
    let attempt = attempts(&runtime)[0];
    let history = runtime.committed_events().len();

    // Another payload under the spent key is another request. Answering it
    // with the first response would report an effect that never ran for it.
    let permit = once(
        &runtime,
        &alice,
        &with_payload(&action, b"another payload"),
        "invoice-7",
    );
    let refused = runtime.execute_prepared(&alice, permit);
    assert_eq!(
        refused,
        Err(ExecutionError::KeyBoundToAnotherAction { attempt })
    );
    assert!(refused
        .unwrap_err()
        .to_string()
        .starts_with("PTR_EXECUTION_DENIED: KeyBoundToAnotherAction"));
    assert_eq!(echo.calls(), 1, "the executor was not reached");
    assert_eq!(
        runtime.committed_events().len(),
        history,
        "and nothing was recorded"
    );
    assert!(runtime.unsettled_effects().is_empty());

    // The request the key was spent on is still answered from history.
    let permit = once(&runtime, &alice, &action, "invoice-7");
    assert_eq!(
        runtime.execute_prepared(&alice, permit).unwrap(),
        b"applied verified payload"
    );
    assert_eq!(echo.calls(), 1);
}

#[test]
fn a_spent_key_is_refused_for_another_principal_rather_than_answered_with_its_receipt() {
    let (mut runtime, action) = fixture();
    let alice_echo = Echo::default();
    let bob_echo = Echo::default();
    let alice = synchronous(&mut runtime, "alice", &action, &alice_echo);
    let bob = synchronous(&mut runtime, "bob", &action, &bob_echo);
    let permit = once(&runtime, &alice, &action, "invoice-7");
    runtime.execute_prepared(&alice, permit).unwrap();
    let attempt = attempts(&runtime)[0];
    let history = runtime.committed_events().len();

    // The same action with a grant of his own: the action digest matches, and
    // the request is still not alice's. Her receipt is not his to be given.
    let permit = once(&runtime, &bob, &action, "invoice-7");
    assert_eq!(
        runtime.execute_prepared(&bob, permit),
        Err(ExecutionError::KeyBoundToAnotherAction { attempt })
    );
    assert_eq!(bob_echo.calls(), 0);
    assert_eq!(runtime.committed_events().len(), history);

    // The control: the refusal is about the key, not about bob's grant.
    let permit = once(&runtime, &bob, &action, "invoice-7-bob");
    assert_eq!(
        runtime.execute_prepared(&bob, permit).unwrap(),
        b"applied verified payload"
    );
    assert_eq!(bob_echo.calls(), 1);
    let permit = once(&runtime, &alice, &action, "invoice-7");
    runtime.execute_prepared(&alice, permit).unwrap();
    assert_eq!(alice_echo.calls(), 1, "alice's retry is still answered");
}

#[test]
fn a_detached_dispatch_under_a_key_spent_on_another_action_is_refused_and_hands_nothing_out() {
    let (mut runtime, action) = fixture();
    let echo = Echo::default();
    let session = detached(&mut runtime, "alice", &action, &echo);
    let permit = once(&runtime, &session, &action, "charge-1");
    let first = runtime.dispatch_detached(&session, permit).unwrap();
    runtime
        .settle_detached(first.attempt, b"charged".to_vec())
        .unwrap();
    let history = runtime.committed_events().len();

    let permit = once(
        &runtime,
        &session,
        &with_payload(&action, b"another charge"),
        "charge-1",
    );
    assert_eq!(
        runtime.dispatch_detached(&session, permit),
        Err(ExecutionError::KeyBoundToAnotherAction {
            attempt: first.attempt
        })
    );
    assert_eq!(echo.calls(), 1, "nothing was handed out");
    assert_eq!(runtime.committed_events().len(), history);
    assert!(runtime.outstanding_detached().is_empty());
}

#[test]
fn a_key_reconciled_as_applied_is_bound_to_the_action_its_attempt_recorded() {
    let (mut runtime, action) = fixture();
    let echo = Echo::default();
    let alice = synchronous(&mut runtime, "alice", &action, &echo);
    let bob = synchronous(&mut runtime, "bob", &action, &Echo::default());
    let uncertain = with_payload(&action, b"uncertain");
    let permit = once(&runtime, &alice, &uncertain, "invoice-7");
    assert!(runtime.execute_prepared(&alice, permit).is_err());
    let attempt = attempts(&runtime)[0];
    runtime
        .reconcile_effect(attempt, true, "receiving system shows it")
        .unwrap();

    // Reconciliation establishes that the effect applied and never what it
    // returned, so the binding comes from the attempt record alone.
    let permit = once(&runtime, &alice, &action, "invoice-7");
    assert_eq!(
        runtime.execute_prepared(&alice, permit),
        Err(ExecutionError::KeyBoundToAnotherAction { attempt })
    );
    let permit = once(&runtime, &bob, &uncertain, "invoice-7");
    assert_eq!(
        runtime.execute_prepared(&bob, permit),
        Err(ExecutionError::KeyBoundToAnotherAction { attempt })
    );
    let permit = once(&runtime, &alice, &uncertain, "invoice-7");
    assert_eq!(
        runtime.execute_prepared(&alice, permit),
        Err(ExecutionError::ResponseNotRetained { attempt })
    );
    assert_eq!(echo.calls(), 1);
}

#[test]
fn a_key_reconciled_as_not_applied_binds_no_action() {
    let (mut runtime, action) = fixture();
    let alice_echo = Echo::default();
    let bob_echo = Echo::default();
    let alice = synchronous(&mut runtime, "alice", &action, &alice_echo);
    let bob = synchronous(&mut runtime, "bob", &action, &bob_echo);
    let uncertain = with_payload(&action, b"uncertain");
    let permit = once(&runtime, &alice, &uncertain, "invoice-7");
    assert!(runtime.execute_prepared(&alice, permit).is_err());
    let first = attempts(&runtime)[0];
    runtime
        .reconcile_effect(first, false, "receiving system has no record")
        .unwrap();

    // Nothing applied under the key, so there is no response to give the
    // wrong request and nothing that could apply twice: another action from
    // another principal may spend it.
    let corrected = with_payload(&action, b"corrected");
    let permit = once(&runtime, &bob, &corrected, "invoice-7");
    assert_eq!(
        runtime.execute_prepared(&bob, permit).unwrap(),
        b"applied corrected"
    );
    let second = attempts(&runtime)[1];

    // From here the key is bound to the action that applied, and to no other.
    let permit = once(&runtime, &alice, &uncertain, "invoice-7");
    assert_eq!(
        runtime.execute_prepared(&alice, permit),
        Err(ExecutionError::KeyBoundToAnotherAction { attempt: second })
    );
    let permit = once(&runtime, &bob, &corrected, "invoice-7");
    assert_eq!(
        runtime.execute_prepared(&bob, permit).unwrap(),
        b"applied corrected"
    );
    assert_eq!(alice_echo.calls(), 1);
    assert_eq!(bob_echo.calls(), 1);
}

#[test]
fn a_spent_key_stays_bound_to_its_action_and_principal_across_a_restart() {
    let temp = Temp::new();
    let (mut runtime, action) = durable_fixture(&temp.log());
    let alice = synchronous(&mut runtime, "alice", &action, &Echo::default());
    let permit = once(&runtime, &alice, &action, "invoice-7");
    runtime.execute_prepared(&alice, permit).unwrap();
    let attempt = attempts(&runtime)[0];
    drop(runtime);

    // The binding is rebuilt from the attempt record, the same way the fence
    // is, so a fresh process holds the key to the same request.
    let mut reopened = reopen(&temp.log(), &action);
    let echo = Echo::default();
    let alice = synchronous(&mut reopened, "alice", &action, &echo);
    let bob = synchronous(&mut reopened, "bob", &action, &echo);
    let permit = once(
        &reopened,
        &alice,
        &with_payload(&action, b"another payload"),
        "invoice-7",
    );
    assert_eq!(
        reopened.execute_prepared(&alice, permit),
        Err(ExecutionError::KeyBoundToAnotherAction { attempt })
    );
    let permit = once(&reopened, &bob, &action, "invoice-7");
    assert_eq!(
        reopened.execute_prepared(&bob, permit),
        Err(ExecutionError::KeyBoundToAnotherAction { attempt })
    );
    let permit = once(&reopened, &alice, &action, "invoice-7");
    assert_eq!(
        reopened.execute_prepared(&alice, permit).unwrap(),
        b"applied verified payload"
    );
    assert_eq!(echo.calls(), 0);
}

#[test]
fn a_key_spent_under_another_project_is_refused_after_a_restart_and_after_compaction() {
    let temp = Temp::new();
    let (runtime, action) = durable_fixture(&temp.log());
    drop(runtime);

    // A capsule belongs to one project, so no live path records one action
    // under two. The attempt is written into the log directly, which is what a
    // restart reads: the same action and principal, recorded under project q.
    let response = b"applied in q".to_vec();
    let attempt = {
        let mut log = FileLedger::open(temp.log()).unwrap();
        let attempt = log
            .append_durable(LedgerEvent::EffectAttempted {
                key: Some("invoice-7".into()),
                project: ProjectId::from("q"),
                principal: "alice".into(),
                target: action.target.clone(),
                operation: action.operation.clone(),
                capability: action.capability.clone(),
                effect: action.effect,
                generation: action.generation,
                revision: action.revision,
                verification: VerificationLevel::Deterministic,
                action_digest: action_digest(&action),
            })
            .unwrap();
        log.append_durable(LedgerEvent::EffectSettled {
            attempt,
            response_digest: integrity::sha256(&response),
            response: Some(response),
        })
        .unwrap();
        attempt
    };

    let mut reopened = reopen(&temp.log(), &action);
    let echo = Echo::default();
    let alice = synchronous(&mut reopened, "alice", &action, &echo);
    let permit = once(&reopened, &alice, &action, "invoice-7");
    assert_eq!(
        reopened.execute_prepared(&alice, permit),
        Err(ExecutionError::KeyBoundToAnotherAction { attempt })
    );

    let mut restored = round_trip(&reopened, &action);
    let alice = synchronous(&mut restored, "alice", &action, &echo);
    let permit = once(&restored, &alice, &action, "invoice-7");
    assert_eq!(
        restored.execute_prepared(&alice, permit),
        Err(ExecutionError::KeyBoundToAnotherAction { attempt })
    );
    assert_eq!(echo.calls(), 0);

    // The control: alice may act in project p under a key nobody spent.
    let permit = once(&restored, &alice, &action, "invoice-8");
    restored.execute_prepared(&alice, permit).unwrap();
    assert_eq!(echo.calls(), 1);
}

#[test]
fn a_restored_runtime_keeps_each_spent_key_bound_to_its_action_principal_and_attempt() {
    let (mut runtime, action) = fixture();
    let alice = synchronous(&mut runtime, "alice", &action, &Echo::default());
    let permit = once(&runtime, &alice, &action, "invoice-7");
    runtime.execute_prepared(&alice, permit).unwrap();
    let applied = attempts(&runtime)[0];
    let uncertain = with_payload(&action, b"uncertain");
    let permit = once(&runtime, &alice, &uncertain, "invoice-8");
    assert!(runtime.execute_prepared(&alice, permit).is_err());
    let unretained = attempts(&runtime)[1];
    runtime
        .reconcile_effect(unretained, true, "receiving system shows it")
        .unwrap();

    // Every record that named these attempts is below the floor now; the
    // snapshot is all that says which request spent each key.
    let mut restored = round_trip(&runtime, &action);
    let echo = Echo::default();
    let alice = synchronous(&mut restored, "alice", &action, &echo);
    let bob = synchronous(&mut restored, "bob", &action, &echo);
    let other = with_payload(&action, b"another payload");
    let permit = once(&restored, &alice, &other, "invoice-7");
    assert_eq!(
        restored.execute_prepared(&alice, permit),
        Err(ExecutionError::KeyBoundToAnotherAction { attempt: applied })
    );
    let permit = once(&restored, &bob, &action, "invoice-7");
    assert_eq!(
        restored.execute_prepared(&bob, permit),
        Err(ExecutionError::KeyBoundToAnotherAction { attempt: applied })
    );
    let permit = once(&restored, &alice, &action, "invoice-7");
    assert_eq!(
        restored.execute_prepared(&alice, permit).unwrap(),
        b"applied verified payload"
    );
    let permit = once(&restored, &alice, &other, "invoice-8");
    assert_eq!(
        restored.execute_prepared(&alice, permit),
        Err(ExecutionError::KeyBoundToAnotherAction {
            attempt: unretained
        })
    );
    let permit = once(&restored, &alice, &uncertain, "invoice-8");
    assert_eq!(
        restored.execute_prepared(&alice, permit),
        Err(ExecutionError::ResponseNotRetained {
            attempt: unretained
        })
    );
    assert_eq!(echo.calls(), 0);
}

#[test]
fn a_detached_retry_names_the_attempt_that_settled_its_own_key_when_another_key_settled_equal_bytes(
) {
    let (mut runtime, action) = fixture();
    let echo = Echo::default();
    let session = detached(&mut runtime, "alice", &action, &echo);
    let mut settled = Vec::new();
    for key in ["charge-1", "charge-2"] {
        let permit = once(&runtime, &session, &action, key);
        let dispatched = runtime.dispatch_detached(&session, permit).unwrap();
        runtime
            .settle_detached(dispatched.attempt, b"ok".to_vec())
            .unwrap();
        settled.push(dispatched.attempt);
    }
    assert_ne!(settled[0], settled[1]);

    // Two keys, equal response bytes. The attempt comes from each key's own
    // entry, so neither retry can be answered with the other's.
    for (key, attempt) in ["charge-1", "charge-2"].into_iter().zip(settled) {
        let permit = once(&runtime, &session, &action, key);
        let retry = runtime.dispatch_detached(&session, permit).unwrap();
        assert!(retry.already_applied, "{key}");
        assert_eq!(retry.attempt, attempt, "{key}");
    }
    assert_eq!(echo.calls(), 2);
}

#[test]
fn a_detached_retry_after_a_compacted_round_trip_names_the_attempt_that_settled_its_key() {
    let (mut runtime, action) = fixture();
    let session = detached(&mut runtime, "alice", &action, &Echo::default());
    let permit = once(&runtime, &session, &action, "charge-1");
    let dispatched = runtime.dispatch_detached(&session, permit).unwrap();
    runtime
        .settle_detached(dispatched.attempt, b"charged".to_vec())
        .unwrap();

    // The settlement record is below the floor after this, so the attempt can
    // only come from what the snapshot carried.
    let mut restored = round_trip(&runtime, &action);
    let echo = Echo::default();
    let session = detached(&mut restored, "alice", &action, &echo);
    let permit = once(&restored, &session, &action, "charge-1");
    let retry = restored.dispatch_detached(&session, permit).unwrap();
    assert!(retry.already_applied);
    assert_eq!(retry.attempt, dispatched.attempt);
    assert_eq!(echo.calls(), 0);
}

#[test]
fn a_refused_retry_names_the_attempt_that_settled_its_key_not_the_first_attempt_under_it() {
    let (mut runtime, action) = fixture();
    let echo = Echo::default();
    let alice = synchronous(&mut runtime, "alice", &action, &echo);
    let uncertain = with_payload(&action, b"uncertain");
    let permit = once(&runtime, &alice, &uncertain, "invoice-7");
    assert!(runtime.execute_prepared(&alice, permit).is_err());
    let first = attempts(&runtime)[0];
    runtime
        .reconcile_effect(first, false, "receiving system has no record")
        .unwrap();
    let permit = once(&runtime, &alice, &uncertain, "invoice-7");
    assert!(runtime.execute_prepared(&alice, permit).is_err());
    let second = attempts(&runtime)[1];
    runtime
        .reconcile_effect(second, true, "receiving system shows it")
        .unwrap();

    // Two attempts under one key, and only the second applied. The first
    // attempt record under the key is not the one that settled it.
    let permit = once(&runtime, &alice, &uncertain, "invoice-7");
    assert_eq!(
        runtime.execute_prepared(&alice, permit),
        Err(ExecutionError::ResponseNotRetained { attempt: second })
    );
    assert_eq!(echo.calls(), 2);
}

/// Alice's retry of the request that spent `invoice-7`, carrying the revision
/// and generation `current` holds, is answered from history without reaching an
/// executor or writing a record, while another payload from her and the same
/// action from bob are still refused.
fn retry_at(runtime: &mut PtrRuntime, current: &ActionIr, attempt: CommitIndex) {
    let echo = Echo::default();
    let alice = synchronous(runtime, "alice", current, &echo);
    let bob = synchronous(runtime, "bob", current, &echo);
    let history = runtime.committed_events().len();
    let permit = once(runtime, &alice, current, "invoice-7");
    assert_eq!(
        runtime.execute_prepared(&alice, permit).unwrap(),
        b"applied verified payload"
    );
    let permit = once(
        runtime,
        &alice,
        &with_payload(current, b"another payload"),
        "invoice-7",
    );
    assert_eq!(
        runtime.execute_prepared(&alice, permit),
        Err(ExecutionError::KeyBoundToAnotherAction { attempt })
    );
    let permit = once(runtime, &bob, current, "invoice-7");
    assert_eq!(
        runtime.execute_prepared(&bob, permit),
        Err(ExecutionError::KeyBoundToAnotherAction { attempt })
    );
    assert_eq!(echo.calls(), 0);
    assert_eq!(runtime.committed_events().len(), history);
}

#[test]
fn a_retry_is_answered_after_the_revision_and_generation_move_and_another_request_is_still_refused()
{
    let temp = Temp::new();
    let (mut runtime, action) = durable_fixture(&temp.log());
    let alice = synchronous(&mut runtime, "alice", &action, &Echo::default());
    let permit = once(&runtime, &alice, &action, "invoice-7");
    runtime.execute_prepared(&alice, permit).unwrap();
    let attempt = attempts(&runtime)[0];

    // An unrelated fact moves the revision, and admission refuses a permit that
    // does not carry the current one. The attempt's digest covers the revision
    // it was issued at, so compared as recorded no retry could match it again,
    // and the requester whose effect applied would be told the key was spent on
    // another action.
    let revision = runtime
        .ingest_text(RequestId::from("unrelated"), "an unrelated fact")
        .unwrap();
    let moved = ActionIr {
        revision,
        ..action.clone()
    };
    retry_at(&mut runtime, &moved, attempt);

    // The same for the target's generation.
    runtime
        .commit(LedgerEvent::CapsuleSuperseded {
            capsule: CapsuleId::from("capsule:a"),
            old: Generation(1),
            new: Generation(2),
        })
        .unwrap();
    let superseded = ActionIr {
        generation: Generation(2),
        ..moved
    };
    retry_at(&mut runtime, &superseded, attempt);

    // The recorded position comes back from the attempt record on a restart,
    // and from the snapshot once that record is below the floor.
    drop(runtime);
    let mut reopened = reopen(&temp.log(), &action);
    retry_at(&mut reopened, &superseded, attempt);
    let mut restored = round_trip(&reopened, &action);
    retry_at(&mut restored, &superseded, attempt);
}

#[test]
fn a_principal_longer_than_a_snapshot_carries_is_refused_before_it_can_spend_a_key() {
    let (mut runtime, action) = fixture();
    let echo = Echo::default();
    let longer = "a".repeat(MAX_PRINCIPAL_BYTES + 1);

    // A spent key's principal travels in every later snapshot, so one this long
    // would make every export after its first keyed effect fail. It is refused
    // where it enters: from the host and from the admission policy.
    let grant = ExecutionGrant::new(
        scope(&action),
        RequiredVerification::Deterministic,
        echo.clone(),
        echo.clone(),
    );
    assert_eq!(
        runtime
            .register_execution_session(longer.as_str(), vec![grant], TTL)
            .err(),
        Some(ExecutionError::InvalidSession)
    );
    let mut policy = AdmissionPolicy::new();
    assert_eq!(
        policy.admit(NodeId::from("peer"), longer, TTL, Vec::new),
        Err(ExecutionError::InvalidSession)
    );
    assert!(!policy.admits(&NodeId::from("peer")));

    // The control: the longest principal allowed spends a key, and the runtime
    // still exports, restores and answers its retry.
    let longest = "a".repeat(MAX_PRINCIPAL_BYTES);
    let session = synchronous(&mut runtime, &longest, &action, &echo);
    let permit = once(&runtime, &session, &action, "invoice-7");
    runtime.execute_prepared(&session, permit).unwrap();
    let mut restored = round_trip(&runtime, &action);
    let session = synchronous(&mut restored, &longest, &action, &echo);
    let permit = once(&restored, &session, &action, "invoice-7");
    assert_eq!(
        restored.execute_prepared(&session, permit).unwrap(),
        b"applied verified payload"
    );
    assert_eq!(echo.calls(), 1);
}

/// Where the execution section starts: after the header and the semantic and
/// lifecycle sections, whose lengths the header gives.
fn execution_section(bytes: &[u8]) -> usize {
    let length = |at: usize| u64::from_le_bytes(bytes[at..at + 8].try_into().unwrap()) as usize;
    80 + length(56) + length(64)
}

/// Seal edited bytes again and anchor them, so only the version can refuse them.
fn reseal(bytes: &mut [u8], trusted: CompactedAnchor) -> CompactedAnchor {
    let end = bytes.len() - 32;
    let digest = integrity::sha256(&bytes[..end]);
    bytes[end..].copy_from_slice(&digest);
    CompactedAnchor { digest, ..trusted }
}

#[test]
fn a_snapshot_whose_keys_bind_no_action_is_refused_by_version() {
    let (mut runtime, action) = fixture();
    let alice = synchronous(&mut runtime, "alice", &action, &Echo::default());
    let permit = once(&runtime, &alice, &action, "invoice-7");
    runtime.execute_prepared(&alice, permit).unwrap();
    let snapshot = runtime.export_compacted_snapshot().unwrap();
    let section = execution_section(snapshot.bytes());

    // The layouts that carried a spent key without the attempt that settled it
    // or what that attempt recorded. Read with this build's meaning, a key from
    // either would answer any action for anyone.
    let mut outer = snapshot.bytes().to_vec();
    outer[..8].copy_from_slice(b"PTRCS002");
    let anchor = reseal(&mut outer, snapshot.anchor());
    let mut inner = snapshot.bytes().to_vec();
    inner[section..section + 8].copy_from_slice(b"PTREX001");
    let inner_anchor = reseal(&mut inner, snapshot.anchor());
    for (bytes, anchor) in [(outer, anchor), (inner, inner_anchor)] {
        assert_eq!(
            PtrRuntime::restore_compacted(PtrConfig::default(), &bytes, anchor, &[]).err(),
            Some(RuntimeError::Compacted(CompactedError::UnsupportedVersion))
        );
    }

    // The control: this build's own layout restores, so the refusals above are
    // about the version and nothing else.
    assert_eq!(&snapshot.bytes()[..8], b"PTRCS003");
    assert_eq!(&snapshot.bytes()[section..section + 8], b"PTREX002");
    assert!(PtrRuntime::restore_compacted(
        PtrConfig::default(),
        snapshot.bytes(),
        snapshot.anchor(),
        &[]
    )
    .is_ok());
}
