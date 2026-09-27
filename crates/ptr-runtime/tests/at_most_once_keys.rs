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
//! could be answered once either had moved. A snapshot carries whatever a
//! settled key needs: the project and principal at any length, since no build
//! has bounded them, and the key as it always has, which is why a key too long
//! for it is refused at preparation and at commit before it can be spent. A
//! log an earlier build wrote may hold a longer key; it still opens wherever
//! stored history is rebuilt, because refusing it would keep the runtime from
//! starting after an upgrade, and the request that spent the key is still
//! answered from history.
//!
//! Every property is checked where the entry is built: live, after an ordinary
//! restart that replays the log, and after a compacted round trip.

mod common;

use common::{fixture, scope, TTL};
use ptr_config::PtrConfig;
use ptr_core::action_head::ActionIr;
use ptr_ledger::{integrity, CommittedEvent, FileLedger, LedgerEvent};
use ptr_runtime::compacted::{CompactedAnchor, CompactedError};
use ptr_runtime::persistence::SnapshotAnchor;
use ptr_runtime::{execution::*, PtrRuntime, RuntimeError};
use ptr_types::{
    CapabilityId, CapsuleId, CommitIndex, Effect, Generation, NodeId, Probability, ProjectId,
    RequestId, Revision, TypeId, VerificationLevel,
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
fn a_principal_of_any_length_spends_a_key_and_a_snapshot_carries_it() {
    let (mut runtime, action) = fixture();
    let echo = Echo::default();
    let long = "a".repeat(64 * 1024);

    // A spent key's principal travels in every later snapshot. No build has
    // bounded a principal, so the snapshot writes it at any length: the
    // admission policy admits a peer under one this long, a session registered
    // under one spends a key, and the runtime still exports, restores and
    // answers the retry, while another principal is refused under the key.
    let mut policy = AdmissionPolicy::new();
    policy
        .admit(NodeId::from("peer"), long.as_str(), TTL, Vec::new)
        .unwrap();
    assert!(policy.admits(&NodeId::from("peer")));

    let session = synchronous(&mut runtime, &long, &action, &echo);
    let permit = once(&runtime, &session, &action, "invoice-7");
    runtime.execute_prepared(&session, permit).unwrap();
    let attempt = attempts(&runtime)[0];
    let mut restored = round_trip(&runtime, &action);
    let session = synchronous(&mut restored, &long, &action, &echo);
    let permit = once(&restored, &session, &action, "invoice-7");
    assert_eq!(
        restored.execute_prepared(&session, permit).unwrap(),
        b"applied verified payload"
    );
    let alice = synchronous(&mut restored, "alice", &action, &echo);
    let permit = once(&restored, &alice, &action, "invoice-7");
    assert_eq!(
        restored.execute_prepared(&alice, permit),
        Err(ExecutionError::KeyBoundToAnotherAction { attempt })
    );
    assert_eq!(echo.calls(), 1);
}

/// An attempt record for `action` as a host commits it directly, or a log
/// replays it, rather than as admission builds it: under `key`, naming
/// `project` and `principal`.
fn recorded(action: &ActionIr, key: Option<&str>, project: &str, principal: &str) -> LedgerEvent {
    LedgerEvent::EffectAttempted {
        key: key.map(str::to_owned),
        project: ProjectId::from(project),
        principal: principal.to_owned(),
        target: action.target.clone(),
        operation: action.operation.clone(),
        capability: action.capability.clone(),
        effect: action.effect,
        generation: action.generation,
        revision: action.revision,
        verification: VerificationLevel::Deterministic,
        action_digest: action_digest(action),
    }
}

/// A recovery snapshot of `history`, framed as `export_recovery_snapshot`
/// frames one, with the anchor a host would retain for it. It is framed here
/// rather than exported, so a history this build would not commit can still be
/// put in one; the history holds effect records only, and those move no
/// revision.
fn recovery_snapshot(history: &[CommittedEvent]) -> (Vec<u8>, SnapshotAnchor) {
    let log = integrity::encode_log(history).unwrap();
    let anchor = integrity::decode_log(&log).unwrap().anchor();
    let mut bytes = b"PTRSN001".to_vec();
    bytes.extend_from_slice(&0_u64.to_le_bytes());
    bytes.extend_from_slice(&anchor.index.0.to_le_bytes());
    bytes.extend_from_slice(&(log.len() as u64).to_le_bytes());
    bytes.extend_from_slice(&anchor.digest);
    bytes.extend_from_slice(&log);
    let digest = integrity::sha256(&bytes);
    bytes.extend_from_slice(&digest);
    let trusted = SnapshotAnchor {
        revision: Revision(0),
        log: anchor,
        digest,
    };
    (bytes, trusted)
}

/// Rebuild a runtime from a history holding `attempt` at index 1, reconciled
/// as `applied` when that is given and left unsettled otherwise, on every path
/// that rebuilds one from stored history, and assert that each opens it. The
/// paths are `PtrRuntime::replay`, `open_durable` and `open_durable_at` over a
/// log holding the history, and `restore_recovery_snapshot` over a recovery
/// snapshot of it, through which `read_recovery_snapshot` and
/// `restore_durable_snapshot` restore too.
fn assert_opens_on_every_rebuild_path(attempt: &LedgerEvent, applied: Option<bool>) {
    let mut events = vec![attempt.clone()];
    if let Some(applied) = applied {
        events.push(LedgerEvent::EffectReconciled {
            attempt: CommitIndex(1),
            applied,
            evidence: "operator".into(),
        });
    }
    let history: Vec<CommittedEvent> = events
        .into_iter()
        .enumerate()
        .map(|(offset, event)| CommittedEvent {
            index: CommitIndex(offset as u64 + 1),
            event,
        })
        .collect();

    let temp = Temp::new();
    let anchor = {
        let mut log = FileLedger::open(temp.log()).unwrap();
        for committed in &history {
            let index = log.append_durable(committed.event.clone()).unwrap();
            assert_eq!(index, committed.index);
        }
        log.anchor().unwrap()
    };
    let (snapshot, trusted) = recovery_snapshot(&history);
    let rebuilt = [
        (
            "replay",
            PtrRuntime::replay(PtrConfig::default(), &history).err(),
        ),
        (
            "open_durable",
            PtrRuntime::open_durable(PtrConfig::default(), temp.log()).err(),
        ),
        (
            "open_durable_at",
            PtrRuntime::open_durable_at(PtrConfig::default(), temp.log(), anchor).err(),
        ),
        (
            "restore_recovery_snapshot",
            PtrRuntime::restore_recovery_snapshot(PtrConfig::default(), &snapshot, trusted).err(),
        ),
    ];
    for (path, outcome) in rebuilt {
        assert_eq!(outcome, None, "{path}, reconciled as applied: {applied:?}");
    }
}

/// How a log ends the attempt it holds.
#[derive(Clone, Copy, Debug)]
enum Ending {
    Settled,
    Reconciled(bool),
    Unsettled,
}

/// A log an earlier build could have written, reopened by this one as after an
/// upgrade: the durable fixture's ground state, then an attempt on the
/// fixture's action under `key`, recorded for `project` and `principal`, ended
/// as `ending`. Returns the log's directory, which has to outlive the runtime,
/// the runtime, the action and the attempt's index.
fn logged_by_an_earlier_build(
    key: &str,
    project: &str,
    principal: &str,
    ending: Ending,
) -> (Temp, PtrRuntime, ActionIr, CommitIndex) {
    let temp = Temp::new();
    let (runtime, action) = durable_fixture(&temp.log());
    drop(runtime);
    let attempt = {
        let mut log = FileLedger::open(temp.log()).unwrap();
        let attempt = log
            .append_durable(recorded(&action, Some(key), project, principal))
            .unwrap();
        let response = b"applied verified payload".to_vec();
        match ending {
            Ending::Settled => {
                log.append_durable(LedgerEvent::EffectSettled {
                    attempt,
                    response_digest: integrity::sha256(&response),
                    response: Some(response),
                })
                .unwrap();
            }
            Ending::Reconciled(applied) => {
                log.append_durable(LedgerEvent::EffectReconciled {
                    attempt,
                    applied,
                    evidence: "operator".into(),
                })
                .unwrap();
            }
            Ending::Unsettled => {}
        }
        attempt
    };
    let reopened = reopen(&temp.log(), &action);
    (temp, reopened, action, attempt)
}

/// A log an earlier build could have written, holding an attempt under `key`
/// for `project` and `principal`, where the key is longer than
/// `MAX_KEY_BYTES` or the project or principal is one no build has bounded,
/// reopens however the attempt ended, and does next what the previous build
/// did, except that the key is bound to the request that spent it:
/// - export fails with PTR_COMPACTED_SECTION_LIMIT once a key past the bound
///   has settled either way, as it did before, and succeeds otherwise; an
///   attempt left unsettled fences the runtime, as any does;
/// - once the key has applied, alice's request in project p under it is
///   answered from history where the attempt was hers, with its response when
///   that was retained and `ResponseNotRetained` otherwise, and is refused as
///   bound to another request where it was another principal's or project's;
///   either way nothing runs;
/// - once it was reconciled as not applied, the key binds nothing: alice's
///   request under it executes, unless the key is past the bound, in which
///   case its attempt is refused at commit with nothing attempted;
/// - a runtime whose attempt was settled still executes under a key nobody
///   spent.
fn assert_a_log_from_an_earlier_build_opens(key: &str, project: &str, principal: &str) {
    let over_long_key = key.len() > MAX_KEY_BYTES;
    let hers = project == "p" && principal == "alice";
    for ending in [
        Ending::Settled,
        Ending::Reconciled(true),
        Ending::Reconciled(false),
        Ending::Unsettled,
    ] {
        let (_temp, mut reopened, action, attempt) =
            logged_by_an_earlier_build(key, project, principal, ending);

        let exported = match ending {
            Ending::Unsettled => Some(RuntimeError::ExecutionFenced),
            _ if over_long_key => Some(RuntimeError::Compacted(CompactedError::SectionLimit)),
            _ => None,
        };
        assert_eq!(
            reopened.export_compacted_snapshot().err(),
            exported,
            "{ending:?}"
        );
        if matches!(ending, Ending::Unsettled) {
            assert_eq!(
                reopened
                    .unsettled_effects()
                    .iter()
                    .map(|effect| effect.attempt)
                    .collect::<Vec<_>>(),
                vec![attempt]
            );
            continue;
        }

        let echo = Echo::default();
        let alice = synchronous(&mut reopened, "alice", &action, &echo);
        let permit = once(&reopened, &alice, &action, key);
        let outcome = reopened.execute_prepared(&alice, permit);
        let executed = match ending {
            Ending::Settled if hers => {
                assert_eq!(outcome.unwrap(), b"applied verified payload");
                false
            }
            Ending::Reconciled(true) if hers => {
                assert_eq!(
                    outcome,
                    Err(ExecutionError::ResponseNotRetained { attempt })
                );
                false
            }
            Ending::Settled | Ending::Reconciled(true) => {
                assert_eq!(
                    outcome,
                    Err(ExecutionError::KeyBoundToAnotherAction { attempt }),
                    "{ending:?}"
                );
                false
            }
            Ending::Reconciled(false) if over_long_key => {
                assert_eq!(
                    outcome,
                    Err(ExecutionError::Audit(Box::new(
                        RuntimeError::InvalidEffectKey {
                            key: key.to_owned()
                        }
                    )))
                );
                assert!(reopened.unsettled_effects().is_empty());
                false
            }
            Ending::Reconciled(false) => {
                outcome.unwrap();
                true
            }
            Ending::Unsettled => unreachable!("an unsettled attempt fences the runtime"),
        };
        assert_eq!(
            attempts(&reopened).len(),
            if executed { 2 } else { 1 },
            "{ending:?}"
        );
        assert_eq!(echo.calls(), usize::from(executed), "{ending:?}");

        if matches!(ending, Ending::Settled) {
            let permit = once(&reopened, &alice, &action, "invoice-8");
            reopened.execute_prepared(&alice, permit).unwrap();
            assert_eq!(echo.calls(), 1);
        }
    }
}

#[test]
fn a_keyed_attempt_under_any_principal_is_committed_and_a_snapshot_carries_it() {
    let (mut runtime, action) = fixture();
    let long = "a".repeat(64 * 1024);

    // A key settled or reconciled as applied carries its attempt's principal
    // into every later snapshot. The layout before carried no principal, and
    // no build has bounded one, so the snapshot writes it at any length, the
    // empty one included. An attempt committed directly under a key with such
    // a principal, or with one no session could be registered as, is recorded
    // as given, opens on every rebuild path however its key ended, and once
    // applied still exports and restores, with the key bound to it: a
    // session's request under the key is refused.
    let mut spent = Vec::new();
    for (key, principal) in [
        ("invoice-1", ""),
        ("invoice-2", long.as_str()),
        ("invoice-3", " alice"),
        ("invoice-4", "bell\u{7}"),
    ] {
        let keyed = recorded(&action, Some(key), "p", principal);
        for applied in [Some(true), Some(false), None] {
            assert_opens_on_every_rebuild_path(&keyed, applied);
        }
        let attempt = runtime.commit(keyed).unwrap();
        runtime.reconcile_effect(attempt, true, "operator").unwrap();
        spent.push((key, attempt));
    }
    let mut restored = round_trip(&runtime, &action);
    let echo = Echo::default();
    let alice = synchronous(&mut restored, "alice", &action, &echo);
    for (key, attempt) in spent {
        let permit = once(&restored, &alice, &action, key);
        assert_eq!(
            restored.execute_prepared(&alice, permit),
            Err(ExecutionError::KeyBoundToAnotherAction { attempt }),
            "{key}"
        );
    }
    assert_eq!(echo.calls(), 0);

    // A log an earlier build wrote with such a principal reopens and compacts,
    // as it did before this layout carried the principal.
    for principal in ["", long.as_str()] {
        assert_a_log_from_an_earlier_build_opens("invoice-7", "p", principal);
    }
}

#[test]
fn a_key_longer_than_a_snapshot_carries_is_refused_before_it_is_spent_and_a_logged_one_is_still_answered(
) {
    let (mut runtime, action) = fixture();
    let echo = Echo::default();
    let alice = synchronous(&mut runtime, "alice", &action, &echo);

    // A snapshot carries every settled key, whatever its outcome, as 1 to
    // MAX_KEY_BYTES bytes, as the layout before it did, so a key this long
    // made every export after its attempt settled fail. Preparation and commit
    // each accepted it; each refuses it now, before it is spent. The longest is
    // the execution wire's 64 KiB field, the most a peer could send.
    for longer in ["k".repeat(MAX_KEY_BYTES + 1), "k".repeat(64 * 1024)] {
        assert_eq!(
            runtime
                .prepare_execution_once(
                    &alice,
                    &ProjectId::from("p"),
                    &action,
                    TTL,
                    longer.as_str()
                )
                .err(),
            Some(ExecutionError::InvalidKey)
        );
        let keyed = recorded(&action, Some(&longer), "p", "alice");
        assert_eq!(
            runtime.commit(keyed.clone()).err(),
            Some(RuntimeError::InvalidEffectKey {
                key: longer.clone()
            })
        );

        // A log an earlier build wrote may hold one, since admission spent any
        // identifier. It opens on every rebuild path however its key ended,
        // because refusing it would keep the runtime from starting after an
        // upgrade, and preparation lets the key through because the runtime
        // holds it, so the request that spent it is answered from history
        // rather than told nothing was attempted. It is never spent again.
        for applied in [Some(true), Some(false), None] {
            assert_opens_on_every_rebuild_path(&keyed, applied);
        }
        assert_a_log_from_an_earlier_build_opens(&longer, "p", "alice");
    }
    assert!(attempts(&runtime).is_empty());
    assert_eq!(echo.calls(), 0);

    // The control: the longest key allowed is rebuilt on every path from a
    // history that holds it, and through admission it is spent, and the
    // runtime still exports, restores and answers its retry from history.
    let longest = "k".repeat(MAX_KEY_BYTES);
    assert_opens_on_every_rebuild_path(
        &recorded(&action, Some(&longest), "p", "alice"),
        Some(true),
    );
    let permit = once(&runtime, &alice, &action, &longest);
    runtime.execute_prepared(&alice, permit).unwrap();
    let mut restored = round_trip(&runtime, &action);
    let alice = synchronous(&mut restored, "alice", &action, &echo);
    let permit = once(&restored, &alice, &action, &longest);
    assert_eq!(
        restored.execute_prepared(&alice, permit).unwrap(),
        b"applied verified payload"
    );
    assert_eq!(echo.calls(), 1);
}

#[test]
fn a_keyed_attempt_under_any_project_is_committed_and_a_snapshot_carries_it() {
    let (mut runtime, action) = fixture();
    let long = "p".repeat(64 * 1024);

    // A settled key carries its attempt's project too, and `ProjectId` checks
    // nothing. The layout before carried no project, so the snapshot writes it
    // at any length, the empty one included: an attempt committed directly
    // under a key with such a project opens on every rebuild path however its
    // key ended, and once applied still exports and restores with the key
    // bound to it, so alice's request in project p under the key is refused.
    let mut spent = Vec::new();
    for (key, project) in [("invoice-1", ""), ("invoice-2", long.as_str())] {
        let keyed = recorded(&action, Some(key), project, "alice");
        for applied in [Some(true), Some(false), None] {
            assert_opens_on_every_rebuild_path(&keyed, applied);
        }
        let attempt = runtime.commit(keyed).unwrap();
        runtime.reconcile_effect(attempt, true, "operator").unwrap();
        spent.push((key, attempt));
    }
    let mut restored = round_trip(&runtime, &action);
    let echo = Echo::default();
    let alice = synchronous(&mut restored, "alice", &action, &echo);
    for (key, attempt) in spent {
        let permit = once(&restored, &alice, &action, key);
        assert_eq!(
            restored.execute_prepared(&alice, permit),
            Err(ExecutionError::KeyBoundToAnotherAction { attempt }),
            "{key}"
        );
    }
    assert_eq!(echo.calls(), 0);

    // A log an earlier build wrote with such a project reopens and compacts,
    // as it did before this layout carried the project.
    for project in ["", long.as_str()] {
        assert_a_log_from_an_earlier_build_opens("invoice-7", project, "alice");
    }
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

#[test]
fn a_snapshot_whose_recorded_project_or_principal_is_malformed_is_refused() {
    let (mut runtime, action) = fixture();
    let alice = synchronous(&mut runtime, "alice", &action, &Echo::default());
    let permit = once(&runtime, &alice, &action, "invoice-7");
    runtime.execute_prepared(&alice, permit).unwrap();
    let snapshot = runtime.export_compacted_snapshot().unwrap();
    let bytes = snapshot.bytes();

    // The section holds no unsettled attempt and one settled key: its magic,
    // the two counts, the key, the outcome tag and the settling attempt, then
    // the project and principal that attempt recorded, each a length and the
    // bytes. They are written at any length, so decoding holds them to their
    // framing: a length past the bytes that remain, and bytes that are not
    // UTF-8, are refused. Each edit is resealed, so only the section can refuse
    // it.
    let project = execution_section(bytes) + 8 + 4 + 4 + (4 + "invoice-7".len()) + 8 + 8;
    let principal = project + 4 + "p".len();
    assert_eq!(&bytes[project..project + 4], &1_u32.to_le_bytes());
    assert_eq!(&bytes[principal..principal + 4], &5_u32.to_le_bytes());
    assert_eq!(&bytes[principal + 4..principal + 9], b"alice");

    let mut past_the_end = bytes.to_vec();
    past_the_end[principal..principal + 4].copy_from_slice(&u32::MAX.to_le_bytes());
    let anchor = reseal(&mut past_the_end, snapshot.anchor());
    assert_eq!(
        PtrRuntime::restore_compacted(PtrConfig::default(), &past_the_end, anchor, &[]).err(),
        Some(RuntimeError::Compacted(CompactedError::LengthMismatch))
    );

    let mut not_utf8 = bytes.to_vec();
    not_utf8[project + 4] = 0xFF;
    let anchor = reseal(&mut not_utf8, snapshot.anchor());
    assert_eq!(
        PtrRuntime::restore_compacted(PtrConfig::default(), &not_utf8, anchor, &[]).err(),
        Some(RuntimeError::Compacted(CompactedError::NoncanonicalSection))
    );

    // The control: the snapshot as exported restores.
    assert!(
        PtrRuntime::restore_compacted(PtrConfig::default(), bytes, snapshot.anchor(), &[]).is_ok()
    );
}
