//! The audited window for work that does not finish inside the call.
//!
//! P0.1's window opened and closed around a synchronous adapter call, so an effect
//! handed to something that answers later was audited as though it had finished when
//! the call returned. That is precisely the ambiguity the audit exists to prevent:
//! "the call returned" and "the effect applied" are different facts, and for detached
//! work they are separated by an unbounded amount of time.
//!
//! So a detached dispatch commits its attempt and leaves it unsettled. The runtime is
//! fenced from the moment the work is handed over until the adapter reports back —
//! and a call that returned unfenced would be claiming the effect had finished.

mod common;

use common::*;
use ptr_config::PtrConfig;
use ptr_core::action_head::ActionIr;
use ptr_ledger::{LedgerEvent, MAX_RETAINED_RESPONSE};
use ptr_runtime::{execution::*, PtrRuntime, RuntimeError};
use ptr_types::{
    CapabilityId, CapsuleId, CommitIndex, Effect, Generation, ProjectId, RequestId, Revision,
    TypeId,
};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

struct Temp(PathBuf);

impl Temp {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "ptr-detached-{}-{}",
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

/// An adapter that takes the work and answers later — or refuses to take it.
#[derive(Clone, Default)]
struct Handoff {
    accepted: Arc<Mutex<Vec<String>>>,
    refuse: Arc<Mutex<bool>>,
}

impl Handoff {
    fn handoffs(&self) -> usize {
        self.accepted.lock().unwrap().len()
    }

    fn refuse_next(&self) {
        *self.refuse.lock().unwrap() = true;
    }
}

impl DetachedExecutor for Handoff {
    fn start(&self, dispatch: VerifiedDispatch<'_>) -> Result<(), String> {
        if *self.refuse.lock().unwrap() {
            return Err("the queue would not take it".to_owned());
        }
        self.accepted
            .lock()
            .unwrap()
            .push(dispatch.principal().to_owned());
        Ok(())
    }
}

/// A session whose single grant dispatches detached.
fn detached_session(
    runtime: &mut PtrRuntime,
    action: &ActionIr,
    probe: &Probe,
    handoff: &Handoff,
) -> ExecutionSession {
    runtime
        .register_execution_session(
            "alice",
            vec![ExecutionGrant::detached(
                scope(action),
                RequiredVerification::FullSemantic,
                TestVerifier(probe.clone()),
                handoff.clone(),
            )],
            TTL,
        )
        .unwrap()
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

/// The in-memory fixture's ground state on a durable ledger, so a test can drop the
/// runtime and open the log again.
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
    runtime
        .permissions_mut()
        .capabilities
        .insert(action.capability.clone());
    runtime.permissions_mut().allow_mutation = true;
    (runtime, action)
}

fn settlements(runtime: &PtrRuntime) -> Vec<(CommitIndex, Option<usize>, [u8; 32])> {
    runtime
        .committed_events()
        .iter()
        .filter_map(|committed| match &committed.event {
            LedgerEvent::EffectSettled {
                attempt,
                response,
                response_digest,
            } => Some((*attempt, response.as_ref().map(Vec::len), *response_digest)),
            _ => None,
        })
        .collect()
}

#[test]
fn detached_work_fences_the_runtime_until_the_adapter_answers() {
    let (mut runtime, action) = fixture();
    let probe = Probe::default();
    let handoff = Handoff::default();
    let session = detached_session(&mut runtime, &action, &probe, &handoff);
    let permit = runtime
        .prepare_execution(&session, &ProjectId::from("p"), &action, TTL)
        .unwrap();

    let dispatched = runtime.dispatch_detached(&session, permit).unwrap();
    assert!(!dispatched.already_applied);
    assert_eq!(
        handoff.handoffs(),
        1,
        "the work was handed over exactly once"
    );
    assert_eq!(runtime.outstanding_detached(), vec![dispatched.attempt]);

    // The window is open, and everything that depends on an unambiguous outcome
    // refuses while it is.
    assert_eq!(runtime.unsettled_effects().len(), 1);
    assert_eq!(
        runtime.unsettled_effects()[0].attempt,
        dispatched.attempt,
        "the fence names the attempt an answer is owed for"
    );
    assert!(runtime
        .prepare_execution(&session, &ProjectId::from("p"), &action, TTL)
        .is_err());
    assert_eq!(runtime.journal_anchor(), Err(RuntimeError::ExecutionFenced));
    assert!(runtime.export_compacted_snapshot().is_err());

    // The adapter answers. The window closes, and the answer is in history with its
    // digest.
    let settled = runtime
        .settle_detached(dispatched.attempt, b"queued and done".to_vec())
        .unwrap();
    assert!(settled > dispatched.attempt);
    assert!(runtime.unsettled_effects().is_empty());
    assert!(runtime.outstanding_detached().is_empty());
    assert!(runtime.journal_anchor().is_ok());

    let recorded = settlements(&runtime);
    assert_eq!(recorded.len(), 1);
    assert_eq!(recorded[0].0, dispatched.attempt);
    assert_eq!(recorded[0].1, Some("queued and done".len()));
    assert_eq!(
        recorded[0].2,
        ptr_ledger::integrity::sha256(b"queued and done")
    );

    // And the session works again afterwards.
    let again = runtime
        .prepare_execution(&session, &ProjectId::from("p"), &action, TTL)
        .unwrap();
    runtime.dispatch_detached(&session, again).unwrap();
    assert_eq!(handoff.handoffs(), 2);
}

#[test]
fn an_adapter_that_would_not_take_the_work_still_leaves_the_window_open() {
    // "Not accepted" is not "not applied": an adapter that failed while handing work
    // off may have handed it off. The attempt is committed before the adapter is
    // called for exactly this case, and the fence stands after it.
    let (mut runtime, action) = fixture();
    let probe = Probe::default();
    let handoff = Handoff::default();
    let session = detached_session(&mut runtime, &action, &probe, &handoff);
    let permit = runtime
        .prepare_execution(&session, &ProjectId::from("p"), &action, TTL)
        .unwrap();
    handoff.refuse_next();

    assert!(matches!(
        runtime.dispatch_detached(&session, permit),
        Err(ExecutionError::Executor(_))
    ));
    assert_eq!(handoff.handoffs(), 0, "it did not take the work");
    assert_eq!(
        runtime.unsettled_effects().len(),
        1,
        "and the attempt that may nevertheless have reached it is recorded"
    );
    assert_eq!(runtime.journal_anchor(), Err(RuntimeError::ExecutionFenced));

    // Only evidence moves it, because the runtime has no way to find out.
    let attempt = runtime.unsettled_effects()[0].attempt;
    runtime
        .reconcile_effect(attempt, false, "queue-audit-showed-nothing-was-enqueued")
        .unwrap();
    assert!(runtime.unsettled_effects().is_empty());
}

#[test]
fn the_grant_decides_how_an_effect_is_dispatched_and_a_refusal_writes_no_record() {
    let (mut runtime, action) = fixture();
    let probe = Probe::default();
    let handoff = Handoff::default();

    // A detached grant cannot be run synchronously.
    let detached = detached_session(&mut runtime, &action, &probe, &handoff);
    let before = runtime.committed_events().len();
    let permit = runtime
        .prepare_execution(&detached, &ProjectId::from("p"), &action, TTL)
        .unwrap();
    assert_eq!(
        runtime.execute_prepared(&detached, permit),
        Err(ExecutionError::DispatchMismatch { detached: true })
    );
    assert_eq!(
        runtime.committed_events().len(),
        before,
        "a refusal writes no record: a record with nothing behind it is a fence with nothing behind it"
    );
    assert!(runtime.unsettled_effects().is_empty());
    assert_eq!(probe.verifications(), 0, "and nothing was verified either");

    // A synchronous grant cannot be dispatched detached.
    let synchronous = session(&mut runtime, &action, &probe, "bob");
    let before = runtime.committed_events().len();
    let permit = runtime
        .prepare_execution(&synchronous, &ProjectId::from("p"), &action, TTL)
        .unwrap();
    assert_eq!(
        runtime.dispatch_detached(&synchronous, permit),
        Err(ExecutionError::DispatchMismatch { detached: false })
    );
    assert_eq!(runtime.committed_events().len(), before);
    assert!(runtime.unsettled_effects().is_empty());
}

#[test]
fn a_detached_attempt_survives_a_restart_as_a_fence_that_only_evidence_moves() {
    let temp = Temp::new();
    let attempt = {
        let (mut runtime, action) = durable_fixture(&temp.log());
        let probe = Probe::default();
        let handoff = Handoff::default();
        let session = detached_session(&mut runtime, &action, &probe, &handoff);
        let permit = runtime
            .prepare_execution(&session, &ProjectId::from("p"), &action, TTL)
            .unwrap();
        runtime.dispatch_detached(&session, permit).unwrap().attempt
    };

    let mut reopened = PtrRuntime::open_durable(PtrConfig::default(), temp.log()).unwrap();
    assert_eq!(
        reopened.unsettled_effects().len(),
        1,
        "the fence comes back by itself"
    );
    assert_eq!(reopened.unsettled_effects()[0].attempt, attempt);

    // What does *not* come back is the knowledge that this attempt was detached. The
    // record says an effect was attempted, not how it was dispatched, and a restarted
    // runtime that accepted an "adapter answer" for work it never handed over would be
    // inventing that distinction.
    assert!(
        reopened.outstanding_detached().is_empty(),
        "a reopened runtime claims no detached attempts"
    );
    assert_eq!(
        reopened.settle_detached(attempt, b"late answer".to_vec()),
        Err(ExecutionError::NotDetached { attempt })
    );
    assert_eq!(reopened.unsettled_effects().len(), 1, "still fenced");

    // Reconciliation is the way forward, which is the same answer a crash inside a
    // synchronous effect gets.
    reopened
        .reconcile_effect(attempt, true, "broker-receipt-9182")
        .unwrap();
    assert!(reopened.unsettled_effects().is_empty());
}

#[test]
fn a_settlement_for_an_attempt_this_runtime_did_not_hand_out_is_refused() {
    let (mut runtime, action) = fixture();
    let probe = Probe::default();
    let handoff = Handoff::default();
    let session = detached_session(&mut runtime, &action, &probe, &handoff);

    // Nothing dispatched at all.
    assert_eq!(
        runtime.settle_detached(CommitIndex(7), b"x".to_vec()),
        Err(ExecutionError::NotDetached {
            attempt: CommitIndex(7)
        })
    );

    // And a settled one cannot be settled twice.
    let permit = runtime
        .prepare_execution(&session, &ProjectId::from("p"), &action, TTL)
        .unwrap();
    let dispatched = runtime.dispatch_detached(&session, permit).unwrap();
    runtime
        .settle_detached(dispatched.attempt, b"done".to_vec())
        .unwrap();
    assert_eq!(
        runtime.settle_detached(dispatched.attempt, b"again".to_vec()),
        Err(ExecutionError::NotDetached {
            attempt: dispatched.attempt
        })
    );
    assert_eq!(settlements(&runtime).len(), 1);
}

#[test]
fn an_at_most_once_key_hands_detached_work_out_once_even_across_retries() {
    let (mut runtime, action) = fixture();
    let probe = Probe::default();
    let handoff = Handoff::default();
    let session = detached_session(&mut runtime, &action, &probe, &handoff);

    let permit = runtime
        .prepare_execution_once(&session, &ProjectId::from("p"), &action, TTL, "charge-1")
        .unwrap();
    let first = runtime.dispatch_detached(&session, permit).unwrap();
    assert_eq!(handoff.handoffs(), 1);

    // While the first is still outstanding, a retry cannot even be prepared: the
    // fence is a stronger guard than the key, and it names the attempt an answer is
    // owed for. So "two live attempts under one key" is unreachable here rather than
    // merely refused — which is worth stating, because the key rule is what would
    // have caught it if the fence had not.
    // `ExecutionPermit` is deliberately not comparable — it is a capability, not a
    // value — so the refusal is matched rather than equated.
    match runtime.prepare_execution_once(&session, &ProjectId::from("p"), &action, TTL, "charge-1")
    {
        Err(ExecutionError::AmbiguousOutcome { attempt }) => {
            assert_eq!(
                attempt, first.attempt,
                "the fence names the outstanding work"
            );
        }
        other => panic!(
            "expected the fence to refuse a retry, got {:?}",
            other.err()
        ),
    }
    assert_eq!(handoff.handoffs(), 1, "the work was not handed out twice");

    runtime
        .settle_detached(first.attempt, b"charged".to_vec())
        .unwrap();

    // After it is settled, a retry under the same key is answered from history and the
    // adapter is not called again.
    let permit = runtime
        .prepare_execution_once(&session, &ProjectId::from("p"), &action, TTL, "charge-1")
        .unwrap();
    let retry = runtime.dispatch_detached(&session, permit).unwrap();
    assert!(retry.already_applied);
    assert_eq!(
        retry.attempt, first.attempt,
        "it names the attempt that did apply"
    );
    assert_eq!(handoff.handoffs(), 1);
    assert_eq!(settlements(&runtime).len(), 1);
}

#[test]
fn an_oversize_detached_answer_keeps_its_digest_without_being_retained() {
    // "We did not keep the response" must never become "we do not know what
    // happened", so the digest is recorded either way — and a later retry under the
    // key is then refused rather than answered with something the effect never
    // produced.
    let (mut runtime, action) = fixture();
    let probe = Probe::default();
    let handoff = Handoff::default();
    let session = detached_session(&mut runtime, &action, &probe, &handoff);
    let permit = runtime
        .prepare_execution_once(&session, &ProjectId::from("p"), &action, TTL, "bulk-1")
        .unwrap();
    let dispatched = runtime.dispatch_detached(&session, permit).unwrap();

    let huge = vec![7_u8; MAX_RETAINED_RESPONSE + 1];
    runtime
        .settle_detached(dispatched.attempt, huge.clone())
        .unwrap();

    let recorded = settlements(&runtime);
    assert_eq!(recorded.len(), 1);
    assert_eq!(recorded[0].1, None, "past the bound, so not retained");
    assert_eq!(recorded[0].2, ptr_ledger::integrity::sha256(&huge));
    assert!(
        runtime.unsettled_effects().is_empty(),
        "and the fence lifted"
    );

    let permit = runtime
        .prepare_execution_once(&session, &ProjectId::from("p"), &action, TTL, "bulk-1")
        .unwrap();
    assert!(matches!(
        runtime.dispatch_detached(&session, permit),
        Err(ExecutionError::ResponseNotRetained { .. })
    ));
    assert_eq!(handoff.handoffs(), 1);
}
