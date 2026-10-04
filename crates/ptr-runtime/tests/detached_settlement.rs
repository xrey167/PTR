//! Detached work a peer asked for is settled by that peer, and by no other (C4 in
//! issue #23).
//!
//! When an admitted peer's session dispatches detached work, the runtime commits the
//! attempt and, before the work is handed out, a `DetachedSettlerBound` record
//! naming the peer the transport authenticated. From then on the outcome is the
//! bound peer's to report — through `settle_detached_from`, which takes an
//! `AuthenticatedPeer` — once, while the attempt is outstanding, and only while the
//! peer is still a member. Everything else stays the host operator's, through
//! reconciliation.
//!
//! These tests hold that without a network: `ptr-execwire` carries the same calls
//! over real connections.

mod common;

use common::*;
use ptr_config::PtrConfig;
use ptr_core::action_head::ActionIr;
use ptr_ledger::integrity::{self, LogAnchor};
use ptr_ledger::{CommittedEvent, LedgerEvent};
use ptr_runtime::compacted::CompactedAnchor;
use ptr_runtime::{execution::*, PtrRuntime, RuntimeError};
use ptr_types::{CapsuleId, CommitIndex, Generation, NodeId, ProjectId, RequestId};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

/// Tests have no network, so they assume what a transport would have proved.
fn authenticated(id: &NodeId) -> AuthenticatedPeer {
    AuthenticatedPeer::assume_authenticated(id.clone())
}

/// The peer that asks for the work, and is bound to settle it.
fn requester() -> NodeId {
    NodeId::from("k-requester")
}

/// Another member under the same grant set, with every right the requester has.
fn other() -> NodeId {
    NodeId::from("k-other-member")
}

/// The grant set both peers are admitted under.
const GRANT_SET: &str = "detached-writer";

struct Temp(PathBuf);

impl Temp {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "ptr-detached-settlement-{}-{}",
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

/// An adapter that takes work and answers later, recording each hand-off.
#[derive(Clone, Default)]
struct Handoff {
    accepted: Arc<Mutex<Vec<String>>>,
}

impl Handoff {
    fn handoffs(&self) -> usize {
        self.accepted.lock().unwrap().len()
    }
}

impl DetachedExecutor for Handoff {
    fn start(&self, dispatch: VerifiedDispatch<'_>) -> Result<(), String> {
        self.accepted
            .lock()
            .unwrap()
            .push(dispatch.principal().to_owned());
        Ok(())
    }
}

/// The grant set: one detached grant for `action`.
fn grant_sets(action: &ActionIr, handoff: &Handoff) -> GrantSets {
    let mut sets = GrantSets::new();
    let scope = scope(action);
    let handoff = handoff.clone();
    let probe = Probe::default();
    sets.define(GRANT_SET, TTL, move || {
        vec![ExecutionGrant::detached(
            scope.clone(),
            RequiredVerification::FullSemantic,
            TestVerifier(probe.clone()),
            handoff.clone(),
        )]
    })
    .unwrap();
    sets
}

/// Grant sets and both members on a runtime that has the fixture's ground state.
fn admit_both(runtime: &mut PtrRuntime, action: &ActionIr, handoff: &Handoff) {
    runtime.install_grant_sets(grant_sets(action, handoff));
    runtime
        .admit_member(requester(), "requester", GRANT_SET)
        .unwrap();
    runtime.admit_member(other(), "other", GRANT_SET).unwrap();
}

/// The fixture with both peers admitted.
fn admitted() -> (PtrRuntime, ActionIr, Handoff) {
    let (mut runtime, action) = fixture();
    let handoff = Handoff::default();
    admit_both(&mut runtime, &action, &handoff);
    (runtime, action, handoff)
}

/// Dispatch the detached action from `peer`'s session, as the wire host does.
fn dispatch_from(
    runtime: &mut PtrRuntime,
    action: &ActionIr,
    peer: &NodeId,
) -> Result<DetachedEffect, ExecutionError> {
    let session = runtime.admit_peer(&authenticated(peer)).unwrap();
    let permit = runtime
        .prepare_execution(&session, &ProjectId::from("p"), action, TTL)
        .unwrap();
    assert!(permit.is_detached(), "the grant decides the dispatch kind");
    runtime.dispatch_detached(&session, permit)
}

/// The settlements a history records: the attempt each closes and its response.
fn settlements(runtime: &PtrRuntime) -> Vec<(CommitIndex, Option<Vec<u8>>)> {
    runtime
        .committed_events()
        .iter()
        .filter_map(|committed| match &committed.event {
            LedgerEvent::EffectSettled {
                attempt, response, ..
            } => Some((*attempt, response.clone())),
            _ => None,
        })
        .collect()
}

#[test]
fn the_bound_settler_settles_and_the_fence_lifts() {
    let (mut runtime, action, handoff) = admitted();
    let dispatched = dispatch_from(&mut runtime, &action, &requester()).unwrap();
    let attempt = dispatched.attempt;
    assert!(!dispatched.already_applied);
    assert_eq!(handoff.handoffs(), 1, "the work was handed out");

    // The binding is the record right after the attempt, and names the peer the
    // transport authenticated — not a principal, not a name from a payload.
    let history = runtime.committed_events();
    let at = history.iter().position(|c| c.index == attempt).unwrap();
    assert_eq!(
        history[at + 1].event,
        LedgerEvent::DetachedSettlerBound {
            attempt,
            settler: requester(),
        }
    );
    assert_eq!(runtime.detached_settler(attempt), Some(&requester()));
    assert_eq!(runtime.outstanding_detached(), vec![attempt]);

    // Before the report: fenced, nothing settled.
    assert_eq!(runtime.unsettled_effects().len(), 1);
    assert_eq!(runtime.journal_anchor(), Err(RuntimeError::ExecutionFenced));
    assert!(settlements(&runtime).is_empty());

    let settled = runtime
        .settle_detached_from(&authenticated(&requester()), attempt, b"done".to_vec())
        .unwrap();
    assert!(settled > attempt);

    // After it: the fence is lifted and the outcome is history.
    assert!(runtime.unsettled_effects().is_empty());
    assert!(runtime.journal_anchor().is_ok());
    assert_eq!(
        settlements(&runtime),
        vec![(attempt, Some(b"done".to_vec()))]
    );
    assert_eq!(runtime.detached_settler(attempt), None);
    assert!(runtime.outstanding_detached().is_empty());
}

#[test]
fn another_admitted_peer_is_refused_and_nothing_is_recorded() {
    let (mut runtime, action, _) = admitted();
    let attempt = dispatch_from(&mut runtime, &action, &requester())
        .unwrap()
        .attempt;
    let before = runtime.committed_events().len();

    assert_eq!(
        runtime.settle_detached_from(&authenticated(&other()), attempt, b"mine".to_vec()),
        Err(ExecutionError::NotSettler {
            attempt,
            peer: other(),
        })
    );
    assert_eq!(runtime.committed_events().len(), before, "nothing recorded");
    assert_eq!(runtime.unsettled_effects().len(), 1, "still fenced");

    // The control: the bound peer, with the same report, is recorded.
    runtime
        .settle_detached_from(&authenticated(&requester()), attempt, b"mine".to_vec())
        .unwrap();
    assert!(runtime.unsettled_effects().is_empty());
}

#[test]
fn a_second_settlement_is_refused() {
    let (mut runtime, action, _) = admitted();
    let attempt = dispatch_from(&mut runtime, &action, &requester())
        .unwrap()
        .attempt;
    // The control: the first settlement is recorded.
    runtime
        .settle_detached_from(&authenticated(&requester()), attempt, b"first".to_vec())
        .unwrap();
    let after = runtime.committed_events().len();

    for response in [b"first".to_vec(), b"second".to_vec()] {
        assert_eq!(
            runtime.settle_detached_from(&authenticated(&requester()), attempt, response),
            Err(ExecutionError::NotDetached { attempt })
        );
    }
    assert_eq!(runtime.committed_events().len(), after);
    assert_eq!(
        settlements(&runtime),
        vec![(attempt, Some(b"first".to_vec()))]
    );
}

#[test]
fn a_settler_withdrawn_during_the_work_is_refused_and_the_host_reconciles() {
    let (mut runtime, action, _) = admitted();
    let attempt = dispatch_from(&mut runtime, &action, &requester())
        .unwrap()
        .attempt;

    // Withdrawal narrows authority and does not wait for the fence: the host can
    // withdraw the settler while its work is outstanding. Admission widens it, and
    // still waits — the control on the fence being in place.
    runtime.withdraw_peer(&requester()).unwrap();
    assert!(matches!(
        runtime.admit_member(NodeId::from("k-late"), "late", GRANT_SET),
        Err(ExecutionError::Audit(error)) if *error == RuntimeError::ExecutionFenced
    ));
    let before = runtime.committed_events().len();
    assert_eq!(
        runtime.settle_detached_from(&authenticated(&requester()), attempt, b"done".to_vec()),
        Err(ExecutionError::SettlerWithdrawn {
            attempt,
            settler: requester(),
        })
    );
    assert_eq!(runtime.committed_events().len(), before, "nothing recorded");
    assert_eq!(runtime.unsettled_effects().len(), 1, "still fenced");

    // Nor can the host readmit the key to let it report: admission waits behind
    // the fence the attempt holds. A withdrawn settler's work is the operator's.
    assert!(matches!(
        runtime.admit_member(requester(), "requester", GRANT_SET),
        Err(ExecutionError::Audit(error)) if *error == RuntimeError::ExecutionFenced
    ));

    // Reconciliation is the way forward, and once it has recorded the outcome the
    // attempt cannot be settled by anybody.
    let authority = runtime.take_reconciliation_authority().unwrap();
    runtime
        .reconcile_effect(&authority, attempt, true, "broker-receipt-77")
        .unwrap();
    assert!(runtime.unsettled_effects().is_empty());
    assert_eq!(
        runtime.settle_detached_from(&authenticated(&requester()), attempt, b"late".to_vec()),
        Err(ExecutionError::NotDetached { attempt })
    );
}

#[test]
fn a_foreign_or_unbound_attempt_is_refused() {
    let (mut runtime, action, handoff) = admitted();

    // An index that is no attempt at all.
    let capsule = CommitIndex(2);
    assert_eq!(
        runtime.settle_detached_from(&authenticated(&requester()), capsule, b"x".to_vec()),
        Err(ExecutionError::NotDetached { attempt: capsule })
    );

    // An attempt bound at another runtime: its index names nothing here.
    let (mut elsewhere, elsewhere_action, _) = admitted();
    let foreign = dispatch_from(&mut elsewhere, &elsewhere_action, &requester())
        .unwrap()
        .attempt;
    assert_eq!(
        runtime.settle_detached_from(&authenticated(&requester()), foreign, b"x".to_vec()),
        Err(ExecutionError::NotDetached { attempt: foreign })
    );

    // Detached work the host dispatched for itself: no peer, no binding. No peer may
    // settle it, and the host settles it in process — the control.
    let local = runtime
        .register_execution_session(
            "host-itself",
            vec![ExecutionGrant::detached(
                scope(&action),
                RequiredVerification::FullSemantic,
                TestVerifier(Probe::default()),
                handoff.clone(),
            )],
            TTL,
        )
        .unwrap();
    let permit = runtime
        .prepare_execution(&local, &ProjectId::from("p"), &action, TTL)
        .unwrap();
    let unbound = runtime.dispatch_detached(&local, permit).unwrap().attempt;
    assert_eq!(runtime.detached_settler(unbound), None);
    assert!(!runtime
        .committed_events()
        .iter()
        .any(|c| matches!(c.event, LedgerEvent::DetachedSettlerBound { .. })));
    assert_eq!(
        runtime.settle_detached_from(&authenticated(&requester()), unbound, b"x".to_vec()),
        Err(ExecutionError::NotDetached { attempt: unbound })
    );
    runtime.settle_detached(unbound, b"x".to_vec()).unwrap();
    assert!(runtime.unsettled_effects().is_empty());
}

#[test]
fn a_bound_attempt_is_not_settled_in_process_and_a_settled_one_is_not_reconciled() {
    let (mut runtime, action, _) = admitted();
    let attempt = dispatch_from(&mut runtime, &action, &requester())
        .unwrap()
        .attempt;
    // The binding says who reports: not the host's in-process call.
    assert_eq!(
        runtime.settle_detached(attempt, b"host says".to_vec()),
        Err(ExecutionError::BoundToSettler {
            attempt,
            settler: requester(),
        })
    );
    assert_eq!(runtime.unsettled_effects().len(), 1);
    runtime
        .settle_detached_from(&authenticated(&requester()), attempt, b"done".to_vec())
        .unwrap();
    // Settled is final: reconciliation finds nothing awaiting it.
    let authority = runtime.take_reconciliation_authority().unwrap();
    assert_eq!(
        runtime.reconcile_effect(&authority, attempt, false, "operator-disagrees"),
        Err(ExecutionError::UnknownAttempt { attempt })
    );
    assert_eq!(
        settlements(&runtime),
        vec![(attempt, Some(b"done".to_vec()))]
    );
}

/// The fixture's ground state on a durable ledger with both peers admitted.
fn durable_admitted(path: &PathBuf, handoff: &Handoff) -> (PtrRuntime, ActionIr) {
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
    let (_, mut action) = fixture();
    action.revision = revision;
    grant_permissions(&mut runtime, &action);
    admit_both(&mut runtime, &action, handoff);
    (runtime, action)
}

fn grant_permissions(runtime: &mut PtrRuntime, action: &ActionIr) {
    runtime
        .permissions_mut()
        .capabilities
        .insert(action.capability.clone());
    runtime.permissions_mut().allow_mutation = true;
}

#[test]
fn the_binding_survives_a_restart() {
    let temp = Temp::new();
    let handoff = Handoff::default();
    let (attempt, action) = {
        let (mut runtime, action) = durable_admitted(&temp.log(), &handoff);
        let attempt = dispatch_from(&mut runtime, &action, &requester())
            .unwrap()
            .attempt;
        (attempt, action)
    };

    let mut reopened = PtrRuntime::open_durable(PtrConfig::default(), temp.log()).unwrap();
    // The fence and the binding both come back from the log. Unlike detached work
    // the host dispatched for itself, this attempt's history says how it was
    // dispatched and to whom its outcome belongs.
    assert_eq!(reopened.unsettled_effects().len(), 1);
    assert_eq!(reopened.detached_settler(attempt), Some(&requester()));
    assert_eq!(reopened.outstanding_detached(), vec![attempt]);

    // Authority is re-derived at use: until the host supplies the grant set the
    // settler's membership names, the settlement is refused.
    assert_eq!(
        reopened.settle_detached_from(&authenticated(&requester()), attempt, b"done".to_vec()),
        Err(ExecutionError::UnknownGrantSet {
            name: GRANT_SET.into()
        })
    );
    reopened.install_grant_sets(grant_sets(&action, &handoff));
    assert_eq!(
        reopened.settle_detached_from(&authenticated(&other()), attempt, b"done".to_vec()),
        Err(ExecutionError::NotSettler {
            attempt,
            peer: other(),
        }),
        "the binding came back with the right peer in it"
    );
    reopened
        .settle_detached_from(&authenticated(&requester()), attempt, b"done".to_vec())
        .unwrap();
    assert!(reopened.unsettled_effects().is_empty());
    drop(reopened);

    // And the settlement is durable too: a second reopen has no fence and no
    // binding, and a repeated report is refused.
    let mut again = PtrRuntime::open_durable(PtrConfig::default(), temp.log()).unwrap();
    again.install_grant_sets(grant_sets(&action, &handoff));
    assert!(again.unsettled_effects().is_empty());
    assert_eq!(again.detached_settler(attempt), None);
    assert_eq!(
        again.settle_detached_from(&authenticated(&requester()), attempt, b"done".to_vec()),
        Err(ExecutionError::NotDetached { attempt })
    );
}

#[test]
fn the_binding_survives_a_compacted_restore() {
    // A fenced runtime exports no snapshot, so the binding crosses a compaction the
    // way its attempt does: as a record above the snapshot's floor, replayed by
    // `restore_compacted` through the validation a live commit applies.
    let (mut runtime, action, handoff) = admitted();
    let snapshot = runtime.export_compacted_snapshot().unwrap();
    let floor = snapshot.anchor().floor.index;
    let attempt = dispatch_from(&mut runtime, &action, &requester())
        .unwrap()
        .attempt;
    let above: Vec<CommittedEvent> = runtime
        .committed_events()
        .iter()
        .filter(|c| c.index > floor)
        .cloned()
        .collect();
    assert!(above
        .iter()
        .any(|c| matches!(c.event, LedgerEvent::DetachedSettlerBound { .. })));

    let mut restored = restore_fixture(snapshot.bytes(), snapshot.anchor(), &above, &action);
    restored.install_grant_sets(grant_sets(&action, &handoff));
    assert_eq!(restored.detached_settler(attempt), Some(&requester()));
    assert!(matches!(
        restored.settle_detached_from(&authenticated(&other()), attempt, b"x".to_vec()),
        Err(ExecutionError::NotSettler { .. })
    ));
    restored
        .settle_detached_from(&authenticated(&requester()), attempt, b"done".to_vec())
        .unwrap();
    assert!(restored.unsettled_effects().is_empty());

    // A snapshot taken after the settlement carries the binding's projection key —
    // history — and binds nothing when restored: the attempt is settled. (Taken
    // from the original runtime, whose journal is whole; a restored one anchors
    // only what it holds above its floor.)
    runtime
        .settle_detached_from(&authenticated(&requester()), attempt, b"done".to_vec())
        .unwrap();
    let settled = runtime.export_compacted_snapshot().unwrap();
    let key = ptr_state::detached_settler_key(attempt.0);
    let mut from_settled = restore_fixture(settled.bytes(), settled.anchor(), &[], &action);
    assert_eq!(
        from_settled
            .materialized_state()
            .values
            .get(&key)
            .map(String::as_str),
        Some(requester().0.as_str())
    );
    from_settled.install_grant_sets(grant_sets(&action, &handoff));
    assert_eq!(from_settled.detached_settler(attempt), None);
    assert_eq!(
        from_settled.settle_detached_from(&authenticated(&requester()), attempt, b"x".to_vec()),
        Err(ExecutionError::NotDetached { attempt })
    );
}

#[test]
fn a_binding_a_dispatch_did_not_write_is_refused_in_replay() {
    let (mut runtime, action, _) = admitted();
    let attempt = dispatch_from(&mut runtime, &action, &requester())
        .unwrap()
        .attempt;
    let history = runtime.committed_events().to_vec();
    let at = history.iter().position(|c| c.index == attempt).unwrap();
    // The control: the history as written replays.
    assert!(PtrRuntime::replay(PtrConfig::default(), &history).is_ok());

    let rebind = |settler: NodeId, bound: CommitIndex| -> Vec<CommittedEvent> {
        let mut edited = history.clone();
        edited[at + 1].event = LedgerEvent::DetachedSettlerBound {
            attempt: bound,
            settler,
        };
        edited
    };
    let refused = |events: &[CommittedEvent]| match PtrRuntime::replay(PtrConfig::default(), events)
    {
        Err(RuntimeError::InvalidSettlerBinding { reason, .. }) => reason,
        other => panic!("expected a refused binding, got {:?}", other.err()),
    };

    // A settler the host never admitted.
    assert_eq!(
        refused(&rebind(NodeId::from("k-stranger"), attempt)),
        "the settler is not a member"
    );
    // A binding that does not follow its attempt: it names an earlier record.
    assert_eq!(
        refused(&rebind(requester(), CommitIndex(attempt.0 - 1))),
        "the binding does not immediately follow its attempt"
    );
    // A second binding for the same attempt, appended after the first.
    let mut twice = history[..=at + 1].to_vec();
    twice.push(CommittedEvent {
        index: CommitIndex(history[at + 1].index.0 + 1),
        event: LedgerEvent::DetachedSettlerBound {
            attempt,
            settler: other(),
        },
    });
    assert_eq!(
        refused(&twice),
        "the binding does not immediately follow its attempt"
    );

    // And no host can commit one directly: an attempt awaiting an outcome fences
    // every ordinary commit.
    assert_eq!(
        runtime.commit(LedgerEvent::DetachedSettlerBound {
            attempt,
            settler: other(),
        }),
        Err(RuntimeError::ExecutionFenced)
    );
}

/// The fixture with both peers admitted, restored from its snapshot at `floor`, so
/// exactly `u64::MAX - floor` more records can be appended.
fn admitted_at_floor(floor: CommitIndex) -> (PtrRuntime, ActionIr, Handoff) {
    let (runtime, action, handoff) = admitted();
    let snapshot = runtime.export_compacted_snapshot().unwrap();
    let floor = LogAnchor {
        index: floor,
        digest: snapshot.anchor().floor.digest,
    };
    let mut bytes = snapshot.bytes().to_vec();
    bytes[16..24].copy_from_slice(&floor.index.0.to_le_bytes());
    let end = bytes.len() - 32;
    let digest = integrity::sha256(&bytes[..end]);
    bytes[end..].copy_from_slice(&digest);
    let trusted = CompactedAnchor {
        revision: snapshot.anchor().revision,
        floor,
        digest,
    };
    let mut restored = restore_fixture(&bytes, trusted, &[], &action);
    restored.install_grant_sets(grant_sets(&action, &handoff));
    (restored, action, handoff)
}

#[test]
fn a_binding_the_host_cannot_record_hands_nothing_out() {
    // One index left: the attempt takes it and the binding finds none. The work is
    // not handed out, because nobody would be entitled to report it.
    let (mut runtime, action, handoff) = admitted_at_floor(CommitIndex(u64::MAX - 1));
    let refused = dispatch_from(&mut runtime, &action, &requester());
    assert!(matches!(
        refused,
        Err(ExecutionError::SettlerNotBound { attempt, .. }) if attempt == CommitIndex(u64::MAX)
    ));
    assert_eq!(handoff.handoffs(), 0, "nothing was handed out");
    assert_eq!(
        runtime.unsettled_effects().len(),
        1,
        "the attempt record stands and fences until reconciled"
    );
    assert_eq!(runtime.detached_settler(CommitIndex(u64::MAX)), None);

    // The control: with room for both records the same dispatch binds and hands out.
    let (mut roomy, action, handoff) = admitted_at_floor(CommitIndex(u64::MAX - 2));
    let attempt = dispatch_from(&mut roomy, &action, &requester())
        .unwrap()
        .attempt;
    assert_eq!(handoff.handoffs(), 1);
    assert_eq!(roomy.detached_settler(attempt), Some(&requester()));
}
