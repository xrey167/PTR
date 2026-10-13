//! Raft's log floor against the ledger's retention floor, with a real runtime as
//! the state machine raft feeds.
//!
//! The raft log is where a committed ledger event lives until a durable snapshot
//! of the state machine describes it. If raft discarded an entry no retained
//! snapshot covers, the event would be retained nowhere — and for an
//! `EffectAttempted`/`EffectSettled` pair that is the memory that an at-most-once
//! key was spent. So these tests put the runtime's own records through raft,
//! compact against a `PTRCS006` snapshot's floor, restart, and ask the restored
//! runtime whether it still remembers.
//!
//! Deterministic and single-member: whether a *follower* beyond the floor is still
//! caught up by snapshot is `tests/wire.rs` over a real connection and
//! `ptr-ledger`'s `raft_cluster.rs` message by message.

use ptr_config::PtrConfig;
use ptr_core::action_head::ActionIr;
use ptr_ledger::{CommittedEvent, LedgerEvent, RaftNode, RetentionFloor};
use ptr_runtime::execution::{
    ActionExecutor, ActionScope, ExecutionGrant, ExecutionSession, RequiredVerification,
    VerifiedDispatch,
};
use ptr_runtime::PtrRuntime;
use ptr_types::{
    CapabilityId, CapsuleId, CommitIndex, Effect, Generation, Probability, ProjectId, RequestId,
    TypeId, VerificationLevel,
};
use ptr_verifier::{VerificationReport, VerificationStatus, Verifier};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

const TTL: Duration = Duration::from_secs(60);

struct Temp(PathBuf);

impl Temp {
    fn new(label: &str) -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "ptr-retention-floor-{label}-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&path).unwrap();
        Self(path)
    }
}

impl Drop for Temp {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// Counts how often the effect actually ran. A fresh one after a restore makes
/// any execution there a *second* application of an effect the world has seen.
#[derive(Clone, Default)]
struct Probe(Arc<AtomicUsize>);

impl Probe {
    fn executions(&self) -> usize {
        self.0.load(Ordering::SeqCst)
    }
}

struct Pass;

impl Verifier<ActionIr> for Pass {
    fn verify(&self, _action: &ActionIr) -> VerificationReport {
        VerificationReport {
            status: VerificationStatus::Pass,
            level: VerificationLevel::Deterministic,
            score: Probability::new(1.0).unwrap(),
            findings: Vec::new(),
        }
    }
}

struct Executor(Probe);

impl ActionExecutor for Executor {
    fn execute(&self, _dispatch: VerifiedDispatch<'_>) -> Result<Vec<u8>, String> {
        self.0 .0.fetch_add(1, Ordering::SeqCst);
        Ok(b"executed".to_vec())
    }
}

/// A runtime with one capsule to act on, and the action that writes it.
fn fixture() -> (PtrRuntime, ActionIr) {
    let mut runtime = PtrRuntime::new(PtrConfig::default()).unwrap();
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
    grant_permissions(&mut runtime, &action);
    (runtime, action)
}

/// Permissions are this process's, not committed history's, so a restored
/// runtime is granted them again.
fn grant_permissions(runtime: &mut PtrRuntime, action: &ActionIr) {
    runtime
        .permissions_mut()
        .capabilities
        .insert(action.capability.clone());
    runtime.permissions_mut().allow_mutation = true;
}

fn session(runtime: &mut PtrRuntime, action: &ActionIr, probe: &Probe) -> ExecutionSession {
    let scope = ActionScope {
        project: ProjectId::from("p"),
        target: action.target.clone(),
        operation: action.operation.clone(),
        capability: action.capability.clone(),
        input_type: action.input_type.clone(),
        effect: action.effect,
    };
    runtime
        .register_execution_session(
            "alice",
            vec![ExecutionGrant::new(
                scope,
                RequiredVerification::FullSemantic,
                Pass,
                Executor(probe.clone()),
            )],
            TTL,
        )
        .unwrap()
}

/// Run the effect once under `key` and return what the adapter answered.
fn execute_once(runtime: &mut PtrRuntime, action: &ActionIr, probe: &Probe, key: &str) -> Vec<u8> {
    let opened = session(runtime, action, probe);
    let permit = runtime
        .prepare_execution_once(&opened, &ProjectId::from("p"), action, TTL, key)
        .unwrap();
    runtime.execute_prepared(&opened, permit).unwrap()
}

/// A single-member group that leads, so every proposal commits on the spot.
fn leader(dir: &std::path::Path) -> RaftNode {
    let mut node = RaftNode::open(dir, 1, &[1]).unwrap();
    assert!(
        node.campaign().unwrap().is_empty(),
        "nobody to ask for votes"
    );
    assert!(node.is_leader());
    node
}

/// Order through raft every record the runtime committed that raft has not
/// carried yet, so raft's log and the runtime's ledger are one history.
fn replicate(node: &mut RaftNode, runtime: &PtrRuntime) {
    let carried = node.applied_events() as usize;
    for committed in &runtime.committed_events()[carried..] {
        assert!(node.propose(committed.event.clone()).unwrap().is_empty());
    }
    assert_eq!(
        node.applied_events(),
        runtime.committed_events().len() as u64,
        "raft numbers the runtime's records with the runtime's own indexes"
    );
}

/// Whether `events` hold the attempt made under `key` and its settlement.
///
/// A settlement names its attempt by commit index rather than by key, so the
/// pair is found through the attempt.
fn holds_settled_attempt(events: &[CommittedEvent], key: &str) -> bool {
    events.iter().any(|attempt| {
        matches!(&attempt.event, LedgerEvent::EffectAttempted { key: Some(k), .. } if k == key)
            && events.iter().any(|settled| {
                matches!(
                    &settled.event,
                    LedgerEvent::EffectSettled { attempt: index, .. } if *index == attempt.index
                )
            })
    })
}

#[test]
fn an_at_most_once_key_survives_raft_compacting_to_the_retention_floor_and_a_restart() {
    let temp = Temp::new("at-most-once");
    let (mut runtime, action) = fixture();
    let probe = Probe::default();
    let answered = execute_once(&mut runtime, &action, &probe, "invoice-7");
    assert_eq!(answered, b"executed");
    assert_eq!(probe.executions(), 1, "the effect applied exactly once");

    let mut node = leader(&temp.0);
    replicate(&mut node, &runtime);
    assert!(
        holds_settled_attempt(node.committed_events(), "invoice-7"),
        "the settled attempt is in raft's log before compaction"
    );

    // The host's durable artifact, and the floor it reports.
    let snapshot = runtime.export_compacted_snapshot().unwrap();
    let floor = RetentionFloor::covering(snapshot.covers());
    let covered = node.raft_committed();
    assert!(
        covered > floor.index().0,
        "raft indexes include the leader's empty entry, so the two floors are \
         different numbers for the same position: {covered} against {}",
        floor.index().0
    );
    node.record_snapshot_within(covered, snapshot.bytes().to_vec(), floor)
        .expect("compacting exactly to what the retained snapshot covers is allowed");

    // The group continues above the floor: a second key is spent after it.
    let later = execute_once(&mut runtime, &action, &probe, "invoice-8");
    assert_eq!(later, b"executed");
    replicate(&mut node, &runtime);
    drop(node);

    // Restart. What raft still holds is exactly the history above the floor, with
    // the indexes the runtime gave it — counted from the snapshot's coverage, not
    // from zero, without anyone having to restate it.
    let reopened = RaftNode::open(&temp.0, 1, &[1]).unwrap();
    let above = reopened.committed_events().to_vec();
    assert_eq!(
        above.first().map(|committed| committed.index),
        Some(CommitIndex(floor.index().0 + 1)),
        "nothing below the floor is held, and nothing above it was lost"
    );
    assert_eq!(
        above.as_slice(),
        &runtime.committed_events()[floor.index().0 as usize..],
        "what raft kept is the runtime's history above the floor, record for record"
    );
    assert!(
        !holds_settled_attempt(&above, "invoice-7"),
        "the first key's attempt left raft's log with the compaction"
    );
    assert!(
        holds_settled_attempt(&above, "invoice-8"),
        "the second key's attempt is above the floor and still in raft's log"
    );

    // The state machine restores from the retained snapshot and raft's remainder.
    let mut restored = PtrRuntime::restore_compacted(
        PtrConfig::default(),
        snapshot.bytes(),
        snapshot.anchor(),
        &above,
    )
    .expect("the snapshot and the log above its floor are one history");
    grant_permissions(&mut restored, &action);

    let after = Probe::default();
    for key in ["invoice-7", "invoice-8"] {
        let answer = execute_once(&mut restored, &action, &after, key);
        assert_eq!(
            answer, b"executed",
            "a retry under {key} is answered with the original outcome"
        );
    }
    assert_eq!(
        after.executions(),
        0,
        "neither key is refunded: one is remembered by the snapshot, one by raft's log"
    );

    // The control: a restored runtime that refused everything would pass the
    // assertions above just as well, so an unspent key must still execute.
    execute_once(&mut restored, &action, &after, "invoice-9");
    assert_eq!(after.executions(), 1, "an unspent key still executes");
}

#[test]
fn raft_refuses_to_discard_a_settled_attempt_the_retained_snapshot_predates() {
    let temp = Temp::new("predates");
    let (mut runtime, action) = fixture();
    let probe = Probe::default();

    // The only artifact the host has retained was taken before the effect.
    let stale = runtime.export_compacted_snapshot().unwrap();
    execute_once(&mut runtime, &action, &probe, "invoice-7");

    let mut node = leader(&temp.0);
    replicate(&mut node, &runtime);
    let covered = node.raft_committed();

    let refused = node
        .record_snapshot_within(
            covered,
            stale.bytes().to_vec(),
            RetentionFloor::covering(stale.covers()),
        )
        .expect_err("the attempt is above the only retained floor");
    assert!(
        refused.contains("PTR_RAFT_COMPACTION_ABOVE_RETENTION_FLOOR"),
        "the refusal is named: {refused}"
    );
    drop(node);

    // Nothing moved: after a restart raft still holds the whole history, settled
    // attempt included, so a runtime replaying it remembers the key.
    let reopened = RaftNode::open(&temp.0, 1, &[1]).unwrap();
    assert_eq!(reopened.committed_events(), runtime.committed_events());
    assert!(holds_settled_attempt(
        reopened.committed_events(),
        "invoice-7"
    ));
}
