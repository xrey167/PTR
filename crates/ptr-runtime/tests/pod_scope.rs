//! Pod scopes as committed history (the Pod-access follow-up to D5 in issue
//! #23).
//!
//! The property under test is the one D5 established for execution membership,
//! now held for Pod access: what a peer may reach is what the ledger says, so a
//! scope withdrawn before a restart is still withdrawn after it, a history
//! replays in order to its last record, a record the runtime would refuse to
//! write is refused when it is read back, and compaction carries the result
//! across its floor. Every refusal is paired with a control that the same
//! runtime accepts, so the tests show a rule rather than a runtime refusing
//! everything.

use ptr_config::PtrConfig;
use ptr_ledger::{CommittedEvent, LedgerEvent, MAX_POD_SCOPE_ENTRIES, MAX_POD_SCOPE_FIELD_BYTES};
use ptr_runtime::{PodScope, PtrRuntime, RuntimeError};
use ptr_types::{CapabilityId, CommitIndex, NodeId, ProjectId, TypeId};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

fn peer() -> NodeId {
    NodeId::from("k51qzi5uqu5dh-pod-peer")
}

fn other() -> NodeId {
    NodeId::from("k51qzi5uqu5dh-other-pod-peer")
}

fn alpha() -> PodScope {
    PodScope::new(ProjectId::from("alpha"))
        .allow(CapabilityId::from("summarize"), TypeId::from("Text"))
        .allow(CapabilityId::from("classify"), TypeId::from("Text"))
}

fn beta() -> PodScope {
    PodScope::new(ProjectId::from("beta")).allow(CapabilityId::from("embed"), TypeId::from("Text"))
}

struct Temp(PathBuf);

impl Temp {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "ptr-pod-scope-{}-{}",
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

fn open(temp: &Temp) -> PtrRuntime {
    PtrRuntime::open_durable(PtrConfig::default(), temp.log()).unwrap()
}

/// `history` with `event` appended at the next index, as a writer below the
/// runtime could have put it in a log.
fn with_record(history: &[CommittedEvent], event: LedgerEvent) -> Vec<CommittedEvent> {
    let mut events = history.to_vec();
    events.push(CommittedEvent {
        index: CommitIndex(events.len() as u64 + 1),
        event,
    });
    events
}

fn admitted(peer: NodeId, scope: &PodScope) -> LedgerEvent {
    LedgerEvent::PodScopeAdmitted {
        peer,
        project: scope.project().clone(),
        entries: scope.entries().cloned().collect(),
    }
}

/// The test #23 asks of any journaled policy: a recovery that replayed only
/// admissions would pass a naive positive test and fail this one.
#[test]
fn a_pod_scope_withdrawn_before_a_restart_is_still_withdrawn_after_it() {
    let temp = Temp::new();
    {
        let mut runtime = open(&temp);
        runtime.admit_pod_scope(peer(), &alpha()).unwrap();
        runtime.admit_pod_scope(other(), &beta()).unwrap();
        // Execution membership for the same peer, to show the two families are
        // withdrawn independently.
        runtime
            .commit(LedgerEvent::PeerAdmitted {
                peer: peer(),
                principal: "alice".into(),
                grant_set: "writer".into(),
            })
            .unwrap();
    }
    {
        let mut runtime = open(&temp);
        assert_eq!(
            runtime.pod_scope(&peer()),
            Some(&alpha()),
            "admission replayed"
        );
        runtime.withdraw_pod_scope(&peer()).unwrap();
    }

    let runtime = open(&temp);
    assert_eq!(
        runtime.pod_scope(&peer()),
        None,
        "the withdrawal is history"
    );
    // The controls: the scope that was not withdrawn is still in force, exactly
    // as it was admitted, and withdrawing a Pod scope left the same peer's
    // execution membership alone.
    assert_eq!(runtime.pod_scope(&other()), Some(&beta()));
    assert!(runtime.is_member(&peer()));
}

#[test]
fn admit_withdraw_admit_replays_to_the_last_pod_scope() {
    let temp = Temp::new();
    {
        let mut runtime = open(&temp);
        runtime.admit_pod_scope(peer(), &alpha()).unwrap();
        runtime.withdraw_pod_scope(&peer()).unwrap();
        runtime.admit_pod_scope(peer(), &beta()).unwrap();
    }

    let runtime = open(&temp);
    assert_eq!(
        runtime.pod_scope(&peer()),
        Some(&beta()),
        "replayed in order, the second admission is the one in force"
    );
    // And the same history replays to the same answer from memory.
    let replayed = PtrRuntime::replay(PtrConfig::default(), runtime.committed_events()).unwrap();
    assert_eq!(replayed.pod_scope(&peer()), Some(&beta()));
}

/// A second admission would silently replace the first, widening or narrowing
/// what the peer reaches without a withdrawal anyone can see. Refused live,
/// with nothing appended, and refused on replay of a history that holds one.
#[test]
fn a_duplicate_pod_scope_admission_is_refused_live_and_on_replay() {
    let mut runtime = PtrRuntime::new(PtrConfig::default()).unwrap();
    runtime.admit_pod_scope(peer(), &alpha()).unwrap();
    let history = runtime.committed_events().to_vec();

    assert_eq!(
        runtime.admit_pod_scope(peer(), &beta()),
        Err(RuntimeError::InvalidPodScope {
            peer: peer(),
            reason: "the peer already has a Pod scope",
        })
    );
    assert_eq!(runtime.committed_events(), history.as_slice());
    assert_eq!(
        runtime.pod_scope(&peer()),
        Some(&alpha()),
        "the first stays"
    );

    let doubled = with_record(&history, admitted(peer(), &beta()));
    assert_eq!(
        PtrRuntime::replay(PtrConfig::default(), &doubled).err(),
        Some(RuntimeError::InvalidPodScope {
            peer: peer(),
            reason: "the peer already has a Pod scope",
        })
    );
    // The control: the same history with one admission replays.
    let replayed = PtrRuntime::replay(PtrConfig::default(), &history).unwrap();
    assert_eq!(replayed.pod_scope(&peer()), Some(&alpha()));
}

#[test]
fn a_withdrawal_of_a_peer_without_a_pod_scope_is_refused_live_and_on_replay() {
    let mut runtime = PtrRuntime::new(PtrConfig::default()).unwrap();
    let refusal = RuntimeError::InvalidPodScope {
        peer: peer(),
        reason: "the peer has no Pod scope",
    };
    assert_eq!(runtime.withdraw_pod_scope(&peer()), Err(refusal.clone()));
    assert!(runtime.committed_events().is_empty());

    let forged = with_record(&[], LedgerEvent::PodScopeWithdrawn { peer: peer() });
    assert_eq!(
        PtrRuntime::replay(PtrConfig::default(), &forged).err(),
        Some(refusal)
    );

    // The control: a withdrawal after an admission is accepted.
    runtime.admit_pod_scope(peer(), &alpha()).unwrap();
    runtime.withdraw_pod_scope(&peer()).unwrap();
    assert_eq!(runtime.pod_scope(&peer()), None);
}

/// A scope the codec would not encode is one the runtime will not commit and
/// will not replay, by the codec's own rule and with its code. The replayed
/// cases are records a writer below the runtime could have appended to an
/// in-memory ledger, which encodes nothing and so checks nothing.
#[test]
fn a_malformed_or_noncanonical_pod_scope_is_refused_live_and_on_replay() {
    let mut runtime = PtrRuntime::new(PtrConfig::default()).unwrap();
    let refused = |peer: NodeId, reason: &'static str| {
        Err::<CommitIndex, _>(RuntimeError::InvalidPodScope { peer, reason })
    };

    let mut too_many = PodScope::new(ProjectId::from("alpha"));
    for index in 0..=MAX_POD_SCOPE_ENTRIES {
        too_many = too_many.allow(
            CapabilityId::from(format!("capability-{index:02}").as_str()),
            TypeId::from("Text"),
        );
    }
    let long = "a".repeat(MAX_POD_SCOPE_FIELD_BYTES + 1);
    let live = [
        (peer(), too_many, "PTR_LEDGER_POD_SCOPE_LIMIT"),
        (
            peer(),
            PodScope::new(ProjectId::from("alpha"))
                .allow(CapabilityId::from(""), TypeId::from("Text")),
            "PTR_LEDGER_POD_SCOPE_FIELD",
        ),
        (
            peer(),
            PodScope::new(ProjectId::from("alpha"))
                .allow(CapabilityId::from("sum\tmarize"), TypeId::from("Text")),
            "PTR_LEDGER_POD_SCOPE_FIELD",
        ),
        (
            peer(),
            PodScope::new(ProjectId::from(" alpha")),
            "PTR_LEDGER_POD_SCOPE_FIELD",
        ),
        (
            peer(),
            PodScope::new(ProjectId::from(long.as_str())),
            "PTR_LEDGER_POD_SCOPE_FIELD",
        ),
        (
            NodeId::from("k-peer\n"),
            alpha(),
            "PTR_LEDGER_POD_SCOPE_FIELD",
        ),
    ];
    for (who, scope, code) in live {
        assert_eq!(
            runtime.admit_pod_scope(who.clone(), &scope),
            refused(who.clone(), code),
            "{scope:?}"
        );
        assert_eq!(runtime.pod_scope(&who), None);
    }
    assert!(
        runtime.committed_events().is_empty(),
        "nothing was appended"
    );

    // Records the builder cannot produce, because a set is always ascending
    // and free of duplicates: written by hand, each refused on replay.
    let summarize = (CapabilityId::from("summarize"), TypeId::from("Text"));
    let classify = (CapabilityId::from("classify"), TypeId::from("Text"));
    for entries in [
        vec![summarize.clone(), classify.clone()],
        vec![classify.clone(), classify.clone()],
    ] {
        let forged = with_record(
            &[],
            LedgerEvent::PodScopeAdmitted {
                peer: peer(),
                project: ProjectId::from("alpha"),
                entries,
            },
        );
        assert_eq!(
            PtrRuntime::replay(PtrConfig::default(), &forged).err(),
            Some(RuntimeError::InvalidPodScope {
                peer: peer(),
                reason: "PTR_LEDGER_POD_SCOPE_ORDER",
            })
        );
    }

    // The control: the canonical record, with the same pairs ascending, replays,
    // and the builder's scope commits. An empty scope is a scope too.
    let canonical = with_record(
        &[],
        LedgerEvent::PodScopeAdmitted {
            peer: peer(),
            project: ProjectId::from("alpha"),
            entries: vec![classify, summarize],
        },
    );
    let replayed = PtrRuntime::replay(PtrConfig::default(), &canonical).unwrap();
    assert_eq!(replayed.pod_scope(&peer()), Some(&alpha()));
    runtime.admit_pod_scope(peer(), &alpha()).unwrap();
    let empty = PodScope::new(ProjectId::from("alpha"));
    runtime.admit_pod_scope(other(), &empty).unwrap();
    assert_eq!(runtime.pod_scope(&other()), Some(&empty));
}

/// Compaction removes the records a scope was committed in; the projection
/// carries the scope across the floor, and a withdrawal crosses it too.
#[test]
fn pod_scopes_survive_compaction_withdrawals_included() {
    let mut runtime = PtrRuntime::new(PtrConfig::default()).unwrap();
    runtime.admit_pod_scope(peer(), &alpha()).unwrap();
    runtime.admit_pod_scope(other(), &beta()).unwrap();
    runtime.withdraw_pod_scope(&peer()).unwrap();

    let snapshot = runtime.export_compacted_snapshot().unwrap();
    assert_eq!(&snapshot.bytes()[..8], b"PTRCS005");
    let mut restored = PtrRuntime::restore_compacted(
        PtrConfig::default(),
        snapshot.bytes(),
        snapshot.anchor(),
        &[],
    )
    .unwrap();

    assert_eq!(restored.pod_scope(&peer()), None, "the withdrawal crossed");
    // The control: the scope that was not withdrawn crossed too, whole.
    assert_eq!(restored.pod_scope(&other()), Some(&beta()));

    // It is history there rather than only a projected row: it can be
    // withdrawn, a second admission of it is refused, and the withdrawn peer
    // can be admitted again.
    assert!(matches!(
        restored.admit_pod_scope(other(), &alpha()),
        Err(RuntimeError::InvalidPodScope { .. })
    ));
    restored.withdraw_pod_scope(&other()).unwrap();
    assert_eq!(restored.pod_scope(&other()), None);
    restored.admit_pod_scope(peer(), &beta()).unwrap();
    assert_eq!(restored.pod_scope(&peer()), Some(&beta()));
}
