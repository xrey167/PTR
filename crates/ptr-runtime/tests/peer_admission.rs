//! Admission of transport-authenticated peers against host policy.
//!
//! The property under test is that a peer never names its own principal and
//! never chooses its own grants, and that withdrawing it takes effect at once
//! rather than when a TTL happens to run out.
//!
//! Since D5 in issue #23, membership is committed history and authority is the
//! host's grant sets: the restart tests at the end hold both halves — a peer
//! withdrawn before a restart is still refused after it, and an admitted one is
//! still admitted but reaches nothing until the host supplies its grant set.

mod common;

use common::*;
use ptr_config::PtrConfig;
use ptr_core::action_head::ActionIr;
use ptr_ledger::LedgerEvent;
use ptr_runtime::{execution::*, PtrRuntime, RuntimeError};
use ptr_types::{
    CapabilityId, CapsuleId, Effect, Generation, NodeId, ProjectId, RequestId, TypeId,
};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

/// Tests have no network, so they assume what a transport would have proved.
fn authenticated(id: NodeId) -> ptr_runtime::execution::AuthenticatedPeer {
    ptr_runtime::execution::AuthenticatedPeer::assume_authenticated(id)
}

fn peer() -> NodeId {
    NodeId::from("k51qzi5uqu5dh-authenticated-peer")
}

/// The grant set named "writer": exactly the grant for `action`.
fn grant_sets_for(action: &ActionIr, probe: &Probe) -> GrantSets {
    let mut sets = GrantSets::new();
    let scope = scope(action);
    let probe = probe.clone();
    sets.define("writer", TTL, move || {
        vec![grant(
            scope.clone(),
            &probe,
            RequiredVerification::FullSemantic,
            ExecutorMode::Success,
        )]
    })
    .unwrap();
    sets
}

/// "writer" again, with a grant that does **not** cover `action`.
///
/// Narrowing is the case that matters: a replacement which still defines the set
/// but grants less. One identical to the old cannot show whether grants were
/// re-derived, because both derivations produce the same answer.
fn narrower_grant_sets_for(action: &ActionIr, probe: &Probe) -> GrantSets {
    let mut sets = GrantSets::new();
    let mut narrowed = scope(action);
    narrowed.operation = "something-else".into();
    let probe = probe.clone();
    sets.define("writer", TTL, move || {
        vec![grant(
            narrowed.clone(),
            &probe,
            RequiredVerification::FullSemantic,
            ExecutorMode::Success,
        )]
    })
    .unwrap();
    sets
}

/// The fixture with `peer()` admitted as "alice" under "writer".
fn admitted_fixture(probe: &Probe) -> (PtrRuntime, ActionIr) {
    let (mut runtime, action) = fixture();
    runtime.install_grant_sets(grant_sets_for(&action, probe));
    runtime.admit_member(peer(), "alice", "writer").unwrap();
    (runtime, action)
}

fn principal_of(runtime: &PtrRuntime) -> Option<String> {
    runtime
        .committed_events()
        .iter()
        .find_map(|committed| match &committed.event {
            LedgerEvent::EffectAttempted { principal, .. } => Some(principal.clone()),
            _ => None,
        })
}

#[test]
fn an_admitted_peer_gets_exactly_the_principal_and_grants_the_policy_names() {
    let probe = Probe::default();
    let (mut runtime, action) = admitted_fixture(&probe);

    let session = runtime.admit_peer(&authenticated(peer())).unwrap();
    let permit = runtime
        .prepare_execution(&session, &ProjectId::from("p"), &action, TTL)
        .unwrap();
    assert_eq!(
        runtime.execute_prepared(&session, permit).unwrap(),
        b"executed"
    );

    // The principal in the audit record is the membership's. There is no parameter
    // anywhere on this path through which a caller could have supplied one.
    assert_eq!(principal_of(&runtime).as_deref(), Some("alice"));
    assert_eq!(probe.executions(), 1);
}

#[test]
fn an_admitted_peer_reaches_nothing_beyond_its_grant_set() {
    let probe = Probe::default();
    let (mut runtime, action) = admitted_fixture(&probe);
    let session = runtime.admit_peer(&authenticated(peer())).unwrap();

    // Same capsule, a different operation: outside the one grant the policy
    // installed, so there is nothing to match.
    let mut other = action.clone();
    other.operation = "delete".into();
    assert!(matches!(
        runtime.prepare_execution(&session, &ProjectId::from("p"), &other, TTL),
        Err(ExecutionError::ScopeDenied)
    ));

    // A different project is refused for the same reason, before any check that
    // could depend on what the caller claims.
    assert!(matches!(
        runtime.prepare_execution(&session, &ProjectId::from("other"), &action, TTL),
        Err(ExecutionError::ScopeDenied)
    ));
    assert_eq!(probe.executions(), 0);
}

#[test]
fn a_peer_the_host_never_admitted_is_not_admitted_at_all() {
    let probe = Probe::default();
    let (mut runtime, _) = admitted_fixture(&probe);

    let stranger = NodeId::from("k51qzi5uqu5dh-stranger");
    assert_eq!(
        runtime.admit_peer(&authenticated(stranger.clone())).err(),
        Some(ExecutionError::UnknownPeer {
            peer: stranger.clone()
        })
    );
    assert_eq!(
        runtime.withdraw_peer(&stranger).err(),
        Some(ExecutionError::UnknownPeer { peer: stranger })
    );
    assert_eq!(probe.executions(), 0);
}

#[test]
fn withdrawing_a_peer_stops_its_live_session_without_waiting_for_a_ttl() {
    let probe = Probe::default();
    let (mut runtime, action) = admitted_fixture(&probe);
    let session = runtime.admit_peer(&authenticated(peer())).unwrap();

    // A permit issued while the peer was still admitted.
    let permit = runtime
        .prepare_execution(&session, &ProjectId::from("p"), &action, TTL)
        .unwrap();

    runtime.withdraw_peer(&peer()).unwrap();

    // The session's TTL has not moved. Its authority has.
    assert_eq!(
        runtime.execute_prepared(&session, permit).err(),
        Some(ExecutionError::PeerNotAdmitted { peer: peer() })
    );
    assert!(matches!(
        runtime.prepare_execution(&session, &ProjectId::from("p"), &action, TTL),
        Err(ExecutionError::PeerNotAdmitted { .. })
    ));
    assert_eq!(
        runtime.admit_peer(&authenticated(peer())).err(),
        Some(ExecutionError::UnknownPeer { peer: peer() })
    );
    assert_eq!(probe.executions(), 0);
}

#[test]
fn replacing_the_grant_sets_suspends_every_session_whose_set_they_no_longer_define() {
    let probe = Probe::default();
    let (mut runtime, action) = admitted_fixture(&probe);
    let session = runtime.admit_peer(&authenticated(peer())).unwrap();
    assert!(runtime
        .prepare_execution(&session, &ProjectId::from("p"), &action, TTL)
        .is_ok());

    // No grant sets: the peer is still a member — that is history — but its
    // membership names a set the host no longer supplies, so it reaches nothing.
    runtime.install_grant_sets(GrantSets::new());
    assert_eq!(
        runtime
            .prepare_execution(&session, &ProjectId::from("p"), &action, TTL)
            .err(),
        Some(ExecutionError::UnknownGrantSet {
            name: "writer".into()
        })
    );
    assert!(runtime.is_member(&peer()));

    // The control: the host supplies the set again and the same session resolves,
    // so the refusal above was the missing authority and not a dropped session.
    runtime.install_grant_sets(grant_sets_for(&action, &probe));
    assert!(runtime
        .prepare_execution(&session, &ProjectId::from("p"), &action, TTL)
        .is_ok());
    assert_eq!(probe.executions(), 0);
}

#[test]
fn a_re_admission_issues_new_sessions_and_revives_none() {
    let probe = Probe::default();
    let (mut runtime, action) = admitted_fixture(&probe);
    let old = runtime.admit_peer(&authenticated(peer())).unwrap();
    runtime.withdraw_peer(&peer()).unwrap();
    runtime.admit_member(peer(), "alice", "writer").unwrap();

    // The old session was bound to the admission that was withdrawn. The new
    // admission is a different record, and it does not bring the old one back.
    assert_eq!(
        runtime
            .prepare_execution(&old, &ProjectId::from("p"), &action, TTL)
            .err(),
        Some(ExecutionError::PeerNotAdmitted { peer: peer() })
    );
    // The control: a session issued under the new admission works.
    let new = runtime.admit_peer(&authenticated(peer())).unwrap();
    assert!(runtime
        .prepare_execution(&new, &ProjectId::from("p"), &action, TTL)
        .is_ok());
}

#[test]
fn a_host_registered_session_is_not_governed_by_membership() {
    let (mut runtime, action) = fixture();
    let probe = Probe::default();
    let session = common::session(&mut runtime, &action, &probe, "operator");

    // Nobody is a member and no grant set is defined, which says nothing about a
    // session the trusted host installed directly.
    runtime.install_grant_sets(GrantSets::new());
    let permit = runtime
        .prepare_execution(&session, &ProjectId::from("p"), &action, TTL)
        .unwrap();
    assert!(runtime.execute_prepared(&session, permit).is_ok());
    assert_eq!(principal_of(&runtime).as_deref(), Some("operator"));
}

#[test]
fn a_peer_cannot_hold_two_memberships_with_different_authority() {
    let probe = Probe::default();
    let (mut runtime, _) = admitted_fixture(&probe);
    let history = runtime.committed_events().len();

    // A second admission would not narrow the first, it would replace it, and the
    // replacement is exactly what nobody decided.
    assert_eq!(
        runtime.admit_member(peer(), "mallory", "writer").err(),
        Some(ExecutionError::DuplicatePeerEntry { peer: peer() })
    );
    assert_eq!(
        runtime.committed_events().len(),
        history,
        "nothing was written"
    );

    // The same record through the host's general commit path is refused by the
    // same rule, so the convenience method is not the only thing standing guard.
    assert!(matches!(
        runtime.commit(LedgerEvent::PeerAdmitted {
            peer: peer(),
            principal: "mallory".into(),
            grant_set: "writer".into(),
        }),
        Err(RuntimeError::InvalidMembership { .. })
    ));
    assert!(runtime.is_member(&peer()));
}

#[test]
fn a_malformed_peer_principal_or_grant_set_is_refused() {
    let probe = Probe::default();
    let (mut runtime, _) = admitted_fixture(&probe);
    let long = "x".repeat(MAX_MEMBER_FIELD_BYTES + 1);
    for (node, principal, set) in [
        (NodeId::from(""), "alice", "writer"),
        (NodeId::from("peer\nid"), "alice", "writer"),
        (NodeId::from("peer"), "", "writer"),
        (NodeId::from("peer"), "alice\n", "writer"),
        (NodeId::from("peer"), "alice", " writer"),
        (NodeId::from(long.as_str()), "alice", "writer"),
    ] {
        assert_eq!(
            runtime.admit_member(node, principal, set).err(),
            Some(ExecutionError::InvalidSession)
        );
    }
    // The control: the bound itself is allowed.
    let longest = "x".repeat(MAX_MEMBER_FIELD_BYTES);
    runtime
        .admit_member(NodeId::from(longest.as_str()), "alice", "writer")
        .unwrap();
}

#[test]
fn an_admission_must_name_a_grant_set_the_host_defines() {
    let probe = Probe::default();
    let (mut runtime, _) = admitted_fixture(&probe);
    let other = NodeId::from("k51qzi5uqu5dh-other");
    assert_eq!(
        runtime.admit_member(other.clone(), "bob", "writre").err(),
        Some(ExecutionError::UnknownGrantSet {
            name: "writre".into()
        })
    );
    assert!(!runtime.is_member(&other));
    let mut sets = GrantSets::new();
    sets.define("writer", TTL, Vec::new).unwrap();
    assert_eq!(
        sets.define("writer", TTL, Vec::new).err(),
        Some(ExecutionError::DuplicateGrantSet {
            name: "writer".into()
        })
    );
}

/// Narrowed grant sets must not leave a live session running under the old grants.
///
/// `session()` re-derives whether the admission *still stands*; it does not
/// re-derive what the peer may do, because `SessionRecord` holds the grants
/// admission returned. So a replacement that keeps the set's name but narrows it
/// would leave the wider set in force until the TTL expired — which is the
/// opposite of this contract's own rule that authority is re-derived at use.
#[test]
fn replacing_the_grant_sets_re_derives_the_grants_a_live_session_runs_under() {
    let probe = Probe::default();
    let (mut runtime, action) = admitted_fixture(&probe);
    let session = runtime.admit_peer(&authenticated(peer())).unwrap();

    // Issued under the grants the first policy gave.
    let permit = runtime
        .prepare_execution(&session, &ProjectId::from("p"), &action, TTL)
        .unwrap();

    // A replacement that still admits the peer but grants it less. The live
    // session must follow the narrowing rather than keep the wider set.
    runtime.install_grant_sets(narrower_grant_sets_for(&action, &probe));

    assert_eq!(
        runtime.execute_prepared(&session, permit),
        Err(ExecutionError::StalePermit),
        "a permit scoped by the grants that have just been replaced is stale"
    );
    assert!(
        matches!(
            runtime.prepare_execution(&session, &ProjectId::from("p"), &action, TTL),
            Err(ExecutionError::ScopeDenied)
        ),
        "the session must run under the narrowed grants, not the ones it was \
         admitted with"
    );
    assert_eq!(probe.executions(), 0, "nothing ran under the old authority");

    // The control: the session is not merely dead. Restore the wider policy and
    // the same handle works again — so the refusal above is the narrowing taking
    // effect rather than the session having been dropped.
    runtime.install_grant_sets(grant_sets_for(&action, &probe));
    let permit = runtime
        .prepare_execution(&session, &ProjectId::from("p"), &action, TTL)
        .expect("the peer is still admitted, so its session still resolves");
    runtime.execute_prepared(&session, permit).unwrap();
    assert_eq!(
        probe.executions(),
        1,
        "under the grants the current policy gives"
    );
}

// ---------------------------------------------------------------------------
// D5: membership is committed history; authority is the host's at every start.
// ---------------------------------------------------------------------------

struct Temp(PathBuf);

impl Temp {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "ptr-peer-admission-{}-{}",
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

fn other() -> NodeId {
    NodeId::from("k51qzi5uqu5dh-other-peer")
}

/// The fixture's ground state on a durable ledger, with `peer()` and `other()`
/// both admitted under "writer".
fn durable_admitted(path: &PathBuf, probe: &Probe) -> ActionIr {
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
    runtime.install_grant_sets(grant_sets_for(&action, probe));
    runtime.admit_member(peer(), "alice", "writer").unwrap();
    runtime.admit_member(other(), "bob", "writer").unwrap();
    action
}

/// Reopen with permissions and grant sets supplied again: both belong to the
/// process, not to the log.
fn reopen(path: &PathBuf, action: &ActionIr, probe: &Probe) -> PtrRuntime {
    let mut runtime = PtrRuntime::open_durable(PtrConfig::default(), path).unwrap();
    runtime
        .permissions_mut()
        .capabilities
        .insert(action.capability.clone());
    runtime.permissions_mut().allow_mutation = true;
    runtime.install_grant_sets(grant_sets_for(action, probe));
    runtime
}

/// The one test #23 says to keep whichever design wins: a recovery that replayed
/// only admissions would pass a naive positive test and fail this one.
#[test]
fn a_peer_withdrawn_before_a_restart_is_still_refused_after_it() {
    let temp = Temp::new();
    let probe = Probe::default();
    let action = durable_admitted(&temp.log(), &probe);
    {
        let mut runtime = reopen(&temp.log(), &action, &probe);
        runtime.withdraw_peer(&peer()).unwrap();
    }

    let mut runtime = reopen(&temp.log(), &action, &probe);
    assert_eq!(
        runtime.admit_peer(&authenticated(peer())).err(),
        Some(ExecutionError::UnknownPeer { peer: peer() }),
        "the withdrawal is history and replays"
    );
    assert!(!runtime.is_member(&peer()));

    // The control: the peer that was not withdrawn is still admitted, under the
    // principal its admission recorded, and executes.
    let session = runtime.admit_peer(&authenticated(other())).unwrap();
    let permit = runtime
        .prepare_execution(&session, &ProjectId::from("p"), &action, TTL)
        .unwrap();
    runtime.execute_prepared(&session, permit).unwrap();
    assert_eq!(principal_of(&runtime).as_deref(), Some("bob"));
    assert_eq!(probe.executions(), 1);
}

#[test]
fn admit_withdraw_admit_replays_to_the_last_admission() {
    let temp = Temp::new();
    let probe = Probe::default();
    let action = durable_admitted(&temp.log(), &probe);
    {
        let mut runtime = reopen(&temp.log(), &action, &probe);
        runtime.withdraw_peer(&peer()).unwrap();
        runtime.admit_member(peer(), "carol", "writer").unwrap();
    }

    let mut runtime = reopen(&temp.log(), &action, &probe);
    let session = runtime.admit_peer(&authenticated(peer())).unwrap();
    let permit = runtime
        .prepare_execution(&session, &ProjectId::from("p"), &action, TTL)
        .unwrap();
    runtime.execute_prepared(&session, permit).unwrap();
    assert_eq!(
        principal_of(&runtime).as_deref(),
        Some("carol"),
        "replayed in order, the second admission is the one in force"
    );
}

#[test]
fn after_a_restart_a_member_reaches_nothing_until_the_host_supplies_its_grant_set() {
    let temp = Temp::new();
    let probe = Probe::default();
    let action = durable_admitted(&temp.log(), &probe);

    let mut runtime = PtrRuntime::open_durable(PtrConfig::default(), temp.log()).unwrap();
    assert!(runtime.is_member(&peer()), "membership replayed");
    assert_eq!(
        runtime.admit_peer(&authenticated(peer())).err(),
        Some(ExecutionError::UnknownGrantSet {
            name: "writer".into()
        }),
        "authority is not history: nothing is granted by default"
    );

    // The control: the host supplies the set, and the same membership admits.
    runtime.install_grant_sets(grant_sets_for(&action, &probe));
    assert!(runtime.admit_peer(&authenticated(peer())).is_ok());
}

#[test]
fn a_history_that_admits_a_peer_twice_is_refused_on_replay() {
    let (runtime, _) = fixture();
    let mut events = runtime.committed_events().to_vec();
    for principal in ["alice", "mallory"] {
        let index = ptr_types::CommitIndex(events.len() as u64 + 1);
        events.push(ptr_ledger::CommittedEvent {
            index,
            event: LedgerEvent::PeerAdmitted {
                peer: peer(),
                principal: principal.into(),
                grant_set: "writer".into(),
            },
        });
    }
    assert!(matches!(
        PtrRuntime::replay(PtrConfig::default(), &events),
        Err(RuntimeError::InvalidMembership { .. })
    ));
    // The control: the same history with one admission replays.
    events.pop();
    let replayed = PtrRuntime::replay(PtrConfig::default(), &events).unwrap();
    assert!(replayed.is_member(&peer()));
}

#[test]
fn a_withdrawal_of_a_peer_that_is_not_a_member_is_refused() {
    let (mut runtime, _) = fixture();
    let history = runtime.committed_events().len();
    assert_eq!(
        runtime.withdraw_peer(&peer()).err(),
        Some(ExecutionError::UnknownPeer { peer: peer() })
    );
    assert!(matches!(
        runtime.commit(LedgerEvent::PeerWithdrawn { peer: peer() }),
        Err(RuntimeError::InvalidMembership { .. })
    ));
    assert_eq!(runtime.committed_events().len(), history);
}

#[test]
fn membership_survives_compaction_withdrawals_included() {
    let probe = Probe::default();
    let (mut runtime, action) = admitted_fixture(&probe);
    runtime.admit_member(other(), "bob", "writer").unwrap();
    runtime.withdraw_peer(&peer()).unwrap();

    let snapshot = runtime.export_compacted_snapshot().unwrap();
    assert_eq!(&snapshot.bytes()[..8], b"PTRCS005");
    let mut restored = restore_fixture(snapshot.bytes(), snapshot.anchor(), &[], &action);
    restored.install_grant_sets(grant_sets_for(&action, &probe));

    assert_eq!(
        restored.admit_peer(&authenticated(peer())).err(),
        Some(ExecutionError::UnknownPeer { peer: peer() }),
        "the withdrawal crossed the floor"
    );
    // The control: the member that was not withdrawn crossed it too, under its
    // own principal.
    let session = restored.admit_peer(&authenticated(other())).unwrap();
    let permit = restored
        .prepare_execution(&session, &ProjectId::from("p"), &action, TTL)
        .unwrap();
    restored.execute_prepared(&session, permit).unwrap();
    assert_eq!(principal_of(&restored).as_deref(), Some("bob"));

    // And it can be withdrawn after the restore, which needs it to be a member
    // there rather than only a projected row.
    restored.withdraw_peer(&other()).unwrap();
    assert!(!restored.is_member(&other()));
}
