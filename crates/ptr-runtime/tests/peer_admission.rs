//! Admission of transport-authenticated peers against host policy.
//!
//! The property under test is that a peer never names its own principal and
//! never chooses its own grants, and that withdrawing it takes effect at once
//! rather than when a TTL happens to run out.

mod common;

use common::*;
use ptr_core::action_head::ActionIr;
use ptr_ledger::LedgerEvent;
use ptr_runtime::{execution::*, PtrRuntime};
use ptr_types::{CapabilityId, Effect, NodeId, ProjectId, TypeId};

fn peer() -> NodeId {
    NodeId::from("k51qzi5uqu5dh-authenticated-peer")
}

/// A policy that admits `peer()` as "alice" with exactly the grant for `action`.
fn policy_for(action: &ActionIr, probe: &Probe) -> AdmissionPolicy {
    let mut policy = AdmissionPolicy::new();
    let scope = scope(action);
    let probe = probe.clone();
    policy
        .admit(peer(), "alice", TTL, move || {
            vec![grant(
                scope.clone(),
                &probe,
                RequiredVerification::FullSemantic,
                ExecutorMode::Success,
            )]
        })
        .unwrap();
    policy
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
    let (mut runtime, action) = fixture();
    let probe = Probe::default();
    runtime.install_admission_policy(policy_for(&action, &probe));

    let session = runtime.admit_peer(&peer()).unwrap();
    let permit = runtime
        .prepare_execution(&session, &ProjectId::from("p"), &action, TTL)
        .unwrap();
    assert_eq!(
        runtime.execute_prepared(&session, permit).unwrap(),
        b"executed"
    );

    // The principal in the audit record is the policy's. There is no parameter
    // anywhere on this path through which a caller could have supplied one.
    assert_eq!(principal_of(&runtime).as_deref(), Some("alice"));
    assert_eq!(probe.executions(), 1);
}

#[test]
fn an_admitted_peer_reaches_nothing_beyond_its_grant_set() {
    let (mut runtime, action) = fixture();
    let probe = Probe::default();
    runtime.install_admission_policy(policy_for(&action, &probe));
    let session = runtime.admit_peer(&peer()).unwrap();

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
fn a_peer_the_policy_never_bound_is_not_admitted_at_all() {
    let (mut runtime, action) = fixture();
    let probe = Probe::default();
    runtime.install_admission_policy(policy_for(&action, &probe));

    let stranger = NodeId::from("k51qzi5uqu5dh-stranger");
    assert_eq!(
        runtime.admit_peer(&stranger).err(),
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
    let (mut runtime, action) = fixture();
    let probe = Probe::default();
    runtime.install_admission_policy(policy_for(&action, &probe));
    let session = runtime.admit_peer(&peer()).unwrap();

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
        runtime.admit_peer(&peer()).err(),
        Some(ExecutionError::UnknownPeer { peer: peer() })
    );
    assert_eq!(probe.executions(), 0);
}

#[test]
fn replacing_the_policy_withdraws_every_session_it_no_longer_admits() {
    let (mut runtime, action) = fixture();
    let probe = Probe::default();
    runtime.install_admission_policy(policy_for(&action, &probe));
    let session = runtime.admit_peer(&peer()).unwrap();
    assert!(runtime
        .prepare_execution(&session, &ProjectId::from("p"), &action, TTL)
        .is_ok());

    // An empty table admits nobody, and the session was never a separate fact
    // that could outlive it.
    runtime.install_admission_policy(AdmissionPolicy::new());
    assert!(matches!(
        runtime.prepare_execution(&session, &ProjectId::from("p"), &action, TTL),
        Err(ExecutionError::PeerNotAdmitted { .. })
    ));

    // Re-admitting the same peer revives nothing: the old session stays dead and
    // a new admission issues a new one.
    runtime.install_admission_policy(policy_for(&action, &probe));
    assert!(runtime
        .prepare_execution(&session, &ProjectId::from("p"), &action, TTL)
        .is_ok());
    assert_eq!(probe.executions(), 0);
}

#[test]
fn a_host_registered_session_is_not_governed_by_the_admission_policy() {
    let (mut runtime, action) = fixture();
    let probe = Probe::default();
    let session = common::session(&mut runtime, &action, &probe, "operator");

    // The policy admits nobody, which says nothing about a session the trusted
    // host installed directly.
    runtime.install_admission_policy(AdmissionPolicy::new());
    let permit = runtime
        .prepare_execution(&session, &ProjectId::from("p"), &action, TTL)
        .unwrap();
    assert!(runtime.execute_prepared(&session, permit).is_ok());
    assert_eq!(principal_of(&runtime).as_deref(), Some("operator"));
}

#[test]
fn a_peer_cannot_hold_two_entries_with_different_authority() {
    let (_, action) = fixture();
    let probe = Probe::default();
    let mut policy = policy_for(&action, &probe);

    // A second entry would not narrow the first, it would replace it, and the
    // replacement is exactly what nobody decided.
    let scope = scope(&action);
    let second = probe.clone();
    assert_eq!(
        policy
            .admit(peer(), "mallory", TTL, move || {
                vec![grant(
                    scope.clone(),
                    &second,
                    RequiredVerification::FullSemantic,
                    ExecutorMode::Success,
                )]
            })
            .err(),
        Some(ExecutionError::DuplicatePeerEntry { peer: peer() })
    );
    assert!(policy.admits(&peer()));
}

#[test]
fn a_malformed_peer_or_principal_is_refused_by_the_policy() {
    let mut policy = AdmissionPolicy::new();
    let scope = ActionScope {
        project: ProjectId::from("p"),
        target: "capsule:a".into(),
        operation: "write".into(),
        capability: CapabilityId::from("file.write"),
        input_type: TypeId::from("Bytes"),
        effect: Effect::Mutation,
    };
    for (node, principal) in [
        (NodeId::from(""), "alice"),
        (NodeId::from("peer\nid"), "alice"),
        (NodeId::from("peer"), ""),
        (NodeId::from("peer"), "alice\n"),
    ] {
        let scope = scope.clone();
        assert_eq!(
            policy
                .admit(node, principal, TTL, move || {
                    vec![grant(
                        scope.clone(),
                        &Probe::default(),
                        RequiredVerification::FullSemantic,
                        ExecutorMode::Success,
                    )]
                })
                .err(),
            Some(ExecutionError::InvalidSession)
        );
    }
}
