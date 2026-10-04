//! What happens to an in-flight permit when the world changes underneath it.
//!
//! A word about what "race" can mean here. `execute_prepared` takes `&mut self`,
//! so nothing — no other session, no host call — can interleave *between* the last
//! admission check and the executor invocation. The interleaving a race needs is
//! prevented by construction rather than by care, which is why these tests do not
//! try to produce one.
//!
//! What remains observable, and is therefore what is tested, is the window between
//! **preparing** a permit and **consuming** it. A permission withdrawn there, a
//! generation revoked or superseded there, a policy replaced there: each must refuse
//! the permit, and refuse it *before* the verifier or the executor is reached. A
//! refusal that arrived after the executor would be a refusal of something that had
//! already happened.
//!
//! A genuinely concurrent race needs two runtimes, which needs the wire — see
//! `docs/architecture/21-scoped-execution.md` and issue #20's open Gate 1c.

mod common;

use common::*;
use ptr_core::action_head::ActionIr;
use ptr_ledger::LedgerEvent;
use ptr_runtime::execution::*;
use ptr_security::AuthorizationDenial;
use ptr_types::{CapsuleId, Generation, NodeId, ProjectId};

/// Tests have no network, so they assume what a transport would have proved.
fn authenticated(id: NodeId) -> ptr_runtime::execution::AuthenticatedPeer {
    ptr_runtime::execution::AuthenticatedPeer::assume_authenticated(id)
}

fn first() -> NodeId {
    NodeId::from("k51qzi5uqu5dh-first-peer")
}

fn second() -> NodeId {
    NodeId::from("k51qzi5uqu5dh-second-peer")
}

/// Grant set "writer" and two peers admitted under it, so one can be withdrawn
/// while the other keeps working.
fn admit_both(runtime: &mut ptr_runtime::PtrRuntime, action: &ActionIr, probe: &Probe) {
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
    runtime.install_grant_sets(sets);
    runtime.admit_member(first(), "alice", "writer").unwrap();
    runtime.admit_member(second(), "bob", "writer").unwrap();
}

#[test]
fn withdrawing_one_peer_in_the_window_refuses_its_permit_and_leaves_the_other_working() {
    let (mut runtime, action) = fixture();
    let probe = Probe::default();
    admit_both(&mut runtime, &action, &probe);

    let alice = runtime.admit_peer(&authenticated(first())).unwrap();
    let bob = runtime.admit_peer(&authenticated(second())).unwrap();
    let alice_permit = runtime
        .prepare_execution(&alice, &ProjectId::from("p"), &action, TTL)
        .unwrap();
    let bob_permit = runtime
        .prepare_execution(&bob, &ProjectId::from("p"), &action, TTL)
        .unwrap();

    // The change comes from outside alice's session: the host withdraws her peer
    // while her permit is in flight.
    runtime.withdraw_peer(&first()).unwrap();

    assert!(
        runtime.execute_prepared(&alice, alice_permit).is_err(),
        "a withdrawn peer's permit is not consumable"
    );
    assert_eq!(
        probe.verifications(),
        0,
        "the verifier must not be reached by a refused permit"
    );
    assert_eq!(probe.executions(), 0);

    // The withdrawal is a committed record, and every commit retires every
    // permit prepared before it: a permit is authorized against the world as it
    // was, and the world has moved. So bob's in-flight permit is stale too —
    assert_eq!(
        runtime.execute_prepared(&bob, bob_permit).err(),
        Some(ExecutionError::StalePermit)
    );
    // — but the withdrawal is about alice's peer, not about bob's: his session
    // still resolves, and a permit prepared now executes.
    let bob_permit = runtime
        .prepare_execution(&bob, &ProjectId::from("p"), &action, TTL)
        .unwrap();
    assert_eq!(
        runtime.execute_prepared(&bob, bob_permit).unwrap(),
        b"executed"
    );
    assert_eq!(probe.executions(), 1);
}

#[test]
fn a_withdrawal_and_re_admission_in_the_window_refuses_the_permit_of_the_first_admission() {
    let (mut runtime, action) = fixture();
    let probe = Probe::default();
    admit_both(&mut runtime, &action, &probe);
    let alice = runtime.admit_peer(&authenticated(first())).unwrap();
    let permit = runtime
        .prepare_execution(&alice, &ProjectId::from("p"), &action, TTL)
        .unwrap();

    // Withdrawn and admitted again while the permit is in flight. The peer is a
    // member at consume time, but not under the admission the session holds.
    runtime.withdraw_peer(&first()).unwrap();
    runtime.admit_member(first(), "alice", "writer").unwrap();

    assert_eq!(
        runtime.execute_prepared(&alice, permit).err(),
        Some(ExecutionError::PeerNotAdmitted { peer: first() }),
        "a permit from a session bound to a withdrawn admission is not consumable"
    );
    assert_eq!(probe.verifications(), 0);
    assert_eq!(probe.executions(), 0);

    // The re-admission does not resurrect the old permit — but it does give a
    // working session.
    let readmitted = runtime.admit_peer(&authenticated(first())).unwrap();
    let fresh = runtime
        .prepare_execution(&readmitted, &ProjectId::from("p"), &action, TTL)
        .unwrap();
    assert_eq!(
        runtime.execute_prepared(&readmitted, fresh).unwrap(),
        b"executed"
    );
}

#[test]
fn a_generation_superseded_in_the_window_is_refused_before_the_verifier() {
    // Supersession is not revocation: nothing was withdrawn, a newer generation
    // simply exists. A permit bound to generation 1 still must not execute against a
    // world where generation 2 is live, because it was authorized against the older
    // one.
    let (mut runtime, action) = fixture();
    let probe = Probe::default();
    let session = session(&mut runtime, &action, &probe, "alice");
    let permit = runtime
        .prepare_execution(&session, &ProjectId::from("p"), &action, TTL)
        .unwrap();

    runtime
        .commit(LedgerEvent::CapsuleCommitted {
            project: ProjectId::from("p"),
            capsule: CapsuleId::from("capsule:a"),
            generation: Generation(2),
        })
        .unwrap();

    assert_eq!(
        runtime.execute_prepared(&session, permit),
        Err(ExecutionError::StalePermit)
    );
    assert_eq!(probe.verifications(), 0);
    assert_eq!(probe.executions(), 0);

    // And preparing again names the reason rather than failing vaguely.
    assert!(matches!(
        runtime.prepare_execution(&session, &ProjectId::from("p"), &action, TTL),
        Err(ExecutionError::AuthorizationDenied(
            AuthorizationDenial::StaleGeneration { .. }
        ))
    ));
}

#[test]
fn a_permit_prepared_before_a_revocation_never_reaches_the_verifier() {
    // The revocation case already has a test for the refusal itself; what it did not
    // assert is that the verifier is never invoked. A verifier that ran would have
    // been handed an action the runtime had already decided against.
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
    assert_eq!(probe.verifications(), 0);
    assert_eq!(probe.executions(), 0);
}

#[test]
fn one_session_s_successful_effect_refuses_another_s_in_flight_permit() {
    // Two sessions, both authorized, both holding permits. The first one executes,
    // which moves the runtime; the second one's permit was authorized against the
    // state before that and is refused rather than silently applied to the state
    // after it.
    let (mut runtime, action) = fixture();
    let probe = Probe::default();
    let alice = session(&mut runtime, &action, &probe, "alice");
    let bob = session(&mut runtime, &action, &probe, "bob");
    let alice_permit = runtime
        .prepare_execution(&alice, &ProjectId::from("p"), &action, TTL)
        .unwrap();
    let bob_permit = runtime
        .prepare_execution(&bob, &ProjectId::from("p"), &action, TTL)
        .unwrap();

    assert_eq!(
        runtime.execute_prepared(&alice, alice_permit).unwrap(),
        b"executed"
    );
    assert_eq!(probe.executions(), 1);

    assert_eq!(
        runtime.execute_prepared(&bob, bob_permit),
        Err(ExecutionError::StalePermit)
    );
    assert_eq!(
        probe.executions(),
        1,
        "the second permit must not reach the executor at all"
    );

    // Bob is not locked out: a permit prepared against the current state works.
    let fresh = runtime
        .prepare_execution(&bob, &ProjectId::from("p"), &action, TTL)
        .unwrap();
    assert_eq!(runtime.execute_prepared(&bob, fresh).unwrap(), b"executed");
    assert_eq!(probe.executions(), 2);
}

#[test]
fn a_host_session_and_an_admitted_peer_do_not_share_a_fate() {
    // Withdrawing a peer must not disturb a session the host registered itself, and
    // revoking a host session must not disturb an admitted peer's.
    let (mut runtime, action) = fixture();
    let probe = Probe::default();
    admit_both(&mut runtime, &action, &probe);
    let host_session = session(&mut runtime, &action, &probe, "operator");
    let peer_session = runtime.admit_peer(&authenticated(first())).unwrap();

    runtime.withdraw_peer(&first()).unwrap();
    let permit = runtime
        .prepare_execution(&host_session, &ProjectId::from("p"), &action, TTL)
        .unwrap();
    assert_eq!(
        runtime.execute_prepared(&host_session, permit).unwrap(),
        b"executed"
    );
    assert!(
        runtime
            .prepare_execution(&peer_session, &ProjectId::from("p"), &action, TTL)
            .is_err(),
        "the withdrawn peer's session is gone"
    );
}
