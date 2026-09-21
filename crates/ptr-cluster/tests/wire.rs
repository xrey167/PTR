//! Raft over a real authenticated connection.
//!
//! Two properties are worth a network test rather than a queue. One: the messages
//! actually traverse `ALPN_RAFT` and a group forms over them. Two — the reason this
//! layer exists — a peer cannot claim to be another member, because the sender is
//! the connection and not the payload.
//!
//! Everything else about raft's behaviour is tested deterministically in
//! `ptr-ledger`, where no socket is involved. A network test that tried to prove a
//! partition would be proving it about the test's timing.

use ptr_cluster::{decode_batch, encode_batch, ClusterError, ClusterMember, MemberAddress};
use ptr_ledger::{encode_message, LedgerEvent, RaftMessage};
use ptr_net::{IrohTransport, ALPN_RAFT};
use ptr_types::{CapsuleId, Generation, ProjectId};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

struct Temp(PathBuf);

impl Temp {
    fn new(label: &str) -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "ptr-wire-{label}-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&path).unwrap();
        Self(path)
    }

    fn member(&self, id: u64) -> PathBuf {
        self.0.join(format!("node-{id}"))
    }
}

impl Drop for Temp {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

const VOTERS: [u64; 2] = [1, 2];

fn event(generation: u64) -> LedgerEvent {
    LedgerEvent::CapsuleCommitted {
        project: ProjectId::from("p"),
        capsule: CapsuleId::from("capsule:a"),
        generation: Generation(generation),
    }
}

/// A message as it would look on the wire, with whatever `from`/`to` are given.
fn frame(from: u64, to: u64) -> Vec<u8> {
    let mut message = RaftMessage::default();
    message.set_msg_type(ptr_ledger::raft_node::message_type_heartbeat());
    message.from = from;
    message.to = to;
    message.term = 1;
    encode_batch(&[encode_message(&message).unwrap()]).unwrap()
}

/// Two members that know each other's keys and addresses.
async fn pair(temp: &Temp) -> (ClusterMember, ClusterMember) {
    let mut one = ClusterMember::bind(&temp.member(1), 1, &VOTERS)
        .await
        .unwrap();
    let mut two = ClusterMember::bind(&temp.member(2), 2, &VOTERS)
        .await
        .unwrap();
    one.admit(MemberAddress {
        id: 2,
        identity: two.identity(),
        address: two.address(),
    });
    two.admit(MemberAddress {
        id: 1,
        identity: one.identity(),
        address: one.address(),
    });
    (one, two)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_group_forms_over_alpn_raft_and_both_members_hold_the_same_history() {
    let temp = Temp::new("group");
    let (one, two) = pair(&temp).await;
    let follower = Arc::new(two);
    let serving = Arc::clone(&follower);
    let server = tokio::spawn(async move {
        loop {
            // Keep serving after a refusal or a transient transport error, the way
            // a real member would. Returning here made a *test harness* decide
            // whether a member was reachable: `serve_once` reports a refused peer
            // and a transport hiccup with the same `Err`, so one of either killed
            // the follower for the rest of the test, and a later round then
            // reported it unreached. That is what made
            // `a_member_that_cannot_be_reached_is_reported_rather_than_failing_the_write`
            // fail on CI while passing here. Every one of these tasks is ended by
            // `abort()`, so there is nothing for this loop to exit for.
            let _ = serving.serve_once().await;
        }
    });

    assert!(
        one.campaign().await.unwrap().is_empty(),
        "both members answered"
    );
    assert!(
        one.is_leader(),
        "a vote carried over the wire elects a leader"
    );

    one.propose(event(1)).await.unwrap();
    one.propose(event(2)).await.unwrap();

    let leader = one.committed_events();
    assert_eq!(leader.len(), 2);
    // The follower has to be driven once more for the commit index to reach it.
    one.tick().await.unwrap();
    assert_eq!(
        follower.committed_events(),
        leader,
        "both members hold the same history, in the same order"
    );
    assert_eq!(follower.term(), one.term());

    server.abort();
    one.close().await;
    follower.close().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_peer_cannot_claim_another_member_s_id() {
    // The case the layer exists for. The impostor holds the key member 1 admitted
    // for member 2, and sends a message that says it comes from member 1.
    let temp = Temp::new("forged");
    let mut member = ClusterMember::bind(&temp.member(1), 1, &VOTERS)
        .await
        .unwrap();
    let impostor = IrohTransport::bind(&[]).await.unwrap();
    member.admit(MemberAddress {
        id: 2,
        identity: impostor.identity(),
        address: impostor.direct_addr(),
    });

    let target = member.address();
    let forged = frame(1, 1);
    let sender = tokio::spawn(async move {
        let _ = impostor.request(target, ALPN_RAFT, &forged, 4096).await;
        impostor.close().await;
    });

    let error = member
        .serve_once()
        .await
        .expect_err("a member's id is not something a peer may choose");
    assert_eq!(
        error,
        ClusterError::ForgedSender {
            authenticated: 2,
            claimed: 1,
        }
    );
    assert_eq!(error.code(), "PTR_CLUSTER_FORGED_SENDER");
    assert!(member.committed_events().is_empty());
    assert_eq!(member.term(), 0, "a refused frame changed no state");

    sender.await.unwrap();
    member.close().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_peer_this_member_never_admitted_is_refused() {
    let temp = Temp::new("stranger");
    let member = ClusterMember::bind(&temp.member(1), 1, &VOTERS)
        .await
        .unwrap();
    let stranger = IrohTransport::bind(&[]).await.unwrap();
    let expected_key = stranger.identity().public_key;

    let target = member.address();
    let valid = frame(2, 1);
    let sender = tokio::spawn(async move {
        let _ = stranger.request(target, ALPN_RAFT, &valid, 4096).await;
        stranger.close().await;
    });

    let error = member
        .serve_once()
        .await
        .expect_err("a well-formed frame from nobody is still from nobody");
    assert_eq!(error, ClusterError::UnknownPeer { key: expected_key });

    sender.await.unwrap();
    member.close().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_damaged_frame_on_the_wire_is_refused_and_changes_nothing() {
    // The transport half of the corruption test: the deterministic tests in
    // ptr-ledger prove the decoder, and this proves that a damaged frame arriving
    // on a real connection is refused before it reaches the node.
    let temp = Temp::new("damaged");
    let mut member = ClusterMember::bind(&temp.member(1), 1, &VOTERS)
        .await
        .unwrap();
    let peer = IrohTransport::bind(&[]).await.unwrap();
    member.admit(MemberAddress {
        id: 2,
        identity: peer.identity(),
        address: peer.direct_addr(),
    });

    let mut damaged = frame(2, 1);
    let last = damaged.len() - 1;
    damaged[last] ^= 0xff;
    damaged.truncate(damaged.len() - 2);

    let target = member.address();
    let sender = tokio::spawn(async move {
        let _ = peer.request(target, ALPN_RAFT, &damaged, 4096).await;
        peer.close().await;
    });

    let error = member
        .serve_once()
        .await
        .expect_err("a frame that disagrees with itself is not a message");
    assert!(
        matches!(
            error,
            ClusterError::Frame(_) | ClusterError::Undecodable { .. }
        ),
        "expected a framing refusal, got {error:?}"
    );
    assert!(member.committed_events().is_empty());
    assert_eq!(member.term(), 0);

    sender.await.unwrap();
    member.close().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_message_addressed_to_another_member_is_refused() {
    let temp = Temp::new("misaddressed");
    let mut member = ClusterMember::bind(&temp.member(1), 1, &VOTERS)
        .await
        .unwrap();
    let peer = IrohTransport::bind(&[]).await.unwrap();
    member.admit(MemberAddress {
        id: 2,
        identity: peer.identity(),
        address: peer.direct_addr(),
    });

    let target = member.address();
    let elsewhere = frame(2, 99);
    let sender = tokio::spawn(async move {
        let _ = peer.request(target, ALPN_RAFT, &elsewhere, 4096).await;
        peer.close().await;
    });

    let error = member
        .serve_once()
        .await
        .expect_err("a message for member 99 is not this member's to apply");
    assert_eq!(
        error,
        ClusterError::Misaddressed {
            to: 99,
            expected: 1
        }
    );

    sender.await.unwrap();
    member.close().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_refused_frame_still_gets_an_answer_rather_than_a_hanging_peer() {
    let temp = Temp::new("answered");
    let member = ClusterMember::bind(&temp.member(1), 1, &VOTERS)
        .await
        .unwrap();
    let stranger = IrohTransport::bind(&[]).await.unwrap();

    let target = member.address();
    let valid = frame(2, 1);
    let sender = tokio::spawn(async move {
        let response = stranger
            .request(target, ALPN_RAFT, &valid, 4096)
            .await
            .expect("a refusal is still an answer");
        stranger.close().await;
        response
    });

    let _ = member.serve_once().await.expect_err("an unadmitted peer");
    let response = sender.await.unwrap();
    assert!(
        decode_batch(&response).unwrap().is_empty(),
        "the answer to a refused frame is an empty batch, not a diagnosis"
    );
    member.close().await;
}

/// A serving member survives a refusal, so a stranger cannot take it off the air.
///
/// This is about the shape of every serve loop in this file, and it is here because
/// getting it wrong cost a red CI run.
/// `a_member_that_cannot_be_reached_is_reported_rather_than_failing_the_write`
/// reported member *2* unreached — the member that was supposed to answer — and the
/// cause was the harness, not the code under test: the loop read
/// `if serve_once().await.is_err() { return }`, and `serve_once` reports a refused
/// peer and a transport hiccup with the same `Err`. One of either ended the
/// follower for the rest of the test, and the next round then found it unreachable,
/// which is exactly what a dead server looks like from outside.
///
/// The property worth pinning is the one a real member has: refusing a frame is not
/// a reason to stop serving. If it were, an unadmitted peer could silence an
/// admitted one by connecting once — availability handed to whoever calls first,
/// which is the opposite of what refusing that peer was for.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_refusal_does_not_take_a_serving_member_off_the_air() {
    let temp = Temp::new("survives");
    let (one, two) = pair(&temp).await;

    let follower = Arc::new(two);
    let serving = Arc::clone(&follower);
    let server = tokio::spawn(async move {
        loop {
            let _ = serving.serve_once().await;
        }
    });

    // A peer the follower never admitted, sending a well-formed frame. `serve_once`
    // answers it and returns `Err`, which is the refusal this test walks over.
    let stranger = IrohTransport::bind(&[]).await.unwrap();
    let refused = stranger
        .request(follower.address(), ALPN_RAFT, &frame(99, 2), 4096)
        .await
        .expect("a refusal is still an answer");
    assert!(
        decode_batch(&refused).unwrap().is_empty(),
        "the stranger got a refusal, so the follower did serve it"
    );
    stranger.close().await;

    // And now the member that *is* admitted. Under a loop that exits on `Err` the
    // follower is already gone and this campaign finds it unreached.
    let unreached = one.campaign().await.unwrap();
    assert!(
        unreached.ids().is_empty(),
        "the follower answered after refusing a stranger, so nothing is unreached: {:?}",
        unreached.ids()
    );
    assert!(one.is_leader(), "two of two answered");

    server.abort();
    one.close().await;
    follower.close().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_member_that_cannot_be_reached_is_reported_rather_than_failing_the_write() {
    // A leader that abandoned a proposal because one follower was down could not
    // commit while any member is down, which is the opposite of what a majority is
    // for. So unreachability is reported and the round continues.
    let temp = Temp::new("unreached");
    const THREE: [u64; 3] = [1, 2, 3];
    let mut one = ClusterMember::bind(&temp.member(1), 1, &THREE)
        .await
        .unwrap();
    let mut two = ClusterMember::bind(&temp.member(2), 2, &THREE)
        .await
        .unwrap();
    let three = ClusterMember::bind(&temp.member(3), 3, &THREE)
        .await
        .unwrap();

    for peer in [
        MemberAddress {
            id: 2,
            identity: two.identity(),
            address: two.address(),
        },
        MemberAddress {
            id: 3,
            identity: three.identity(),
            address: three.address(),
        },
    ] {
        one.admit(peer);
    }
    two.admit(MemberAddress {
        id: 1,
        identity: one.identity(),
        address: one.address(),
    });

    // Member 3 is gone, not merely quiet: a bound endpoint that never accepts would
    // leave the leader waiting instead of failing, which is a different situation.
    three.close().await;
    // A deadline shorter than the 5s default, because this test is about the
    // reporting rather than about how patient the default is - but not a tight
    // one. At 250ms this went red on CI with `unreached == [2, 3]`: member 2 is
    // healthy and serving, and its *first* answer still costs a QUIC handshake,
    // which on a loaded runner can outlast a quarter second. That made the
    // assertion below a claim about latency rather than about reporting, which
    // is not the property this test exists for. Member 3 is closed, so the
    // deadline is what bounds it and the test still ends promptly.
    one.with_request_timeout(Duration::from_secs(2));

    let follower = Arc::new(two);
    let serving = Arc::clone(&follower);
    let server = tokio::spawn(async move {
        loop {
            // Keep serving after a refusal or a transient transport error, the way
            // a real member would. Returning here made a *test harness* decide
            // whether a member was reachable: `serve_once` reports a refused peer
            // and a transport hiccup with the same `Err`, so one of either killed
            // the follower for the rest of the test, and a later round then
            // reported it unreached. That is what made
            // `a_member_that_cannot_be_reached_is_reported_rather_than_failing_the_write`
            // fail on CI while passing here. Every one of these tasks is ended by
            // `abort()`, so there is nothing for this loop to exit for.
            let _ = serving.serve_once().await;
        }
    });

    let unreached = one.campaign().await.unwrap();
    assert_eq!(unreached.ids(), vec![3], "member 3 did not answer the vote");
    assert!(
        !unreached.0.is_empty(),
        "every attempt toward it is recorded, not just the member"
    );
    assert!(one.is_leader(), "two of three is still a majority");

    let unreached = one.propose(event(1)).await.unwrap();
    // raft pauses probing a peer that has not answered, so this round may not even
    // attempt member 3. Either way the property is the same one: the write is decided
    // by the majority that did answer.
    assert!(
        unreached.ids().is_empty() || unreached.ids() == vec![3],
        "unexpected report: {:?}",
        unreached.ids()
    );
    assert_eq!(
        one.committed_events().len(),
        1,
        "the write was decided by the majority that answered"
    );

    server.abort();
    one.close().await;
    follower.close().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_ptrcs001_snapshot_travels_as_the_payload_and_restores_on_the_far_side() {
    // "Reusing PTRCS002 rather than inventing a second artifact" is the requirement,
    // so the payload here is a real compacted snapshot exported by a real runtime,
    // and the far side restores a runtime from the bytes that arrived.
    //
    // The artifact's anchor travels out of band, which over this transport means
    // "from the authenticated leader". That is weaker than an anchor retained
    // independently, and deliberately so rather than by omission: a compromised
    // leader could send a consistent pair. What it rules out is a *stranger* doing
    // so, because the sender is the connection.
    use ptr_config::PtrConfig;
    use ptr_runtime::PtrRuntime;
    use ptr_semdb::SemanticDelta;

    let temp = Temp::new("ptrcs001");
    const THREE: [u64; 3] = [1, 2, 3];

    // The application state the snapshot describes.
    let mut runtime = PtrRuntime::new(PtrConfig::default()).unwrap();
    let mut delta = SemanticDelta::default();
    delta
        .upserts
        .insert("plan".into(), "carried by raft".into());
    runtime
        .apply_semantic_delta(runtime.revision(), delta)
        .unwrap();
    let exported = runtime.export_compacted_snapshot().unwrap();
    let payload = exported.bytes().to_vec();
    let anchor = exported.anchor();
    assert_eq!(
        &payload[..8],
        b"PTRCS002",
        "the payload is the artifact itself"
    );

    let mut one = ClusterMember::bind(&temp.member(1), 1, &THREE)
        .await
        .unwrap();
    let mut two = ClusterMember::bind(&temp.member(2), 2, &THREE)
        .await
        .unwrap();
    let behind = ClusterMember::bind(&temp.member(3), 3, &THREE)
        .await
        .unwrap();
    one.admit(MemberAddress {
        id: 2,
        identity: two.identity(),
        address: two.address(),
    });
    one.admit(MemberAddress {
        id: 3,
        identity: behind.identity(),
        address: behind.address(),
    });
    two.admit(MemberAddress {
        id: 1,
        identity: one.identity(),
        address: one.address(),
    });
    behind.close().await;
    // Same reasoning as the deadline in
    // `a_member_that_cannot_be_reached_is_reported_rather_than_failing_the_write`:
    // short enough to bound a member that is gone, long enough that a healthy
    // member's first answer is never mistaken for silence.
    one.with_request_timeout(Duration::from_secs(2));

    let follower = Arc::new(two);
    let serving = Arc::clone(&follower);
    let server = tokio::spawn(async move {
        loop {
            // Keep serving after a refusal or a transient transport error, the way
            // a real member would. Returning here made a *test harness* decide
            // whether a member was reachable: `serve_once` reports a refused peer
            // and a transport hiccup with the same `Err`, so one of either killed
            // the follower for the rest of the test, and a later round then
            // reported it unreached. That is what made
            // `a_member_that_cannot_be_reached_is_reported_rather_than_failing_the_write`
            // fail on CI while passing here. Every one of these tasks is ended by
            // `abort()`, so there is nothing for this loop to exit for.
            let _ = serving.serve_once().await;
        }
    });

    one.campaign().await.unwrap();
    for generation in 1..=3 {
        one.propose(event(generation)).await.unwrap();
    }
    assert_eq!(one.committed_events().len(), 3);

    // The leader records its application state at the position it has committed and
    // discards the log below it, so member 3 can no longer be caught up by replay.
    let covered = one.raft_committed();
    one.record_snapshot(covered, payload.clone()).unwrap();

    // Member 3 comes back — a restart, so a new endpoint with a new key over the same
    // durable state, re-admitted by the leader.
    let mut restarted = ClusterMember::bind(&temp.member(3), 3, &THREE)
        .await
        .unwrap();
    restarted.admit(MemberAddress {
        id: 1,
        identity: one.identity(),
        address: one.address(),
    });
    one.admit(MemberAddress {
        id: 3,
        identity: restarted.identity(),
        address: restarted.address(),
    });

    let returning = Arc::new(restarted);
    let serving_three = Arc::clone(&returning);
    let third = tokio::spawn(async move {
        loop {
            // Same reason as the loops above: a refusal and a transport hiccup are
            // the same `Err`, and exiting on either makes the harness, not the
            // test, decide whether this member is reachable.
            let _ = serving_three.serve_once().await;
        }
    });

    // Drive until the snapshot lands, bounded so a failure is a failure rather than a
    // hang. Each round is a heartbeat the leader sends to a member it cannot replay to.
    let mut installed = None;
    for _ in 0..16 {
        one.tick().await.unwrap();
        if let Some(snapshot) = returning.take_installed_snapshot() {
            installed = Some(snapshot);
            break;
        }
    }
    let installed = installed.expect("a member the leader cannot replay to receives a snapshot");
    assert_eq!(
        installed.state, payload,
        "the artifact arrives byte for byte, unread by the layer that carried it"
    );

    // And it is a usable artifact on the far side: a runtime restores from exactly
    // those bytes, against the anchor the leader also sent.
    let restored =
        PtrRuntime::restore_compacted(PtrConfig::default(), &installed.state, anchor, &[])
            .expect("the bytes that arrived are a restorable snapshot");
    assert_eq!(
        restored.snapshot().value("plan"),
        Some(&ptr_semdb::SemanticValue::from("carried by raft")),
        "the state the snapshot described is the state that came back"
    );

    third.abort();
    server.abort();
    one.close().await;
    follower.close().await;
    returning.close().await;
}
