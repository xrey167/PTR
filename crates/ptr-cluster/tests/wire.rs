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
            if serving.serve_once().await.is_err() {
                return;
            }
        }
    });

    one.campaign().await.unwrap();
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
