use ptr_ledger::{LedgerEvent, SingleNodeRaftConsensus};
use ptr_types::{CapsuleId, Generation, ProjectId};

#[test]
fn single_node_raft_commits_ptr_events_in_order() {
    let mut consensus = SingleNodeRaftConsensus::new(1).unwrap();
    assert!(consensus.is_leader());

    let first = consensus
        .propose(LedgerEvent::CapsuleCommitted {
            project: ProjectId::from("p"),
            capsule: CapsuleId::from("capsule:a"),
            generation: Generation(1),
        })
        .unwrap();
    let second = consensus
        .propose(LedgerEvent::Revoked {
            subject: "capsule:a".into(),
            generation: Generation(1),
        })
        .unwrap();

    assert_eq!(first.commit_index.0, 1);
    assert_eq!(second.commit_index.0, 2);
    assert!(second.raft_index > first.raft_index);
    assert_eq!(consensus.committed_events().len(), 2);
}
