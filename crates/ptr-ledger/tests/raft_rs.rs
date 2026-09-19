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

#[test]
fn prost_codec_preserves_entry_wire_bytes_and_rejects_malformed_payloads() {
    use raft::codec::Message as _;
    let mut entry = raft::eraftpb::Entry::default();
    entry.set_term(7);
    entry.set_index(9);
    entry.set_data(vec![1, 2, 3]);
    let golden = vec![0x10, 7, 0x18, 9, 0x22, 3, 1, 2, 3];
    assert_eq!(entry.write_to_bytes().unwrap(), golden);
    assert_eq!(entry.compute_size() as usize, golden.len());
    let mut decoded = raft::eraftpb::Entry::default();
    decoded.merge_from_bytes(&golden).unwrap();
    assert_eq!(decoded, entry);
    for malformed in [vec![0x22, 0xff], vec![0x10, 0x80], vec![0xff; 32]] {
        let mut decoded = raft::eraftpb::Entry::default();
        assert!(decoded.merge_from_bytes(&malformed).is_err());
    }
}
