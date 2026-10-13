use ptr_ledger::{CommittedEvent, LedgerEvent};
use ptr_state::{pod_head_key, projection_entries, MaterializedState, POD_HEAD_PREFIX};
use ptr_types::{
    CommitIndex, Generation, NamespaceId, PodAddress, PodId, PodIdentity, PodRevisionAddress,
    ProjectId,
};

fn address(generation: u64) -> PodRevisionAddress {
    PodRevisionAddress {
        address: PodAddress::new(
            ProjectId::from("project"),
            NamespaceId::from("ns"),
            PodId::from("reader"),
        )
        .unwrap(),
        semantic_revision: [6; 32],
        generation: Generation(generation),
    }
}

fn advance(index: u64, old: Option<u64>, new: u64) -> CommittedEvent {
    CommittedEvent {
        index: CommitIndex(index),
        event: LedgerEvent::PodHeadAdvanced {
            identity: PodIdentity::from("pod:stable"),
            old: old.map(address),
            new: address(new),
        },
    }
}

#[test]
fn a_head_move_projects_the_canonical_address_under_the_identity_key() {
    let entries = projection_entries(&advance(1, None, 1));
    assert_eq!(
        entries,
        vec![(
            pod_head_key("pod:stable"),
            address(1).canonical_uri().unwrap()
        )]
    );
    assert!(entries[0].0.starts_with(POD_HEAD_PREFIX));
}

#[test]
fn replaying_the_moves_leaves_the_last_head_in_the_materialized_state() {
    let mut state = MaterializedState::default();
    state.apply(&advance(1, None, 1));
    state.apply(&advance(2, Some(1), 2));
    assert_eq!(
        state.values.get(&pod_head_key("pod:stable")),
        Some(&address(2).canonical_uri().unwrap())
    );
    assert_eq!(state.last_applied, 2);
}
