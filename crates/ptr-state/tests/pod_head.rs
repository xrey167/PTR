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
            project: ProjectId::from("project"),
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
            pod_head_key("project", "pod:stable"),
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
        state.values.get(&pod_head_key("project", "pod:stable")),
        Some(&address(2).canonical_uri().unwrap())
    );
    assert_eq!(state.last_applied, 2);
}

#[test]
fn an_address_that_does_not_render_projects_a_value_no_reader_accepts() {
    let mut broken = address(1);
    broken.address.pod_id = PodId::from("a/b");
    let entries = projection_entries(&CommittedEvent {
        index: CommitIndex(1),
        event: LedgerEvent::PodHeadAdvanced {
            project: ProjectId::from("project"),
            identity: PodIdentity::from("pod:stable"),
            old: None,
            new: broken,
        },
    });
    assert_eq!(
        entries,
        vec![(
            pod_head_key("project", "pod:stable"),
            ptr_state::POD_HEAD_UNRENDERABLE.to_owned()
        )]
    );
    assert!(PodRevisionAddress::parse(&entries[0].1).is_err());
}

#[test]
fn the_same_identity_in_two_projects_projects_to_two_keys() {
    let in_project = |project: &str, index: u64| {
        let mut new = address(1);
        new.address.project = ProjectId::from(project);
        CommittedEvent {
            index: CommitIndex(index),
            event: LedgerEvent::PodHeadAdvanced {
                project: ProjectId::from(project),
                identity: PodIdentity::from("pod:stable"),
                old: None,
                new,
            },
        }
    };
    let (alpha, beta) = (in_project("alpha", 1), in_project("beta", 2));
    let mut state = MaterializedState::default();
    state.apply(&alpha);
    state.apply(&beta);
    assert_eq!(
        pod_head_key("alpha", "pod:stable"),
        "pod-head:alpha/pod:stable"
    );
    assert_eq!(state.values.len(), 2);
    for project in ["alpha", "beta"] {
        let stored = state
            .values
            .get(&pod_head_key(project, "pod:stable"))
            .unwrap();
        assert_eq!(
            PodRevisionAddress::parse(stored).unwrap().address.project,
            ProjectId::from(project)
        );
    }
}
