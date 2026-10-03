use ptr_memory::{
    InMemoryKvStateRuntime, KvStateMetadata, KvStateRuntime, KvValidity, RuleRetentionModel,
    TokenBudget,
};
use ptr_runtime::{
    DeviceHealth, DeviceRecord, KvUseError, KvUseRequest, NodeHealth, NodeRecord, PlacementError,
    PodPlacementController, RuntimeKvAuthority,
};
use ptr_types::{
    AdapterVersion, ArtifactId, DeviceId, EventRole, Generation, ModelVersion, NamespaceId, NodeId,
    PodId, PolicyVersion, Probability, RawEventId, SessionId, StateId, Timestamp,
};

#[test]
fn kv_use_ticket_is_rejected_after_placement_fence() {
    let mut placement = PodPlacementController::default();
    placement
        .register_node(NodeRecord {
            node_id: NodeId::from("node"),
            zone: "z".into(),
            health: NodeHealth::Healthy,
            last_heartbeat: Timestamp(1),
            devices: vec![DeviceRecord {
                device_id: DeviceId::from("cuda:0"),
                vram_bytes: 10,
                used_vram_bytes: 0,
                health: DeviceHealth::Healthy,
            }],
        })
        .unwrap();
    let assigned = placement
        .assign(
            PodId::from("pod"),
            ArtifactId::from("artifact"),
            Generation(1),
            NodeId::from("node"),
            DeviceId::from("cuda:0"),
            1,
        )
        .unwrap();
    let lease = placement
        .issue_lease(StateId::from("state"), &assigned.pod_id)
        .unwrap();
    let mut authority = RuntimeKvAuthority::default();
    let request = KvUseRequest {
        state_id: StateId::from("state"),
        model: ModelVersion::from("model"),
        adapter: AdapterVersion::from("adapter"),
        generation: Generation(1),
        context_digest: [7; 32],
    };
    let state_metadata = metadata("state", [7; 32]);
    let ticket = authority
        .prepare(&placement, &state_metadata, lease, request)
        .unwrap();
    placement.mark_node_failed(&NodeId::from("node")).unwrap();
    assert!(matches!(
        authority.complete(&placement, ticket),
        Err(KvUseError::Placement(_))
    ));
}

fn setup() -> (PodPlacementController, ptr_runtime::Placement) {
    let mut placement = PodPlacementController::default();
    placement
        .register_node(NodeRecord {
            node_id: NodeId::from("node"),
            zone: "z".into(),
            health: NodeHealth::Healthy,
            last_heartbeat: Timestamp(1),
            devices: vec![DeviceRecord {
                device_id: DeviceId::from("cuda:0"),
                vram_bytes: 10,
                used_vram_bytes: 0,
                health: DeviceHealth::Healthy,
            }],
        })
        .unwrap();
    let assigned = placement
        .assign(
            PodId::from("pod"),
            ArtifactId::from("artifact"),
            Generation(1),
            NodeId::from("node"),
            DeviceId::from("cuda:0"),
            1,
        )
        .unwrap();
    (placement, assigned)
}

fn request(state: &str, generation: u64, digest: [u8; 32]) -> KvUseRequest {
    KvUseRequest {
        state_id: StateId::from(state),
        model: ModelVersion::from("model"),
        adapter: AdapterVersion::from("adapter"),
        generation: Generation(generation),
        context_digest: digest,
    }
}

fn metadata(state: &str, digest: [u8; 32]) -> KvStateMetadata {
    KvStateMetadata {
        state_id: StateId::from(state),
        parent_state_id: None,
        session: SessionId::from("session"),
        model_version: ModelVersion::from("model"),
        adapter_version: AdapterVersion::from("adapter"),
        snapshot_revision: ptr_types::Revision(0),
        pod_generations: vec![Generation(1)],
        dependency_digests: vec![digest],
        context_digest: digest,
        validity: KvValidity::Valid,
    }
}

#[test]
fn authority_completes_one_valid_kv_use_and_releases_state() {
    let (mut placement, assigned) = setup();
    let lease = placement
        .issue_lease(StateId::from("state"), &assigned.pod_id)
        .unwrap();
    let mut authority = RuntimeKvAuthority::default();
    let expected = request("state", 1, [4; 32]);
    let state_metadata = metadata("state", [4; 32]);
    let ticket = authority
        .prepare(&placement, &state_metadata, lease, expected.clone())
        .unwrap();
    assert_eq!(ticket.state_id(), &StateId::from("state"));
    assert_eq!(ticket.context_digest(), [4; 32]);
    assert_eq!(authority.complete(&placement, ticket).unwrap(), expected);

    let lease = placement
        .issue_lease(StateId::from("state"), &assigned.pod_id)
        .unwrap();
    authority
        .prepare(
            &placement,
            &metadata("state", [5; 32]),
            lease,
            request("state", 1, [5; 32]),
        )
        .unwrap();
}

#[test]
fn authority_rejects_duplicate_use_and_state_mismatch() {
    let (mut placement, assigned) = setup();
    let lease = placement
        .issue_lease(StateId::from("state"), &assigned.pod_id)
        .unwrap();
    let mut authority = RuntimeKvAuthority::default();
    let ticket = authority
        .prepare(
            &placement,
            &metadata("state", [1; 32]),
            lease,
            request("state", 1, [1; 32]),
        )
        .unwrap();
    let second_lease = placement
        .issue_lease(StateId::from("state"), &assigned.pod_id)
        .unwrap();
    assert_eq!(
        authority.prepare(
            &placement,
            &metadata("state", [2; 32]),
            second_lease,
            request("state", 1, [2; 32]),
        ),
        Err(KvUseError::AlreadyInUse)
    );
    let mismatch_lease = placement
        .issue_lease(StateId::from("mismatch"), &assigned.pod_id)
        .unwrap();
    assert_eq!(
        authority.prepare(
            &placement,
            &metadata("mismatch", [3; 32]),
            mismatch_lease,
            request("mismatch", 2, [3; 32]),
        ),
        Err(KvUseError::StateMismatch)
    );
    let _ = ticket;
}

#[test]
fn stale_completion_releases_the_authority_slot_for_recovery() {
    let (mut placement, assigned) = setup();
    placement
        .register_node(NodeRecord {
            node_id: NodeId::from("node-b"),
            zone: "z".into(),
            health: NodeHealth::Healthy,
            last_heartbeat: Timestamp(1),
            devices: vec![DeviceRecord {
                device_id: DeviceId::from("cuda:1"),
                vram_bytes: 10,
                used_vram_bytes: 0,
                health: DeviceHealth::Healthy,
            }],
        })
        .unwrap();
    let lease = placement
        .issue_lease(StateId::from("state"), &assigned.pod_id)
        .unwrap();
    let mut authority = RuntimeKvAuthority::default();
    let ticket = authority
        .prepare(
            &placement,
            &metadata("state", [9; 32]),
            lease,
            request("state", 1, [9; 32]),
        )
        .unwrap();

    placement
        .assign(
            assigned.pod_id.clone(),
            ArtifactId::from("artifact"),
            Generation(1),
            NodeId::from("node-b"),
            DeviceId::from("cuda:1"),
            1,
        )
        .unwrap();
    assert!(matches!(
        authority.complete(&placement, ticket),
        Err(KvUseError::Placement(PlacementError::StaleLease))
    ));

    let recovered_lease = placement
        .issue_lease(StateId::from("state"), &assigned.pod_id)
        .unwrap();
    authority
        .prepare(
            &placement,
            &metadata("state", [10; 32]),
            recovered_lease,
            request("state", 1, [10; 32]),
        )
        .unwrap();
}

#[test]
fn memory_placement_kv_and_fence_form_one_authorized_flow() {
    let mut memory = ptr_runtime::memory::ContextMemoryController::new(
        RuleRetentionModel,
        Probability::new(0.5).unwrap(),
        PolicyVersion::from("policy"),
    );
    let object = memory
        .ingest(
            ptr_memory::RawEvent::new(
                RawEventId::from("raw-1"),
                SessionId::from("session"),
                EventRole::Tool,
                b"memory context".to_vec(),
                Timestamp(1),
            ),
            NamespaceId::from("ptr.test"),
        )
        .unwrap();
    let context = memory
        .compile(std::slice::from_ref(&object.id), TokenBudget(10))
        .unwrap();
    let mut kv = InMemoryKvStateRuntime::new(
        SessionId::from("session"),
        ModelVersion::from("model"),
        AdapterVersion::from("adapter"),
    );
    let kv_state = kv.recompute(context.clone()).unwrap();
    let kv_metadata = kv.metadata(&kv_state).unwrap();

    let (mut placement, assigned) = setup();
    let lease = placement
        .issue_lease(StateId::from("kv-state"), &assigned.pod_id)
        .unwrap();
    let mut authority = RuntimeKvAuthority::default();
    let ticket = authority
        .prepare(
            &placement,
            &KvStateMetadata {
                state_id: StateId::from("kv-state"),
                ..kv_metadata
            },
            lease,
            request("kv-state", 1, context.digest),
        )
        .unwrap();
    placement.mark_node_failed(&NodeId::from("node")).unwrap();
    assert!(matches!(
        authority.complete(&placement, ticket),
        Err(KvUseError::Placement(PlacementError::StaleLease))
    ));
}
