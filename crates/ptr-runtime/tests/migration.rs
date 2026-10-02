use ptr_pods::{
    InMemoryKvTensorBackend, KvTensorBackend, KvTensorDType, KvTensorSchema, TensorRef,
};
use ptr_runtime::{
    DeviceHealth, DeviceRecord, KvMigrationController, MigrationError, MigrationState, NodeHealth,
    NodeRecord, PodPlacementController,
};
use ptr_types::{
    AdapterVersion, ArtifactId, DeviceId, Generation, ModelVersion, NodeId, PodId, StateId,
    Timestamp,
};

fn node(id: &str, device: &str) -> NodeRecord {
    NodeRecord {
        node_id: NodeId::from(id),
        zone: id.into(),
        health: NodeHealth::Healthy,
        last_heartbeat: Timestamp(1),
        devices: vec![DeviceRecord {
            device_id: DeviceId::from(device),
            vram_bytes: 32,
            used_vram_bytes: 0,
            health: DeviceHealth::Healthy,
        }],
    }
}

fn schema(device: &str) -> KvTensorSchema {
    KvTensorSchema {
        model: ModelVersion::from("model"),
        adapter: AdapterVersion::from("adapter"),
        device: DeviceId::from(device),
        layer_count: 1,
        attention_heads: 1,
        head_dim: 2,
        batch_size: 1,
        dtype: KvTensorDType::F32,
    }
}

#[test]
fn snapshot_restore_and_commit_move_a_fenced_state() {
    let mut placement = PodPlacementController::default();
    placement.register_node(node("node-a", "cuda:0")).unwrap();
    placement.register_node(node("node-b", "cuda:1")).unwrap();
    let first = placement
        .assign(
            PodId::from("pod"),
            ArtifactId::from("artifact"),
            Generation(1),
            NodeId::from("node-a"),
            DeviceId::from("cuda:0"),
            1,
        )
        .unwrap();
    let source = placement
        .issue_lease(StateId::from("state"), &first.pod_id)
        .unwrap();

    let backend = InMemoryKvTensorBackend;
    let mut cache = backend.allocate(schema("cuda:0"), 4).unwrap();
    backend
        .append(
            &mut cache,
            &[TensorRef {
                layer: 0,
                shape: vec![1, 2],
                values: vec![1.0, 2.0],
            }],
            &[TensorRef {
                layer: 0,
                shape: vec![1, 2],
                values: vec![3.0, 4.0],
            }],
        )
        .unwrap();
    let snapshot = backend.snapshot(&cache).unwrap();

    let mut migration = KvMigrationController::default();
    let started = migration.begin(&mut placement, source).unwrap();
    assert_eq!(started.state, MigrationState::Frozen);
    migration.snapshot(started.migration_id, &snapshot).unwrap();

    let second = placement
        .assign(
            PodId::from("pod"),
            ArtifactId::from("artifact"),
            Generation(1),
            NodeId::from("node-b"),
            DeviceId::from("cuda:1"),
            1,
        )
        .unwrap();
    let target = placement
        .issue_lease(StateId::from("state"), &second.pod_id)
        .unwrap();
    migration
        .restore(
            &mut placement,
            started.migration_id,
            &target,
            &backend,
            &snapshot,
        )
        .unwrap();
    let committed = migration
        .commit(&mut placement, started.migration_id, &target)
        .unwrap();
    assert_eq!(committed.state, MigrationState::Committed);
}

#[test]
fn corrupted_snapshot_cannot_start_restore() {
    let mut placement = PodPlacementController::default();
    placement.register_node(node("node-a", "cuda:0")).unwrap();
    let assigned = placement
        .assign(
            PodId::from("pod"),
            ArtifactId::from("artifact"),
            Generation(1),
            NodeId::from("node-a"),
            DeviceId::from("cuda:0"),
            1,
        )
        .unwrap();
    let source = placement
        .issue_lease(StateId::from("state"), &assigned.pod_id)
        .unwrap();
    let backend = InMemoryKvTensorBackend;
    let mut cache = backend.allocate(schema("cuda:0"), 2).unwrap();
    backend
        .append(
            &mut cache,
            &[TensorRef {
                layer: 0,
                shape: vec![1, 2],
                values: vec![1.0, 2.0],
            }],
            &[TensorRef {
                layer: 0,
                shape: vec![1, 2],
                values: vec![3.0, 4.0],
            }],
        )
        .unwrap();
    let mut snapshot = backend.snapshot(&cache).unwrap();
    snapshot.digest[0] ^= 1;
    let mut migration = KvMigrationController::default();
    let record = migration.begin(&mut placement, source).unwrap();
    assert_eq!(
        migration.snapshot(record.migration_id, &snapshot),
        Err(MigrationError::SnapshotDigestMismatch)
    );
}

#[test]
fn restore_before_snapshot_is_rejected_as_an_invalid_transition() {
    let (mut placement, source, mut migration) = migration_fixture();
    let started = migration.begin(&mut placement, source).unwrap();
    let target = target(&mut placement);
    let backend = InMemoryKvTensorBackend;
    let mut cache = backend.allocate(schema("cuda:0"), 1).unwrap();
    backend
        .append(
            &mut cache,
            &[TensorRef {
                layer: 0,
                shape: vec![1, 2],
                values: vec![1.0, 2.0],
            }],
            &[TensorRef {
                layer: 0,
                shape: vec![1, 2],
                values: vec![3.0, 4.0],
            }],
        )
        .unwrap();
    let snapshot = backend.snapshot(&cache).unwrap();

    assert_eq!(
        migration.restore(
            &mut placement,
            started.migration_id,
            &target,
            &backend,
            &snapshot,
        ),
        Err(MigrationError::InvalidState(MigrationState::Frozen))
    );
}

#[test]
fn commit_before_restore_is_rejected_as_an_invalid_transition() {
    let (mut placement, source, mut migration) = migration_fixture();
    let started = migration.begin(&mut placement, source).unwrap();
    let target = target(&mut placement);

    assert_eq!(
        migration.commit(&mut placement, started.migration_id, &target),
        Err(MigrationError::InvalidState(MigrationState::Frozen))
    );
}

#[test]
fn abort_prevents_later_snapshot_and_restore() {
    let (mut placement, source, mut migration) = migration_fixture();
    let started = migration.begin(&mut placement, source).unwrap();
    let aborted = migration
        .abort(&mut placement, started.migration_id)
        .unwrap();
    assert_eq!(aborted.state, MigrationState::Aborted);

    let backend = InMemoryKvTensorBackend;
    let cache = backend.allocate(schema("cuda:0"), 1).unwrap();
    let snapshot = backend.snapshot(&cache).unwrap();
    assert_eq!(
        migration.snapshot(started.migration_id, &snapshot),
        Err(MigrationError::InvalidState(MigrationState::Aborted))
    );
}

#[test]
fn abort_is_not_repeatable_after_the_migration_is_already_aborted() {
    let (mut placement, source, mut migration) = migration_fixture();
    let started = migration.begin(&mut placement, source).unwrap();
    migration
        .abort(&mut placement, started.migration_id)
        .unwrap();

    assert_eq!(
        migration.abort(&mut placement, started.migration_id),
        Err(MigrationError::InvalidState(MigrationState::Aborted))
    );
}

#[test]
fn commit_is_not_repeatable_after_the_migration_is_already_committed() {
    let (mut placement, source, mut migration) = migration_fixture();
    let started = migration.begin(&mut placement, source).unwrap();
    let backend = InMemoryKvTensorBackend;
    let mut cache = backend.allocate(schema("cuda:0"), 1).unwrap();
    backend
        .append(
            &mut cache,
            &[TensorRef {
                layer: 0,
                shape: vec![1, 2],
                values: vec![1.0, 2.0],
            }],
            &[TensorRef {
                layer: 0,
                shape: vec![1, 2],
                values: vec![3.0, 4.0],
            }],
        )
        .unwrap();
    let snapshot = backend.snapshot(&cache).unwrap();
    migration.snapshot(started.migration_id, &snapshot).unwrap();
    let target = target(&mut placement);
    migration
        .restore(
            &mut placement,
            started.migration_id,
            &target,
            &backend,
            &snapshot,
        )
        .unwrap();
    migration
        .commit(&mut placement, started.migration_id, &target)
        .unwrap();

    assert_eq!(
        migration.commit(&mut placement, started.migration_id, &target),
        Err(MigrationError::InvalidState(MigrationState::Committed))
    );
}

#[test]
fn stale_source_lease_is_rejected_after_node_rebalance() {
    let mut placement = PodPlacementController::default();
    placement.register_node(node("node-a", "cuda:0")).unwrap();
    placement.register_node(node("node-b", "cuda:0")).unwrap();
    let first = placement
        .assign(
            PodId::from("pod"),
            ArtifactId::from("artifact"),
            Generation(1),
            NodeId::from("node-a"),
            DeviceId::from("cuda:0"),
            1,
        )
        .unwrap();
    let stale = placement
        .issue_lease(StateId::from("state"), &first.pod_id)
        .unwrap();
    placement
        .assign(
            PodId::from("pod"),
            ArtifactId::from("artifact"),
            Generation(1),
            NodeId::from("node-b"),
            DeviceId::from("cuda:0"),
            1,
        )
        .unwrap();

    assert_eq!(
        KvMigrationController::default().begin(&mut placement, stale),
        Err(MigrationError::Placement(
            ptr_runtime::PlacementError::StaleLease
        ))
    );
}

#[test]
fn restore_rejects_a_target_with_a_different_generation() {
    let (mut placement, source, mut migration) = migration_fixture();
    let started = migration.begin(&mut placement, source).unwrap();
    let backend = InMemoryKvTensorBackend;
    let cache = backend.allocate(schema("cuda:0"), 1).unwrap();
    let snapshot = backend.snapshot(&cache).unwrap();
    migration.snapshot(started.migration_id, &snapshot).unwrap();
    let target_placement = placement
        .assign(
            PodId::from("pod"),
            ArtifactId::from("artifact"),
            Generation(2),
            NodeId::from("node-b"),
            DeviceId::from("cuda:0"),
            1,
        )
        .unwrap();
    let target = placement
        .issue_lease(StateId::from("state"), &target_placement.pod_id)
        .unwrap();

    assert_eq!(
        migration.restore(
            &mut placement,
            started.migration_id,
            &target,
            &backend,
            &snapshot,
        ),
        Err(MigrationError::GenerationMismatch)
    );
}

#[test]
fn snapshot_restore_and_commit_use_the_restored_backend_state() {
    let (mut placement, source, mut migration) = migration_fixture();
    let backend = InMemoryKvTensorBackend;
    let mut cache = backend.allocate(schema("cuda:0"), 2).unwrap();
    backend
        .append(
            &mut cache,
            &[TensorRef {
                layer: 0,
                shape: vec![1, 2],
                values: vec![1.0, 2.0],
            }],
            &[TensorRef {
                layer: 0,
                shape: vec![1, 2],
                values: vec![3.0, 4.0],
            }],
        )
        .unwrap();
    let snapshot = backend.snapshot(&cache).unwrap();
    let started = migration.begin(&mut placement, source).unwrap();
    migration.snapshot(started.migration_id, &snapshot).unwrap();
    let restored_cache = backend
        .restore(snapshot.clone(), &DeviceId::from("cuda:0"))
        .unwrap();
    assert_eq!(restored_cache.sequence_length, snapshot.sequence_length);

    let target = target(&mut placement);
    migration
        .restore(
            &mut placement,
            started.migration_id,
            &target,
            &backend,
            &snapshot,
        )
        .unwrap();
    let committed = migration
        .commit(&mut placement, started.migration_id, &target)
        .unwrap();
    assert_eq!(committed.state, MigrationState::Committed);
    assert_eq!(committed.snapshot_digest, Some(snapshot.digest));
}

#[test]
fn begin_freezes_source_until_reassignment_or_abort() {
    let (mut placement, source, mut migration) = migration_fixture();
    let pod_id = source.pod_id().clone();
    let started = migration.begin(&mut placement, source).unwrap();

    assert_eq!(
        placement.issue_lease(StateId::from("state"), &pod_id),
        Err(ptr_runtime::PlacementError::StateFrozen)
    );

    migration
        .abort(&mut placement, started.migration_id)
        .unwrap();
    assert!(placement
        .issue_lease(StateId::from("state"), &pod_id)
        .is_ok());
}

fn migration_fixture() -> (
    PodPlacementController,
    ptr_runtime::FencedStateLease,
    KvMigrationController,
) {
    let mut placement = PodPlacementController::default();
    placement.register_node(node("node-a", "cuda:0")).unwrap();
    placement.register_node(node("node-b", "cuda:0")).unwrap();
    let source_placement = placement
        .assign(
            PodId::from("pod"),
            ArtifactId::from("artifact"),
            Generation(1),
            NodeId::from("node-a"),
            DeviceId::from("cuda:0"),
            1,
        )
        .unwrap();
    let source = placement
        .issue_lease(StateId::from("state"), &source_placement.pod_id)
        .unwrap();
    (placement, source, KvMigrationController::default())
}

fn target(placement: &mut PodPlacementController) -> ptr_runtime::FencedStateLease {
    let target = placement
        .assign(
            PodId::from("pod"),
            ArtifactId::from("artifact"),
            Generation(1),
            NodeId::from("node-b"),
            DeviceId::from("cuda:0"),
            1,
        )
        .unwrap();
    placement
        .issue_lease(StateId::from("state"), &target.pod_id)
        .unwrap()
}
