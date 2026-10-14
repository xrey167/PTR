use ptr_pods::{
    InMemoryKvTensorBackend, KvTensorBackend, KvTensorDType, KvTensorSchema, TensorRef,
};
use ptr_runtime::{
    DeviceHealth, DeviceRecord, KvMigrationController, MigrationError, MigrationState, NodeHealth,
    NodeRecord, PodPlacementController, ProjectPlacementRequest,
};
use ptr_types::{
    AdapterVersion, ArtifactId, DeviceId, Generation, ModelVersion, NodeId, PodId, ProjectId,
    StateId, Timestamp,
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
        key_value_heads: 1,
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

fn assign_in(
    placement: &mut PodPlacementController,
    project: &ProjectId,
    pod: &str,
    node: &str,
    device: &str,
) {
    placement
        .assign_in(ProjectPlacementRequest {
            project: project.clone(),
            pod_id: PodId::from(pod),
            artifact_id: ArtifactId::from("artifact"),
            generation: Generation(1),
            node_id: NodeId::from(node),
            device_id: DeviceId::from(device),
            required_vram: 1,
        })
        .unwrap();
}

/// Two nodes; `pod` placed in `project` on node-a with a lease of `state`.
fn project_fixture(project: &ProjectId) -> (PodPlacementController, ptr_runtime::FencedStateLease) {
    let mut placement = PodPlacementController::default();
    placement.register_node(node("node-a", "cuda:0")).unwrap();
    placement.register_node(node("node-b", "cuda:1")).unwrap();
    assign_in(&mut placement, project, "pod", "node-a", "cuda:0");
    let source = placement
        .issue_lease_in(project, StateId::from("state"), &PodId::from("pod"))
        .unwrap();
    (placement, source)
}

#[test]
fn a_project_migration_lifts_its_own_freeze_on_abort() {
    let project = ProjectId::from("a");
    let (mut placement, source) = project_fixture(&project);
    let mut migration = KvMigrationController::default();
    let started = migration.begin(&mut placement, source).unwrap();
    assert_eq!(
        placement.issue_lease_in(&project, StateId::from("state"), &PodId::from("pod")),
        Err(ptr_runtime::PlacementError::StateFrozen)
    );
    migration
        .abort(&mut placement, started.migration_id)
        .unwrap();
    assert!(placement
        .issue_lease_in(&project, StateId::from("state"), &PodId::from("pod"))
        .is_ok());
}

#[test]
fn a_project_migration_lifts_its_own_freeze_on_commit() {
    let project = ProjectId::from("a");
    let (mut placement, source) = project_fixture(&project);
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
    // Issuing the project target lease below clears the project's own freeze on the state (the
    // placement epoch differs), so a bystander is what shows which namespace `commit` unfreezes:
    // a legacy freeze of the same state id, which only a legacy unfreeze would lift.
    placement
        .assign(
            PodId::from("legacy-pod"),
            ArtifactId::from("artifact"),
            Generation(1),
            NodeId::from("node-a"),
            DeviceId::from("cuda:0"),
            1,
        )
        .unwrap();
    let legacy_source = placement
        .issue_lease(StateId::from("state"), &PodId::from("legacy-pod"))
        .unwrap();
    KvMigrationController::default()
        .begin(&mut placement, legacy_source)
        .unwrap();
    let mut migration = KvMigrationController::default();
    let started = migration.begin(&mut placement, source).unwrap();
    migration.snapshot(started.migration_id, &snapshot).unwrap();
    // The target is another Pod on node-b.
    assign_in(&mut placement, &project, "pod-b", "node-b", "cuda:1");
    let target = placement
        .issue_lease_in(&project, StateId::from("state"), &PodId::from("pod-b"))
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
    assert_eq!(
        placement.issue_lease(StateId::from("state"), &PodId::from("legacy-pod")),
        Err(ptr_runtime::PlacementError::StateFrozen)
    );
    migration
        .commit(&mut placement, started.migration_id, &target)
        .unwrap();
    assert!(placement
        .issue_lease_in(&project, StateId::from("state"), &PodId::from("pod"))
        .is_ok());
    // The project commit lifted the project's freeze only: the legacy freeze is still in force.
    assert_eq!(
        placement.issue_lease(StateId::from("state"), &PodId::from("legacy-pod")),
        Err(ptr_runtime::PlacementError::StateFrozen)
    );
}

#[test]
fn a_legacy_target_lease_cannot_restore_or_commit_a_project_migration() {
    let project = ProjectId::from("a");
    let (mut placement, source) = project_fixture(&project);
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
    migration.snapshot(started.migration_id, &snapshot).unwrap();
    // A legacy-namespace lease of the same state and generation on the other node.
    placement
        .assign(
            PodId::from("legacy-pod"),
            ArtifactId::from("artifact"),
            Generation(1),
            NodeId::from("node-b"),
            DeviceId::from("cuda:1"),
            1,
        )
        .unwrap();
    let legacy_target = placement
        .issue_lease(StateId::from("state"), &PodId::from("legacy-pod"))
        .unwrap();
    let refused = Err(MigrationError::Placement(
        ptr_runtime::PlacementError::StaleLease,
    ));
    assert_eq!(
        migration
            .restore(
                &mut placement,
                started.migration_id,
                &legacy_target,
                &backend,
                &snapshot
            )
            .map(|_| ()),
        refused
    );
    assert_eq!(
        migration.commit(&mut placement, started.migration_id, &legacy_target),
        Err(MigrationError::InvalidState(MigrationState::Snapshotted))
    );
    // The project's freeze is still in force.
    assert_eq!(
        placement.issue_lease_in(&project, StateId::from("state"), &PodId::from("pod")),
        Err(ptr_runtime::PlacementError::StateFrozen)
    );
}

#[test]
fn a_project_migration_leaves_a_legacy_freeze_of_the_same_state_alone_and_the_reverse() {
    let project = ProjectId::from("a");
    let (mut placement, project_source) = project_fixture(&project);
    // A legacy placement and lease of the same state id, frozen by a legacy migration.
    placement
        .assign(
            PodId::from("legacy-pod"),
            ArtifactId::from("artifact"),
            Generation(1),
            NodeId::from("node-b"),
            DeviceId::from("cuda:1"),
            1,
        )
        .unwrap();
    let legacy_source = placement
        .issue_lease(StateId::from("state"), &PodId::from("legacy-pod"))
        .unwrap();
    let mut legacy = KvMigrationController::default();
    let legacy_started = legacy.begin(&mut placement, legacy_source).unwrap();
    let mut scoped = KvMigrationController::default();
    let project_started = scoped.begin(&mut placement, project_source).unwrap();

    // Aborting the project migration leaves the legacy freeze in force.
    scoped
        .abort(&mut placement, project_started.migration_id)
        .unwrap();
    assert_eq!(
        placement.issue_lease(StateId::from("state"), &PodId::from("legacy-pod")),
        Err(ptr_runtime::PlacementError::StateFrozen)
    );
    assert!(placement
        .issue_lease_in(&project, StateId::from("state"), &PodId::from("pod"))
        .is_ok());

    // And the reverse: freeze the project state again, abort the legacy migration.
    let project_source = placement
        .issue_lease_in(&project, StateId::from("state"), &PodId::from("pod"))
        .unwrap();
    scoped.begin(&mut placement, project_source).unwrap();
    legacy
        .abort(&mut placement, legacy_started.migration_id)
        .unwrap();
    assert_eq!(
        placement.issue_lease_in(&project, StateId::from("state"), &PodId::from("pod")),
        Err(ptr_runtime::PlacementError::StateFrozen)
    );
    assert!(placement
        .issue_lease(StateId::from("state"), &PodId::from("legacy-pod"))
        .is_ok());
}
