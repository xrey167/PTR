use ptr_pods::{
    InMemoryKvTensorBackend, InMemoryPagedKvBackend, KvPageSize, KvTensorDType, KvTensorSchema,
    KvTierBinding, TensorRef,
};
use ptr_runtime::{
    DeviceHealth, DeviceRecord, ExecutionScope, ManagedKvPageContext, ManagedKvRegistry,
    NodeHealth, NodeRecord, PodPlacementController, PtrRuntime, ScopeCleanupCoordinator,
    ScopeState, TensorKvError,
};
use ptr_types::{
    AdapterVersion, ArtifactId, DeviceId, Generation, ModelVersion, NodeId, PodId, PrincipalId,
    ProjectId, Revision, ScopeId, ScopeLeaseBinding, SessionId, StateId, Timestamp,
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
fn registry_binds_real_backend_cache_to_opaque_handle_and_lease() {
    let mut placement = PodPlacementController::default();
    placement.register_node(node("node-a", "cuda:0")).unwrap();
    let pod = placement
        .assign(
            PodId::from("pod"),
            ArtifactId::from("artifact"),
            Generation(1),
            NodeId::from("node-a"),
            DeviceId::from("cuda:0"),
            1,
        )
        .unwrap();
    let mut registry = ManagedKvRegistry::new(InMemoryKvTensorBackend);
    let handle = registry
        .allocate(
            &mut placement,
            StateId::from("state"),
            &pod.pod_id,
            schema("cuda:0"),
            4,
        )
        .unwrap();
    registry
        .append(
            &placement,
            &handle,
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
    assert_eq!(registry.metadata(&handle).unwrap().sequence_length, 1);
    assert_eq!(
        registry
            .snapshot(&placement, &handle)
            .unwrap()
            .sequence_length,
        1
    );
}

#[test]
fn stale_placement_fences_cache_append_and_release() {
    let mut placement = PodPlacementController::default();
    placement.register_node(node("node-a", "cuda:0")).unwrap();
    placement.register_node(node("node-b", "cuda:1")).unwrap();
    let pod = placement
        .assign(
            PodId::from("pod"),
            ArtifactId::from("artifact"),
            Generation(1),
            NodeId::from("node-a"),
            DeviceId::from("cuda:0"),
            1,
        )
        .unwrap();
    let mut registry = ManagedKvRegistry::new(InMemoryKvTensorBackend);
    let handle = registry
        .allocate(
            &mut placement,
            StateId::from("state"),
            &pod.pod_id,
            schema("cuda:0"),
            4,
        )
        .unwrap();
    placement
        .assign(
            pod.pod_id.clone(),
            ArtifactId::from("artifact"),
            Generation(1),
            NodeId::from("node-b"),
            DeviceId::from("cuda:1"),
            1,
        )
        .unwrap();
    let result = registry.append(
        &placement,
        &handle,
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
    );
    assert_eq!(result, Err(TensorKvError::StaleLease));
    assert_eq!(
        registry.release(&mut placement, &handle),
        Err(TensorKvError::StaleLease)
    );
}

#[test]
fn invalidated_handle_cannot_reenter_backend() {
    let mut placement = PodPlacementController::default();
    placement.register_node(node("node-a", "cuda:0")).unwrap();
    let pod = placement
        .assign(
            PodId::from("pod"),
            ArtifactId::from("artifact"),
            Generation(1),
            NodeId::from("node-a"),
            DeviceId::from("cuda:0"),
            1,
        )
        .unwrap();
    let mut registry = ManagedKvRegistry::new(InMemoryKvTensorBackend);
    let handle = registry
        .allocate(
            &mut placement,
            StateId::from("state"),
            &pod.pod_id,
            schema("cuda:0"),
            4,
        )
        .unwrap();
    registry.invalidate(&mut placement, &handle).unwrap();
    assert_eq!(
        registry.snapshot(&placement, &handle),
        Err(TensorKvError::Invalidated)
    );
}

struct RecoveryCoordinator<'a> {
    placement: &'a mut PodPlacementController,
    registry: &'a mut ManagedKvRegistry<ptr_pods::InMemoryKvTensorBackend>,
    handle: ptr_runtime::ManagedKvHandle,
    steps: Vec<&'static str>,
}

impl ScopeCleanupCoordinator for RecoveryCoordinator<'_> {
    fn stop_intake(&mut self, _: &ExecutionScope) -> Result<(), ptr_runtime::CleanupError> {
        self.steps.push("stop_intake");
        Ok(())
    }

    fn cancel_children(&mut self, _: &ExecutionScope) -> Result<(), ptr_runtime::CleanupError> {
        self.steps.push("cancel_children");
        Ok(())
    }

    fn drain_queues(&mut self, _: &ExecutionScope) -> Result<(), ptr_runtime::CleanupError> {
        self.steps.push("drain_queues");
        Ok(())
    }

    fn release_pod_lease(&mut self, _: &ExecutionScope) -> Result<(), ptr_runtime::CleanupError> {
        self.steps.push("release_pod_lease");
        self.registry
            .invalidate(self.placement, &self.handle)
            .map_err(|error| ptr_runtime::CleanupError {
                step: "release_pod_lease",
                message: format!("KV invalidation failed: {error:?}"),
            })
    }

    fn release_resource_lease(
        &mut self,
        _: &ExecutionScope,
    ) -> Result<(), ptr_runtime::CleanupError> {
        self.steps.push("release_resource_lease");
        Ok(())
    }

    fn close_session(&mut self, _: &ExecutionScope) -> Result<(), ptr_runtime::CleanupError> {
        self.steps.push("close_session");
        Ok(())
    }
}

#[test]
fn uncertain_request_recovery_fences_old_kv_and_allows_new_root_state() {
    let mut placement = PodPlacementController::default();
    placement.register_node(node("node-a", "cuda:0")).unwrap();
    let pod = placement
        .assign(
            PodId::from("pod"),
            ArtifactId::from("artifact"),
            Generation(1),
            NodeId::from("node-a"),
            DeviceId::from("cuda:0"),
            1,
        )
        .unwrap();
    let mut registry = ManagedKvRegistry::new(ptr_pods::InMemoryKvTensorBackend);
    let old_handle = registry
        .allocate(
            &mut placement,
            StateId::from("old-state"),
            &pod.pod_id,
            schema("cuda:0"),
            4,
        )
        .unwrap();
    let old_metadata = registry.metadata(&old_handle).unwrap();
    let scope_id = ScopeId::from("pod-call");
    let scope = ExecutionScope {
        id: scope_id.clone(),
        parent: None,
        session: SessionId::from("session"),
        project: ProjectId::from("project"),
        created_at: Timestamp(1),
        deadline: Some(Timestamp(10)),
        state: ScopeState::Created,
        cancellation_requested: false,
    };
    let scope_lease = ScopeLeaseBinding {
        state_id: Some(StateId::from("old-state")),
        pod_id: Some(PodId::from("pod")),
        placement_epoch: Some(old_metadata.placement_epoch.0),
        fencing_token: Some(old_metadata.fencing_token.0),
        generation: Some(Generation(1)),
        resource_lease_id: Some("resource-old".into()),
    };
    let mut runtime = PtrRuntime::new(ptr_config::PtrConfig::default()).unwrap();
    runtime.create_scope(scope, scope_lease.clone()).unwrap();
    runtime
        .transition_scope(&scope_id, ScopeState::Admitted, scope_lease.clone(), None)
        .unwrap();
    runtime
        .transition_scope(&scope_id, ScopeState::Started, scope_lease.clone(), None)
        .unwrap();

    let mut recovery = RecoveryCoordinator {
        placement: &mut placement,
        registry: &mut registry,
        handle: old_handle.clone(),
        steps: Vec::new(),
    };
    runtime
        .recover_uncertain_request(&scope_id, 777, &mut recovery, scope_lease)
        .unwrap();
    assert_eq!(recovery.steps.len(), 6);
    drop(recovery);
    assert_eq!(
        registry.snapshot(&placement, &old_handle),
        Err(TensorKvError::Invalidated)
    );

    let next_placement = placement
        .assign(
            PodId::from("pod"),
            ArtifactId::from("artifact"),
            Generation(2),
            NodeId::from("node-a"),
            DeviceId::from("cuda:0"),
            1,
        )
        .unwrap();
    let new_handle = registry
        .recompute(
            &mut placement,
            StateId::from("new-state"),
            &next_placement.pod_id,
            schema("cuda:0"),
            4,
        )
        .unwrap();
    let new_metadata = registry.metadata(&new_handle).unwrap();
    assert_ne!(new_metadata.fencing_token, old_metadata.fencing_token);
    assert_eq!(new_metadata.generation, Generation(2));
}

#[test]
fn managed_paged_cache_binds_runtime_lease_and_snapshots_without_blocking_continuation() {
    let mut placement = PodPlacementController::default();
    placement.register_node(node("node-a", "cuda:0")).unwrap();
    let pod = placement
        .assign(
            PodId::from("paged-pod"),
            ArtifactId::from("paged-artifact"),
            Generation(3),
            NodeId::from("node-a"),
            DeviceId::from("cuda:0"),
            1,
        )
        .unwrap();
    let mut registry =
        ManagedKvRegistry::new(InMemoryPagedKvBackend::new(KvPageSize::new(2).unwrap()));
    let handle = registry
        .allocate_paged(
            &mut placement,
            StateId::from("paged-state"),
            &pod.pod_id,
            schema("cuda:0"),
            8,
            ManagedKvPageContext {
                execution_manifest: [9; 32],
                principal: PrincipalId::from("alice"),
                page_tokens: KvPageSize::new(2).unwrap(),
            },
        )
        .unwrap();
    registry
        .append(
            &placement,
            &handle,
            &[TensorRef {
                layer: 0,
                shape: vec![3, 2],
                values: vec![1.0; 6],
            }],
            &[TensorRef {
                layer: 0,
                shape: vec![3, 2],
                values: vec![2.0; 6],
            }],
        )
        .unwrap();
    let lease = registry.seal_online_snapshot(&placement, &handle).unwrap();
    registry
        .append(
            &placement,
            &handle,
            &[TensorRef {
                layer: 0,
                shape: vec![1, 2],
                values: vec![3.0; 2],
            }],
            &[TensorRef {
                layer: 0,
                shape: vec![1, 2],
                values: vec![4.0; 2],
            }],
        )
        .unwrap();
    let snapshot = registry.materialize_online_snapshot(&lease).unwrap();
    assert_eq!(snapshot.sequence_length, 3);
    assert_eq!(registry.metadata(&handle).unwrap().sequence_length, 4);
    assert_eq!(registry.page_metrics(&handle).unwrap().cow_copies, 1);

    let old_token = registry.metadata(&handle).unwrap().fencing_token;
    registry.invalidate(&mut placement, &handle).unwrap();
    assert_eq!(
        registry.append(
            &placement,
            &handle,
            &[TensorRef {
                layer: 0,
                shape: vec![1, 2],
                values: vec![0.0; 2],
            }],
            &[TensorRef {
                layer: 0,
                shape: vec![1, 2],
                values: vec![0.0; 2],
            }]
        ),
        Err(TensorKvError::Invalidated)
    );
    let restored = registry
        .restore_paged_snapshot(
            &mut placement,
            StateId::from("restored-paged-state"),
            &pod.pod_id,
            snapshot,
        )
        .unwrap();
    let restored_metadata = registry.metadata(&restored).unwrap();
    assert_eq!(restored_metadata.sequence_length, 3);
    assert_ne!(restored_metadata.fencing_token, old_token);
}

#[test]
fn online_snapshot_revalidates_fencing_before_paged_tier_publication() {
    let mut placement = PodPlacementController::default();
    placement.register_node(node("node-a", "cuda:0")).unwrap();
    let pod = placement
        .assign(
            PodId::from("paged-pod"),
            ArtifactId::from("paged-artifact"),
            Generation(1),
            NodeId::from("node-a"),
            DeviceId::from("cuda:0"),
            1,
        )
        .unwrap();
    let mut registry =
        ManagedKvRegistry::new(InMemoryPagedKvBackend::new(KvPageSize::new(2).unwrap()));
    let handle = registry
        .allocate_paged(
            &mut placement,
            StateId::from("snapshot-state"),
            &pod.pod_id,
            schema("cuda:0"),
            4,
            ManagedKvPageContext {
                execution_manifest: [4; 32],
                principal: PrincipalId::from("alice"),
                page_tokens: KvPageSize::new(2).unwrap(),
            },
        )
        .unwrap();
    registry
        .append(
            &placement,
            &handle,
            &[TensorRef {
                layer: 0,
                shape: vec![2, 2],
                values: vec![1.0; 4],
            }],
            &[TensorRef {
                layer: 0,
                shape: vec![2, 2],
                values: vec![2.0; 4],
            }],
        )
        .unwrap();
    let snapshot = registry.seal_online_snapshot(&placement, &handle).unwrap();
    let binding = KvTierBinding {
        logical_id: snapshot.state_id().0.clone(),
        generation: snapshot.generation(),
        revision: Revision(1),
        context_digest: [6; 32],
        placement_epoch: snapshot.placement_epoch().0,
        fencing_token: snapshot.fencing_token().0,
    };
    let object = registry
        .prepare_online_snapshot_to_tier(&placement, &snapshot, &binding)
        .unwrap();
    assert_eq!(object.chunks.len(), 2);

    placement
        .assign(
            pod.pod_id.clone(),
            ArtifactId::from("paged-artifact"),
            Generation(1),
            NodeId::from("node-a"),
            DeviceId::from("cuda:0"),
            1,
        )
        .unwrap();
    assert_eq!(
        registry.prepare_online_snapshot_to_tier(&placement, &snapshot, &binding),
        Err(TensorKvError::StaleLease)
    );
}
