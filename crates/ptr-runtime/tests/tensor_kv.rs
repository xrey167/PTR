use ptr_pods::{InMemoryKvTensorBackend, KvTensorDType, KvTensorSchema, TensorRef};
use ptr_runtime::{
    DeviceHealth, DeviceRecord, ExecutionScope, ManagedKvRegistry, NodeHealth, NodeRecord,
    PodPlacementController, PtrRuntime, ScopeCleanupCoordinator, ScopeState, TensorKvError,
};
use ptr_types::{
    AdapterVersion, ArtifactId, DeviceId, Generation, ModelVersion, NodeId, PodId, ProjectId,
    ScopeId, ScopeLeaseBinding, SessionId, StateId, Timestamp,
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
