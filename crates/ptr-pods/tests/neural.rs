use ptr_pods::{
    ArtifactCatalog, DynPod, ExecutorFactory, InMemoryArtifactCatalog, InMemoryResourceGovernor,
    LeaseState, NeuralPodAdapter, NeuralPodDescriptor, NeuralPodExecutor, NeuralPodLease,
    NeuralPodType, PodLifecycle, PodManifest, PodRegistry, PodWireRequest, PodWireResponse,
    ReferenceExecutorFactory, ReferenceNeuralExecutor, ResourceError, ResourceGovernor,
    ResourceRequirements, TensorContract, TensorDType, TransportError, TypedPayload,
};
use ptr_types::ProjectId;
use ptr_types::{ArtifactId, CapabilityId, Generation, PodId, RequestId, Revision, TypeId};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

fn descriptor() -> NeuralPodDescriptor {
    NeuralPodDescriptor {
        artifact_id: ArtifactId::from("artifact-1"),
        pod_id: PodId::from("model-1"),
        manifest_hash: [7; 32],
        generation: Generation(1),
        pod_type: NeuralPodType::Model,
        input_schema: TypeId::from("text"),
        output_schema: TypeId::from("answer"),
        capabilities: vec![CapabilityId::from("answer")],
        provenance: vec!["raw:e1".into()],
        lifecycle: PodLifecycle::Ready,
        resources: ResourceRequirements {
            ram_bytes: 1,
            vram_bytes: 2,
            device: None,
            max_concurrency: 1,
        },
        tensor: TensorContract {
            dtype: TensorDType::F32,
            input_len: 1,
            output_len: 1,
        },
    }
}

#[test]
fn descriptor_requires_provenance_and_ready_lifecycle() {
    let pod = descriptor();
    assert!(pod.can_activate());
    assert!(pod.admits_output_type(&TypeId::from("answer")));
    assert!(!pod.admits_output_type(&TypeId::from("other")));
}

#[test]
fn descriptor_without_provenance_cannot_activate() {
    let mut pod = descriptor();
    pod.provenance.clear();
    assert!(!pod.can_activate());
}

#[test]
fn lease_rejects_parallel_use_and_revocation() {
    let mut lease = NeuralPodLease {
        pod_id: PodId::from("model-1"),
        generation: Generation(1),
        state: LeaseState::Ready,
    };
    lease.begin().unwrap();
    assert!(lease.begin().is_err());
    lease.finish().unwrap();
    lease.revoke();
    assert!(lease.begin().is_err());
}

#[test]
fn descriptor_rejects_zero_manifest_hash() {
    let mut pod = descriptor();
    pod.manifest_hash = [0; 32];
    assert!(!pod.can_activate());
}

#[test]
fn artifact_catalog_rejects_revoked_generation() {
    let pod = descriptor();
    let mut catalog = InMemoryArtifactCatalog::default();
    catalog.admit(pod.clone()).unwrap();
    catalog.verify_lineage(&pod).unwrap();
    catalog
        .invalidate(&pod.artifact_id, pod.generation)
        .unwrap();
    assert!(catalog.resolve(&pod.artifact_id, pod.generation).is_err());
}

#[test]
fn artifact_catalog_rejects_conflicting_same_generation() {
    let pod = descriptor();
    let mut other = pod.clone();
    other.provenance = vec!["different".into()];
    let mut catalog = InMemoryArtifactCatalog::default();
    catalog.admit(pod).unwrap();
    assert!(matches!(
        catalog.admit(other),
        Err(ptr_pods::ArtifactError::ConflictingGeneration)
    ));
}

#[test]
fn resource_governor_enforces_capacity_and_release() {
    let requirements = ResourceRequirements {
        ram_bytes: 10,
        vram_bytes: 20,
        device: None,
        max_concurrency: 1,
    };
    let mut governor = InMemoryResourceGovernor::new(10, 20);
    let lease = governor.reserve(&requirements).unwrap();
    assert!(governor.reserve(&requirements).is_err());
    governor.release(lease).unwrap();
    assert!(governor.reserve(&requirements).is_ok());
}

#[test]
fn pod_wire_binds_manifest_generation_and_types() {
    let pod = descriptor();
    let request = PodWireRequest {
        request_id: RequestId::from("req-1"),
        artifact_id: pod.artifact_id.clone(),
        manifest_hash: pod.manifest_hash,
        generation: pod.generation,
        input_type: pod.input_schema.clone(),
        expect_protocol: 1,
        revision: Revision(7),
        payload: TypedPayload {
            type_id: pod.input_schema.clone(),
            bytes: b"input".to_vec(),
        },
    };
    request.validate(&pod, 1).unwrap();
    let response = PodWireResponse {
        request_id: request.request_id.clone(),
        generation: pod.generation,
        revision: request.revision,
        payload: TypedPayload {
            type_id: pod.output_schema.clone(),
            bytes: b"output".to_vec(),
        },
    };
    response.validate(&request, &pod).unwrap();
    let mut stale = response;
    stale.revision = Revision(8);
    assert_eq!(
        stale.validate(&request, &pod),
        Err(TransportError::RevisionMismatch)
    );
}

#[test]
fn reference_executor_enforces_typed_inference_and_release() {
    let pod = descriptor();
    let executor = ReferenceNeuralExecutor::new(pod.clone()).unwrap();
    let mut lease = executor.activate(&pod).unwrap();
    let output = executor
        .infer(
            &mut lease,
            TypedPayload {
                type_id: pod.input_schema.clone(),
                bytes: b"hello".to_vec(),
            },
        )
        .unwrap();
    assert_eq!(output.type_id, pod.output_schema);
    executor.release(lease.clone()).unwrap();
    executor.release(lease).unwrap();
}

#[test]
fn terminal_lifecycle_states_cannot_transition_to_another_state() {
    let mut pod = descriptor();
    pod.transition_to(PodLifecycle::Released).unwrap();
    pod.transition_to(PodLifecycle::Released).unwrap();
    assert!(pod.transition_to(PodLifecycle::Ready).is_err());

    let mut revoked = descriptor();
    revoked.transition_to(PodLifecycle::Revoked).unwrap();
    revoked.transition_to(PodLifecycle::Revoked).unwrap();
    assert!(revoked.transition_to(PodLifecycle::Failed).is_err());
}

#[test]
fn resource_release_is_idempotent_but_rejects_mismatched_retries() {
    let requirements = ResourceRequirements {
        ram_bytes: 1,
        vram_bytes: 0,
        device: None,
        max_concurrency: 1,
    };
    let mut governor = InMemoryResourceGovernor::new(1, 0);
    let lease = governor.reserve(&requirements).unwrap();
    governor.release(lease.clone()).unwrap();
    governor.release(lease.clone()).unwrap();

    let mut mismatch = lease;
    mismatch.requirements.ram_bytes = 2;
    assert_eq!(
        governor.release(mismatch),
        Err(ResourceError::LeaseMismatch)
    );
}

#[test]
fn adapter_registers_and_routes_through_existing_registry() {
    let mut pod = descriptor();
    let manifest = PodManifest {
        project: ProjectId::from("project"),
        id: pod.pod_id.clone(),
        capabilities: vec![CapabilityId::from("answer")],
        accepts: vec![pod.input_schema.clone()],
        produces: vec![pod.output_schema.clone()],
        effects: vec![],
        protocol_version: 1,
    };
    pod.manifest_hash = manifest.digest();
    let adapter = NeuralPodAdapter::new(
        manifest,
        pod.clone(),
        ReferenceNeuralExecutor::new(pod.clone()).unwrap(),
    )
    .unwrap();
    let mut registry = PodRegistry::default();
    registry.register(Arc::new(adapter));
    let resolved = registry
        .resolve(
            &ProjectId::from("project"),
            &CapabilityId::from("answer"),
            &pod.input_schema,
        )
        .unwrap();
    let output = resolved
        .invoke(TypedPayload {
            type_id: pod.input_schema,
            bytes: b"input".to_vec(),
        })
        .unwrap();
    assert_eq!(output.type_id, pod.output_schema);
}

#[test]
fn reference_executor_factory_admits_descriptor() {
    let pod = descriptor();
    let executor = ReferenceExecutorFactory.create(&pod).unwrap();
    assert!(executor.health());
}

#[test]
fn adapter_aborts_failed_inference_to_recover_the_lease() {
    struct FailingExecutor {
        aborted: Arc<AtomicUsize>,
    }

    impl NeuralPodExecutor for FailingExecutor {
        type Lease = NeuralPodLease;
        type Error = &'static str;

        fn activate(&self, descriptor: &NeuralPodDescriptor) -> Result<Self::Lease, Self::Error> {
            Ok(NeuralPodLease {
                pod_id: descriptor.pod_id.clone(),
                generation: descriptor.generation,
                state: LeaseState::Ready,
            })
        }

        fn infer(
            &self,
            _lease: &mut Self::Lease,
            _input: TypedPayload,
        ) -> Result<TypedPayload, Self::Error> {
            Err("inference failed")
        }

        fn release(&self, mut lease: Self::Lease) -> Result<(), Self::Error> {
            lease.release().map_err(|_| "release failed")
        }

        fn abort(&self, lease: Self::Lease) -> Result<(), Self::Error> {
            self.aborted.fetch_add(1, Ordering::SeqCst);
            self.release(lease)
        }

        fn health(&self) -> bool {
            true
        }
    }

    let mut descriptor = descriptor();
    let manifest = PodManifest {
        project: ProjectId::from("project"),
        id: descriptor.pod_id.clone(),
        capabilities: vec![CapabilityId::from("answer")],
        accepts: vec![descriptor.input_schema.clone()],
        produces: vec![descriptor.output_schema.clone()],
        effects: vec![],
        protocol_version: 1,
    };
    descriptor.manifest_hash = manifest.digest();
    let aborted = Arc::new(AtomicUsize::new(0));
    let adapter = NeuralPodAdapter::new(
        manifest,
        descriptor.clone(),
        FailingExecutor {
            aborted: Arc::clone(&aborted),
        },
    )
    .unwrap();

    let error = adapter
        .invoke(TypedPayload {
            type_id: descriptor.input_schema,
            bytes: b"input".to_vec(),
        })
        .unwrap_err();
    assert!(error.contains("inference failed"));
    assert_eq!(aborted.load(Ordering::SeqCst), 1);
}

#[test]
fn adapter_rejects_manifest_input_type_mismatch() {
    let pod = descriptor();
    let mut manifest = PodManifest {
        project: ProjectId::from("project"),
        id: pod.pod_id.clone(),
        capabilities: vec![],
        accepts: vec![TypeId::from("wrong-input")],
        produces: vec![pod.output_schema.clone()],
        effects: vec![],
        protocol_version: 1,
    };
    manifest.protocol_version = 1;
    assert!(matches!(
        NeuralPodAdapter::new(
            manifest,
            pod.clone(),
            ReferenceNeuralExecutor::new(pod).unwrap(),
        ),
        Err(ptr_pods::AdapterError::ManifestInputMismatch)
    ));
}

#[test]
fn adapter_rejects_manifest_output_type_mismatch() {
    let pod = descriptor();
    let manifest = PodManifest {
        project: ProjectId::from("project"),
        id: pod.pod_id.clone(),
        capabilities: vec![],
        accepts: vec![pod.input_schema.clone()],
        produces: vec![TypeId::from("wrong-output")],
        effects: vec![],
        protocol_version: 1,
    };
    assert!(matches!(
        NeuralPodAdapter::new(
            manifest,
            pod.clone(),
            ReferenceNeuralExecutor::new(pod).unwrap(),
        ),
        Err(ptr_pods::AdapterError::ManifestOutputMismatch)
    ));
}

#[test]
fn adapter_rejects_manifest_hash_mismatch() {
    let pod = descriptor();
    let manifest = PodManifest {
        project: ProjectId::from("project"),
        id: pod.pod_id.clone(),
        capabilities: vec![],
        accepts: vec![pod.input_schema.clone()],
        produces: vec![pod.output_schema.clone()],
        effects: vec![],
        protocol_version: 1,
    };
    assert!(matches!(
        NeuralPodAdapter::new(
            manifest,
            pod.clone(),
            ReferenceNeuralExecutor::new(pod).unwrap(),
        ),
        Err(ptr_pods::AdapterError::ManifestHashMismatch)
    ));
}
