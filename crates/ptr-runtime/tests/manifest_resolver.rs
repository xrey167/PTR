use ptr_config::PtrConfig;
use ptr_memory::{ConflictState, Freshness, KnowledgeLifecycle, KnowledgeObject, KnowledgeStore};
use ptr_pods::{
    ArtifactCatalog, ExecutionManifest, InMemoryArtifactCatalog, LineageBinding,
    NeuralPodDescriptor, NeuralPodType, PodLifecycle, ResourceRequirements, TensorContract,
    TensorDType,
};
use ptr_runtime::ManifestBindingAuthority;
use ptr_runtime::PtrRuntime;
use ptr_types::{
    ArtifactId as ArtifactKey, EntityId, Generation, KnowledgeObjectId, NamespaceId, PrincipalId,
    Probability, ProvenanceRef, Revision,
};
use std::time::{SystemTime, UNIX_EPOCH};

fn probability(value: f32) -> Probability {
    Probability::new(value).unwrap()
}

#[derive(Clone)]
struct Authority {
    principal: PrincipalId,
    policy: Revision,
    snapshot_revision: Revision,
    snapshot_digest: [u8; 32],
}

impl ManifestBindingAuthority for Authority {
    fn principal_admitted(&self, principal: &PrincipalId) -> bool {
        principal == &self.principal
    }

    fn current_policy_revision(&self) -> Revision {
        self.policy
    }

    fn snapshot_digest(&self, revision: Revision) -> Option<[u8; 32]> {
        (revision == self.snapshot_revision).then_some(self.snapshot_digest)
    }
}

fn knowledge() -> (KnowledgeStore, KnowledgeObject) {
    let object = KnowledgeObject {
        id: KnowledgeObjectId::from("fact"),
        namespace: NamespaceId::from("ptr.test"),
        entities: vec![EntityId::from("entity")],
        relations: vec![],
        sources: vec![],
        provenance: vec![ProvenanceRef {
            source: ptr_types::EvidenceId::from("test"),
            note: None,
        }],
        generation: Generation(1),
        supersedes: None,
        dependencies: vec![],
        confidence: probability(1.0),
        relevance: probability(1.0),
        freshness: Freshness::Current,
        conflict: ConflictState::None,
        lifecycle: KnowledgeLifecycle::Hot,
        retrieval_keys: vec![],
        token_cost: 1,
        validity: Default::default(),
    };
    let mut store = KnowledgeStore::default();
    store.register_object(object.clone()).unwrap();
    (store, object)
}

fn descriptor() -> NeuralPodDescriptor {
    NeuralPodDescriptor {
        artifact_id: ArtifactKey::from("artifact-1"),
        pod_id: ptr_types::PodId::from("pod-1"),
        manifest_hash: [7; 32],
        generation: Generation(1),
        pod_type: NeuralPodType::Model,
        input_schema: ptr_types::TypeId::from("input"),
        output_schema: ptr_types::TypeId::from("output"),
        capabilities: vec![ptr_types::CapabilityId::from("infer")],
        provenance: vec!["raw:test".into()],
        lifecycle: PodLifecycle::Ready,
        resources: ResourceRequirements {
            ram_bytes: 1,
            vram_bytes: 1,
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

fn manifest(object: &KnowledgeObject, pod: &NeuralPodDescriptor) -> ExecutionManifest {
    ExecutionManifest::build(
        Generation(1),
        vec![LineageBinding {
            key: object.id.0.clone(),
            generation: object.generation,
            digest: object.content_digest(),
        }],
        vec![LineageBinding {
            key: pod.artifact_id.0.clone(),
            generation: pod.generation,
            digest: pod.lineage_digest(),
        }],
        vec!["raw:test".into()],
        None,
        Revision(0),
        [9; 32],
        PrincipalId::from("principal"),
        Revision(1),
    )
    .unwrap()
}

#[test]
fn runtime_resolves_manifest_against_knowledge_and_artifact_catalog() {
    let (knowledge, object) = knowledge();
    let pod = descriptor();
    let mut artifacts = InMemoryArtifactCatalog::default();
    artifacts.admit(pod.clone()).unwrap();
    let candidate = manifest(&object, &pod);
    let mut runtime = PtrRuntime::new(PtrConfig::default()).unwrap();

    let validated = runtime
        .admit_execution_manifest(
            candidate.clone(),
            &knowledge,
            &artifacts,
            ptr_types::Timestamp(0),
        )
        .unwrap();
    assert_eq!(validated.manifest, candidate);
    assert_eq!(
        runtime.resolve_execution_manifest(&candidate.manifest_digest),
        Ok(validated.clone())
    );

    // Re-admission is idempotent and cannot replace the admitted record.
    assert_eq!(
        runtime.admit_execution_manifest(
            candidate.clone(),
            &knowledge,
            &artifacts,
            ptr_types::Timestamp(0)
        ),
        Ok(validated)
    );
}

#[test]
fn runtime_rejects_lineage_tampering_and_revocation() {
    let (knowledge, object) = knowledge();
    let pod = descriptor();
    let mut artifacts = InMemoryArtifactCatalog::default();
    artifacts.admit(pod.clone()).unwrap();
    let mut candidate = manifest(&object, &pod);
    candidate.knowledge[0].digest = [8; 32];
    // The manifest's own digest must also be recomputed by build; tampering is
    // therefore rejected before any lineage lookup.
    assert!(runtime_error(candidate, &knowledge, &artifacts).is_err());

    let candidate = manifest(&object, &pod);
    let mut runtime = PtrRuntime::new(PtrConfig::default()).unwrap();
    runtime
        .admit_execution_manifest(
            candidate.clone(),
            &knowledge,
            &artifacts,
            ptr_types::Timestamp(0),
        )
        .unwrap();
    runtime
        .revoke_execution_manifest(candidate.manifest_digest)
        .unwrap();
    assert!(runtime
        .resolve_execution_manifest(&candidate.manifest_digest)
        .is_err());
    assert!(runtime
        .revoke_execution_manifest(candidate.manifest_digest)
        .is_ok());
}

fn runtime_error(
    candidate: ExecutionManifest,
    knowledge: &KnowledgeStore,
    artifacts: &InMemoryArtifactCatalog,
) -> Result<(), ptr_runtime::RuntimeManifestError> {
    let mut runtime = PtrRuntime::new(PtrConfig::default()).unwrap();
    runtime
        .admit_execution_manifest(candidate, knowledge, artifacts, ptr_types::Timestamp(0))
        .map(|_| ())
}

#[test]
fn runtime_rejects_unknown_knowledge_generation() {
    let (knowledge, object) = knowledge();
    let pod = descriptor();
    let mut artifacts = InMemoryArtifactCatalog::default();
    artifacts.admit(pod.clone()).unwrap();
    let mut candidate = manifest(&object, &pod);
    candidate.knowledge[0].generation = Generation(2);
    candidate.manifest_digest = [1; 32];
    let mut runtime = PtrRuntime::new(PtrConfig::default()).unwrap();
    assert!(runtime
        .admit_execution_manifest(candidate, &knowledge, &artifacts, ptr_types::Timestamp(0))
        .is_err());
}

#[test]
fn runtime_rejects_revoked_artifact_generation() {
    let (knowledge, object) = knowledge();
    let pod = descriptor();
    let mut artifacts = InMemoryArtifactCatalog::default();
    artifacts.admit(pod.clone()).unwrap();
    artifacts
        .invalidate(&ArtifactKey::from("artifact-1"), Generation(1))
        .unwrap();
    let candidate = manifest(&object, &pod);
    let mut runtime = PtrRuntime::new(PtrConfig::default()).unwrap();
    assert!(runtime
        .admit_execution_manifest(candidate, &knowledge, &artifacts, ptr_types::Timestamp(0))
        .is_err());
}

#[test]
fn runtime_resolves_and_rejects_reader_artifact_lineage() {
    let (knowledge, object) = knowledge();
    let pod = descriptor();
    let mut artifacts = InMemoryArtifactCatalog::default();
    artifacts.admit(pod.clone()).unwrap();

    let valid = ExecutionManifest::build(
        Generation(1),
        vec![LineageBinding {
            key: object.id.0.clone(),
            generation: object.generation,
            digest: object.content_digest(),
        }],
        vec![LineageBinding {
            key: pod.artifact_id.0.clone(),
            generation: pod.generation,
            digest: pod.lineage_digest(),
        }],
        vec!["raw:test".into()],
        Some(LineageBinding {
            key: pod.artifact_id.0.clone(),
            generation: pod.generation,
            digest: pod.lineage_digest(),
        }),
        Revision(0),
        [9; 32],
        PrincipalId::from("principal"),
        Revision(1),
    )
    .unwrap();
    let mut runtime = PtrRuntime::new(PtrConfig::default()).unwrap();
    assert!(runtime
        .admit_execution_manifest(valid, &knowledge, &artifacts, ptr_types::Timestamp(0))
        .is_ok());

    let invalid_reader = ExecutionManifest::build(
        Generation(1),
        vec![LineageBinding {
            key: object.id.0.clone(),
            generation: object.generation,
            digest: object.content_digest(),
        }],
        vec![LineageBinding {
            key: pod.artifact_id.0.clone(),
            generation: pod.generation,
            digest: pod.lineage_digest(),
        }],
        vec!["raw:test".into()],
        Some(LineageBinding {
            key: "reader-not-admitted".into(),
            generation: Generation(1),
            digest: pod.lineage_digest(),
        }),
        Revision(0),
        [9; 32],
        PrincipalId::from("principal"),
        Revision(1),
    )
    .unwrap();
    assert!(matches!(
        runtime.admit_execution_manifest(
            invalid_reader,
            &knowledge,
            &artifacts,
            ptr_types::Timestamp(0)
        ),
        Err(ptr_runtime::RuntimeManifestError::MissingReaderGeneration { .. })
    ));
}

#[test]
fn runtime_resolves_and_rejects_adapter_artifact_lineage() {
    let (knowledge, object) = knowledge();
    let pod = descriptor();
    let mut artifacts = InMemoryArtifactCatalog::default();
    artifacts.admit(pod.clone()).unwrap();
    let build = |key: &str| {
        ExecutionManifest::build_with_adapter(
            Generation(1),
            vec![LineageBinding {
                key: object.id.0.clone(),
                generation: object.generation,
                digest: object.content_digest(),
            }],
            vec![LineageBinding {
                key: pod.artifact_id.0.clone(),
                generation: pod.generation,
                digest: pod.lineage_digest(),
            }],
            vec!["raw:test".into()],
            None,
            Some(LineageBinding {
                key: key.into(),
                generation: Generation(1),
                digest: pod.lineage_digest(),
            }),
            Revision(0),
            [9; 32],
            PrincipalId::from("principal"),
            Revision(1),
        )
        .unwrap()
    };

    let mut runtime = PtrRuntime::new(PtrConfig::default()).unwrap();
    assert!(runtime
        .admit_execution_manifest(
            build("artifact-1"),
            &knowledge,
            &artifacts,
            ptr_types::Timestamp(0)
        )
        .is_ok());
    assert!(matches!(
        runtime.admit_execution_manifest(
            build("adapter-not-admitted"),
            &knowledge,
            &artifacts,
            ptr_types::Timestamp(0)
        ),
        Err(ptr_runtime::RuntimeManifestError::MissingAdapterGeneration { .. })
    ));
}

#[test]
fn strict_manifest_admission_binds_principal_policy_and_snapshot() {
    let (knowledge, object) = knowledge();
    let pod = descriptor();
    let mut artifacts = InMemoryArtifactCatalog::default();
    artifacts.admit(pod.clone()).unwrap();
    let candidate = manifest(&object, &pod);
    let authority = Authority {
        principal: PrincipalId::from("principal"),
        policy: Revision(1),
        snapshot_revision: Revision(0),
        snapshot_digest: [9; 32],
    };
    let mut runtime = PtrRuntime::new(PtrConfig::default()).unwrap();
    assert!(runtime
        .admit_execution_manifest_with_authority(
            candidate.clone(),
            &knowledge,
            &artifacts,
            &authority,
            ptr_types::Timestamp(0),
        )
        .is_ok());

    let events = runtime.committed_events().len();
    let bad_policy = Authority {
        policy: Revision(2),
        ..authority.clone()
    };
    assert!(matches!(
        runtime.admit_execution_manifest_with_authority(
            candidate.clone(),
            &knowledge,
            &artifacts,
            &bad_policy,
            ptr_types::Timestamp(0),
        ),
        Err(ptr_runtime::RuntimeManifestError::PolicyRevisionMismatch { .. })
    ));
    assert_eq!(runtime.committed_events().len(), events);

    let bad_snapshot = Authority {
        snapshot_digest: [8; 32],
        ..authority
    };
    assert!(matches!(
        runtime.admit_execution_manifest_with_authority(
            candidate,
            &knowledge,
            &artifacts,
            &bad_snapshot,
            ptr_types::Timestamp(0),
        ),
        Err(ptr_runtime::RuntimeManifestError::SnapshotDigestMismatch { .. })
    ));
}

#[test]
fn manifest_admission_and_revocation_replay_from_durable_ledger() {
    let path = std::env::temp_dir().join(format!(
        "ptr-manifest-{}.ledger",
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let (knowledge, object) = knowledge();
    let pod = descriptor();
    let mut artifacts = InMemoryArtifactCatalog::default();
    artifacts.admit(pod.clone()).unwrap();
    let candidate = manifest(&object, &pod);
    {
        let mut runtime = PtrRuntime::open_durable(PtrConfig::default(), &path).unwrap();
        runtime
            .admit_execution_manifest(
                candidate.clone(),
                &knowledge,
                &artifacts,
                ptr_types::Timestamp(0),
            )
            .unwrap();
    }
    let mut reopened = PtrRuntime::open_durable(PtrConfig::default(), &path).unwrap();
    assert_eq!(
        reopened
            .resolve_execution_manifest(&candidate.manifest_digest)
            .unwrap()
            .manifest,
        candidate
    );
    let authority = reopened.manifest_authority_registry();
    let authority = authority.read().unwrap();
    assert!(authority.principal_admitted(&candidate.principal));
    assert_eq!(
        authority.current_policy_revision(),
        candidate.policy_revision
    );
    assert_eq!(
        authority.snapshot_digest(candidate.snapshot_revision),
        Some(candidate.snapshot_digest)
    );
    reopened
        .revoke_execution_manifest(candidate.manifest_digest)
        .unwrap();
    drop(reopened);
    let reopened = PtrRuntime::open_durable(PtrConfig::default(), &path).unwrap();
    assert!(reopened
        .resolve_execution_manifest(&candidate.manifest_digest)
        .is_err());
    let _ = std::fs::remove_file(path);
}

fn bounded_knowledge(from: u64, until: u64) -> (KnowledgeStore, KnowledgeObject) {
    let (_, mut object) = knowledge();
    object.validity = ptr_types::KnowledgeValidity {
        valid_from: Some(ptr_types::Timestamp(from)),
        valid_until: Some(ptr_types::Timestamp(until)),
    };
    let mut store = KnowledgeStore::default();
    store.register_object(object.clone()).unwrap();
    (store, object)
}

#[test]
fn an_expired_or_not_yet_valid_object_cannot_be_bound_into_a_manifest() {
    let (knowledge, object) = bounded_knowledge(10, 20);
    let pod = descriptor();
    let mut artifacts = InMemoryArtifactCatalog::default();
    artifacts.admit(pod.clone()).unwrap();
    let candidate = manifest(&object, &pod);
    for now in [9, 20, 21] {
        let mut runtime = PtrRuntime::new(PtrConfig::default()).unwrap();
        let refused = runtime.admit_execution_manifest(
            candidate.clone(),
            &knowledge,
            &artifacts,
            ptr_types::Timestamp(now),
        );
        assert!(
            matches!(
                refused,
                Err(ptr_runtime::RuntimeManifestError::ExpiredKnowledgeGeneration { .. })
            ),
            "now {now}: {refused:?}"
        );
        assert!(
            runtime.committed_events().is_empty(),
            "nothing is committed for a refused binding"
        );
    }
    let mut runtime = PtrRuntime::new(PtrConfig::default()).unwrap();
    assert!(runtime
        .admit_execution_manifest(candidate, &knowledge, &artifacts, ptr_types::Timestamp(10))
        .is_ok());
}

#[test]
fn a_manifest_admitted_while_its_object_held_is_not_resolved_after_the_object_ends() {
    let (knowledge, object) = bounded_knowledge(10, 20);
    let pod = descriptor();
    let mut artifacts = InMemoryArtifactCatalog::default();
    artifacts.admit(pod.clone()).unwrap();
    let candidate = manifest(&object, &pod);
    let mut runtime = PtrRuntime::new(PtrConfig::default()).unwrap();
    runtime
        .admit_execution_manifest(
            candidate.clone(),
            &knowledge,
            &artifacts,
            ptr_types::Timestamp(19),
        )
        .unwrap();
    let digest = candidate.manifest_digest;
    assert!(runtime
        .resolve_execution_manifest_at(&digest, &knowledge, ptr_types::Timestamp(19))
        .is_ok());
    let late = runtime.resolve_execution_manifest_at(&digest, &knowledge, ptr_types::Timestamp(20));
    assert!(
        matches!(
            late,
            Err(ptr_runtime::RuntimeManifestError::ExpiredKnowledgeGeneration { .. })
        ),
        "{late:?}"
    );
    // An object invalidated after admission is refused too.
    let mut knowledge = knowledge;
    knowledge.invalidate(object.id.clone()).unwrap();
    assert!(matches!(
        runtime.resolve_execution_manifest_at(&digest, &knowledge, ptr_types::Timestamp(19)),
        Err(ptr_runtime::RuntimeManifestError::RevokedKnowledgeGeneration { .. })
    ));
}

#[test]
fn the_knowledge_backed_validity_check_refuses_once_a_bound_object_ends() {
    use ptr_runtime::ManifestValidity;
    use std::sync::{Arc, RwLock};

    let (knowledge, object) = bounded_knowledge(10, 20);
    let pod = descriptor();
    let mut artifacts = InMemoryArtifactCatalog::default();
    artifacts.admit(pod.clone()).unwrap();
    let candidate = manifest(&object, &pod);
    let mut runtime = PtrRuntime::new(PtrConfig::default()).unwrap();
    runtime
        .admit_execution_manifest(
            candidate.clone(),
            &knowledge,
            &artifacts,
            ptr_types::Timestamp(15),
        )
        .unwrap();
    let digest = candidate.manifest_digest;
    let validity = ptr_runtime::KnowledgeManifestValidity::new(
        runtime.execution_manifest_registry(),
        Arc::new(RwLock::new(knowledge)),
    );
    assert_eq!(
        validity.ensure_valid_at(&digest, ptr_types::Timestamp(19)),
        Ok(())
    );
    assert!(matches!(
        validity.ensure_valid_at(&digest, ptr_types::Timestamp(20)),
        Err(ptr_runtime::RuntimeManifestError::ExpiredKnowledgeGeneration { .. })
    ));
    assert!(matches!(
        validity.ensure_valid_at(&[7; 32], ptr_types::Timestamp(15)),
        Err(ptr_runtime::RuntimeManifestError::UnknownManifestDigest(_))
    ));
    runtime.revoke_execution_manifest(digest).unwrap();
    assert!(validity
        .ensure_valid_at(&digest, ptr_types::Timestamp(15))
        .is_err());
}
