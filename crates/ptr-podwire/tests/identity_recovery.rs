use ptr_config::PtrConfig;
use ptr_ledger::LedgerEvent;
use ptr_net::{EndpointAddr, IrohTransport, PeerAddress, PeerBook, ALPN_PODWIRE};
use ptr_pods::{
    DynPod, ExecutionManifest, InMemoryKvTensorBackend, KvTensorDType, KvTensorSchema,
    LineageBinding, PodManifest, PodRegistry, TensorRef,
};
use ptr_podwire::{
    decode_request_v2, encode_answer_v2, request_digest, PodAccessPolicy, PodAddressBinding,
    PodAdmissionBinding, PodAnswerV2, PodClient, PodHost, PodOutcome, PodRequestV2, PodScope,
    PodWireError, RefusalCode,
};
use ptr_protocol::TypedPayload;
use ptr_runtime::{
    DeviceHealth, DeviceRecord, ExecutionScope, IdentityAdmissionController, ManagedKvRegistry,
    NodeHealth, NodeRecord, PodPlacementController, PtrRuntime, RuntimeRecoveryAdapter,
    ScopeCleanupCoordinator, ScopeState,
};
use ptr_storage::{
    FileProtectedStateStore, GenerationAnchor, ProtectedRecord, ProtectedStateStore,
    SoftwareKeyProvider, StateDomain,
};
use ptr_types::{
    AdmissionDecision, AdmissionPolicy, AdmissionRequest, AuthenticationLevel, CapabilityId,
    Effect, Generation, IdentityContext, PrincipalId, ProjectId, Revision, ScopeId,
    ScopeLeaseBinding, SessionId, StateId, Timestamp, TypeId, VerificationLevel,
};
use ptr_verifier::{VerificationReport, VerificationStatus, Verifier};
use sha2::{Digest as ShaDigest, Sha256};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

const PROJECT: &str = "e2e-project";
const POD: &str = "e2e-pod";
const CAPABILITY: &str = "Infer<Document>";
const INPUT: &str = "Document";
const OUTPUT: &str = "Answer";

fn execution_manifest(
    manifest: &PodManifest,
    generation: Generation,
    revision: Revision,
) -> ExecutionManifest {
    ExecutionManifest::build(
        generation,
        vec![LineageBinding {
            key: "knowledge".into(),
            generation,
            digest: [1; 32],
        }],
        vec![LineageBinding {
            key: manifest.id.0.clone(),
            generation,
            digest: manifest.digest(),
        }],
        vec!["identity-recovery-test".into()],
        None,
        revision,
        [2; 32],
        PrincipalId::from("identity-recovery-test"),
        Revision(1),
    )
    .unwrap()
}

#[derive(Clone)]
struct Policy;

impl AdmissionPolicy for Policy {
    type Error = &'static str;

    fn decide(&self, request: &AdmissionRequest) -> Result<AdmissionDecision, Self::Error> {
        Ok(AdmissionDecision::Allowed {
            project: request.project.clone(),
            capabilities: vec![request.capability.clone()],
            expires_at: Timestamp(90),
            policy_revision: Revision(7),
        })
    }
}

struct Pass;

impl Verifier<TypedPayload> for Pass {
    fn verify(&self, _: &TypedPayload) -> VerificationReport {
        VerificationReport {
            status: VerificationStatus::Pass,
            level: VerificationLevel::Deterministic,
            score: ptr_types::Probability::new(1.0).unwrap(),
            findings: vec![],
        }
    }
}

struct EchoPod {
    manifest: PodManifest,
    calls: Arc<Mutex<usize>>,
}

impl DynPod for EchoPod {
    fn manifest(&self) -> &PodManifest {
        &self.manifest
    }

    fn invoke(&self, input: TypedPayload) -> Result<TypedPayload, String> {
        *self.calls.lock().unwrap() += 1;
        Ok(TypedPayload {
            type_id: TypeId::from(OUTPUT),
            bytes: input.bytes,
        })
    }
}

fn manifest() -> PodManifest {
    PodManifest {
        project: ProjectId::from(PROJECT),
        id: ptr_types::PodId::from(POD),
        capabilities: vec![CapabilityId::from(CAPABILITY)],
        accepts: vec![TypeId::from(INPUT)],
        produces: vec![TypeId::from(OUTPUT)],
        effects: vec![Effect::Pure],
        protocol_version: 1,
    }
}

fn located(address: EndpointAddr) -> PeerAddress {
    let mut book = PeerBook::new();
    book.record(address.clone());
    book.locate(&ptr_types::NodeId(address.id.to_string()))
        .unwrap()
}

fn identity(session: &SessionId) -> IdentityContext {
    let mut value = IdentityContext {
        subject: "e2e-user".into(),
        issuer: "local-idp".into(),
        groups: vec!["operators".into()],
        authentication_level: AuthenticationLevel::MultiFactor,
        session_id: session.clone(),
        issued_at: Timestamp(1),
        expires_at: Timestamp(100),
        identity_digest: [0; 32],
    };
    value.identity_digest = value.canonical_digest();
    value
}

fn admission_request(identity: IdentityContext, session: SessionId) -> AdmissionRequest {
    AdmissionRequest {
        identity,
        project: ProjectId::from(PROJECT),
        capability: CapabilityId::from(CAPABILITY),
        input_type: TypeId::from(INPUT),
        pod: Some(ptr_types::PodId::from(POD)),
        session,
    }
}

#[allow(clippy::too_many_arguments)]
fn request(
    addressed_to: String,
    id: u64,
    manifest: &PodManifest,
    identity: &IdentityContext,
    generation: Generation,
    revision: Revision,
    placement_epoch: u64,
    fencing_token: u128,
    policy_revision: Revision,
) -> PodRequestV2 {
    PodRequestV2 {
        address: Some(PodAddressBinding {
            source: None,
            destination: ptr_types::PodRevisionAddress {
                address: ptr_types::PodAddress::new(
                    ProjectId::from(PROJECT),
                    "test".into(),
                    ptr_types::PodId::from(POD),
                )
                .unwrap(),
                semantic_revision: manifest.digest(),
                generation,
            },
            trace_id: "identity-recovery".into(),
            hop_limit: 8,
            visited: Vec::new(),
            protocol: None,
            mesh: None,
        }),
        execution_manifest: Some(
            execution_manifest(manifest, generation, revision).manifest_digest,
        ),
        addressed_to,
        request_id: id,
        artifact_id: ptr_types::ArtifactId::from("e2e-artifact"),
        manifest_hash: manifest.digest(),
        generation,
        revision,
        placement_epoch: Some(placement_epoch),
        fencing_token: Some(fencing_token),
        identity_digest: identity.identity_digest,
        session_id: identity.session_id.clone(),
        policy_revision,
        capability: CapabilityId::from(CAPABILITY),
        expect_protocol: 1,
        payload: TypedPayload {
            type_id: TypeId::from(INPUT),
            bytes: b"e2e-input".to_vec(),
        },
    }
}

fn node() -> NodeRecord {
    NodeRecord {
        node_id: ptr_types::NodeId::from("node-a"),
        zone: "test".into(),
        health: NodeHealth::Healthy,
        last_heartbeat: Timestamp(1),
        devices: vec![DeviceRecord {
            device_id: ptr_types::DeviceId::from("cuda:0"),
            vram_bytes: 32,
            used_vram_bytes: 0,
            health: DeviceHealth::Healthy,
        }],
    }
}

fn schema() -> KvTensorSchema {
    KvTensorSchema {
        model: ptr_types::ModelVersion::from("model-e2e"),
        adapter: ptr_types::AdapterVersion::from("adapter-e2e"),
        device: ptr_types::DeviceId::from("cuda:0"),
        layer_count: 1,
        attention_heads: 1,
        key_value_heads: 1,
        head_dim: 2,
        batch_size: 1,
        dtype: KvTensorDType::F32,
    }
}

fn digest(bytes: &[u8]) -> [u8; 32] {
    Sha256::digest(bytes).into()
}

fn temp_path(prefix: &str, suffix: &str) -> PathBuf {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!("ptr-{prefix}-{nonce}.{suffix}"))
}

#[derive(Default)]
struct Cleanup<'a> {
    placement: Option<&'a mut PodPlacementController>,
    kv: Option<&'a mut ManagedKvRegistry<InMemoryKvTensorBackend>>,
    handle: Option<ptr_runtime::ManagedKvHandle>,
    steps: Vec<&'static str>,
}

impl ScopeCleanupCoordinator for Cleanup<'_> {
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
        self.kv
            .as_deref_mut()
            .unwrap()
            .invalidate(
                self.placement.as_deref_mut().unwrap(),
                self.handle.as_ref().unwrap(),
            )
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

fn scope(id: &str, session: &SessionId, lease: ScopeLeaseBinding) -> ExecutionScope {
    let _ = lease;
    ExecutionScope {
        id: ScopeId::from(id),
        parent: None,
        session: session.clone(),
        project: ProjectId::from(PROJECT),
        created_at: Timestamp(1),
        deadline: Some(Timestamp(80)),
        state: ScopeState::Created,
        cancellation_requested: false,
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn identity_admission_podwire_protected_kv_ledger_and_recovery_are_bound() {
    let ledger_path = temp_path("identity-recovery", "ledger");
    let state_path = temp_path("identity-recovery", "state");
    let key_id = ptr_types::KeyId::from("e2e-key");
    let keys = SoftwareKeyProvider::default().with_key(key_id.clone(), [9; 32]);

    let session_id = SessionId::from("session-old");
    let identity = identity(&session_id);
    let admission_input = admission_request(identity.clone(), session_id.clone());
    let admission = IdentityAdmissionController::new(Policy, Revision(7))
        .admit(&admission_input, Timestamp(10))
        .unwrap();
    let admission_binding =
        PodAdmissionBinding::from_identity(&identity, admission.policy_revision).unwrap();

    let mut placement = PodPlacementController::default();
    placement.register_node(node()).unwrap();
    let placement_record = placement
        .assign(
            ptr_types::PodId::from(POD),
            ptr_types::ArtifactId::from("e2e-artifact"),
            Generation(1),
            ptr_types::NodeId::from("node-a"),
            ptr_types::DeviceId::from("cuda:0"),
            1,
        )
        .unwrap();

    let mut kv = ManagedKvRegistry::new(InMemoryKvTensorBackend);
    let old_handle = kv
        .allocate(
            &mut placement,
            StateId::from("kv-old"),
            &ptr_types::PodId::from(POD),
            schema(),
            4,
        )
        .unwrap();
    kv.append(
        &placement,
        &old_handle,
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
    let old_metadata = kv.metadata(&old_handle).unwrap();
    let snapshot = kv.snapshot(&placement, &old_handle).unwrap();
    assert!(snapshot.verify_digest());

    let protected_payload = b"serialized-kv-state-generation-1".to_vec();
    let mut anchor_bytes = b"kv-old".to_vec();
    anchor_bytes.extend_from_slice(&1u64.to_le_bytes());
    let record = ProtectedRecord {
        domain: StateDomain::KvSnapshot,
        logical_id: "kv-old".into(),
        generation: Generation(1),
        revision: Revision(1),
        plaintext_digest: digest(&protected_payload),
        payload: protected_payload.clone(),
        key_id: key_id.clone(),
        anchor: GenerationAnchor {
            logical_id: "kv-old".into(),
            generation: Generation(1),
            previous_digest: None,
            anchor_digest: digest(&anchor_bytes),
        },
    };
    let protected_handle = {
        let mut store = FileProtectedStateStore::open(&state_path, keys).unwrap();
        store.seal(record).unwrap()
    };
    let report = FileProtectedStateStore::open(
        &state_path,
        SoftwareKeyProvider::default().with_key(key_id.clone(), [9; 32]),
    )
    .unwrap()
    .verify(&protected_handle)
    .unwrap();
    assert!(report.valid);
    assert_eq!(report.generation, Generation(1));

    let lease = ScopeLeaseBinding {
        state_id: Some(StateId::from("kv-old")),
        pod_id: Some(ptr_types::PodId::from(POD)),
        placement_epoch: Some(old_metadata.placement_epoch.0),
        fencing_token: Some(old_metadata.fencing_token.0),
        generation: Some(Generation(1)),
        resource_lease_id: Some("resource-old".into()),
    };
    let mut runtime = PtrRuntime::open_durable(PtrConfig::default(), &ledger_path).unwrap();
    let scope_id = ScopeId::from("pod-call-old");
    runtime
        .create_scope(
            scope(&scope_id.0, &session_id, lease.clone()),
            lease.clone(),
        )
        .unwrap();
    runtime
        .transition_scope(&scope_id, ScopeState::Admitted, lease.clone(), None)
        .unwrap();
    runtime
        .transition_scope(&scope_id, ScopeState::Started, lease.clone(), None)
        .unwrap();
    runtime
        .commit(LedgerEvent::ProtectedStateCommitted {
            domain: 3,
            logical_id: protected_handle.logical_id.clone(),
            generation: protected_handle.generation,
            revision: protected_handle.revision,
            plaintext_digest: digest(&protected_payload),
            ciphertext_digest: protected_handle.ciphertext_digest,
            anchor_digest: protected_handle.anchor_digest,
        })
        .unwrap();

    let calls = Arc::new(Mutex::new(0));
    let mut registry = PodRegistry::default();
    registry.register(Arc::new(EchoPod {
        manifest: manifest(),
        calls: calls.clone(),
    }));
    let client = PodClient::bind().await.unwrap();
    let mut policy = PodAccessPolicy::new();
    policy
        .admit(
            ptr_types::NodeId(client.identity().public_key.clone()),
            PodScope::new(ProjectId::from(PROJECT))
                .allow(CapabilityId::from(CAPABILITY), TypeId::from(INPUT)),
        )
        .unwrap();
    let host = PodHost::bind(registry, policy, Pass).await.unwrap();
    let binding = host
        .bind_v2_admitted_addressed(
            &ProjectId::from(PROJECT),
            &ptr_types::PodId::from(POD),
            ptr_types::ArtifactId::from("e2e-artifact"),
            Generation(1),
            Revision(1),
            placement_record.epoch.0,
            old_metadata.fencing_token.0,
            admission_binding,
        )
        .unwrap()
        .with_execution_manifest(execution_manifest(&manifest(), Generation(1), Revision(1)))
        .unwrap();
    let address = located(host.address());
    let server = tokio::spawn(async move { host.serve_session_v2(&binding, 5).await.unwrap() });
    let session = client.connect_session(address, 5).await.unwrap();
    let addressed_to = session.peer().public_key.clone();
    let good = request(
        addressed_to.clone(),
        1,
        &manifest(),
        &identity,
        Generation(1),
        Revision(1),
        placement_record.epoch.0,
        old_metadata.fencing_token.0,
        Revision(7),
    );
    let answer = session.request_v2(&good).await.unwrap();
    assert!(matches!(answer.outcome, PodOutcome::Answered { .. }));

    for (id, mutate) in [(2, "identity"), (3, "session"), (4, "policy"), (5, "epoch")] {
        let mut bad = good.clone();
        bad.request_id = id;
        match mutate {
            "identity" => bad.identity_digest[0] ^= 1,
            "session" => bad.session_id = SessionId::from("other-session"),
            "policy" => bad.policy_revision = Revision(8),
            "epoch" => bad.placement_epoch = Some(placement_record.epoch.0 + 1),
            _ => unreachable!(),
        }
        let refused = session.request_v2(&bad).await.unwrap();
        assert!(matches!(
            refused.outcome,
            PodOutcome::Refused {
                code: RefusalCode::Unavailable
            }
        ));
    }
    session.close().await;
    assert_eq!(server.await.unwrap().len(), 5);
    assert_eq!(*calls.lock().unwrap(), 1);

    drop(runtime);

    let uncertain_raw = IrohTransport::bind(&[ALPN_PODWIRE]).await.unwrap();
    let uncertain_address = located(uncertain_raw.direct_addr());
    let responder = uncertain_raw.identity().public_key.clone();
    let responder_task = tokio::spawn(async move {
        let session = uncertain_raw.accept_session().await.unwrap();
        let incoming = session.accept_request(1 << 20).await.unwrap();
        let asked = decode_request_v2(&incoming.payload).unwrap();
        let answer = PodAnswerV2 {
            address: asked.address.clone(),
            execution_manifest: asked.execution_manifest,
            responder,
            request_id: asked.request_id,
            request_digest: request_digest(&incoming.payload),
            generation: Generation(2),
            revision: asked.revision,
            placement_epoch: asked.placement_epoch,
            fencing_token: asked.fencing_token,
            identity_digest: asked.identity_digest,
            session_id: asked.session_id,
            policy_revision: asked.policy_revision,
            outcome: PodOutcome::Refused {
                code: RefusalCode::Unavailable,
            },
        };
        incoming
            .respond(&encode_answer_v2(&answer).unwrap())
            .await
            .unwrap();
        session.wait_closed().await;
    });
    let uncertain_client = PodClient::bind().await.unwrap();
    let uncertain_session = uncertain_client
        .connect_session(uncertain_address, 1)
        .await
        .unwrap();
    let uncertain_request = request(
        uncertain_session.peer().public_key.clone(),
        777,
        &manifest(),
        &identity,
        Generation(1),
        Revision(1),
        placement_record.epoch.0,
        old_metadata.fencing_token.0,
        Revision(7),
    );
    assert!(matches!(
        uncertain_session.request_v2(&uncertain_request).await,
        Err(PodWireError::WrongGeneration { .. })
    ));
    assert_eq!(
        uncertain_session.uncertain_requests(),
        (Vec::new(), vec![777])
    );
    assert_eq!(
        uncertain_session.request_v2(&uncertain_request).await,
        Err(PodWireError::RequestUncertain { request_id: 777 })
    );

    let mut reopened = PtrRuntime::open_durable(PtrConfig::default(), &ledger_path).unwrap();
    assert!(reopened.scopes().recovery_required(&scope_id));
    let mut cleanup = Cleanup {
        placement: Some(&mut placement),
        kv: Some(&mut kv),
        handle: Some(old_handle.clone()),
        steps: Vec::new(),
    };
    uncertain_session
        .recover_uncertain_request(
            777,
            scope_id.clone(),
            &mut RuntimeRecoveryAdapter::new(
                &mut reopened,
                scope_id.clone(),
                &mut cleanup,
                lease.clone(),
            ),
        )
        .unwrap();
    assert_eq!(
        cleanup.steps,
        vec![
            "stop_intake",
            "cancel_children",
            "drain_queues",
            "release_pod_lease",
            "release_resource_lease",
            "close_session"
        ]
    );
    assert_eq!(
        kv.snapshot(&placement, &old_handle),
        Err(ptr_runtime::TensorKvError::Invalidated)
    );
    assert_eq!(
        reopened.scopes().get(&scope_id).unwrap().state,
        ScopeState::Released
    );
    assert!(reopened.committed_events().iter().any(|event| matches!(
        &event.event,
        LedgerEvent::ProtectedStateCommitted { logical_id, generation, .. }
            if logical_id == "kv-old" && *generation == Generation(1)
    )));
    let reopened_store = FileProtectedStateStore::open(
        &state_path,
        SoftwareKeyProvider::default().with_key(key_id.clone(), [9; 32]),
    )
    .unwrap();
    assert!(reopened_store.verify(&protected_handle).unwrap().valid);
    uncertain_session.close().await;
    responder_task.await.unwrap();

    let next_placement = placement
        .assign(
            ptr_types::PodId::from(POD),
            ptr_types::ArtifactId::from("e2e-artifact"),
            Generation(2),
            ptr_types::NodeId::from("node-a"),
            ptr_types::DeviceId::from("cuda:0"),
            1,
        )
        .unwrap();
    let new_handle = kv
        .recompute(
            &mut placement,
            StateId::from("kv-new"),
            &ptr_types::PodId::from(POD),
            schema(),
            4,
        )
        .unwrap();
    let new_metadata = kv.metadata(&new_handle).unwrap();
    assert_ne!(new_metadata.fencing_token, old_metadata.fencing_token);
    assert_eq!(new_metadata.generation, Generation(2));
    assert_ne!(next_placement.epoch, placement_record.epoch);

    let next_identity = self::identity(&SessionId::from("session-new"));
    let next_admission = IdentityAdmissionController::new(Policy, Revision(7))
        .admit(
            &admission_request(next_identity.clone(), next_identity.session_id.clone()),
            Timestamp(10),
        )
        .unwrap();
    assert_ne!(next_identity.session_id, identity.session_id);
    assert_eq!(next_admission.policy_revision, Revision(7));

    let next_binding =
        PodAdmissionBinding::from_identity(&next_identity, next_admission.policy_revision).unwrap();
    let mut next_registry = PodRegistry::default();
    next_registry.register(Arc::new(EchoPod {
        manifest: manifest(),
        calls: calls.clone(),
    }));
    let next_client = PodClient::bind().await.unwrap();
    let mut next_policy = PodAccessPolicy::new();
    next_policy
        .admit(
            ptr_types::NodeId(next_client.identity().public_key.clone()),
            PodScope::new(ProjectId::from(PROJECT))
                .allow(CapabilityId::from(CAPABILITY), TypeId::from(INPUT)),
        )
        .unwrap();
    let next_host = PodHost::bind(next_registry, next_policy, Pass)
        .await
        .unwrap();
    let next_wire_binding = next_host
        .bind_v2_admitted_addressed(
            &ProjectId::from(PROJECT),
            &ptr_types::PodId::from(POD),
            ptr_types::ArtifactId::from("e2e-artifact"),
            Generation(2),
            Revision(2),
            next_placement.epoch.0,
            new_metadata.fencing_token.0,
            next_binding,
        )
        .unwrap()
        .with_execution_manifest(execution_manifest(&manifest(), Generation(2), Revision(2)))
        .unwrap();
    let next_address = located(next_host.address());
    let next_server = tokio::spawn(async move {
        next_host
            .serve_session_v2(&next_wire_binding, 1)
            .await
            .unwrap()
    });
    let next_session = next_client.connect_session(next_address, 1).await.unwrap();
    let next_request = request(
        next_session.peer().public_key.clone(),
        900,
        &manifest(),
        &next_identity,
        Generation(2),
        Revision(2),
        next_placement.epoch.0,
        new_metadata.fencing_token.0,
        Revision(7),
    );
    let next_answer = next_session.request_v2(&next_request).await.unwrap();
    assert!(matches!(next_answer.outcome, PodOutcome::Answered { .. }));
    next_session.close().await;
    assert_eq!(next_server.await.unwrap().len(), 1);
    assert_eq!(*calls.lock().unwrap(), 2);

    drop(reopened);
    std::fs::remove_file(ledger_path).unwrap();
    std::fs::remove_file(state_path).unwrap();
}
