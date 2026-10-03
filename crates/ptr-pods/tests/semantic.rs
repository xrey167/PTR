use ptr_pods::{
    merge_hypotheses, ArtifactLifecycle, ConnectionScope, DeliveryMode, DuplexSession,
    EgressPolicy, ExecutionManifest, LifecycleGate, LineageBinding, MessagePattern, ModelVariant,
    PodCache, PodCacheKey, PodHypothesis, PodKind, PodLink, PodOutput, PodOutputKind,
    PodResourceProfile, PodSemanticManifest, PodTurnEvent, ProtocolBinding, SemanticPodLifecycle,
};
use ptr_protocol::TypedPayload;
use ptr_types::{
    ArtifactId, CapabilityId, EvidenceId, Generation, NamespaceId, PodAddress, PodId, PodIdentity,
    PodRevisionAddress, PrincipalId, Probability, ProjectId, ProvenanceRef, Revision, ScopeId,
    SessionId, StateId, Timestamp, TraceId, TypeId, VerificationLevel,
};
use ptr_verifier::{VerificationReport, VerificationStatus};
use std::time::Duration;

fn revision_address(generation: Generation) -> PodRevisionAddress {
    PodRevisionAddress {
        address: PodAddress::new(
            ProjectId::from("project"),
            NamespaceId::from("ns"),
            PodId::from("pod"),
        )
        .unwrap(),
        semantic_revision: [7; 32],
        generation,
    }
}

fn manifest() -> PodSemanticManifest {
    PodSemanticManifest {
        pod_id: PodId::from("pod"),
        pod_identity: PodIdentity::from("pod:stable"),
        project: ProjectId::from("project"),
        namespace: NamespaceId::from("ns"),
        generation: Generation(1),
        artifact_id: ArtifactId::from("artifact"),
        kind: PodKind::LanguageModel,
        semantic_role: "reader".into(),
        domain: "reasoning".into(),
        capabilities: vec![CapabilityId::from("infer")],
        accepts: vec![TypeId::from("prompt")],
        produces: vec![TypeId::from("answer")],
        model_variant: Some(ModelVariant::Base),
        model_family: Some("qwen".into()),
        interface: Some("text->text".into()),
        adapter_identity: None,
        resources: PodResourceProfile {
            ram_bytes: 1,
            vram_bytes: 1,
            max_concurrency: 1,
        },
        protocols: vec![ProtocolBinding::InProcess],
        effects: vec![],
        provenance: vec![ProvenanceRef {
            source: EvidenceId::from("origin"),
            note: None,
        }],
        principals: vec![PrincipalId::from("local")],
        lifecycle: SemanticPodLifecycle::Active,
    }
}

fn event(sequence: u64) -> PodTurnEvent {
    PodTurnEvent {
        kind: ptr_pods::PodTurnKind::RequestStarted,
        sequence,
        session_id: SessionId::from("session"),
        scope_id: ScopeId::from("scope"),
        turn_id: 1,
        request_id: ptr_types::RequestId::from("request"),
        pod: revision_address(Generation(1)),
        generation: Generation(1),
        manifest_digest: [1; 32],
        artifact_digest: [2; 32],
        input_type: Some(TypeId::from("prompt")),
        output_type: Some(TypeId::from("answer")),
        state_before: Some(StateId::from("before")),
        state_after: None,
        revision: Revision(1),
    }
}

#[test]
fn semantic_manifest_is_immutable_and_digest_bound() {
    let manifest = manifest();
    let digest = manifest.digest().unwrap();
    let mut changed = manifest.clone();
    changed.capabilities.push(CapabilityId::from("other"));
    assert_ne!(changed.digest().unwrap(), digest);
    assert!(matches!(changed.validate(), Ok(())));
}

#[test]
fn semantic_manifest_digest_is_independent_of_set_order() {
    let mut left = manifest();
    left.capabilities = vec![CapabilityId::from("z"), CapabilityId::from("a")];
    left.accepts = vec![TypeId::from("b"), TypeId::from("a")];
    let mut right = left.clone();
    right.capabilities.reverse();
    right.accepts.reverse();
    assert_eq!(left.digest().unwrap(), right.digest().unwrap());
}

#[test]
fn protocol_profiles_capture_transport_architecture_without_changing_identity() {
    let moq = ProtocolBinding::Moq.profile();
    assert_eq!(moq.pattern, MessagePattern::Duplex);
    assert!(moq.multiplexed);
    assert_eq!(moq.connection, ConnectionScope::Session);

    let ipc = ProtocolBinding::Ipc.profile();
    assert_eq!(ipc.pattern, MessagePattern::Pipeline);
    assert_eq!(ipc.connection, ConnectionScope::Persistent);

    let mavlink = ProtocolBinding::Mavlink.profile();
    assert!(mavlink.signed);
    assert_eq!(mavlink.max_frame_bytes, 280);

    let udp = ProtocolBinding::Udp.profile();
    assert_eq!(udp.delivery, DeliveryMode::UnreliableDatagram);

    let webrtc = ProtocolBinding::WebRtc.profile();
    assert_eq!(webrtc.pattern, MessagePattern::Duplex);
    assert_eq!(webrtc.connection, ConnectionScope::Session);
    assert!(webrtc.multiplexed);

    let json_rpc = ProtocolBinding::JsonRpc.profile();
    assert_eq!(json_rpc.pattern, MessagePattern::RequestResponse);
    assert!(json_rpc.multiplexed);
}

#[test]
fn execution_manifest_validates_lineage_and_tampering() {
    let manifest = ExecutionManifest::build(
        Generation(1),
        vec![LineageBinding {
            key: "knowledge".into(),
            generation: Generation(1),
            digest: [1; 32],
        }],
        vec![LineageBinding {
            key: "artifact".into(),
            generation: Generation(1),
            digest: [2; 32],
        }],
        vec!["origin".into()],
        None,
        Revision(3),
        [3; 32],
        PrincipalId::from("local"),
        Revision(4),
    )
    .unwrap();
    assert!(manifest.validate().is_ok());
    let mut tampered = manifest.clone();
    tampered.snapshot_revision = Revision(4);
    assert!(tampered.validate().is_err());
}

#[test]
fn execution_manifest_rejects_duplicate_lineage_keys() {
    let result = ExecutionManifest::build(
        Generation(1),
        vec![
            LineageBinding {
                key: "knowledge".into(),
                generation: Generation(1),
                digest: [1; 32],
            },
            LineageBinding {
                key: "knowledge".into(),
                generation: Generation(2),
                digest: [2; 32],
            },
        ],
        vec![LineageBinding {
            key: "artifact".into(),
            generation: Generation(1),
            digest: [3; 32],
        }],
        vec!["origin".into()],
        None,
        Revision(1),
        [4; 32],
        PrincipalId::from("local"),
        Revision(1),
    );
    assert_eq!(
        result,
        Err(ptr_pods::ExecutionManifestError::DuplicateBinding)
    );
}

#[test]
fn execution_manifest_lineage_order_is_canonical() {
    let make = |knowledge| {
        ExecutionManifest::build(
            Generation(1),
            knowledge,
            vec![LineageBinding {
                key: "artifact".into(),
                generation: Generation(1),
                digest: [3; 32],
            }],
            vec!["origin".into()],
            None,
            Revision(1),
            [4; 32],
            PrincipalId::from("local"),
            Revision(1),
        )
        .unwrap()
    };
    let first = make(vec![
        LineageBinding {
            key: "b".into(),
            generation: Generation(1),
            digest: [2; 32],
        },
        LineageBinding {
            key: "a".into(),
            generation: Generation(1),
            digest: [1; 32],
        },
    ]);
    let second = make(vec![
        LineageBinding {
            key: "a".into(),
            generation: Generation(1),
            digest: [1; 32],
        },
        LineageBinding {
            key: "b".into(),
            generation: Generation(1),
            digest: [2; 32],
        },
    ]);
    assert_eq!(first.manifest_digest, second.manifest_digest);
}

#[test]
fn lifecycle_gate_requires_admission_for_activation() {
    assert!(LifecycleGate::transition(
        ArtifactLifecycle::Approved,
        ArtifactLifecycle::Active,
        true,
        true,
        true,
        true
    )
    .is_ok());
    assert!(LifecycleGate::transition(
        ArtifactLifecycle::Approved,
        ArtifactLifecycle::Active,
        true,
        true,
        true,
        false
    )
    .is_err());
    assert!(LifecycleGate::transition(
        ArtifactLifecycle::Active,
        ArtifactLifecycle::Candidate,
        true,
        true,
        true,
        true
    )
    .is_err());
}

#[test]
fn pod_link_binds_artifact_acl_protocol_and_cycle_state() {
    let target = revision_address(Generation(1));
    let source = PodAddress::new(
        ProjectId::from("project"),
        NamespaceId::from("ns"),
        PodId::from("source"),
    )
    .unwrap();
    let link = PodLink {
        trace_id: "trace".into(),
        source,
        target,
        artifact_id: ArtifactId::from("artifact"),
        execution_manifest: [7; 32],
        protocol: ProtocolBinding::Ssh,
        capability: CapabilityId::from("infer"),
        acl: vec![PrincipalId::from("alice")],
        deadline: Timestamp(20),
        hop_limit: 3,
        visited: Vec::new(),
        attestation: [9; 32],
        egress_policy: EgressPolicy {
            hosts: vec!["worker".into()],
            ports: vec![22],
            topics: vec![],
        },
    };
    assert!(link
        .validate(
            &ArtifactId::from("artifact"),
            &[CapabilityId::from("infer")],
            &PrincipalId::from("alice"),
            Timestamp(1)
        )
        .is_ok());
    assert!(link
        .validate_against_manifest(
            &ArtifactId::from("artifact"),
            &[7; 32],
            &[CapabilityId::from("infer")],
            &PrincipalId::from("alice"),
            Timestamp(1)
        )
        .is_ok());
    assert!(matches!(
        link.validate_against_manifest(
            &ArtifactId::from("artifact"),
            &[8; 32],
            &[CapabilityId::from("infer")],
            &PrincipalId::from("alice"),
            Timestamp(1)
        ),
        Err(ptr_pods::PodLinkError::ExecutionManifestMismatch)
    ));
    assert!(link
        .validate(
            &ArtifactId::from("other"),
            &[CapabilityId::from("infer")],
            &PrincipalId::from("alice"),
            Timestamp(1)
        )
        .is_err());
}

#[test]
fn duplex_events_are_sequenced_and_resume_from_cursor() {
    let mut session =
        DuplexSession::new(SessionId::from("session"), TraceId::from("trace")).unwrap();
    session.emit(event(1)).unwrap();
    session.commit_turn(event(2)).unwrap();
    session.interrupt(event(3)).unwrap();
    let replay = session.resume(1).unwrap();
    assert_eq!(replay.len(), 2);
    assert_eq!(session.epoch, 1);
}

#[test]
fn outputs_require_provenance_and_promotion_verification() {
    let output = PodOutput {
        kind: PodOutputKind::StateDelta,
        payload: TypedPayload {
            type_id: TypeId::from("answer"),
            bytes: b"ok".to_vec(),
        },
        generation: Generation(1),
        manifest_digest: [1; 32],
        artifact_digest: [2; 32],
        provenance: vec![ProvenanceRef {
            source: EvidenceId::from("origin"),
            note: None,
        }],
        dependencies: vec![],
        revision: Revision(1),
        verified: false,
    };
    assert!(output.validate(&TypeId::from("answer"), true).is_err());
}

#[test]
fn hypotheses_merge_only_verified_same_generation_and_deduplicate_evidence() {
    let report = || VerificationReport {
        status: VerificationStatus::Pass,
        level: VerificationLevel::Deterministic,
        score: Probability::new(0.9).unwrap(),
        findings: vec![],
    };
    let hypothesis = |branch_id: &str, confidence: f32| PodHypothesis {
        branch_id: branch_id.into(),
        pod_id: PodId::from("pod"),
        generation: Generation(1),
        evidence: vec![ProvenanceRef {
            source: EvidenceId::from("origin"),
            note: None,
        }],
        output: TypedPayload {
            type_id: TypeId::from("answer"),
            bytes: b"answer".to_vec(),
        },
        confidence: Probability::new(confidence).unwrap(),
        latency: Duration::from_millis(5),
        verification: report(),
    };
    let merged = merge_hypotheses(vec![hypothesis("b", 0.7), hypothesis("a", 0.9)]).unwrap();
    assert_eq!(merged.source_branches, vec!["a", "b"]);
    assert_eq!(merged.evidence.len(), 1);
}

#[test]
fn cache_key_keeps_protocol_and_semantic_revision_invalidation_explicit() {
    let key = PodCacheKey {
        pod_identity: PodIdentity::from("pod:stable"),
        semantic_revision: [1; 32],
        execution_manifest: [2; 32],
        capability: CapabilityId::from("infer"),
        input_digest: [3; 32],
        knowledge_revision: Revision(1),
        principal: PrincipalId::from("alice"),
        protocol: ProtocolBinding::IrohQuic,
    };
    let mut cache = PodCache::default();
    cache.insert(
        key.clone(),
        TypedPayload {
            type_id: TypeId::from("answer"),
            bytes: vec![1],
        },
    );
    assert!(cache.get(&key).is_some());
    assert_eq!(
        cache.invalidate(|candidate| candidate.semantic_revision == [1; 32]),
        1
    );
}
