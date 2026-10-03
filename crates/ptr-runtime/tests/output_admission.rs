use ptr_branch::{AutoThreshold, PolicyRecord, TriagePolicy};
use ptr_config::PtrConfig;
use ptr_core::action_head::ActionIr;
use ptr_pods::{ExecutionManifest, LineageBinding, PodOutput, PodOutputKind};
use ptr_protocol::TypedPayload;
use ptr_runtime::{
    execution::RequiredVerification, ExecutionScope, MergeAuthority, PodOutputAdmission,
    PodOutputAdmissionRequest, PtrRuntime, ScopeState, SemanticChange, SemanticGrant,
};
use ptr_types::{
    CapabilityId, Effect, Generation, PrincipalId, Probability, ProjectId, ProvenanceRef,
    RequestId, Revision, ScopeId, ScopeLeaseBinding, SessionId, Timestamp, TypeId,
    VerificationLevel,
};
use ptr_verifier::{NamedVerifier, VerificationReport, VerificationStatus, Verifier};

struct Pass;

impl Verifier<TypedPayload> for Pass {
    fn verify(&self, _: &TypedPayload) -> VerificationReport {
        VerificationReport {
            status: VerificationStatus::Pass,
            level: VerificationLevel::Deterministic,
            score: Probability::new(1.0).unwrap(),
            findings: vec![],
        }
    }
}

#[derive(Clone)]
struct MergePass;

impl<'a> Verifier<SemanticChange<'a>> for MergePass {
    fn verify(&self, _: &SemanticChange<'a>) -> VerificationReport {
        VerificationReport {
            status: VerificationStatus::Pass,
            level: VerificationLevel::Deterministic,
            score: Probability::new(1.0).unwrap(),
            findings: vec![],
        }
    }
}

impl<'a> NamedVerifier<SemanticChange<'a>> for MergePass {
    fn name(&self) -> &'static str {
        "pod-merge"
    }
}

fn manifest() -> ExecutionManifest {
    ExecutionManifest::build(
        Generation(1),
        vec![LineageBinding {
            key: "knowledge:fact".into(),
            generation: Generation(1),
            digest: [3; 32],
        }],
        vec![LineageBinding {
            key: "artifact:pod".into(),
            generation: Generation(1),
            digest: [2; 32],
        }],
        vec!["raw:test".into()],
        None,
        Revision(1),
        [4; 32],
        PrincipalId::from("principal"),
        Revision(1),
    )
    .unwrap()
}

fn runtime_with_scope() -> PtrRuntime {
    let mut runtime = PtrRuntime::new(PtrConfig::default()).unwrap();
    runtime
        .ingest_text(RequestId::from("request"), "ground truth")
        .unwrap();
    runtime
        .create_scope(
            ExecutionScope {
                id: ScopeId::from("scope"),
                parent: None,
                session: SessionId::from("session"),
                project: ProjectId::from("project"),
                created_at: Timestamp(1),
                deadline: None,
                state: ScopeState::Created,
                cancellation_requested: false,
            },
            ScopeLeaseBinding::default(),
        )
        .unwrap();
    runtime
}

fn request(kind: PodOutputKind) -> PodOutputAdmissionRequest {
    let manifest = manifest();
    let manifest_digest = manifest.manifest_digest;
    PodOutputAdmissionRequest {
        request_id: RequestId::from("request"),
        manifest_digest: manifest.manifest_digest,
        manifest,
        pod_id: ptr_types::PodId::from("pod"),
        output: PodOutput {
            kind,
            payload: TypedPayload {
                type_id: TypeId::from("text"),
                bytes: b"observation".to_vec(),
            },
            generation: Generation(1),
            manifest_digest,
            artifact_digest: [2; 32],
            provenance: vec![ProvenanceRef {
                source: ptr_types::EvidenceId::from("evidence"),
                note: None,
            }],
            dependencies: vec![[3; 32]],
            revision: Revision(1),
            verified: false,
        },
        expected_type: TypeId::from("text"),
        session_id: SessionId::from("session"),
        scope_id: ScopeId::from("scope"),
        placement_epoch: ptr_runtime::PlacementEpoch(1),
        fencing_token: ptr_runtime::FencingToken(1),
    }
}

fn encode_action(action: &ActionIr) -> Vec<u8> {
    let mut bytes = vec![1u8];
    let string = |bytes: &mut Vec<u8>, value: &str| {
        bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
        bytes.extend_from_slice(value.as_bytes());
    };
    let body = |bytes: &mut Vec<u8>, value: &[u8]| {
        bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
        bytes.extend_from_slice(value);
    };
    string(&mut bytes, &action.operation);
    string(&mut bytes, &action.target);
    string(&mut bytes, &action.capability.0);
    string(&mut bytes, &action.input_type.0);
    bytes.push(match action.effect {
        Effect::Pure => 1,
        Effect::Read => 2,
        Effect::Mutation => 3,
        Effect::External => 4,
        Effect::Irreversible => 5,
    });
    bytes.extend_from_slice(&action.generation.0.to_le_bytes());
    bytes.extend_from_slice(&action.revision.0.to_le_bytes());
    body(&mut bytes, &action.payload);
    bytes
}

#[test]
fn valid_observation_is_admitted_and_idempotent() {
    let mut runtime = runtime_with_scope();
    let first = runtime.admit_pod_output(request(PodOutputKind::Observation), &Pass);
    assert!(matches!(
        first,
        Ok(PodOutputAdmission::ObservationCandidate { .. })
    ));
    let candidate_key = match &first {
        Ok(PodOutputAdmission::ObservationCandidate { semantic_key, .. }) => semantic_key,
        _ => unreachable!(),
    };
    assert!(runtime.snapshot().payload(candidate_key).is_some());
    assert!(runtime
        .snapshot()
        .payload(&ptr_runtime::semantic::pod_output_key(
            &RequestId::from("request"),
            &ptr_types::PodId::from("pod"),
        ))
        .is_none());
    let event_count = runtime.committed_events().len();
    let second = runtime.admit_pod_output(request(PodOutputKind::Observation), &Pass);
    assert_eq!(first, second);
    assert_eq!(runtime.committed_events().len(), event_count);
}

#[test]
fn wrong_manifest_and_missing_provenance_are_rejected() {
    let mut runtime = runtime_with_scope();
    let mut bad_manifest = request(PodOutputKind::Observation);
    bad_manifest.manifest_digest = [9; 32];
    assert!(matches!(
        runtime.admit_pod_output(bad_manifest, &Pass),
        Err(ptr_runtime::RuntimeError::PodOutputAdmission(_))
    ));

    let mut missing_provenance = request(PodOutputKind::Observation);
    missing_provenance.output.provenance.clear();
    assert!(matches!(
        runtime.admit_pod_output(missing_provenance, &Pass),
        Err(ptr_runtime::RuntimeError::PodOutputAdmission(_))
    ));
}

#[test]
fn action_and_hypothesis_are_admitted_without_direct_effects() {
    let mut runtime = runtime_with_scope();
    runtime
        .commit(ptr_ledger::LedgerEvent::CapsuleCommitted {
            project: ProjectId::from("project"),
            capsule: ptr_types::CapsuleId::from("capsule:a"),
            generation: Generation(1),
        })
        .unwrap();
    runtime
        .permissions_mut()
        .capabilities
        .insert(CapabilityId::from("file.write"));
    runtime.permissions_mut().allow_mutation = true;
    let mut action_request = request(PodOutputKind::ActionProposal);
    action_request.request_id = RequestId::from("action");
    action_request.output.payload.bytes = encode_action(&ActionIr {
        operation: "write".into(),
        target: "capsule:a".into(),
        capability: CapabilityId::from("file.write"),
        effect: Effect::Mutation,
        input_type: TypeId::from("Bytes"),
        generation: Generation(1),
        revision: Revision(1),
        payload: b"proposal".to_vec(),
    });
    let action = runtime.admit_pod_output(action_request, &Pass).unwrap();
    assert!(matches!(action, PodOutputAdmission::ActionProposal { .. }));
    let hypothesis = runtime
        .admit_pod_output(
            {
                let mut request = request(PodOutputKind::Hypothesis);
                request.request_id = RequestId::from("hypothesis");
                request
            },
            &Pass,
        )
        .unwrap();
    let branch_id = match hypothesis {
        PodOutputAdmission::Hypothesis { branch_id } => branch_id,
        other => panic!("unexpected admission: {other:?}"),
    };
    assert!(runtime.hypothesis(&branch_id).is_some());
    let replayed = PtrRuntime::replay(PtrConfig::default(), runtime.committed_events()).unwrap();
    assert!(replayed.hypothesis(&branch_id).is_some());
    assert!(matches!(
        runtime.merge_pod_hypothesis(
            &branch_id,
            MergeAuthority::Triage {
                score: Probability::new(1.0).unwrap(),
            },
        ),
        Err(ptr_runtime::RuntimeError::NoSemanticGrant)
    ));
    runtime
        .install_semantic_grant(
            SemanticGrant::new(RequiredVerification::Deterministic)
                .with_verifier(MergePass)
                .with_merge_policy(
                    PolicyRecord::manual(
                        "pod-merge-policy",
                        TriagePolicy::new(AutoThreshold::AtLeast(0.8), 0.0).unwrap(),
                    )
                    .unwrap(),
                    7,
                ),
        )
        .unwrap();
    assert!(matches!(
        runtime.merge_pod_hypothesis(
            &branch_id,
            MergeAuthority::Triage {
                score: Probability::new(1.0).unwrap(),
            },
        ),
        Ok(ptr_runtime::MergeOutcome::Committed(_))
    ));
    assert!(runtime
        .snapshot()
        .payload(&format!("pod-hypothesis:{branch_id}"))
        .is_some());
}

#[test]
fn verified_state_delta_requires_verification_and_is_promoted_after_admission() {
    let mut runtime = runtime_with_scope();
    let mut request = request(PodOutputKind::StateDelta);
    request.request_id = RequestId::from("state");
    runtime
        .ingest_text(RequestId::from("state"), "state input")
        .unwrap();
    request.output.verified = true;
    let admitted = runtime.admit_pod_output(request, &Pass).unwrap();
    assert!(matches!(admitted, PodOutputAdmission::StateDelta { .. }));
    assert!(runtime.committed_events().iter().any(|event| matches!(
        event.event,
        ptr_ledger::LedgerEvent::PodOutputAdmitted { .. }
    )));
}

#[test]
fn unknown_dependency_is_rejected_before_ledger_append() {
    let mut runtime = runtime_with_scope();
    let mut request = request(PodOutputKind::Observation);
    request.output.dependencies = vec![[99; 32]];
    assert!(matches!(
        runtime.admit_pod_output(request, &Pass),
        Err(ptr_runtime::RuntimeError::PodOutputAdmission(_))
    ));
    assert!(!runtime.committed_events().iter().any(|event| matches!(
        event.event,
        ptr_ledger::LedgerEvent::PodOutputAdmitted { .. }
    )));
}

#[test]
fn admitted_output_replays_without_recreating_the_payload() {
    let mut runtime = runtime_with_scope();
    runtime
        .admit_pod_output(request(PodOutputKind::Observation), &Pass)
        .unwrap();
    let events = runtime.committed_events().to_vec();
    let replayed = PtrRuntime::replay(PtrConfig::default(), &events).unwrap();
    assert_eq!(replayed.committed_events(), events.as_slice());
    assert!(replayed.committed_events().iter().any(|event| matches!(
        &event.event,
        ptr_ledger::LedgerEvent::PodOutputAdmitted { .. }
    )));
}
