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
fn a_hypothesis_over_the_ledger_limits_is_refused_before_admission_is_committed() {
    let mut runtime = runtime_with_scope();
    let mut request = request(PodOutputKind::Hypothesis);
    request.output.provenance = (0..ptr_ledger::MAX_HYPOTHESIS_PROVENANCE + 1)
        .map(|_| ProvenanceRef {
            source: ptr_types::EvidenceId::from("evidence"),
            note: None,
        })
        .collect();
    assert!(runtime.admit_pod_output(request.clone(), &Pass).is_err());
    // Neither record may be left behind: an admission without its hypothesis
    // would make a retry report success for a hypothesis that does not exist.
    assert!(!runtime.committed_events().iter().any(|event| matches!(
        event.event,
        ptr_ledger::LedgerEvent::PodOutputAdmitted { .. }
            | ptr_ledger::LedgerEvent::PodHypothesisCommitted { .. }
    )));
    assert!(runtime.admit_pod_output(request, &Pass).is_err());
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

/// A runtime that holds only the admission record of an earlier attempt and
/// not the step after it, as when promotion failed after the record was
/// committed. The record is taken from a complete run of the same request.
fn runtime_with_only_the_admission_record(complete: &PtrRuntime) -> PtrRuntime {
    let record = complete
        .committed_events()
        .iter()
        .find(|committed| {
            matches!(
                committed.event,
                ptr_ledger::LedgerEvent::PodOutputAdmitted { .. }
            )
        })
        .expect("a complete run commits an admission record")
        .event
        .clone();
    let mut runtime = runtime_with_scope();
    runtime.commit(record).unwrap();
    runtime
}

#[test]
fn a_retry_completes_a_promotion_that_failed_after_the_admission_record() {
    let mut first = runtime_with_scope();
    let admitted = first
        .admit_pod_output(request(PodOutputKind::Observation), &Pass)
        .unwrap();
    let PodOutputAdmission::ObservationCandidate { semantic_key, .. } = &admitted else {
        panic!("expected a candidate");
    };
    let mut runtime = runtime_with_only_the_admission_record(&first);
    assert!(
        runtime.snapshot().payload(semantic_key).is_none(),
        "the promotion must be missing before the retry"
    );
    let retried = runtime
        .admit_pod_output(request(PodOutputKind::Observation), &Pass)
        .unwrap();
    assert!(matches!(
        retried,
        PodOutputAdmission::ObservationCandidate { .. }
    ));
    assert!(
        runtime.snapshot().payload(semantic_key).is_some(),
        "the retry must carry out the missing promotion"
    );
    // The admission record is not written a second time.
    let admissions = runtime
        .committed_events()
        .iter()
        .filter(|committed| {
            matches!(
                committed.event,
                ptr_ledger::LedgerEvent::PodOutputAdmitted { .. }
            )
        })
        .count();
    assert_eq!(admissions, 1);
    // And once complete, a further retry is a plain replay.
    let event_count = runtime.committed_events().len();
    runtime
        .admit_pod_output(request(PodOutputKind::Observation), &Pass)
        .unwrap();
    assert_eq!(runtime.committed_events().len(), event_count);
}

#[test]
fn a_retry_completes_a_missing_hypothesis_record() {
    let mut first = runtime_with_scope();
    first
        .admit_pod_output(request(PodOutputKind::Hypothesis), &Pass)
        .unwrap();
    let mut runtime = runtime_with_only_the_admission_record(&first);
    let has_hypothesis = |runtime: &PtrRuntime| {
        runtime.committed_events().iter().any(|committed| {
            matches!(
                committed.event,
                ptr_ledger::LedgerEvent::PodHypothesisCommitted { .. }
            )
        })
    };
    assert!(!has_hypothesis(&runtime));
    runtime
        .admit_pod_output(request(PodOutputKind::Hypothesis), &Pass)
        .unwrap();
    assert!(has_hypothesis(&runtime));
}

#[test]
fn a_retry_from_another_scope_cannot_complete_someone_elses_admission() {
    let mut first = runtime_with_scope();
    first
        .admit_pod_output(request(PodOutputKind::Hypothesis), &Pass)
        .unwrap();
    let mut runtime = runtime_with_only_the_admission_record(&first);
    runtime
        .create_scope(
            ExecutionScope {
                id: ScopeId::from("other-scope"),
                parent: None,
                session: SessionId::from("other-session"),
                project: ProjectId::from("project"),
                created_at: Timestamp(1),
                deadline: None,
                state: ScopeState::Created,
                cancellation_requested: false,
            },
            ScopeLeaseBinding::default(),
        )
        .unwrap();
    // The same request id, Pod and output, submitted from a second scope.
    let mut stranger = request(PodOutputKind::Hypothesis);
    stranger.scope_id = ScopeId::from("other-scope");
    stranger.session_id = SessionId::from("other-session");
    assert!(matches!(
        runtime.admit_pod_output(stranger, &Pass),
        Err(ptr_runtime::RuntimeError::PodOutputAdmission(_))
    ));
    // Nothing was written for the second scope under the first one's admission.
    assert!(!runtime.committed_events().iter().any(|committed| matches!(
        committed.event,
        ptr_ledger::LedgerEvent::PodHypothesisCommitted { .. }
    )));
    // The original scope can still complete its own admission.
    runtime
        .admit_pod_output(request(PodOutputKind::Hypothesis), &Pass)
        .unwrap();
}

#[test]
fn a_promotion_written_before_the_admission_is_not_its_follow_up() {
    let state_request = || {
        let mut request = request(PodOutputKind::StateDelta);
        request.request_id = RequestId::from("state");
        request.output.verified = true;
        request
    };
    let mut first = runtime_with_scope();
    first
        .ingest_text(RequestId::from("state"), "state input")
        .unwrap();
    first.admit_pod_output(state_request(), &Pass).unwrap();
    let admission = first
        .committed_events()
        .iter()
        .find(|committed| {
            matches!(
                committed.event,
                ptr_ledger::LedgerEvent::PodOutputAdmitted { .. }
            )
        })
        .unwrap()
        .event
        .clone();
    let promotions = |runtime: &PtrRuntime| {
        runtime
            .committed_events()
            .iter()
            .filter(|committed| {
                matches!(
                    &committed.event,
                    ptr_ledger::LedgerEvent::SemanticDeltaCommitted {
                        origin: ptr_ledger::SemanticOrigin::PodOutput { .. },
                        ..
                    }
                )
            })
            .count()
    };

    // A promotion for the same request and Pod that did not come through this
    // admission, written before the admission record, which has nothing after it.
    let pod = ptr_pods::PodManifest {
        project: ProjectId::from("project"),
        id: ptr_types::PodId::from("pod"),
        capabilities: vec![],
        accepts: vec![TypeId::from("text")],
        produces: vec![TypeId::from("text")],
        effects: vec![ptr_types::Effect::Pure],
        protocol_version: 1,
    };
    let mut runtime = runtime_with_scope();
    runtime
        .ingest_text(RequestId::from("state"), "state input")
        .unwrap();
    runtime
        .promote_verified_pod_output(
            &RequestId::from("state"),
            &pod,
            &TypedPayload {
                type_id: TypeId::from("text"),
                bytes: b"an earlier output".to_vec(),
            },
            &Pass,
        )
        .unwrap();
    runtime.commit(admission).unwrap();
    assert_eq!(promotions(&runtime), 1);

    // The retry must not take that earlier promotion for this admission's follow-up.
    runtime.admit_pod_output(state_request(), &Pass).unwrap();
    assert_eq!(
        promotions(&runtime),
        2,
        "the retry carries out its own promotion after the admission"
    );
}

#[test]
fn a_later_promotion_with_another_payload_is_not_the_admissions_follow_up() {
    let state_request = || {
        let mut request = request(PodOutputKind::StateDelta);
        request.request_id = RequestId::from("state");
        request.output.verified = true;
        request
    };
    let mut first = runtime_with_scope();
    first
        .ingest_text(RequestId::from("state"), "state input")
        .unwrap();
    first.admit_pod_output(state_request(), &Pass).unwrap();
    let admission = first
        .committed_events()
        .iter()
        .find(|committed| {
            matches!(
                committed.event,
                ptr_ledger::LedgerEvent::PodOutputAdmitted { .. }
            )
        })
        .unwrap()
        .event
        .clone();

    let pod = ptr_pods::PodManifest {
        project: ProjectId::from("project"),
        id: ptr_types::PodId::from("pod"),
        capabilities: vec![],
        accepts: vec![TypeId::from("text")],
        produces: vec![TypeId::from("text")],
        effects: vec![ptr_types::Effect::Pure],
        protocol_version: 1,
    };
    let mut runtime = runtime_with_scope();
    runtime
        .ingest_text(RequestId::from("state"), "state input")
        .unwrap();
    runtime.commit(admission).unwrap();
    // After the admission, a promotion for the same request and Pod that carries
    // some other output reaches the ledger through the public path.
    runtime
        .promote_verified_pod_output(
            &RequestId::from("state"),
            &pod,
            &TypedPayload {
                type_id: TypeId::from("text"),
                bytes: b"another output".to_vec(),
            },
            &Pass,
        )
        .unwrap();

    // The retry must still promote what was admitted.
    runtime.admit_pod_output(state_request(), &Pass).unwrap();
    let key = ptr_runtime::semantic::pod_output_key(
        &RequestId::from("state"),
        &ptr_types::PodId::from("pod"),
    );
    let snapshot = runtime.snapshot();
    let promoted = snapshot.payload(&key).expect("a promoted value");
    assert_eq!(promoted.bytes, b"observation".to_vec());
}
