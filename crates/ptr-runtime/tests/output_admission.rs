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
        kv_taint: ptr_model_api::KvReuseTaint::Exact,
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

#[test]
fn a_foreign_hypothesis_record_is_not_the_admissions_follow_up() {
    let mut first = runtime_with_scope();
    first
        .admit_pod_output(request(PodOutputKind::Hypothesis), &Pass)
        .unwrap();
    let mut foreign = first
        .committed_events()
        .iter()
        .find_map(|committed| match &committed.event {
            event @ ptr_ledger::LedgerEvent::PodHypothesisCommitted { .. } => Some(event.clone()),
            _ => None,
        })
        .unwrap();
    let ptr_ledger::LedgerEvent::PodHypothesisCommitted { branch_id, .. } = &mut foreign else {
        unreachable!();
    };
    // Same request, Pod and output digest, but not the branch this admission
    // derives: those fields are not part of the output digest.
    *branch_id = "pod-branch:someone-else".into();

    let mut runtime = runtime_with_only_the_admission_record(&first);
    runtime.commit(foreign).unwrap();
    let hypotheses = |runtime: &PtrRuntime| {
        runtime
            .committed_events()
            .iter()
            .filter(|committed| {
                matches!(
                    committed.event,
                    ptr_ledger::LedgerEvent::PodHypothesisCommitted { .. }
                )
            })
            .count()
    };
    assert_eq!(hypotheses(&runtime), 1);
    let admitted = runtime
        .admit_pod_output(request(PodOutputKind::Hypothesis), &Pass)
        .unwrap();
    let PodOutputAdmission::Hypothesis { branch_id } = admitted else {
        panic!("expected a hypothesis");
    };
    // The retry commits the branch it reports instead of replaying on the foreign one.
    assert_eq!(hypotheses(&runtime), 2);
    assert!(runtime.committed_events().iter().any(|committed| matches!(
        &committed.event,
        ptr_ledger::LedgerEvent::PodHypothesisCommitted { branch_id: recorded, .. }
            if recorded == &branch_id
    )));
}

/// Issue #141: admission records the version-2 output digest; a record committed before
/// version 2 carries the version-1 digest and still replays; any other digest is refused.
#[test]
fn a_hypothesis_record_replays_under_the_current_or_the_legacy_output_digest_only() {
    let mut first = runtime_with_scope();
    first
        .admit_pod_output(request(PodOutputKind::Hypothesis), &Pass)
        .unwrap();
    let recorded = first
        .committed_events()
        .iter()
        .find_map(|committed| match &committed.event {
            event @ ptr_ledger::LedgerEvent::PodHypothesisCommitted { .. } => Some(event.clone()),
            _ => None,
        })
        .unwrap();
    let output = request(PodOutputKind::Hypothesis).output;
    let ptr_ledger::LedgerEvent::PodHypothesisCommitted { output_digest, .. } = &recorded else {
        unreachable!();
    };
    assert_eq!(*output_digest, output.digest());
    assert_ne!(*output_digest, output.legacy_v1_digest());

    let with_digest = |digest: ptr_types::Digest| {
        let mut event = recorded.clone();
        let ptr_ledger::LedgerEvent::PodHypothesisCommitted { output_digest, .. } = &mut event
        else {
            unreachable!();
        };
        *output_digest = digest;
        event
    };
    let admission = first
        .committed_events()
        .iter()
        .find(|c| matches!(c.event, ptr_ledger::LedgerEvent::PodOutputAdmitted { .. }))
        .unwrap()
        .event
        .clone();
    // Already in a log: the version-1 digest replays; any other digest is refused.
    let replayed = runtime_with(vec![
        admission.clone(),
        with_digest(output.legacy_v1_digest()),
    ]);
    assert!(replayed.committed_events().len() > 2);
    let mut events = runtime_with_scope().committed_events().to_vec();
    for event in [admission, with_digest([7; 32])] {
        let index = ptr_types::CommitIndex(events.len() as u64 + 1);
        events.push(ptr_ledger::CommittedEvent { index, event });
    }
    assert!(PtrRuntime::replay(PtrConfig::default(), &events).is_err());

    // Review of #141: a new record through the public commit path must carry version 2; the
    // version-1 digest is refused there and nothing is appended.
    let mut fresh = runtime_with_only_the_admission_record(&first);
    let before = fresh.committed_events().len();
    let refused = fresh.commit(with_digest(output.legacy_v1_digest()));
    assert!(
        matches!(&refused, Err(ptr_runtime::RuntimeError::PodOutputAdmission(m)) if m.contains("digest mismatch")),
        "{refused:?}"
    );
    assert_eq!(fresh.committed_events().len(), before);
    fresh.commit(with_digest(output.digest())).unwrap();
    assert!(fresh.commit(with_digest([7; 32])).is_err());
}

/// The admission (and, for a hypothesis, its follow-up) a pre-version-2 build would have written
/// for `request`: the records of a complete run with the output digest replaced by version 1.
fn legacy_records(
    request: &PodOutputAdmissionRequest,
    with_followup: bool,
) -> Vec<ptr_ledger::LedgerEvent> {
    let legacy = request.output.legacy_v1_digest();
    let hex: String = legacy.iter().map(|b| format!("{b:02x}")).collect();
    let mut first = runtime_with_scope();
    first.admit_pod_output(request.clone(), &Pass).unwrap();
    let mut out = Vec::new();
    for committed in first.committed_events() {
        match &committed.event {
            event @ ptr_ledger::LedgerEvent::PodOutputAdmitted { .. } => {
                let mut event = event.clone();
                if let ptr_ledger::LedgerEvent::PodOutputAdmitted { output_digest, .. } = &mut event
                {
                    *output_digest = legacy;
                }
                out.push(event);
            }
            event @ ptr_ledger::LedgerEvent::PodHypothesisCommitted { .. } if with_followup => {
                let mut event = event.clone();
                if let ptr_ledger::LedgerEvent::PodHypothesisCommitted {
                    output_digest,
                    branch_id,
                    ..
                } = &mut event
                {
                    *output_digest = legacy;
                    *branch_id = format!("pod-branch:request:pod:{hex}");
                }
                out.push(event);
            }
            _ => {}
        }
    }
    out
}

/// A runtime whose log holds the scoped fixture followed by `records`, written by an earlier
/// build: the records are replayed, as a log already on disk is, not committed anew (a new
/// record must carry the version-2 digest).
fn runtime_with(records: Vec<ptr_ledger::LedgerEvent>) -> PtrRuntime {
    // The scope is created live after the replay: a scope replayed from the log would need
    // recovery before it takes any work, which is not what these tests are about.
    let mut base = PtrRuntime::new(PtrConfig::default()).unwrap();
    base.ingest_text(RequestId::from("request"), "ground truth")
        .unwrap();
    let mut events = base.committed_events().to_vec();
    for event in records {
        let index = ptr_types::CommitIndex(events.len() as u64 + 1);
        events.push(ptr_ledger::CommittedEvent { index, event });
    }
    let mut runtime = PtrRuntime::replay(PtrConfig::default(), &events).unwrap();
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

/// Review of #141: a retry of an admission recorded under the version-1 output digest returns
/// the recorded result only when its follow-up is already in the ledger, and appends nothing.
#[test]
fn a_legacy_admission_with_its_follow_up_answers_a_retry_idempotently() {
    let hypothesis = request(PodOutputKind::Hypothesis);
    let legacy = hypothesis.output.legacy_v1_digest();
    let hex: String = legacy.iter().map(|b| format!("{b:02x}")).collect();
    let mut runtime = runtime_with(legacy_records(&hypothesis, true));
    let events = runtime.committed_events().len();
    for _ in 0..2 {
        let retried = runtime.admit_pod_output(hypothesis.clone(), &Pass).unwrap();
        assert_eq!(
            retried,
            PodOutputAdmission::Hypothesis {
                branch_id: format!("pod-branch:request:pod:{hex}")
            }
        );
        assert_eq!(runtime.committed_events().len(), events);
    }
}

/// Review of #141 (P1): version 1 is not injective. A lone version-1 admission whose follow-up
/// is missing is not completed by a retry, neither for the identical output nor for a different
/// output that has the same version-1 digest: the retry is refused with a typed error and
/// nothing is appended. A different output with another digest still conflicts, and a fresh
/// request is unaffected.
#[test]
fn an_incomplete_legacy_admission_is_refused_for_the_identical_and_a_colliding_output() {
    // Two different outputs that share their version-1 digest (see ptr-pods tests/output.rs).
    let source = "abcdefghijklmnopqrstuvw";
    let mut forged = [0u8; 32];
    forged[..8].copy_from_slice(&23u64.to_le_bytes());
    forged[8..31].copy_from_slice(source.as_bytes());
    let kept = ProvenanceRef {
        source: ptr_types::EvidenceId::from("zzz"),
        note: None,
    };
    // Both carry a manifest that also binds the forged digest, so it is a known dependency.
    let with_forged_binding = |mut request: PodOutputAdmissionRequest| {
        let manifest = ExecutionManifest::build(
            Generation(1),
            vec![
                LineageBinding {
                    key: "knowledge:fact".into(),
                    generation: Generation(1),
                    digest: [3; 32],
                },
                LineageBinding {
                    key: "knowledge:forged".into(),
                    generation: Generation(1),
                    digest: forged,
                },
            ],
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
        .unwrap();
        request.manifest_digest = manifest.manifest_digest;
        request.output.manifest_digest = manifest.manifest_digest;
        request.manifest = manifest;
        request
    };
    let mut admitted = with_forged_binding(request(PodOutputKind::Observation));
    admitted.output.dependencies = vec![forged];
    admitted.output.provenance = vec![kept.clone()];
    let mut colliding = with_forged_binding(request(PodOutputKind::Observation));
    colliding.output.dependencies = Vec::new();
    colliding.output.provenance = vec![
        ProvenanceRef {
            source: ptr_types::EvidenceId::from(source),
            note: None,
        },
        kept,
    ];
    assert_ne!(admitted.output, colliding.output);
    assert_eq!(
        admitted.output.legacy_v1_digest(),
        colliding.output.legacy_v1_digest()
    );

    // Crash between the admission record and its promotion: only the record is in the ledger.
    let mut runtime = runtime_with(legacy_records(&admitted, false));
    let events = runtime.committed_events().len();
    for retry in [admitted.clone(), colliding.clone()] {
        let refused = runtime.admit_pod_output(retry, &Pass);
        assert!(
            matches!(
                &refused,
                Err(ptr_runtime::RuntimeError::LegacyAdmissionIncomplete { request_id, pod_id })
                    if request_id == &RequestId::from("request") && pod_id.0 == "pod"
            ),
            "{refused:?}"
        );
        assert_eq!(runtime.committed_events().len(), events, "nothing appended");
    }
    // Review of #141 (round 5): a retry that carries an approximate KV taint is refused just
    // the same, before its taint is recorded: nothing at all is appended.
    for retry in [admitted.clone(), colliding] {
        let mut tainted = retry;
        tainted.kv_taint = ptr_model_api::KvReuseTaint::Approximate {
            report_digest: [5; 32],
        };
        let refused = runtime.admit_pod_output(tainted, &Pass);
        assert!(
            matches!(
                &refused,
                Err(ptr_runtime::RuntimeError::LegacyAdmissionIncomplete { .. })
            ),
            "{refused:?}"
        );
        assert_eq!(runtime.committed_events().len(), events, "nothing appended");
        assert!(!runtime.committed_events().iter().any(|committed| matches!(
            committed.event,
            ptr_ledger::LedgerEvent::KvTaintCommitted { .. }
        )));
    }
    // A different output with another digest is still a reused request id.
    let mut other = request(PodOutputKind::Observation);
    other.output.payload.bytes = b"another output".to_vec();
    let conflict = runtime.admit_pod_output(other, &Pass);
    assert!(
        matches!(&conflict, Err(ptr_runtime::RuntimeError::PodOutputAdmission(message)) if message.contains("different output")),
        "{conflict:?}"
    );
    // A fresh request is admitted under version 2 as usual.
    let mut fresh = request(PodOutputKind::Observation);
    fresh.request_id = RequestId::from("fresh");
    runtime
        .ingest_text(RequestId::from("fresh"), "ground truth")
        .unwrap();
    let PodOutputAdmission::ObservationCandidate { semantic_key, .. } =
        runtime.admit_pod_output(fresh.clone(), &Pass).unwrap()
    else {
        panic!("expected a candidate");
    };
    assert_eq!(
        semantic_key,
        ptr_runtime::semantic::pod_candidate_key(
            &RequestId::from("fresh"),
            &ptr_types::PodId::from("pod"),
            &fresh.output.digest(),
        )
    );
}

/// Review of #141: a hypothesis output marked verified is refused before anything is appended
/// (its record would be rebuilt unverified and fail its digest, leaving an admission no retry
/// could complete); an ordinary hypothesis still admits and replays.
#[test]
fn a_hypothesis_marked_verified_is_refused_before_any_append() {
    let mut runtime = runtime_with_scope();
    let events = runtime.committed_events().len();
    let mut verified = request(PodOutputKind::Hypothesis);
    verified.output.verified = true;
    let refused = runtime.admit_pod_output(verified, &Pass);
    assert!(
        matches!(&refused, Err(ptr_runtime::RuntimeError::PodOutputAdmission(m)) if m.contains("cannot be marked verified")),
        "{refused:?}"
    );
    assert_eq!(runtime.committed_events().len(), events, "nothing appended");
    let admitted = runtime
        .admit_pod_output(request(PodOutputKind::Hypothesis), &Pass)
        .unwrap();
    assert!(matches!(admitted, PodOutputAdmission::Hypothesis { .. }));
    PtrRuntime::replay(PtrConfig::default(), runtime.committed_events()).unwrap();
}

fn install_merge_grant(runtime: &mut PtrRuntime) {
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
}

fn triage() -> MergeAuthority {
    MergeAuthority::Triage {
        score: Probability::new(1.0).unwrap(),
    }
}

/// Issue #153 (3): a retry answered from an admission recorded under the version-1 digest is
/// taint-contained like a version-2 retry. With an approximate KV taint it is answered with the
/// recorded branch, behind one taint record (the version-1 and version-2 PodOutput subjects and
/// the payload's Content subject), so the hypothesis is never merged, also after a replay. The
/// same retry without the taint appends nothing and its branch merges.
#[test]
fn a_legacy_retry_with_an_approximate_taint_is_taint_contained() {
    let hypothesis = request(PodOutputKind::Hypothesis);
    let legacy = hypothesis.output.legacy_v1_digest();
    let hex: String = legacy.iter().map(|b| format!("{b:02x}")).collect();
    let answer = PodOutputAdmission::Hypothesis {
        branch_id: format!("pod-branch:request:pod:{hex}"),
    };
    let branch = format!("pod-branch:request:pod:{hex}");
    let is_taint = |event: &ptr_ledger::LedgerEvent| {
        matches!(event, ptr_ledger::LedgerEvent::KvTaintCommitted { .. })
    };

    // Control: an exact retry appends nothing and its branch merges.
    let mut clean = runtime_with(legacy_records(&hypothesis, true));
    let before = clean.committed_events().len();
    assert_eq!(
        clean.admit_pod_output(hypothesis.clone(), &Pass).unwrap(),
        answer
    );
    assert_eq!(clean.committed_events().len(), before);
    install_merge_grant(&mut clean);
    assert!(matches!(
        clean.merge_pod_hypothesis(&branch, triage()),
        Ok(ptr_runtime::MergeOutcome::Committed(_))
    ));

    let mut runtime = runtime_with(legacy_records(&hypothesis, true));
    let mut tainted = hypothesis.clone();
    tainted.kv_taint = ptr_model_api::KvReuseTaint::Approximate {
        report_digest: [5; 32],
    };
    let before = runtime.committed_events().len();
    assert_eq!(
        runtime.admit_pod_output(tainted.clone(), &Pass).unwrap(),
        answer
    );
    assert_eq!(runtime.committed_events().len(), before + 1);
    let last = &runtime.committed_events()[before].event;
    let ptr_ledger::LedgerEvent::KvTaintCommitted { subjects, .. } = last else {
        panic!("expected a taint record, got {last:?}");
    };
    assert_eq!(subjects.len(), 3);
    let kinds: Vec<_> = subjects.iter().map(|(kind, _)| *kind).collect();
    assert_eq!(
        kinds
            .iter()
            .filter(|kind| **kind == ptr_ledger::KvTaintSubjectKind::PodOutput)
            .count(),
        2
    );
    assert!(kinds.contains(&ptr_ledger::KvTaintSubjectKind::Content));
    // A second tainted retry finds every subject live: nothing more is appended.
    assert_eq!(runtime.admit_pod_output(tainted, &Pass).unwrap(), answer);
    assert_eq!(runtime.committed_events().len(), before + 1);
    assert_eq!(
        runtime
            .committed_events()
            .iter()
            .filter(|committed| is_taint(&committed.event))
            .count(),
        1
    );
    let replayed = PtrRuntime::replay(PtrConfig::default(), runtime.committed_events()).unwrap();
    for mut runtime in [runtime, replayed] {
        install_merge_grant(&mut runtime);
        let events = runtime.committed_events().len();
        let refused = runtime.merge_pod_hypothesis(&branch, triage());
        assert!(
            matches!(
                &refused,
                Err(ptr_runtime::RuntimeError::ApproximateSemanticContent {
                    boundary: ptr_runtime::kv_taint::SemanticTaintBoundary::BranchMerge,
                    report_digest,
                }) if *report_digest == [5; 32]
            ),
            "{refused:?}"
        );
        assert_eq!(runtime.committed_events().len(), events);
    }
}

/// Issue #153 (3): a hypothesis record holds the whole output, so a retry answered from a
/// version-1 admission is compared with it exactly, not only by the non-injective version-1
/// digest. A different output that shares the version-1 digest is a reused request id and is
/// refused, with or without a taint, appending nothing; the identical output is answered.
#[test]
fn a_colliding_retry_of_a_complete_legacy_hypothesis_is_refused() {
    let source = "abcdefghijklmnopqrstuvw";
    let mut forged = [0u8; 32];
    forged[..8].copy_from_slice(&23u64.to_le_bytes());
    forged[8..31].copy_from_slice(source.as_bytes());
    let kept = ProvenanceRef {
        source: ptr_types::EvidenceId::from("zzz"),
        note: None,
    };
    let manifest = ExecutionManifest::build(
        Generation(1),
        vec![
            LineageBinding {
                key: "knowledge:fact".into(),
                generation: Generation(1),
                digest: [3; 32],
            },
            LineageBinding {
                key: "knowledge:forged".into(),
                generation: Generation(1),
                digest: forged,
            },
        ],
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
    .unwrap();
    let mut admitted = request(PodOutputKind::Hypothesis);
    admitted.manifest_digest = manifest.manifest_digest;
    admitted.output.manifest_digest = manifest.manifest_digest;
    admitted.manifest = manifest;
    let mut colliding = admitted.clone();
    admitted.output.dependencies = vec![forged];
    admitted.output.provenance = vec![kept.clone()];
    colliding.output.dependencies = Vec::new();
    colliding.output.provenance = vec![
        ProvenanceRef {
            source: ptr_types::EvidenceId::from(source),
            note: None,
        },
        kept,
    ];
    assert_ne!(admitted.output, colliding.output);
    assert_eq!(
        admitted.output.legacy_v1_digest(),
        colliding.output.legacy_v1_digest()
    );

    let mut runtime = runtime_with(legacy_records(&admitted, true));
    let events = runtime.committed_events().len();
    let mut tainted = colliding.clone();
    tainted.kv_taint = ptr_model_api::KvReuseTaint::Approximate {
        report_digest: [5; 32],
    };
    for retry in [colliding, tainted] {
        let refused = runtime.admit_pod_output(retry, &Pass);
        assert!(
            matches!(&refused, Err(ptr_runtime::RuntimeError::PodOutputAdmission(message)) if message.contains("different output")),
            "{refused:?}"
        );
        assert_eq!(runtime.committed_events().len(), events, "nothing appended");
    }
    assert!(matches!(
        runtime.admit_pod_output(admitted, &Pass).unwrap(),
        PodOutputAdmission::Hypothesis { .. }
    ));
    assert_eq!(runtime.committed_events().len(), events);
}
