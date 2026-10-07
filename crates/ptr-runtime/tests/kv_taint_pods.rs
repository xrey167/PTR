//! Approximate-KV taint containment at the Pod-output, SemDB and merge boundaries: tainted
//! outputs of promoting or acting kinds are refused before the verifier runs, the other kinds
//! are stored as tainted candidates or hypotheses behind a taint record, and tainted content
//! can be neither promoted, host-written nor merged, whatever the merge authority.

use ptr_branch::{AutoThreshold, Branch, BranchId, PolicyRecord, TriagePolicy};
use ptr_config::PtrConfig;
use ptr_ledger::{KvTaintSubjectKind, LedgerEvent, SemanticOrigin};
use ptr_model_api::{
    ByteTokenizer, KvReuseMode, KvReusePolicy, KvReuseReport, KvReuseTaint, ModelRun,
    PromptSegment, SegmentedModelRequest, Tokenizer as _,
};
use ptr_pods::{ExecutionManifest, LineageBinding, PodManifest, PodOutput, PodOutputKind};
use ptr_protocol::TypedPayload;
use ptr_runtime::execution::RequiredVerification;
use ptr_runtime::kv_taint::{KvTaintRequest, SemanticTaintBoundary};
use ptr_runtime::{
    ExecutionScope, MergeAuthority, MergeOutcome, PodOutputAdmission, PodOutputAdmissionRequest,
    PtrRuntime, RuntimeError, ScopeState, SemanticChange, SemanticGrant,
};
use ptr_semdb::{SemanticDelta, SemanticPayload, SemanticValue};
use ptr_types::{
    Effect, Generation, PodId, PrincipalId, Probability, ProjectId, ProvenanceRef, RequestId,
    Revision, ScopeId, ScopeLeaseBinding, SemanticContext, SessionId, Timestamp, TypeId,
    VerificationLevel,
};
use ptr_verifier::{NamedVerifier, VerificationReport, VerificationStatus, Verifier};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

const REPORT: [u8; 32] = [7; 32];
const PAYLOAD: &[u8] = b"observation";

fn pass_report() -> VerificationReport {
    VerificationReport {
        status: VerificationStatus::Pass,
        level: VerificationLevel::Deterministic,
        score: Probability::new(1.0).unwrap(),
        findings: vec![],
    }
}

/// Passing verifier that counts how often it was asked.
#[derive(Clone, Default)]
struct Counting(Arc<AtomicUsize>);

impl Counting {
    fn calls(&self) -> usize {
        self.0.load(Ordering::SeqCst)
    }
}

impl Verifier<TypedPayload> for Counting {
    fn verify(&self, _: &TypedPayload) -> VerificationReport {
        self.0.fetch_add(1, Ordering::SeqCst);
        pass_report()
    }
}

#[derive(Clone)]
struct MergePass;

impl<'a> Verifier<SemanticChange<'a>> for MergePass {
    fn verify(&self, _: &SemanticChange<'a>) -> VerificationReport {
        pass_report()
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

fn approximate() -> KvReuseTaint {
    KvReuseTaint::Approximate {
        report_digest: REPORT,
    }
}

fn request(kind: PodOutputKind, kv_taint: KvReuseTaint) -> PodOutputAdmissionRequest {
    let manifest = manifest();
    let manifest_digest = manifest.manifest_digest;
    PodOutputAdmissionRequest {
        request_id: RequestId::from("request"),
        manifest_digest: manifest.manifest_digest,
        manifest,
        pod_id: PodId::from("pod"),
        output: PodOutput {
            kind,
            payload: TypedPayload {
                type_id: TypeId::from("text"),
                bytes: PAYLOAD.to_vec(),
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
            verified: matches!(
                kind,
                PodOutputKind::StateDelta | PodOutputKind::VerifiedResult
            ),
        },
        expected_type: TypeId::from("text"),
        session_id: SessionId::from("session"),
        scope_id: ScopeId::from("scope"),
        placement_epoch: ptr_runtime::PlacementEpoch(1),
        fencing_token: ptr_runtime::FencingToken(1),
        kv_taint,
    }
}

fn pod_manifest() -> PodManifest {
    PodManifest {
        project: ProjectId::from("project"),
        id: PodId::from("pod"),
        capabilities: vec![],
        accepts: vec![TypeId::from("text")],
        produces: vec![TypeId::from("text")],
        effects: vec![Effect::Pure],
        protocol_version: 1,
    }
}

fn text(bytes: &[u8]) -> TypedPayload {
    TypedPayload {
        type_id: TypeId::from("text"),
        bytes: bytes.to_vec(),
    }
}

fn segmented(id: &str) -> SegmentedModelRequest {
    let tokenizer = ByteTokenizer::new();
    let segment = PromptSegment::raw_from_text(&tokenizer, "hello taint").unwrap();
    SegmentedModelRequest::new(
        RequestId::from(id),
        SemanticContext {
            revision: Revision(0),
            entries: Vec::new(),
        },
        tokenizer.identity().clone(),
        vec![segment],
    )
    .unwrap()
}

/// The runtime ingests an Approximate model run for `id`, recording the request subject.
fn ingest_approximate_run(runtime: &mut PtrRuntime, id: &str) {
    runtime
        .ingest_model_run_for(
            &KvTaintRequest {
                request: &segmented(id),
                producer: "stub-approximate-producer/1",
                generation_parameters: b"max_new_tokens=8",
            },
            &ModelRun {
                events: Vec::new(),
                kv_report: Some(
                    KvReuseReport::new(
                        16,
                        KvReuseMode::Approximate,
                        64,
                        KvReusePolicy::AllowApproximate,
                    )
                    .unwrap(),
                ),
            },
        )
        .unwrap();
}

fn position(runtime: &PtrRuntime, wanted: impl Fn(&LedgerEvent) -> bool) -> Option<usize> {
    runtime
        .committed_events()
        .iter()
        .position(|committed| wanted(&committed.event))
}

fn taint_record_count(runtime: &PtrRuntime) -> usize {
    runtime
        .committed_events()
        .iter()
        .filter(|committed| matches!(committed.event, LedgerEvent::KvTaintCommitted { .. }))
        .count()
}

fn is_approximate_semantic(
    result: &Result<impl std::fmt::Debug, RuntimeError>,
    expected: SemanticTaintBoundary,
) -> bool {
    matches!(
        result,
        Err(RuntimeError::ApproximateSemanticContent { boundary, report_digest })
            if *boundary == expected && *report_digest == REPORT
    )
}

fn install_grant(runtime: &mut PtrRuntime) {
    runtime
        .install_semantic_grant(
            SemanticGrant::new(RequiredVerification::Deterministic)
                .with_verifier(MergePass)
                .allow_host_writes()
                .with_merge_policy(
                    PolicyRecord::manual(
                        "pod-merge-policy",
                        TriagePolicy::new(AutoThreshold::AtLeast(0.8), 0.0).unwrap(),
                    )
                    .unwrap(),
                    7,
                )
                .with_reviewer(PrincipalId::from("reviewer-1")),
        )
        .unwrap();
}

#[test]
fn promoting_and_acting_kinds_are_refused_before_the_verifier_and_the_ledger() {
    for kind in [
        PodOutputKind::StateDelta,
        PodOutputKind::VerifiedResult,
        PodOutputKind::ActionProposal,
    ] {
        let mut runtime = runtime_with_scope();
        let before = runtime.committed_events().len();
        let verifier = Counting::default();
        let result = runtime.admit_pod_output(request(kind, approximate()), &verifier);
        assert!(
            matches!(
                &result,
                Err(RuntimeError::ApproximatePodOutput { kind: refused, report_digest })
                    if *refused == kind && *report_digest == REPORT
            ),
            "{kind:?}: {result:?}"
        );
        assert_eq!(verifier.calls(), 0, "{kind:?}: the verifier must not run");
        assert_eq!(
            runtime.committed_events().len(),
            before,
            "{kind:?}: appended"
        );
    }
}

#[test]
fn candidate_kinds_and_hypotheses_are_stored_behind_a_taint_record() {
    for kind in [
        PodOutputKind::Observation,
        PodOutputKind::Candidate,
        PodOutputKind::ToolResult,
        PodOutputKind::EnvironmentObservation,
        PodOutputKind::Hypothesis,
    ] {
        let mut runtime = runtime_with_scope();
        let admitted = runtime
            .admit_pod_output(request(kind, approximate()), &Counting::default())
            .unwrap_or_else(|error| panic!("{kind:?}: {error:?}"));
        match (kind, &admitted) {
            (PodOutputKind::Hypothesis, PodOutputAdmission::Hypothesis { .. }) => {}
            (_, PodOutputAdmission::ObservationCandidate { .. }) => {}
            other => panic!("{kind:?}: unexpected admission {other:?}"),
        }
        // Exactly one taint record, naming the PodOutput and the payload Content subject, and
        // it precedes the admission and the candidate or hypothesis record.
        assert_eq!(taint_record_count(&runtime), 1, "{kind:?}");
        let taint_at = position(&runtime, |e| {
            matches!(e, LedgerEvent::KvTaintCommitted { .. })
        })
        .unwrap();
        let admitted_at = position(&runtime, |e| {
            matches!(e, LedgerEvent::PodOutputAdmitted { .. })
        })
        .unwrap();
        let stored_at = position(&runtime, |e| match e {
            LedgerEvent::PodHypothesisCommitted { .. } => kind == PodOutputKind::Hypothesis,
            LedgerEvent::SemanticDeltaCommitted {
                origin: SemanticOrigin::PodCandidate { .. },
                ..
            } => kind != PodOutputKind::Hypothesis,
            _ => false,
        })
        .unwrap();
        assert!(
            taint_at < admitted_at && admitted_at < stored_at,
            "{kind:?}"
        );
        let LedgerEvent::KvTaintCommitted { subjects, .. } =
            &runtime.committed_events()[taint_at].event
        else {
            unreachable!()
        };
        let kinds: Vec<_> = subjects.iter().map(|(kind, _)| *kind).collect();
        assert!(kinds.contains(&KvTaintSubjectKind::PodOutput), "{kind:?}");
        assert!(kinds.contains(&KvTaintSubjectKind::Content), "{kind:?}");

        // The stored content stays refused later, also after a replay, while other content
        // from the same Pod is promoted as before.
        let replayed =
            PtrRuntime::replay(PtrConfig::default(), runtime.committed_events()).unwrap();
        for mut runtime in [runtime, replayed] {
            runtime
                .ingest_text(RequestId::from("later"), "later input")
                .unwrap();
            let before = runtime.committed_events().len();
            let refused = runtime.promote_verified_pod_output(
                &RequestId::from("later"),
                &pod_manifest(),
                &text(PAYLOAD),
                &Counting::default(),
            );
            assert!(
                is_approximate_semantic(&refused, SemanticTaintBoundary::PromoteVerifiedPodOutput),
                "{kind:?}: {refused:?}"
            );
            assert_eq!(runtime.committed_events().len(), before);
            runtime
                .promote_verified_pod_output(
                    &RequestId::from("later"),
                    &pod_manifest(),
                    &text(b"unrelated output"),
                    &Counting::default(),
                )
                .unwrap();
        }
    }
}

#[test]
fn a_caller_saying_exact_does_not_shed_what_the_runtime_recorded() {
    let mut runtime = runtime_with_scope();
    runtime
        .ingest_text(RequestId::from("clean"), "clean input")
        .unwrap();
    ingest_approximate_run(&mut runtime, "request");

    for kind in [
        PodOutputKind::StateDelta,
        PodOutputKind::VerifiedResult,
        PodOutputKind::ActionProposal,
    ] {
        let before = runtime.committed_events().len();
        let verifier = Counting::default();
        let result = runtime.admit_pod_output(request(kind, KvReuseTaint::Exact), &verifier);
        assert!(
            matches!(&result, Err(RuntimeError::ApproximatePodOutput { .. })),
            "{kind:?}: {result:?}"
        );
        assert_eq!(verifier.calls(), 0);
        assert_eq!(runtime.committed_events().len(), before);
    }

    // A candidate of the same request is stored, tainted, although the caller said Exact.
    let before = taint_record_count(&runtime);
    runtime
        .admit_pod_output(
            request(PodOutputKind::Observation, KvReuseTaint::Exact),
            &Counting::default(),
        )
        .unwrap();
    assert_eq!(taint_record_count(&runtime), before + 1);
    runtime
        .ingest_text(RequestId::from("later"), "later input")
        .unwrap();
    assert!(runtime
        .promote_verified_pod_output(
            &RequestId::from("later"),
            &pod_manifest(),
            &text(PAYLOAD),
            &Counting::default(),
        )
        .is_err());

    // Another request that was never tainted behaves as before.
    let mut clean = request(PodOutputKind::StateDelta, KvReuseTaint::Exact);
    clean.request_id = RequestId::from("clean");
    clean.output.payload.bytes = b"clean state".to_vec();
    assert!(matches!(
        runtime.admit_pod_output(clean, &Counting::default()),
        Ok(PodOutputAdmission::StateDelta { .. })
    ));
}

#[test]
fn exact_pod_outputs_are_unchanged_and_write_no_taint_record() {
    let mut runtime = runtime_with_scope();
    runtime
        .ingest_text(RequestId::from("state"), "state input")
        .unwrap();
    runtime
        .admit_pod_output(
            request(PodOutputKind::Observation, KvReuseTaint::Exact),
            &Counting::default(),
        )
        .unwrap();
    let mut hypothesis = request(PodOutputKind::Hypothesis, KvReuseTaint::Exact);
    hypothesis.request_id = RequestId::from("hypothesis");
    runtime
        .admit_pod_output(hypothesis, &Counting::default())
        .unwrap();
    let mut state = request(PodOutputKind::StateDelta, KvReuseTaint::Exact);
    state.request_id = RequestId::from("state");
    runtime
        .admit_pod_output(state, &Counting::default())
        .unwrap();
    assert_eq!(taint_record_count(&runtime), 0);
}

fn merge_authorities() -> Vec<(&'static str, MergeAuthority)> {
    vec![
        (
            "triage 1.0",
            MergeAuthority::Triage {
                score: Probability::new(1.0).unwrap(),
            },
        ),
        (
            "reviewed",
            MergeAuthority::Reviewed {
                plan_digest: [0; 32],
                reviewer: PrincipalId::from("reviewer-1"),
            },
        ),
    ]
}

#[test]
fn a_branch_holding_a_tainted_hypothesis_is_never_merged_or_previewed() {
    let mut runtime = runtime_with_scope();
    install_grant(&mut runtime);
    let branch_id = match runtime
        .admit_pod_output(
            request(PodOutputKind::Hypothesis, approximate()),
            &Counting::default(),
        )
        .unwrap()
    {
        PodOutputAdmission::Hypothesis { branch_id } => branch_id,
        other => panic!("unexpected admission: {other:?}"),
    };
    let before = runtime.committed_events().len();
    for (name, authority) in merge_authorities() {
        let refused = runtime.merge_pod_hypothesis(&branch_id, authority);
        assert!(
            is_approximate_semantic(&refused, SemanticTaintBoundary::BranchMerge),
            "{name}: {refused:?}"
        );
    }
    // The same branch sealed by hand (merge_branch and preview_merge directly).
    let mut work = Branch::open(
        BranchId::from(branch_id.as_str()),
        PrincipalId::from("pod"),
        runtime.snapshot(),
    );
    work.read("pod-hypothesis:x").unwrap();
    work.put(
        "pod-hypothesis:x",
        SemanticValue::Payload(SemanticPayload {
            type_id: TypeId::from("text"),
            source: "pod".into(),
            bytes: PAYLOAD.to_vec(),
        }),
    )
    .unwrap();
    let sealed = work.seal().unwrap();
    for (name, authority) in merge_authorities() {
        let refused = runtime.merge_branch(&sealed, authority);
        assert!(
            is_approximate_semantic(&refused, SemanticTaintBoundary::BranchMerge),
            "{name}: {refused:?}"
        );
    }
    assert!(is_approximate_semantic(
        &runtime.preview_merge(&sealed),
        SemanticTaintBoundary::BranchMerge
    ));
    // A branch under another id that puts the tainted content as text is refused as well.
    let mut work = Branch::open(
        BranchId::from("copy-branch"),
        PrincipalId::from("agent"),
        runtime.snapshot(),
    );
    work.read("note:copy").unwrap();
    work.put("note:copy", "observation".into()).unwrap();
    let copy = work.seal().unwrap();
    assert!(is_approximate_semantic(
        &runtime.preview_merge(&copy),
        SemanticTaintBoundary::BranchMerge
    ));
    assert!(is_approximate_semantic(
        &runtime.merge_branch(
            &copy,
            MergeAuthority::Triage {
                score: Probability::new(1.0).unwrap()
            }
        ),
        SemanticTaintBoundary::BranchMerge
    ));
    assert_eq!(runtime.committed_events().len(), before, "nothing appended");

    // Positive control: a clean branch merges under Triage 1.0, and previews.
    let mut work = Branch::open(
        BranchId::from("clean-branch"),
        PrincipalId::from("agent"),
        runtime.snapshot(),
    );
    work.read("price:sku-1").unwrap();
    work.put("price:sku-1", "11".into()).unwrap();
    let clean = work.seal().unwrap();
    assert!(runtime.preview_merge(&clean).is_ok());
    assert!(matches!(
        runtime.merge_branch(
            &clean,
            MergeAuthority::Triage {
                score: Probability::new(1.0).unwrap()
            }
        ),
        Ok(MergeOutcome::Committed(_))
    ));
}

#[test]
fn a_host_write_of_tainted_content_is_refused_and_other_values_are_accepted() {
    let mut runtime = runtime_with_scope();
    install_grant(&mut runtime);
    runtime
        .admit_pod_output(
            request(PodOutputKind::Observation, approximate()),
            &Counting::default(),
        )
        .unwrap();
    let principal = PrincipalId::from("operator");
    let write = |runtime: &mut PtrRuntime, value: SemanticValue| {
        let mut delta = SemanticDelta::default();
        delta.upserts.insert("note:host".into(), value);
        let revision = runtime.revision();
        runtime.apply_verified_semantic_delta(revision, delta, &principal)
    };
    let before = runtime.committed_events().len();
    let as_text = write(&mut runtime, "observation".into());
    assert!(
        is_approximate_semantic(&as_text, SemanticTaintBoundary::HostWrite),
        "{as_text:?}"
    );
    let as_payload = write(
        &mut runtime,
        SemanticValue::Payload(SemanticPayload {
            type_id: TypeId::from("text"),
            source: "operator".into(),
            bytes: PAYLOAD.to_vec(),
        }),
    );
    assert!(is_approximate_semantic(
        &as_payload,
        SemanticTaintBoundary::HostWrite
    ));
    assert_eq!(runtime.committed_events().len(), before);
    assert!(write(&mut runtime, "a different value".into()).is_ok());
}
