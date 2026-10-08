//! Evidence binding: a gate report is evidence for an expert change only if its subject names the
//! artifact the change produces or acts on, the change's base model and the evaluation manifest
//! the host expects. Admission and readmission share one check; replay repeats the artifact and
//! base model checks (the grant, and with it the expected evaluation manifest, is not recorded).

mod expert_support;

use expert_support::{binding, prime, requirement, BASE_MODEL_DIGEST, EVAL_MANIFEST};
use ptr_config::PtrConfig;
use ptr_ledger::{
    integrity, CommittedEvent, ExpertFenceAction, ExpertFenceCause, ExpertFenceEvidence,
    ExpertFenceResume, LedgerEvent,
};
use ptr_lineage::{
    AccuracyMatrix, AdapterId, BoundGateReport, ForgettingGate, GateSubject, PublicSuite,
};
use ptr_pods::{
    BaseModelDigest, BaseModelId, BaseModelRef, ExpertCommitRecord, ExpertLevel, ExpertPodSpec,
    ExpertProposal, ExpertTarget, ExpertTargetInit, ManifestDigest, ProposerRef, RationaleDigest,
    RetiredReason, StructuralOp,
};
use ptr_runtime::expert::{
    expert_evidence_artifact_digest, expert_evidence_base_model_digest,
    expert_readmission_subject_ids, ledger_op_kind, ExpertAdmissionError, ExpertAdmissionGrant,
    ExpertFenceGrant, ExpertFenceRequest,
};
use ptr_runtime::{PtrRuntime, RuntimeError};
use ptr_types::{CommitIndex, Generation, PodId, ProjectId};

const LAYER: u16 = 3;
const EPOCH: u64 = 7;
const QUARANTINE: ExpertFenceCause = ExpertFenceCause::AuthorityQuarantine;

fn project() -> ProjectId {
    ProjectId::from("proj")
}

fn base() -> BaseModelRef {
    BaseModelRef::new(
        BaseModelId::new("base-a").unwrap(),
        BaseModelDigest::new(BASE_MODEL_DIGEST).unwrap(),
        1,
    )
    .unwrap()
}

fn proposer() -> ProposerRef {
    ProposerRef::new(project(), PodId::from("meta"), Generation(1)).unwrap()
}

fn gate() -> ForgettingGate {
    ForgettingGate {
        max_average_forgetting: 0.5,
        max_task_forgetting: 0.5,
        min_backward_transfer: -0.5,
        max_public_regression: 0.02,
    }
}

fn gate_digest() -> [u8; 32] {
    gate().config_digest().unwrap()
}

fn grant() -> ExpertAdmissionGrant {
    ExpertAdmissionGrant::new(
        project(),
        proposer(),
        requirement(EPOCH, gate_digest()),
        binding(),
    )
}

fn gate_grant() -> ExpertFenceGrant {
    ExpertFenceGrant::gate(project(), requirement(EPOCH, gate_digest()), binding())
}

fn revocation_grant() -> ExpertFenceGrant {
    ExpertFenceGrant::revocation(project(), binding())
}

fn bind(subject: GateSubject) -> BoundGateReport {
    let matrix = AccuracyMatrix::new(vec![vec![0.9, 0.0], vec![0.9, 0.9]]).unwrap();
    gate()
        .evaluate(
            &AdapterId::from("candidate"),
            &matrix,
            PublicSuite {
                serving: 0.8,
                candidate: 0.8,
            },
        )
        .unwrap()
        .bind(subject)
}

/// What a report's subject gets wrong.
#[derive(Clone, Copy, Debug)]
enum Wrong {
    Artifact,
    BaseModel,
    EvalManifest,
}

const WRONGS: [Wrong; 3] = [Wrong::Artifact, Wrong::BaseModel, Wrong::EvalManifest];

fn refusal_of(wrong: Wrong) -> ExpertAdmissionError {
    match wrong {
        Wrong::Artifact => ExpertAdmissionError::EvidenceArtifactMismatch,
        Wrong::BaseModel => ExpertAdmissionError::EvidenceBaseModelMismatch,
        Wrong::EvalManifest => ExpertAdmissionError::EvidenceEvalManifestMismatch,
    }
}

/// The digest triple of a correct subject, with `wrong` replaced by a digest nobody expects.
fn digests(
    artifact: [u8; 32],
    base_model: [u8; 32],
    wrong: Option<Wrong>,
) -> ([u8; 32], [u8; 32], [u8; 32]) {
    let artifact = if matches!(wrong, Some(Wrong::Artifact)) {
        [0xA1; 32]
    } else {
        artifact
    };
    let base_model = if matches!(wrong, Some(Wrong::BaseModel)) {
        [0xB2; 32]
    } else {
        base_model
    };
    let eval = if matches!(wrong, Some(Wrong::EvalManifest)) {
        [0xC3; 32]
    } else {
        EVAL_MANIFEST
    };
    (artifact, base_model, eval)
}

fn subject_of(proposal: &ExpertProposal, wrong: Option<Wrong>) -> GateSubject {
    let (artifact, base_model, eval) = digests(
        expert_evidence_artifact_digest(proposal.op()),
        expert_evidence_base_model_digest(proposal.op()).unwrap(),
        wrong,
    );
    GateSubject::new(
        *proposal.proposal_id().as_bytes(),
        *proposal.operation_id().as_bytes(),
        artifact,
        base_model,
        eval,
    )
}

fn level_of(pod: &str) -> ExpertLevel {
    ExpertLevel::Token {
        layer: LAYER,
        index: u16::from(pod.as_bytes()[0]),
    }
}

fn spec_at(pod: &str, generation: u64) -> ExpertPodSpec {
    let mut digest = [1u8; 32];
    digest[0] = pod.as_bytes()[0];
    digest[1] = generation as u8;
    ExpertPodSpec::new(
        project(),
        PodId::from(pod),
        ManifestDigest::new(digest).unwrap(),
        level_of(pod),
        base(),
        Generation(generation),
    )
    .unwrap()
}

fn proposal(op: StructuralOp) -> ExpertProposal {
    ExpertProposal::new(
        project(),
        proposer(),
        CommitIndex(1),
        op,
        RationaleDigest::new([7; 32]).unwrap(),
    )
    .unwrap()
}

fn runtime() -> PtrRuntime {
    let mut rt = PtrRuntime::new(PtrConfig::default()).unwrap();
    prime(&mut rt);
    rt
}

fn admit_with(
    rt: &mut PtrRuntime,
    op: StructuralOp,
    wrong: Option<Wrong>,
) -> Result<CommitIndex, RuntimeError> {
    let proposal = proposal(op);
    let report = bind(subject_of(&proposal, wrong));
    rt.admit_expert_proposal(&proposal, &report, &grant())
}

fn admit(rt: &mut PtrRuntime, op: StructuralOp) {
    admit_with(rt, op, None).unwrap();
}

fn target(rt: &PtrRuntime, pod: &str) -> ExpertTarget {
    let entry = rt
        .expert_registry()
        .get(&project(), &PodId::from(pod))
        .unwrap_or_else(|| panic!("no expert {pod}"))
        .clone();
    ExpertTarget::new(ExpertTargetInit {
        project: project(),
        pod: PodId::from(pod),
        generation: entry.generation,
        level: entry.level,
        base_model: entry.base_model,
        manifest_digest: entry.manifest_digest,
        state: entry.state,
        state_version: entry.state_version,
    })
    .unwrap()
}

fn birth(rt: &mut PtrRuntime, pod: &str) {
    admit(
        rt,
        StructuralOp::Birth {
            result: spec_at(pod, 1),
        },
    );
}

#[derive(Clone, Copy, Debug)]
enum Scenario {
    Birth,
    Merge,
    Split,
    Freeze,
    Unfreeze,
    ReactivateRetired,
    PruneOne,
    PruneMany,
    Retire,
}

const SCENARIOS: [Scenario; 9] = [
    Scenario::Birth,
    Scenario::Merge,
    Scenario::Split,
    Scenario::Freeze,
    Scenario::Unfreeze,
    Scenario::ReactivateRetired,
    Scenario::PruneOne,
    Scenario::PruneMany,
    Scenario::Retire,
];

/// A runtime whose registry holds what `scenario` operates on.
fn seeded(scenario: Scenario) -> PtrRuntime {
    let mut rt = runtime();
    match scenario {
        Scenario::Birth => {}
        Scenario::Merge | Scenario::PruneMany => {
            birth(&mut rt, "a");
            birth(&mut rt, "b");
        }
        Scenario::Split | Scenario::Freeze | Scenario::Retire | Scenario::PruneOne => {
            birth(&mut rt, "a");
        }
        Scenario::Unfreeze => {
            birth(&mut rt, "a");
            let freeze = StructuralOp::Freeze {
                target: target(&rt, "a"),
            };
            admit(&mut rt, freeze);
        }
        Scenario::ReactivateRetired => {
            birth(&mut rt, "a");
            let retire = StructuralOp::Retire {
                target: target(&rt, "a"),
                reason: RetiredReason::GateFailed,
            };
            admit(&mut rt, retire);
        }
    }
    rt
}

fn operation(scenario: Scenario, rt: &PtrRuntime) -> StructuralOp {
    match scenario {
        Scenario::Birth => StructuralOp::Birth {
            result: spec_at("n", 1),
        },
        Scenario::Merge => StructuralOp::Merge {
            sources: vec![target(rt, "a"), target(rt, "b")],
            result: spec_at("m", 1),
        },
        Scenario::Split => StructuralOp::Split {
            source: target(rt, "a"),
            results: vec![spec_at("x", 1), spec_at("y", 1)],
        },
        Scenario::Freeze => StructuralOp::Freeze {
            target: target(rt, "a"),
        },
        Scenario::Unfreeze => StructuralOp::Reactivate {
            target: target(rt, "a"),
            result: None,
        },
        Scenario::ReactivateRetired => {
            let target = target(rt, "a");
            let next = spec_at("a", target.generation().0 + 1);
            StructuralOp::Reactivate {
                target,
                result: Some(next),
            }
        }
        Scenario::PruneOne => StructuralOp::Prune {
            targets: vec![target(rt, "a")],
        },
        Scenario::PruneMany => StructuralOp::Prune {
            targets: vec![target(rt, "a"), target(rt, "b")],
        },
        Scenario::Retire => StructuralOp::Retire {
            target: target(rt, "a"),
            reason: RetiredReason::Superseded,
        },
    }
}

fn expert_error(result: Result<CommitIndex, RuntimeError>) -> ExpertAdmissionError {
    match result {
        Err(RuntimeError::Expert(error)) => error,
        other => panic!("expected an expert refusal, got {other:?}"),
    }
}

/// The manifest digest `spec_at(pod, generation)` gives its expert.
fn manifest_of(pod: &str, generation: u64) -> [u8; 32] {
    let mut digest = [1u8; 32];
    digest[0] = pod.as_bytes()[0];
    digest[1] = generation as u8;
    digest
}

/// The documented encoding of a digest set: length-prefixed domain, count, digests in order.
fn set_digest(domain: &str, digests: &[[u8; 32]]) -> [u8; 32] {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&(domain.len() as u64).to_le_bytes());
    bytes.extend_from_slice(domain.as_bytes());
    bytes.extend_from_slice(&(digests.len() as u64).to_le_bytes());
    for digest in digests {
        bytes.extend_from_slice(digest);
    }
    integrity::sha256(&bytes)
}

// ---- the expected digests -----------------------------------------------------------------

#[test]
fn the_artifact_digest_is_the_result_or_operand_manifest_digest_per_operation() {
    let one = manifest_of;
    for scenario in SCENARIOS {
        let rt = seeded(scenario);
        let op = operation(scenario, &rt);
        let expected = match scenario {
            Scenario::Birth => one("n", 1),
            Scenario::Merge => one("m", 1),
            Scenario::Split => set_digest(
                "ptr.expert.evidence.artifact-set.split-results.v1",
                &[one("x", 1), one("y", 1)],
            ),
            Scenario::Freeze | Scenario::Unfreeze | Scenario::Retire | Scenario::PruneOne => {
                one("a", 1)
            }
            Scenario::ReactivateRetired => one("a", 2),
            Scenario::PruneMany => set_digest(
                "ptr.expert.evidence.artifact-set.prune-targets.v1",
                &[one("a", 1), one("b", 1)],
            ),
        };
        assert_eq!(
            expert_evidence_artifact_digest(&op),
            expected,
            "{scenario:?}"
        );
        assert_eq!(
            expert_evidence_base_model_digest(&op),
            Some(BASE_MODEL_DIGEST),
            "{scenario:?}"
        );
    }
}

#[test]
fn set_digests_are_ordered_domain_separated_and_never_a_single_manifest_digest() {
    let rt = seeded(Scenario::Split);
    let split = |first: &str, second: &str| StructuralOp::Split {
        source: target(&rt, "a"),
        results: vec![spec_at(first, 1), spec_at(second, 1)],
    };
    let forward = expert_evidence_artifact_digest(&split("x", "y"));
    let reversed = expert_evidence_artifact_digest(&split("y", "x"));
    assert_ne!(forward, reversed, "the order of the results is part of it");
    for single in [spec_at("x", 1), spec_at("y", 1)] {
        assert_ne!(forward, *single.manifest_digest().as_bytes());
    }
    // The same two digests under the prune domain give another digest.
    let x = *spec_at("x", 1).manifest_digest().as_bytes();
    let y = *spec_at("y", 1).manifest_digest().as_bytes();
    assert_ne!(
        set_digest("ptr.expert.evidence.artifact-set.split-results.v1", &[x, y]),
        set_digest("ptr.expert.evidence.artifact-set.prune-targets.v1", &[x, y])
    );
}

// ---- admission ---------------------------------------------------------------------------

#[test]
fn a_report_bound_exactly_is_admitted_for_every_operation() {
    for scenario in SCENARIOS {
        let mut rt = seeded(scenario);
        let before = rt.committed_events().len();
        let op = operation(scenario, &rt);
        admit_with(&mut rt, op, None)
            .unwrap_or_else(|error| panic!("{scenario:?}: control refused: {error:?}"));
        assert_eq!(rt.committed_events().len(), before + 1, "{scenario:?}");
    }
}

#[test]
fn a_report_for_another_artifact_base_model_or_evaluation_is_refused_and_appends_nothing() {
    for scenario in SCENARIOS {
        for wrong in WRONGS {
            let mut rt = seeded(scenario);
            let before = rt.committed_events().len();
            let registry = rt.expert_registry().clone();
            let op = operation(scenario, &rt);
            assert_eq!(
                expert_error(admit_with(&mut rt, op.clone(), Some(wrong))),
                refusal_of(wrong),
                "{scenario:?} {wrong:?}"
            );
            assert_eq!(
                rt.committed_events().len(),
                before,
                "{scenario:?} {wrong:?}"
            );
            assert_eq!(rt.expert_registry(), &registry, "{scenario:?} {wrong:?}");
            // The refusal was the binding alone: the exact binding is admitted next.
            admit_with(&mut rt, op, None)
                .unwrap_or_else(|error| panic!("{scenario:?} {wrong:?}: {error:?}"));
        }
    }
}

#[test]
fn a_report_bound_to_a_part_of_the_artifact_is_refused() {
    // A merge is evidence for its result, not for a source it consumes.
    let mut merge = seeded(Scenario::Merge);
    let op = operation(Scenario::Merge, &merge);
    let merge_proposal = proposal(op);
    let source = *target(&merge, "a").manifest_digest().as_bytes();
    let report = bind(GateSubject::new(
        *merge_proposal.proposal_id().as_bytes(),
        *merge_proposal.operation_id().as_bytes(),
        source,
        BASE_MODEL_DIGEST,
        EVAL_MANIFEST,
    ));
    assert_eq!(
        expert_error(merge.admit_expert_proposal(&merge_proposal, &report, &grant())),
        ExpertAdmissionError::EvidenceArtifactMismatch
    );
    // A split is evidence for the ordered set of its results, not for the first result alone.
    let mut split = seeded(Scenario::Split);
    let op = operation(Scenario::Split, &split);
    let split_proposal = proposal(op);
    let first = *spec_at("x", 1).manifest_digest().as_bytes();
    let report = bind(GateSubject::new(
        *split_proposal.proposal_id().as_bytes(),
        *split_proposal.operation_id().as_bytes(),
        first,
        BASE_MODEL_DIGEST,
        EVAL_MANIFEST,
    ));
    assert_eq!(
        expert_error(split.admit_expert_proposal(&split_proposal, &report, &grant())),
        ExpertAdmissionError::EvidenceArtifactMismatch
    );
    assert_eq!(split.committed_events().len(), 1, "only the seeding birth");
}

#[test]
fn the_binding_is_checked_in_a_fixed_order_after_the_ids() {
    let mut rt = seeded(Scenario::Freeze);
    let op = operation(Scenario::Freeze, &rt);
    let proposal = proposal(op);
    let all_wrong = bind(GateSubject::new(
        *proposal.proposal_id().as_bytes(),
        *proposal.operation_id().as_bytes(),
        [0xA1; 32],
        [0xB2; 32],
        [0xC3; 32],
    ));
    assert_eq!(
        expert_error(rt.admit_expert_proposal(&proposal, &all_wrong, &grant())),
        ExpertAdmissionError::EvidenceArtifactMismatch
    );
    let base_and_eval_wrong = bind(GateSubject::new(
        *proposal.proposal_id().as_bytes(),
        *proposal.operation_id().as_bytes(),
        expert_evidence_artifact_digest(proposal.op()),
        [0xB2; 32],
        [0xC3; 32],
    ));
    assert_eq!(
        expert_error(rt.admit_expert_proposal(&proposal, &base_and_eval_wrong, &grant())),
        ExpertAdmissionError::EvidenceBaseModelMismatch
    );
    // Another operation's ids come before any digest.
    let other_operation = bind(GateSubject::new(
        *proposal.proposal_id().as_bytes(),
        [0xEE; 32],
        [0xA1; 32],
        [0xB2; 32],
        [0xC3; 32],
    ));
    assert_eq!(
        expert_error(rt.admit_expert_proposal(&proposal, &other_operation, &grant())),
        ExpertAdmissionError::EvidenceOperationMismatch
    );
}

// ---- readmission --------------------------------------------------------------------------

fn request(pod: &str, version: u64) -> ExpertFenceRequest {
    ExpertFenceRequest {
        project: project(),
        pod: PodId::from(pod),
        generation: Generation(1),
        expected_state_version: version,
        cause: QUARANTINE,
        revocation: None,
    }
}

fn readmission_report(
    request: &ExpertFenceRequest,
    artifact: [u8; 32],
    wrong: Option<Wrong>,
) -> BoundGateReport {
    let (proposal_id, operation_id) = expert_readmission_subject_ids(
        &request.project,
        &request.pod,
        request.generation,
        request.expected_state_version,
        request.expected_state_version + 1,
        request.cause,
    );
    let (artifact, base_model, eval) = digests(artifact, BASE_MODEL_DIGEST, wrong);
    bind(GateSubject::new(
        proposal_id,
        operation_id,
        artifact,
        base_model,
        eval,
    ))
}

fn fenced_pair() -> PtrRuntime {
    let mut rt = runtime();
    birth(&mut rt, "x");
    birth(&mut rt, "y");
    rt.fence_expert(&request("x", 1), &revocation_grant())
        .unwrap();
    rt
}

fn manifest(rt: &PtrRuntime, pod: &str) -> [u8; 32] {
    *target(rt, pod).manifest_digest().as_bytes()
}

#[test]
fn a_readmission_is_bound_to_the_expert_its_base_model_and_the_expected_evaluation() {
    for wrong in WRONGS {
        let mut rt = fenced_pair();
        let before = rt.committed_events().len();
        let req = request("x", 2);
        let report = readmission_report(&req, manifest(&rt, "x"), Some(wrong));
        assert_eq!(
            expert_error(rt.readmit_expert(&req, &report, &gate_grant())),
            refusal_of(wrong),
            "{wrong:?}"
        );
        assert_eq!(rt.committed_events().len(), before, "{wrong:?}");
    }
    // The manifest of another expert is not the manifest of this one.
    let mut rt = fenced_pair();
    let req = request("x", 2);
    let report = readmission_report(&req, manifest(&rt, "y"), None);
    assert_eq!(
        expert_error(rt.readmit_expert(&req, &report, &gate_grant())),
        ExpertAdmissionError::EvidenceArtifactMismatch
    );
    // Positive control: the exact binding readmits.
    let report = readmission_report(&req, manifest(&rt, "x"), None);
    rt.readmit_expert(&req, &report, &gate_grant()).unwrap();
}

#[test]
fn a_readmission_checks_in_the_same_order_as_an_admission() {
    let mut rt = fenced_pair();
    let req = request("x", 2);
    let (proposal_id, operation_id) =
        expert_readmission_subject_ids(&req.project, &req.pod, req.generation, 2, 3, req.cause);
    let all_wrong = bind(GateSubject::new(
        proposal_id,
        operation_id,
        [0xA1; 32],
        [0xB2; 32],
        [0xC3; 32],
    ));
    assert_eq!(
        expert_error(rt.readmit_expert(&req, &all_wrong, &gate_grant())),
        ExpertAdmissionError::EvidenceArtifactMismatch
    );
    let artifact = manifest(&rt, "x");
    let base_and_eval = bind(GateSubject::new(
        proposal_id,
        operation_id,
        artifact,
        [0xB2; 32],
        [0xC3; 32],
    ));
    assert_eq!(
        expert_error(rt.readmit_expert(&req, &base_and_eval, &gate_grant())),
        ExpertAdmissionError::EvidenceBaseModelMismatch
    );
}

// ---- replay -------------------------------------------------------------------------------

fn forged_birth(subject_tweak: impl FnOnce(&ExpertProposal) -> GateSubject) -> Vec<CommittedEvent> {
    let proposal = proposal(StructuralOp::Birth {
        result: spec_at("a", 1),
    });
    let report = bind(subject_tweak(&proposal));
    let record = ExpertCommitRecord::new_with_evidence(
        proposal.clone(),
        report.digest(),
        EPOCH,
        gate_digest(),
        report.encode_canonical(),
    )
    .unwrap();
    vec![CommittedEvent {
        index: CommitIndex(1),
        event: LedgerEvent::ExpertStructuralCommitted {
            project: proposal.project().clone(),
            op_kind: ledger_op_kind(proposal.op().kind()),
            proposal_id: *proposal.proposal_id().as_bytes(),
            operation_id: *proposal.operation_id().as_bytes(),
            evidence_digest: *record.evidence_digest(),
            record: record.encode_canonical(),
        },
    }]
}

fn subject_with(
    proposal: &ExpertProposal,
    artifact: [u8; 32],
    base_model: [u8; 32],
    eval: [u8; 32],
) -> GateSubject {
    GateSubject::new(
        *proposal.proposal_id().as_bytes(),
        *proposal.operation_id().as_bytes(),
        artifact,
        base_model,
        eval,
    )
}

#[test]
fn replay_refuses_a_hand_forged_record_with_a_wrong_artifact_or_base_model() {
    // Control: the honest record replays.
    let honest = forged_birth(|proposal| {
        subject_with(
            proposal,
            expert_evidence_artifact_digest(proposal.op()),
            BASE_MODEL_DIGEST,
            EVAL_MANIFEST,
        )
    });
    let replayed = PtrRuntime::replay(PtrConfig::default(), &honest).unwrap();
    assert_eq!(replayed.expert_registry().len(), 1);

    let wrong_artifact = forged_birth(|proposal| {
        subject_with(proposal, [0xA1; 32], BASE_MODEL_DIGEST, EVAL_MANIFEST)
    });
    assert!(matches!(
        PtrRuntime::replay(PtrConfig::default(), &wrong_artifact),
        Err(RuntimeError::Expert(
            ExpertAdmissionError::EvidenceArtifactMismatch
        ))
    ));
    let wrong_base = forged_birth(|proposal| {
        subject_with(
            proposal,
            expert_evidence_artifact_digest(proposal.op()),
            [0xB2; 32],
            EVAL_MANIFEST,
        )
    });
    assert!(matches!(
        PtrRuntime::replay(PtrConfig::default(), &wrong_base),
        Err(RuntimeError::Expert(
            ExpertAdmissionError::EvidenceBaseModelMismatch
        ))
    ));
}

#[test]
fn replay_cannot_re_check_the_evaluation_manifest_because_the_grant_is_not_recorded() {
    let other_manifest = forged_birth(|proposal| {
        subject_with(
            proposal,
            expert_evidence_artifact_digest(proposal.op()),
            BASE_MODEL_DIGEST,
            [0xC3; 32],
        )
    });
    let replayed = PtrRuntime::replay(PtrConfig::default(), &other_manifest).unwrap();
    assert_eq!(replayed.expert_registry().len(), 1);
}

#[test]
fn replay_refuses_a_hand_forged_readmission_with_a_wrong_artifact_or_base_model() {
    let rt = fenced_pair();
    let artifact = manifest(&rt, "x");
    let history = rt.committed_events().to_vec();
    let req = request("x", 2);
    let forged = |artifact: [u8; 32], base_model: [u8; 32], history: &[CommittedEvent]| {
        let (proposal_id, operation_id) =
            expert_readmission_subject_ids(&req.project, &req.pod, req.generation, 2, 3, req.cause);
        let report = bind(GateSubject::new(
            proposal_id,
            operation_id,
            artifact,
            base_model,
            EVAL_MANIFEST,
        ));
        let mut events = history.to_vec();
        events.push(CommittedEvent {
            index: CommitIndex(history.len() as u64 + 1),
            event: LedgerEvent::ExpertFenceCommitted {
                project: project(),
                pod: PodId::from("x"),
                action: ExpertFenceAction::Readmit,
                cause: QUARANTINE,
                resume: ExpertFenceResume::Serving,
                generation: Generation(1),
                expected_state_version: 2,
                epoch: 3,
                revocation: None,
                evidence: Some(ExpertFenceEvidence {
                    gate_epoch: EPOCH,
                    required_gate_config_digest: gate_digest(),
                    report: report.encode_canonical(),
                }),
            },
        });
        events
    };
    // Control: the honest readmission replays.
    let honest = forged(artifact, BASE_MODEL_DIGEST, &history);
    PtrRuntime::replay(PtrConfig::default(), &honest).unwrap();
    // The forgeries do not.
    let wrong_artifact = forged([0xA1; 32], BASE_MODEL_DIGEST, &history);
    assert!(matches!(
        PtrRuntime::replay(PtrConfig::default(), &wrong_artifact),
        Err(RuntimeError::Expert(
            ExpertAdmissionError::EvidenceArtifactMismatch
        ))
    ));
    let wrong_base = forged(artifact, [0xB2; 32], &history);
    assert!(matches!(
        PtrRuntime::replay(PtrConfig::default(), &wrong_base),
        Err(RuntimeError::Expert(
            ExpertAdmissionError::EvidenceBaseModelMismatch
        ))
    ));
}
