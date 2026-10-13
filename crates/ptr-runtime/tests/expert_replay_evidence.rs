//! Replay of expert structural records re-verifies the gate evidence the record embeds: a record
//! whose digest was only made self-consistent is refused, and version 1 records (digest only) are
//! accepted as unverified legacy and flagged.

use ptr_config::PtrConfig;
use ptr_ledger::{CommittedEvent, LedgerEvent};
use ptr_lineage::{
    AccuracyMatrix, AdapterId, BoundGateReport, ForgettingGate, GateSubject, PublicSuite,
};
use ptr_pods::{
    BaseModelDigest, BaseModelId, BaseModelRef, ExpertCommitRecord, ExpertLevel, ExpertPodSpec,
    ExpertProposal, ExpertTarget, ExpertTargetInit, ManifestDigest, ProposerRef, RationaleDigest,
    RetiredReason, StructuralOp,
};
use ptr_runtime::expert::{
    expert_evidence_artifact_digest, expert_evidence_base_model_digest, ledger_op_kind,
    ExpertAdmissionError, ExpertAdmissionGrant, ExpertEvidenceStatus,
};
use ptr_runtime::{PtrRuntime, RuntimeError};
use ptr_types::{CommitIndex, Generation, PodId, ProjectId};

mod expert_support;

use expert_support::{binding, prime, requirement, EVAL_MANIFEST};

const LAYER: u16 = 3;
const EPOCH: u64 = 7;

fn project() -> ProjectId {
    ProjectId::from("proj")
}

fn base() -> BaseModelRef {
    BaseModelRef::new(
        BaseModelId::new("base-a").unwrap(),
        BaseModelDigest::new([9; 32]).unwrap(),
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

fn other_gate() -> ForgettingGate {
    ForgettingGate {
        max_average_forgetting: 0.9,
        ..gate()
    }
}

fn passing() -> PublicSuite {
    PublicSuite {
        serving: 0.8,
        candidate: 0.8,
    }
}

fn failing() -> PublicSuite {
    PublicSuite {
        serving: 0.9,
        candidate: 0.1,
    }
}

fn bind(
    subject: GateSubject,
    gate: &ForgettingGate,
    public: PublicSuite,
    adapter: &str,
) -> BoundGateReport {
    let matrix = AccuracyMatrix::new(vec![vec![0.9, 0.0], vec![0.9, 0.9]]).unwrap();
    gate.evaluate(&AdapterId::from(adapter), &matrix, public)
        .unwrap()
        .bind(subject)
}

fn subject_of(proposal: &ExpertProposal) -> GateSubject {
    GateSubject::new(
        *proposal.proposal_id().as_bytes(),
        *proposal.operation_id().as_bytes(),
        expert_evidence_artifact_digest(proposal.op()),
        expert_evidence_base_model_digest(proposal.op()).unwrap(),
        EVAL_MANIFEST,
    )
}

fn grant() -> ExpertAdmissionGrant {
    ExpertAdmissionGrant::new(
        project(),
        proposer(),
        requirement(EPOCH, gate().config_digest().unwrap()),
        binding(),
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

fn admit(rt: &mut PtrRuntime, op: StructuralOp) -> Result<CommitIndex, RuntimeError> {
    prime(rt);
    let proposal = proposal(op);
    let report = bind(subject_of(&proposal), &gate(), passing(), "candidate");
    rt.admit_expert_proposal(&proposal, &report, &grant())
}

fn target(rt: &PtrRuntime, pod: &str) -> ExpertTarget {
    let entry = rt
        .expert_registry()
        .get(&project(), &PodId::from(pod))
        .unwrap();
    ExpertTarget::new(ExpertTargetInit {
        project: project(),
        pod: PodId::from(pod),
        generation: entry.generation,
        level: entry.level,
        base_model: entry.base_model.clone(),
        manifest_digest: entry.manifest_digest,
        state: entry.state,
        state_version: entry.state_version,
    })
    .unwrap()
}

/// All seven structural operations in one history.
fn full_history() -> PtrRuntime {
    let mut rt = runtime();
    for pod in ["a", "b", "c", "d"] {
        admit(
            &mut rt,
            StructuralOp::Birth {
                result: spec_at(pod, 1),
            },
        )
        .unwrap();
    }
    let merge = StructuralOp::Merge {
        sources: vec![target(&rt, "a"), target(&rt, "b")],
        result: spec_at("m", 1),
    };
    admit(&mut rt, merge).unwrap();
    let split = StructuralOp::Split {
        source: target(&rt, "c"),
        results: vec![spec_at("x", 1), spec_at("y", 1)],
    };
    admit(&mut rt, split).unwrap();
    let freeze = StructuralOp::Freeze {
        target: target(&rt, "m"),
    };
    admit(&mut rt, freeze).unwrap();
    let unfreeze = StructuralOp::Reactivate {
        target: target(&rt, "m"),
        result: None,
    };
    admit(&mut rt, unfreeze).unwrap();
    let retire = StructuralOp::Retire {
        target: target(&rt, "x"),
        reason: RetiredReason::GateFailed,
    };
    admit(&mut rt, retire).unwrap();
    let revive = StructuralOp::Reactivate {
        target: target(&rt, "x"),
        result: Some(spec_at("x", 2)),
    };
    admit(&mut rt, revive).unwrap();
    let prune = StructuralOp::Prune {
        targets: vec![target(&rt, "y"), target(&rt, "d")],
    };
    admit(&mut rt, prune).unwrap();
    rt
}

fn record_of(event: &LedgerEvent) -> ExpertCommitRecord {
    let LedgerEvent::ExpertStructuralCommitted { record, .. } = event else {
        panic!("not a structural event");
    };
    ExpertCommitRecord::decode_canonical(record).unwrap()
}

/// The event that commits `record`, every id derived from it.
fn event_of(record: &ExpertCommitRecord) -> LedgerEvent {
    let proposal = record.proposal();
    LedgerEvent::ExpertStructuralCommitted {
        project: proposal.project().clone(),
        op_kind: ledger_op_kind(proposal.op().kind()),
        proposal_id: *proposal.proposal_id().as_bytes(),
        operation_id: *proposal.operation_id().as_bytes(),
        evidence_digest: *record.evidence_digest(),
        record: record.encode_canonical(),
    }
}

/// A record for `proposal` embedding exactly `report_bytes`, with `digest` as its evidence digest.
fn forged(
    proposal: &ExpertProposal,
    digest: [u8; 32],
    required: [u8; 32],
    epoch: u64,
    report_bytes: Vec<u8>,
) -> LedgerEvent {
    event_of(
        &ExpertCommitRecord::new_with_evidence(
            proposal.clone(),
            digest,
            epoch,
            required,
            report_bytes,
        )
        .unwrap(),
    )
}

/// A forged record that embeds `report` honestly: its digest and bytes belong together.
fn forged_from(
    proposal: &ExpertProposal,
    report: &BoundGateReport,
    required: [u8; 32],
    epoch: u64,
) -> LedgerEvent {
    forged(
        proposal,
        report.digest(),
        required,
        epoch,
        report.encode_canonical(),
    )
}

fn with_first_event(rt: &PtrRuntime, event: LedgerEvent) -> Vec<CommittedEvent> {
    let mut events = rt.committed_events().to_vec();
    events[0].event = event;
    events
}

fn replay_error(events: &[CommittedEvent]) -> RuntimeError {
    match PtrRuntime::replay(PtrConfig::default(), events) {
        Ok(_) => panic!("replay must fail"),
        Err(error) => error,
    }
}

fn expert(error: ExpertAdmissionError) -> RuntimeError {
    RuntimeError::Expert(error)
}

fn gate_digest() -> [u8; 32] {
    gate().config_digest().unwrap()
}

fn first_proposal(rt: &PtrRuntime) -> ExpertProposal {
    record_of(&rt.committed_events()[0].event)
        .proposal()
        .clone()
}

#[test]
fn every_structural_operation_replays_to_an_identical_verified_projection() {
    let rt = full_history();
    let events = rt.committed_events().to_vec();
    assert_eq!(events.len(), 11);
    for committed in &events {
        let record = record_of(&committed.event);
        assert!(record.evidence().is_some(), "admission writes version 2");
        assert!(record.encode_canonical().starts_with(b"PTREXR02"));
    }
    let replayed = PtrRuntime::replay(PtrConfig::default(), &events).unwrap();
    assert_eq!(replayed.expert_registry(), rt.expert_registry());
    assert_eq!(replayed.committed_events(), rt.committed_events());
    for (_, pod, _) in replayed.expert_registry().iter() {
        assert_eq!(
            replayed
                .expert_registry()
                .evidence_status(&project(), &PodId::from(pod)),
            Some(ExpertEvidenceStatus::Verified)
        );
    }
}

#[test]
fn replay_refuses_an_embedded_report_whose_digest_is_not_the_evidence_digest() {
    let rt = full_history();
    let proposal = first_proposal(&rt);
    // A passing, correctly bound report, but not the one the evidence digest names.
    let other = bind(subject_of(&proposal), &gate(), passing(), "another-adapter");
    let honest = bind(subject_of(&proposal), &gate(), passing(), "candidate");
    assert_ne!(other.digest(), honest.digest());
    let event = forged(
        &proposal,
        honest.digest(),
        gate_digest(),
        EPOCH,
        other.encode_canonical(),
    );
    assert_eq!(
        replay_error(&with_first_event(&rt, event)),
        expert(ExpertAdmissionError::EvidenceDigestMismatch)
    );
}

#[test]
fn replay_refuses_a_report_whose_stored_digest_or_bytes_were_altered() {
    let rt = full_history();
    let proposal = first_proposal(&rt);
    let report = bind(subject_of(&proposal), &gate(), passing(), "candidate");
    let mut bytes = report.encode_canonical();
    let last = bytes.len() - 1;
    bytes[last] ^= 1;
    let event = forged(&proposal, report.digest(), gate_digest(), EPOCH, bytes);
    assert!(matches!(
        replay_error(&with_first_event(&rt, event)),
        RuntimeError::Expert(ExpertAdmissionError::EvidenceUndecodable(_))
    ));
    let event = forged(
        &proposal,
        report.digest(),
        gate_digest(),
        EPOCH,
        b"not a report".to_vec(),
    );
    assert!(matches!(
        replay_error(&with_first_event(&rt, event)),
        RuntimeError::Expert(ExpertAdmissionError::EvidenceUndecodable(_))
    ));
}

#[test]
fn replay_refuses_a_record_that_carries_a_failed_report() {
    let rt = full_history();
    let proposal = first_proposal(&rt);
    let failed = bind(subject_of(&proposal), &gate(), failing(), "candidate");
    assert!(!failed.passed());
    let event = forged_from(&proposal, &failed, gate_digest(), EPOCH);
    assert_eq!(
        replay_error(&with_first_event(&rt, event)),
        expert(ExpertAdmissionError::GateFailed)
    );
}

#[test]
fn replay_refuses_a_report_bound_to_another_proposal_or_operation() {
    let rt = full_history();
    let proposal = first_proposal(&rt);
    let mut other_proposal = subject_of(&proposal);
    other_proposal.proposal_id = [0xAB; 32];
    let report = bind(other_proposal, &gate(), passing(), "candidate");
    let event = forged_from(&proposal, &report, gate_digest(), EPOCH);
    assert_eq!(
        replay_error(&with_first_event(&rt, event)),
        expert(ExpertAdmissionError::EvidenceProposalMismatch)
    );
    let mut other_operation = subject_of(&proposal);
    other_operation.operation_id = [0xCD; 32];
    let report = bind(other_operation, &gate(), passing(), "candidate");
    let event = forged_from(&proposal, &report, gate_digest(), EPOCH);
    assert_eq!(
        replay_error(&with_first_event(&rt, event)),
        expert(ExpertAdmissionError::EvidenceOperationMismatch)
    );
}

#[test]
fn replay_refuses_a_gate_config_digest_that_is_not_the_one_recorded_for_the_epoch() {
    let rt = full_history();
    let proposal = first_proposal(&rt);
    // The report was produced under another configuration than the record says was required.
    let report = bind(subject_of(&proposal), &other_gate(), passing(), "candidate");
    assert!(report.passed());
    let event = forged_from(&proposal, &report, gate_digest(), EPOCH);
    assert_eq!(
        replay_error(&with_first_event(&rt, event)),
        expert(ExpertAdmissionError::GateConfigMismatch)
    );
}

#[test]
fn replay_refuses_two_records_of_one_epoch_that_disagree_on_the_required_configuration() {
    let rt = full_history();
    let events = rt.committed_events().to_vec();
    // Record 2 is internally consistent under the other configuration, same epoch as record 1.
    let proposal = record_of(&events[1].event).proposal().clone();
    let report = bind(subject_of(&proposal), &other_gate(), passing(), "candidate");
    let other_digest = other_gate().config_digest().unwrap();
    let mut tampered = events;
    tampered[1].event = forged_from(&proposal, &report, other_digest, EPOCH);
    assert_eq!(
        replay_error(&tampered),
        expert(ExpertAdmissionError::EpochGateConfigConflict { epoch: EPOCH })
    );
    // The same record under its own epoch is a different, consistent configuration.
    let mut other_epoch = rt.committed_events().to_vec();
    other_epoch[1].event = forged_from(&proposal, &report, other_digest, EPOCH + 1);
    assert!(!matches!(
        PtrRuntime::replay(PtrConfig::default(), &other_epoch),
        Err(RuntimeError::Expert(
            ExpertAdmissionError::EpochGateConfigConflict { .. }
        ))
    ));
}

#[test]
fn a_version_1_record_replays_as_unverified_legacy_and_is_flagged() {
    let rt = full_history();
    let proposal = first_proposal(&rt);
    let legacy = ExpertCommitRecord::new(proposal.clone(), [0x44; 32], EPOCH).unwrap();
    assert!(legacy.evidence().is_none());
    assert!(legacy.encode_canonical().starts_with(b"PTREXR01"));
    let birth_a = event_of(&legacy);
    // Only the first birth is rewritten: the rest of the history is unaffected by it.
    let events = with_first_event(&rt, birth_a);
    let replayed = PtrRuntime::replay(PtrConfig::default(), &events).unwrap();
    let status = |pod: &str| {
        replayed
            .expert_registry()
            .evidence_status(&project(), &PodId::from(pod))
    };
    // `a` was consumed by the merge, which is verified and rewrote it.
    assert_eq!(status("a"), Some(ExpertEvidenceStatus::Verified));
    assert_eq!(status("nope"), None);
    // Replayed alone, the legacy birth leaves its expert flagged.
    let alone = PtrRuntime::replay(PtrConfig::default(), &events[..1]).unwrap();
    assert_eq!(
        alone
            .expert_registry()
            .evidence_status(&project(), &PodId::from("a")),
        Some(ExpertEvidenceStatus::UnverifiedLegacy)
    );
    // A verified admission on top rewrites the mark.
    let mut alone = alone;
    let freeze = StructuralOp::Freeze {
        target: target(&alone, "a"),
    };
    admit(&mut alone, freeze).unwrap();
    assert_eq!(
        alone
            .expert_registry()
            .evidence_status(&project(), &PodId::from("a")),
        Some(ExpertEvidenceStatus::Verified)
    );
    // The legacy mark survives a registry comparison: it is not equal to the verified one.
    let verified = PtrRuntime::replay(PtrConfig::default(), &rt.committed_events()[..1]).unwrap();
    assert_ne!(
        PtrRuntime::replay(PtrConfig::default(), &events[..1])
            .unwrap()
            .expert_registry(),
        verified.expert_registry()
    );
}

#[test]
fn a_legacy_record_is_still_checked_against_the_registry() {
    let rt = full_history();
    let proposal = first_proposal(&rt);
    let legacy = ExpertCommitRecord::new(proposal, [0x44; 32], EPOCH).unwrap();
    let mut events = rt.committed_events().to_vec();
    // The birth is repeated as a legacy record: its result pod is already registered.
    events.insert(
        1,
        CommittedEvent {
            index: CommitIndex(2),
            event: event_of(&legacy),
        },
    );
    let events: Vec<CommittedEvent> = events
        .into_iter()
        .enumerate()
        .map(|(i, c)| CommittedEvent {
            index: CommitIndex(i as u64 + 1),
            event: c.event,
        })
        .collect();
    assert_eq!(
        replay_error(&events),
        expert(ExpertAdmissionError::ResultPodOccupied {
            pod: "a".to_owned()
        })
    );
}
