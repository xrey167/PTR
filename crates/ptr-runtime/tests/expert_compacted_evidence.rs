//! A compaction floor keeps how each expert's evidence was checked: legacy marks and the per-epoch
//! gate configuration memo survive snapshot and restore, PTRCS005 restores conservatively, and a
//! tampered evidence block is refused.

use ptr_config::PtrConfig;
use ptr_ledger::{integrity, CommittedEvent, LedgerEvent};
use ptr_lineage::{AccuracyMatrix, AdapterId, ForgettingGate, GateSubject, PublicSuite};
use ptr_pods::{
    BaseModelDigest, BaseModelId, BaseModelRef, ExpertCommitRecord, ExpertLevel, ExpertPodSpec,
    ExpertProposal, ProposerRef, RationaleDigest, StructuralOp,
};
use ptr_runtime::compacted::{CompactedAnchor, CompactedError};
use ptr_runtime::expert::{
    ledger_op_kind, ExpertAdmissionError, ExpertAdmissionGrant, ExpertEvidenceStatus,
};
use ptr_runtime::{PtrRuntime, RuntimeError};
use ptr_types::{CommitIndex, Generation, PodId, ProjectId};

const EPOCH: u64 = 7;

fn project() -> ProjectId {
    ProjectId::from("proj")
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

fn spec(pod: &str) -> ExpertPodSpec {
    let mut digest = [1u8; 32];
    digest[0] = pod.as_bytes()[0];
    ExpertPodSpec::new(
        project(),
        PodId::from(pod),
        ptr_pods::ManifestDigest::new(digest).unwrap(),
        ExpertLevel::Token {
            layer: 3,
            index: u16::from(pod.as_bytes()[0]),
        },
        BaseModelRef::new(
            BaseModelId::new("base-a").unwrap(),
            BaseModelDigest::new([9; 32]).unwrap(),
            1,
        )
        .unwrap(),
        Generation(1),
    )
    .unwrap()
}

fn proposal_for(pod: &str) -> ExpertProposal {
    ExpertProposal::new(
        project(),
        proposer(),
        CommitIndex(1),
        StructuralOp::Birth { result: spec(pod) },
        RationaleDigest::new([7; 32]).unwrap(),
    )
    .unwrap()
}

fn admit_birth(
    rt: &mut PtrRuntime,
    pod: &str,
    gate: &ForgettingGate,
) -> Result<CommitIndex, RuntimeError> {
    let proposal = proposal_for(pod);
    let matrix = AccuracyMatrix::new(vec![vec![0.9, 0.0], vec![0.9, 0.9]]).unwrap();
    let public = PublicSuite {
        serving: 0.8,
        candidate: 0.8,
    };
    let report = gate
        .evaluate(&AdapterId::from("candidate"), &matrix, public)
        .unwrap()
        .bind(GateSubject::new(
            *proposal.proposal_id().as_bytes(),
            *proposal.operation_id().as_bytes(),
            [3; 32],
            [4; 32],
            [5; 32],
        ));
    let grant =
        ExpertAdmissionGrant::new(project(), proposer(), EPOCH, gate.config_digest().unwrap());
    rt.admit_expert_proposal(&proposal, &report, &grant)
}

/// A version 1 birth of `pod`, as a history written before evidence was embedded holds it.
fn legacy_birth(pod: &str, index: u64) -> CommittedEvent {
    let proposal = proposal_for(pod);
    let record = ExpertCommitRecord::new(proposal.clone(), [0x44; 32], EPOCH).unwrap();
    CommittedEvent {
        index: CommitIndex(index),
        event: LedgerEvent::ExpertStructuralCommitted {
            project: project(),
            op_kind: ledger_op_kind(proposal.op().kind()),
            proposal_id: *proposal.proposal_id().as_bytes(),
            operation_id: *proposal.operation_id().as_bytes(),
            evidence_digest: *record.evidence_digest(),
            record: record.encode_canonical(),
        },
    }
}

/// `xx` unverified legacy, `yy` verified under epoch 7.
fn mixed() -> PtrRuntime {
    let mut rt = PtrRuntime::replay(PtrConfig::default(), &[legacy_birth("xx", 1)]).unwrap();
    admit_birth(&mut rt, "yy", &gate()).unwrap();
    rt
}

fn status(rt: &PtrRuntime, pod: &str) -> Option<ExpertEvidenceStatus> {
    rt.expert_registry()
        .evidence_status(&project(), &PodId::from(pod))
}

fn restore_bytes(
    bytes: &[u8],
    anchor: CompactedAnchor,
    tail: &[CommittedEvent],
) -> Result<PtrRuntime, RuntimeError> {
    PtrRuntime::restore_compacted(PtrConfig::default(), bytes, anchor, tail)
}

fn resealed(mut bytes: Vec<u8>, trusted: CompactedAnchor) -> (Vec<u8>, CompactedAnchor) {
    let end = bytes.len() - 32;
    let digest = integrity::sha256(&bytes[..end]);
    bytes[end..].copy_from_slice(&digest);
    (bytes, CompactedAnchor { digest, ..trusted })
}

fn evidence_block(bytes: &[u8]) -> usize {
    bytes
        .windows(8)
        .rposition(|w| w == b"PTREV001")
        .expect("evidence block")
}

#[test]
fn legacy_marks_and_the_epoch_memo_survive_compaction_and_restore() {
    let rt = mixed();
    assert_eq!(
        status(&rt, "xx"),
        Some(ExpertEvidenceStatus::UnverifiedLegacy)
    );
    assert_eq!(status(&rt, "yy"), Some(ExpertEvidenceStatus::Verified));
    let snapshot = rt.export_compacted_snapshot().unwrap();
    assert_eq!(&snapshot.bytes()[..8], b"PTRCS006");
    let restored = restore_bytes(snapshot.bytes(), snapshot.anchor(), &[]).unwrap();
    assert_eq!(restored.expert_registry(), rt.expert_registry());
    assert_eq!(
        status(&restored, "xx"),
        Some(ExpertEvidenceStatus::UnverifiedLegacy)
    );
    assert_eq!(
        status(&restored, "yy"),
        Some(ExpertEvidenceStatus::Verified)
    );
}

#[test]
fn after_restore_a_record_of_the_same_epoch_with_another_config_is_refused() {
    let rt = mixed();
    let snapshot = rt.export_compacted_snapshot().unwrap();
    let mut restored = restore_bytes(snapshot.bytes(), snapshot.anchor(), &[]).unwrap();
    assert_eq!(
        admit_birth(&mut restored, "zz", &other_gate()).unwrap_err(),
        RuntimeError::Expert(ExpertAdmissionError::EpochGateConfigConflict { epoch: EPOCH })
    );
    admit_birth(&mut restored, "zz", &gate()).unwrap();
}

#[test]
fn snapshot_plus_tail_equals_full_replay_including_marks() {
    let mut full = PtrRuntime::replay(PtrConfig::default(), &[legacy_birth("xx", 1)]).unwrap();
    admit_birth(&mut full, "yy", &gate()).unwrap();
    let snapshot = full.export_compacted_snapshot().unwrap();
    admit_birth(&mut full, "zz", &gate()).unwrap();
    let tail = full.committed_events()[2..].to_vec();
    let restored = restore_bytes(snapshot.bytes(), snapshot.anchor(), &tail).unwrap();
    assert_eq!(restored.expert_registry(), full.expert_registry());
    let replayed = PtrRuntime::replay(PtrConfig::default(), full.committed_events()).unwrap();
    assert_eq!(replayed.expert_registry(), restored.expert_registry());
    assert_eq!(
        status(&restored, "xx"),
        Some(ExpertEvidenceStatus::UnverifiedLegacy)
    );
    assert_eq!(
        status(&restored, "zz"),
        Some(ExpertEvidenceStatus::Verified)
    );
}

/// The same snapshot in the PTRCS005 layout: no evidence block.
fn as_ptrcs005(bytes: &[u8], anchor: CompactedAnchor) -> (Vec<u8>, CompactedAnchor) {
    let semantic = u64::from_le_bytes(bytes[56..64].try_into().unwrap()) as usize;
    let lifecycle = u64::from_le_bytes(bytes[64..72].try_into().unwrap()) as usize;
    let lifecycle_start = 80 + semantic;
    let lifecycle_end = lifecycle_start + lifecycle;
    let block = evidence_block(bytes);
    let mut out = bytes[..block].to_vec();
    out.extend_from_slice(&bytes[lifecycle_end..]);
    out[..8].copy_from_slice(b"PTRCS005");
    out[lifecycle_start..lifecycle_start + 8].copy_from_slice(b"PTRLC003");
    let shortened = (block - lifecycle_start) as u64;
    out[64..72].copy_from_slice(&shortened.to_le_bytes());
    resealed(out, anchor)
}

#[test]
fn a_ptrcs005_snapshot_restores_every_expert_as_unverified_legacy_with_an_empty_memo() {
    let rt = mixed();
    let snapshot = rt.export_compacted_snapshot().unwrap();
    let (bytes, anchor) = as_ptrcs005(snapshot.bytes(), snapshot.anchor());
    let mut restored = restore_bytes(&bytes, anchor, &[]).unwrap();
    assert_eq!(
        status(&restored, "xx"),
        Some(ExpertEvidenceStatus::UnverifiedLegacy)
    );
    assert_eq!(
        status(&restored, "yy"),
        Some(ExpertEvidenceStatus::UnverifiedLegacy)
    );
    assert_ne!(restored.expert_registry(), rt.expert_registry());
    // The memo is empty, so the first verified record of an epoch sets it again.
    admit_birth(&mut restored, "zz", &other_gate()).unwrap();
    assert_eq!(
        status(&restored, "zz"),
        Some(ExpertEvidenceStatus::Verified)
    );
}

fn tampered_error(change: impl FnOnce(&mut Vec<u8>, usize)) -> Option<RuntimeError> {
    let snapshot = mixed().export_compacted_snapshot().unwrap();
    let mut bytes = snapshot.bytes().to_vec();
    let block = evidence_block(&bytes);
    change(&mut bytes, block);
    let (bytes, anchor) = resealed(bytes, snapshot.anchor());
    restore_bytes(&bytes, anchor, &[]).err()
}

#[test]
fn a_tampered_or_non_canonical_evidence_block_is_refused() {
    // Block: magic, count (4), "proj" (4+4), "xx" (4+2), memo count (4), epoch (8), digest (32).
    let mark_pod = |block: usize| block + 8 + 4 + 8 + 4;
    let memo_digest = |block: usize| block + 8 + 4 + 8 + 6 + 4 + 8;
    let noncanonical = Some(RuntimeError::Compacted(CompactedError::NoncanonicalSection));
    // A mark naming an expert the registry does not hold.
    assert_eq!(
        tampered_error(|b, block| b[mark_pod(block)] = b'z'),
        noncanonical
    );
    // A zero configuration digest in the memo.
    assert_eq!(
        tampered_error(|b, block| {
            let at = memo_digest(block);
            b[at..at + 32].fill(0);
        }),
        noncanonical
    );
    // A wrong block magic.
    assert_eq!(
        tampered_error(|b, block| b[block] ^= 1),
        Some(RuntimeError::Compacted(CompactedError::UnsupportedVersion))
    );
    // A trailing byte the section lengths do not account for is refused too.
    assert!(tampered_error(|b, block| b[memo_digest(block) + 31] ^= 1).is_none());
}
