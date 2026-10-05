//! An expert structural record is authority over which experts serve. The public `commit`
//! refuses it; replay accepts only a record that decodes and whose event fields were derived
//! from it, so a record that did not come through admission fails with a precise error.

use ptr_config::PtrConfig;
use ptr_ledger::{CommittedEvent, ExpertOpKind, FileLedger, LedgerEvent};
use ptr_runtime::expert::{ExpertAdmissionError, ExpertEventField};
use ptr_runtime::{PtrRuntime, RuntimeError};
use ptr_types::{CommitIndex, ProjectId, Revision};
use std::path::PathBuf;

fn structural(op_kind: ExpertOpKind) -> LedgerEvent {
    LedgerEvent::ExpertStructuralCommitted {
        project: ProjectId::from("p"),
        op_kind,
        proposal_id: [1; 32],
        operation_id: [2; 32],
        evidence_digest: [3; 32],
        record: b"PTREXR01-opaque".to_vec(),
    }
}

fn ordinary() -> LedgerEvent {
    LedgerEvent::Revoked {
        subject: "nothing".into(),
        generation: ptr_types::Generation(1),
    }
}

#[test]
fn direct_commit_of_a_structural_record_is_refused_and_appends_nothing() {
    for op_kind in (0..=6).map(|c| ExpertOpKind::from_code(c).unwrap()) {
        let mut runtime = PtrRuntime::new(PtrConfig::default()).unwrap();
        assert_eq!(
            runtime.commit(structural(op_kind)).err(),
            Some(RuntimeError::StructuralRecordOutsideAdmissionPath)
        );
        assert!(runtime.committed_events().is_empty());
        assert_eq!(runtime.revision(), Revision(0));
        // The refusal consumed no index: the next ordinary commit is the first record.
        assert_eq!(runtime.commit(ordinary()).unwrap(), CommitIndex(1));
        assert_eq!(runtime.committed_events().len(), 1);
    }
}

#[test]
fn a_history_holding_an_opaque_structural_record_fails_on_replay_as_an_invalid_record() {
    let history = [
        CommittedEvent {
            index: CommitIndex(1),
            event: ordinary(),
        },
        CommittedEvent {
            index: CommitIndex(2),
            event: structural(ExpertOpKind::Retire),
        },
    ];
    assert!(matches!(
        PtrRuntime::replay(PtrConfig::default(), &history).err(),
        Some(RuntimeError::Expert(ExpertAdmissionError::InvalidRecord(_)))
    ));
}

#[test]
fn a_durable_log_holding_an_opaque_structural_record_does_not_open_and_is_left_unchanged() {
    let dir: PathBuf = std::env::temp_dir().join(format!(
        "ptr-expert-record-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir(&dir).unwrap();
    let path = dir.join("ledger");
    let mut log = FileLedger::open(&path).unwrap();
    log.append_durable(structural(ExpertOpKind::Birth)).unwrap();
    drop(log);
    let before = std::fs::read(&path).unwrap();
    assert!(matches!(
        PtrRuntime::open_durable(PtrConfig::default(), &path),
        Err(RuntimeError::Expert(ExpertAdmissionError::InvalidRecord(_)))
    ));
    assert_eq!(before, std::fs::read(&path).unwrap());
    std::fs::remove_dir_all(&dir).unwrap();
}

/// A record that decodes but was not produced by admission: its event ids are made up.
#[test]
fn a_decodable_record_with_ids_that_do_not_match_it_fails_on_replay_with_the_first_mismatch() {
    use ptr_pods::{
        BaseModelDigest, BaseModelId, BaseModelRef, ExpertCommitRecord, ExpertLevel, ExpertPodSpec,
        ExpertProposal, ManifestDigest, ProposerRef, RationaleDigest, StructuralOp,
    };
    use ptr_types::{Generation, PodId};
    let project = ProjectId::from("p");
    let spec = ExpertPodSpec::new(
        project.clone(),
        PodId::from("a"),
        ManifestDigest::new([2; 32]).unwrap(),
        ExpertLevel::Segment,
        BaseModelRef::new(
            BaseModelId::new("base").unwrap(),
            BaseModelDigest::new([9; 32]).unwrap(),
            1,
        )
        .unwrap(),
        Generation(1),
    )
    .unwrap();
    let proposal = ExpertProposal::new(
        project.clone(),
        ProposerRef::new(project.clone(), PodId::from("meta"), Generation(1)).unwrap(),
        CommitIndex(1),
        StructuralOp::Birth { result: spec },
        RationaleDigest::new([7; 32]).unwrap(),
    )
    .unwrap();
    let record = ExpertCommitRecord::new(proposal.clone(), [3; 32], 1).unwrap();
    let event = |proposal_id, evidence_digest| LedgerEvent::ExpertStructuralCommitted {
        project: project.clone(),
        op_kind: ExpertOpKind::Birth,
        proposal_id,
        operation_id: *proposal.operation_id().as_bytes(),
        evidence_digest,
        record: record.encode_canonical(),
    };
    let replay = |event| {
        PtrRuntime::replay(
            PtrConfig::default(),
            &[CommittedEvent {
                index: CommitIndex(1),
                event,
            }],
        )
        .err()
    };
    assert_eq!(
        replay(event([1; 32], [3; 32])),
        Some(RuntimeError::Expert(ExpertAdmissionError::EventMismatch {
            field: ExpertEventField::ProposalId
        }))
    );
    assert_eq!(
        replay(event(*proposal.proposal_id().as_bytes(), [4; 32])),
        Some(RuntimeError::Expert(ExpertAdmissionError::EventMismatch {
            field: ExpertEventField::EvidenceDigest
        }))
    );
    // Consistent in every field it carries, the record replays: admission is what decides
    // whether it should have been written, and replay re-checks only what the log can show.
    let replayed = PtrRuntime::replay(
        PtrConfig::default(),
        &[CommittedEvent {
            index: CommitIndex(1),
            event: event(*proposal.proposal_id().as_bytes(), [3; 32]),
        }],
    )
    .unwrap();
    assert_eq!(replayed.expert_registry().len(), 1);
}
