//! An expert structural record is authority over which experts serve. Until the runtime's
//! admission path exists, the public `commit` refuses it and replay fails closed.

use ptr_config::PtrConfig;
use ptr_ledger::{CommittedEvent, ExpertOpKind, FileLedger, LedgerEvent};
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
fn a_history_holding_a_structural_record_fails_closed_on_replay() {
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
    assert_eq!(
        PtrRuntime::replay(PtrConfig::default(), &history).err(),
        Some(RuntimeError::StructuralRecordOutsideAdmissionPath)
    );
}

#[test]
fn a_durable_log_holding_a_structural_record_does_not_open_and_is_left_unchanged() {
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
        Err(RuntimeError::StructuralRecordOutsideAdmissionPath)
    ));
    assert_eq!(before, std::fs::read(&path).unwrap());
    std::fs::remove_dir_all(&dir).unwrap();
}
