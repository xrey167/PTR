//! A taint record changes which approximate results are refused. The public `commit` refuses it,
//! like an expert record: only the runtime's own taint path may write one.

use ptr_config::PtrConfig;
use ptr_ledger::{KvTaintChange, KvTaintSubjectKind, LedgerEvent};
use ptr_runtime::{PtrRuntime, RuntimeError};
use ptr_types::{CommitIndex, Generation, Revision};

fn taint(change: KvTaintChange) -> LedgerEvent {
    LedgerEvent::KvTaintCommitted {
        subjects: vec![(KvTaintSubjectKind::Action, [1; 32])],
        change,
    }
}

#[test]
fn direct_commit_of_a_taint_record_is_refused_and_appends_nothing() {
    let changes = [
        KvTaintChange::Record {
            report_digest: [2; 32],
            recipe_digest: [3; 32],
        },
        KvTaintChange::Clear {
            recipe_digest: [3; 32],
            recompute_digest: [4; 32],
        },
    ];
    for change in changes {
        let mut runtime = PtrRuntime::new(PtrConfig::default()).unwrap();
        assert_eq!(
            runtime.commit(taint(change)).err(),
            Some(RuntimeError::StructuralRecordOutsideAdmissionPath)
        );
        assert!(runtime.committed_events().is_empty());
        assert_eq!(runtime.revision(), Revision(0));
        // The refusal consumed no index.
        let ordinary = LedgerEvent::Revoked {
            subject: "nothing".into(),
            generation: Generation(1),
        };
        assert_eq!(runtime.commit(ordinary).unwrap(), CommitIndex(1));
    }
}
