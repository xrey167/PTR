use ptr_ledger::{CommittedEvent, LedgerEvent};
use ptr_state::{ApplyOutcome, TursoMaterializedState};
use ptr_types::{CommitIndex, Generation};
use std::time::{SystemTime, UNIX_EPOCH};

fn directory() -> std::path::PathBuf {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!("ptr-turso-state-{nonce}"))
}

#[tokio::test]
async fn turso_state_is_monotonic_and_survives_reopen() {
    let dir = directory();
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("state.db");
    let db_path = path.to_string_lossy().to_string();

    {
        let mut state = TursoMaterializedState::open(&db_path).await.unwrap();
        let first = CommittedEvent {
            index: CommitIndex(1),
            event: LedgerEvent::HardConstraintCommitted {
                key: "no-network".into(),
                generation: Generation(3),
            },
        };
        assert_eq!(state.try_apply(&first).await.unwrap(), ApplyOutcome::Applied);
        assert_eq!(
            state.get("constraint:no-network").await.unwrap(),
            Some("3".into())
        );
        assert_eq!(state.try_apply(&first).await.unwrap(), ApplyOutcome::Duplicate);

        let gap = CommittedEvent {
            index: CommitIndex(3),
            event: LedgerEvent::Revoked {
                subject: "capsule:a".into(),
                generation: Generation(1),
            },
        };
        assert_eq!(state.try_apply(&gap).await.unwrap(), ApplyOutcome::Gap);
        assert_eq!(state.last_applied(), 1);
    }

    {
        let state = TursoMaterializedState::open(&db_path).await.unwrap();
        assert_eq!(state.last_applied(), 1);
        assert_eq!(
            state.get("constraint:no-network").await.unwrap(),
            Some("3".into())
        );
    }

    std::fs::remove_dir_all(dir).unwrap();
}
