use ptr_ledger::{LedgerEvent, RaftEngineLedger};
use ptr_types::{CapsuleId, Generation, ProjectId};
use std::time::{SystemTime, UNIX_EPOCH};

fn dir() -> std::path::PathBuf {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!("ptr-raft-engine-ledger-{nonce}"))
}

#[test]
fn durable_events_reopen_in_exact_commit_order() {
    let dir = dir();
    let expected = vec![
        LedgerEvent::CapsuleCommitted {
            project: ProjectId::from("p"),
            capsule: CapsuleId::from("capsule:a"),
            generation: Generation(1),
        },
        LedgerEvent::Revoked {
            subject: "capsule:a".into(),
            generation: Generation(1),
        },
    ];

    {
        let mut ledger = RaftEngineLedger::open(&dir).unwrap();
        for event in &expected {
            ledger.append_durable(event.clone()).unwrap();
        }
        ledger.sync().unwrap();
        assert_eq!(ledger.events().len(), 2);
    }

    {
        let reopened = RaftEngineLedger::open(&dir).unwrap();
        let actual = reopened
            .events()
            .iter()
            .map(|event| event.event.clone())
            .collect::<Vec<_>>();
        assert_eq!(actual, expected);
        assert_eq!(reopened.events()[0].index.0, 1);
        assert_eq!(reopened.events()[1].index.0, 2);
    }

    std::fs::remove_dir_all(dir).unwrap();
}
