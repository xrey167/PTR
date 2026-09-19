use ptr_ledger::{FileLedger, LedgerEvent};
use ptr_types::{CapsuleId, CommitIndex, Generation, ProjectId, Revision};
use std::fs::OpenOptions;
use std::io::Write;
use std::time::{SystemTime, UNIX_EPOCH};

fn path(name: &str) -> std::path::PathBuf {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!("ptr-{name}-{nonce}.ledger"))
}

#[test]
fn all_event_variants_roundtrip_across_reopen() {
    let path = path("roundtrip");
    let expected = vec![
        LedgerEvent::SemanticDeltaCommitted {
            base_revision: Revision(2),
            revision: Revision(3),
            encoded_delta: vec![0, 1, 255],
        },
        LedgerEvent::CapsuleCommitted {
            project: ProjectId::from("p"),
            capsule: CapsuleId::from("c"),
            generation: Generation(1),
        },
        LedgerEvent::CapsuleSuperseded {
            capsule: CapsuleId::from("c"),
            old: Generation(1),
            new: Generation(2),
        },
        LedgerEvent::Revoked {
            subject: "c".into(),
            generation: Generation(2),
        },
        LedgerEvent::HardConstraintCommitted {
            key: "immutable".into(),
            generation: Generation(3),
        },
        LedgerEvent::VerifierAttested {
            subject: "candidate".into(),
            passed: true,
        },
        LedgerEvent::ProcedurePromoted {
            id: "proc".into(),
            generation: Generation(4),
        },
        LedgerEvent::ProcedureRevoked {
            id: "proc".into(),
            generation: Generation(4),
        },
        LedgerEvent::SnapshotCommitted {
            revision: 9,
            covers: CommitIndex(7),
        },
    ];

    {
        let mut ledger = FileLedger::open(&path).unwrap();
        for event in &expected {
            ledger.append_durable(event.clone()).unwrap();
        }
    }

    let reopened = FileLedger::open(&path).unwrap();
    let actual = reopened
        .events()
        .iter()
        .map(|event| event.event.clone())
        .collect::<Vec<_>>();
    assert_eq!(actual, expected);
    drop(reopened);
    std::fs::remove_file(path).unwrap();
}

#[test]
fn partial_trailing_record_requires_explicit_anchored_recovery() {
    let path = path("partial-tail");
    let (trusted, history) = {
        let mut ledger = FileLedger::open(&path).unwrap();
        ledger
            .append_durable(LedgerEvent::Revoked {
                subject: "capsule:a".into(),
                generation: Generation(7),
            })
            .unwrap();
        (ledger.anchor().unwrap(), ledger.events().to_vec())
    };
    let durable = std::fs::read(&path).unwrap();
    let mut next = history;
    next.push(ptr_ledger::CommittedEvent {
        index: CommitIndex(2),
        event: LedgerEvent::VerifierAttested {
            subject: "pending".into(),
            passed: false,
        },
    });
    let all = ptr_ledger::integrity::encode_log(&next).unwrap();
    {
        let mut file = OpenOptions::new().append(true).open(&path).unwrap();
        file.write_all(&all[durable.len()..all.len() - 1]).unwrap();
        file.flush().unwrap();
    }
    let damaged = std::fs::read(&path).unwrap();
    assert!(FileLedger::open(&path).is_err());
    assert_eq!(std::fs::read(&path).unwrap(), damaged);
    let reopened = FileLedger::recover_unacknowledged_tail(&path, trusted).unwrap();
    assert_eq!(reopened.events().len(), 1);
    drop(reopened);
    assert_eq!(std::fs::read(&path).unwrap(), durable);
    std::fs::remove_file(path).unwrap();
}
