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
fn partial_trailing_record_is_truncated_without_admission() {
    let path = path("partial-tail");
    {
        let mut ledger = FileLedger::open(&path).unwrap();
        ledger
            .append_durable(LedgerEvent::Revoked {
                subject: "capsule:a".into(),
                generation: Generation(7),
            })
            .unwrap();
    }

    let durable_len = std::fs::metadata(&path).unwrap().len();
    {
        let mut file = OpenOptions::new().append(true).open(&path).unwrap();
        file.write_all(&100_u32.to_le_bytes()).unwrap();
        file.write_all(b"partial").unwrap();
        file.flush().unwrap();
    }
    assert!(std::fs::metadata(&path).unwrap().len() > durable_len);

    let reopened = FileLedger::open(&path).unwrap();
    assert_eq!(reopened.events().len(), 1);
    assert_eq!(std::fs::metadata(&path).unwrap().len(), durable_len);
    drop(reopened);
    std::fs::remove_file(path).unwrap();
}
