use ptr_ledger::integrity::MAX_RECORD_BYTES;
use ptr_ledger::{Attestation, LedgerEvent, RaftEngineLedger, SemanticOrigin};
use ptr_types::{CapsuleId, Generation, ProjectId, Revision, VerificationLevel};
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

fn host_write(required: VerificationLevel, verifiers: &[&str]) -> LedgerEvent {
    LedgerEvent::SemanticDeltaCommitted {
        base_revision: Revision(0),
        revision: Revision(1),
        encoded_delta: vec![1, 2, 3],
        origin: SemanticOrigin::Host {
            principal: "operator".into(),
            verification: Attestation {
                required,
                level: VerificationLevel::FullSemantic,
                verifiers: verifiers.iter().map(|name| (*name).to_owned()).collect(),
                findings: Vec::new(),
            },
        },
    }
}

/// Events the decoder would refuse, or that exceed the record bound, each
/// with the code that refuses it.
fn refused_events() -> Vec<(LedgerEvent, &'static str)> {
    vec![
        (
            host_write(VerificationLevel::FullSemantic, &[]),
            "PTR_LEDGER_ATTESTATION_LIMIT",
        ),
        (
            host_write(VerificationLevel::SampleVerified, &["schema"]),
            "PTR_LEDGER_ATTESTATION_REQUIREMENT",
        ),
        (
            // The tag, both revisions and the delta's length take 21 bytes.
            LedgerEvent::SemanticDeltaCommitted {
                base_revision: Revision(0),
                revision: Revision(1),
                encoded_delta: vec![0; MAX_RECORD_BYTES - 20],
                origin: SemanticOrigin::Legacy,
            },
            "PTR_LOG_PAYLOAD_LIMIT",
        ),
    ]
}

#[test]
fn an_origin_the_decoder_refuses_is_refused_before_the_engine_writes_it() {
    let dir = dir();
    let accepted = host_write(VerificationLevel::FullSemantic, &["schema"]);
    {
        let mut ledger = RaftEngineLedger::open(&dir).unwrap();
        for (event, code) in refused_events() {
            let error = ledger.append_durable(event).unwrap_err();
            assert_eq!(error.to_string(), code);
            assert!(ledger.events().is_empty());
        }
        ledger.append_durable(accepted.clone()).unwrap();
        ledger.sync().unwrap();
    }
    let reopened = RaftEngineLedger::open(&dir).unwrap();
    assert_eq!(reopened.events().len(), 1);
    assert_eq!(reopened.events()[0].index.0, 1);
    assert_eq!(reopened.events()[0].event, accepted);
    drop(reopened);
    std::fs::remove_dir_all(dir).unwrap();
}
