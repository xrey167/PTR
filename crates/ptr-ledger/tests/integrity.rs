use ptr_ledger::{
    check_encodable, integrity::*, Attestation, CommittedEvent, FileLedger, LedgerEvent,
    MergeAuthorityRecord, MergeRecord, SemanticOrigin,
};
use ptr_types::{CommitIndex, Generation, Revision, VerificationLevel};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

struct Temp(PathBuf);
impl Temp {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "ptr-integrity-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&path).unwrap();
        Self(path)
    }
    fn path(&self) -> PathBuf {
        self.0.join("log")
    }
}
impl Drop for Temp {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}
fn event(n: u64) -> CommittedEvent {
    CommittedEvent {
        index: CommitIndex(n),
        event: LedgerEvent::Revoked {
            subject: format!("subject:{n}"),
            generation: Generation(n),
        },
    }
}

#[test]
fn all_single_bit_mutations_of_complete_log_are_rejected() {
    let original = encode_log(&[event(1), event(2), event(3)]).unwrap();
    for offset in 0..original.len() {
        for bit in 0..8 {
            let mut bytes = original.clone();
            bytes[offset] ^= 1 << bit;
            assert!(decode_log(&bytes).is_err(), "offset {offset} bit {bit}");
        }
    }
    assert_eq!(
        decode_log(&original).unwrap().events(),
        [event(1), event(2), event(3)]
    );
}

#[test]
fn strict_open_never_rewrites_corruption_and_recovery_never_discards_complete_records() {
    let tmp = Temp::new();
    let prefix = encode_log(&[event(1)]).unwrap();
    let trusted = decode_log(&prefix).unwrap().anchor();
    let complete = encode_log(&[event(1), event(2)]).unwrap();
    for offset in [
        0,
        prefix.len() + 16,
        prefix.len() + FRAME_HEADER_BYTES,
        complete.len() - 1,
    ] {
        let mut bytes = complete.clone();
        bytes[offset] ^= 1;
        std::fs::write(tmp.path(), &bytes).unwrap();
        assert!(FileLedger::open(tmp.path()).is_err());
        assert!(FileLedger::recover_unacknowledged_tail(tmp.path(), trusted).is_err());
        assert_eq!(std::fs::read(tmp.path()).unwrap(), bytes);
    }
    std::fs::write(tmp.path(), &complete).unwrap();
    assert!(FileLedger::recover_unacknowledged_tail(tmp.path(), trusted).is_err());
    assert_eq!(std::fs::read(tmp.path()).unwrap(), complete);
}

#[test]
fn every_incomplete_frame_cut_requires_matching_independent_anchor() {
    let tmp = Temp::new();
    let prefix = encode_log(&[event(1)]).unwrap();
    let trusted = decode_log(&prefix).unwrap().anchor();
    let complete = encode_log(&[event(1), event(2)]).unwrap();
    let future = decode_log(&complete).unwrap().anchor();
    for cut in prefix.len() + 1..complete.len() {
        let bytes = &complete[..cut];
        std::fs::write(tmp.path(), bytes).unwrap();
        assert!(FileLedger::open(tmp.path()).is_err(), "cut {cut}");
        assert_eq!(std::fs::read(tmp.path()).unwrap(), bytes);
        assert!(FileLedger::recover_unacknowledged_tail(tmp.path(), future).is_err());
        assert_eq!(std::fs::read(tmp.path()).unwrap(), bytes);
        let recovered = FileLedger::recover_unacknowledged_tail(tmp.path(), trusted).unwrap();
        assert_eq!(recovered.events(), [event(1)]);
        drop(recovered);
        assert_eq!(std::fs::read(tmp.path()).unwrap(), prefix);
    }
}

#[test]
fn trusted_anchor_detects_complete_suffix_loss_and_rehashed_alternative_history() {
    let tmp = Temp::new();
    let complete = encode_log(&[event(1), event(2)]).unwrap();
    let trusted = decode_log(&complete).unwrap().anchor();
    let mut other = event(2);
    other.event = LedgerEvent::Revoked {
        subject: "different".into(),
        generation: Generation(2),
    };
    for bytes in [
        encode_log(&[event(1)]).unwrap(),
        encode_log(&[event(1), other]).unwrap(),
        LOG_MAGIC.to_vec(),
        vec![],
    ] {
        std::fs::write(tmp.path(), &bytes).unwrap();
        assert!(FileLedger::open_at(tmp.path(), trusted).is_err());
        assert_eq!(std::fs::read(tmp.path()).unwrap(), bytes);
    }
    std::fs::remove_file(tmp.path()).unwrap();
    assert!(FileLedger::open_at(tmp.path(), trusted).is_err());
    assert!(!tmp.path().exists());
}

#[test]
fn order_duplicates_lengths_versions_and_appended_garbage_fail_closed() {
    let prefix = encode_log(&[event(1)]).unwrap();
    let full = encode_log(&[event(1), event(2)]).unwrap();
    let first = &prefix[LOG_MAGIC.len()..];
    let second = &full[prefix.len()..];
    for bytes in [
        [LOG_MAGIC.as_slice(), second, first].concat(),
        [&full[..], second].concat(),
        [&full[..], b"garbage"].concat(),
    ] {
        assert!(decode_log(&bytes).is_err());
    }
    assert!(encode_log(&[event(2)]).is_err());
    let mut bytes = full.clone();
    let start = prefix.len();
    bytes[start + 16..start + 20].copy_from_slice(&u32::MAX.to_le_bytes());
    let header_hash = sha256(&[b"PTRHDR02".as_slice(), &bytes[start..start + 52]].concat());
    bytes[start + 52..start + 84].copy_from_slice(&header_hash);
    assert!(decode_log(&bytes).is_err());
    for cut in 0..LOG_MAGIC.len() {
        assert!(decode_log(&LOG_MAGIC[..cut]).is_err());
    }
    assert!(decode_log(b"PTRLOG03").is_err());
}

#[test]
fn create_new_and_rejected_oversize_append_do_not_destroy_existing_data() {
    let tmp = Temp::new();
    let bytes = encode_log(&[event(1)]).unwrap();
    let trusted = decode_log(&bytes).unwrap().anchor();
    let mut ledger = FileLedger::create_from_log(tmp.path(), &bytes, trusted).unwrap();
    assert!(FileLedger::create_from_log(tmp.path(), LOG_MAGIC, LogAnchor::empty()).is_err());
    assert!(ledger
        .append_durable(LedgerEvent::SemanticDeltaCommitted {
            base_revision: Revision(0),
            revision: Revision(1),
            encoded_delta: vec![0; MAX_RECORD_BYTES],
            origin: SemanticOrigin::Legacy,
        })
        .is_err());
    assert_eq!(ledger.anchor().unwrap(), trusted);
    assert_eq!(
        ledger.append_durable(event(2).event).unwrap(),
        CommitIndex(2)
    );
    drop(ledger);
    assert_eq!(FileLedger::open(tmp.path()).unwrap().events().len(), 2);
}

#[test]
fn legacy_inspection_is_explicit_strict_and_never_an_automatic_downgrade() {
    let tmp = Temp::new();
    // Historical v1 VerifierAttested("x", true): tag, string length, string, bool.
    let legacy = [7, 0, 0, 0, 4, 1, 0, 0, 0, b'x', 1];
    std::fs::write(tmp.path(), legacy).unwrap();
    assert!(FileLedger::open(tmp.path()).is_err());
    let events = FileLedger::read_legacy(tmp.path()).unwrap();
    assert_eq!(events[0].index, CommitIndex(1));
    assert_eq!(
        events[0].event,
        LedgerEvent::VerifierAttested {
            subject: "x".into(),
            passed: true
        }
    );
    assert_eq!(std::fs::read(tmp.path()).unwrap(), legacy);
    for end in 1..legacy.len() {
        assert!(decode_legacy_log(&legacy[..end]).is_err());
    }
}

#[test]
fn independent_python_hashlib_golden_record_matches_exact_format() {
    let committed = CommittedEvent {
        index: CommitIndex(1),
        event: LedgerEvent::VerifierAttested {
            subject: "x".into(),
            passed: true,
        },
    };
    let encoded = encode_log(&[committed]).unwrap();
    use std::fmt::Write;
    let mut actual = String::new();
    for byte in &encoded {
        write!(&mut actual, "{byte:02x}").unwrap();
    }
    assert_eq!(actual, "5054524c4f47303250545246523030320100000000000000070000005ce1c23d7d9aced70d84663f51b3b393e2f0de428e8eee92259b97a85a313c52186d0b61654ec8eb7f852dabef3826c7f49554c17939e5608893018ac2cabdf5040100000078013518dacb5a52f38b39e5e2b16d9400bc2838487ff0871a45713b2c5501d84f92505452454e443032");
}

#[test]
fn migration_guard_excludes_a_second_writer_until_dropped() {
    let tmp = Temp::new();
    std::fs::write(tmp.path(), [7, 0, 0, 0, 4, 1, 0, 0, 0, b'x', 1]).unwrap();
    let legacy = FileLedger::open_legacy_for_migration(tmp.path()).unwrap();
    assert_eq!(legacy.events().len(), 1);
    let error = FileLedger::read_legacy(tmp.path()).unwrap_err();
    assert_eq!(error.kind(), std::io::ErrorKind::WouldBlock);
    drop(legacy);
    assert!(FileLedger::read_legacy(tmp.path()).is_ok());
}

fn semantic(n: u64, origin: SemanticOrigin) -> CommittedEvent {
    CommittedEvent {
        index: CommitIndex(n),
        event: LedgerEvent::SemanticDeltaCommitted {
            base_revision: Revision(n - 1),
            revision: Revision(n),
            encoded_delta: vec![u8::try_from(n).unwrap(); 3],
            origin,
        },
    }
}

fn attestation() -> Attestation {
    Attestation {
        required: VerificationLevel::FullSemantic,
        level: VerificationLevel::FullSemantic,
        verifiers: vec!["schema".into()],
        findings: Vec::new(),
    }
}

fn merge(authority: MergeAuthorityRecord, rebased: &[&str]) -> SemanticOrigin {
    SemanticOrigin::Merge(MergeRecord {
        branch: "branch-7".into(),
        author: "alice".into(),
        seal: [1; 32],
        plan: [2; 32],
        dependencies: [3; 32],
        rebased: rebased.iter().map(|key| (*key).to_owned()).collect(),
        verification: attestation(),
        authority,
    })
}

#[test]
fn a_log_mixing_tag_eight_and_tag_twelve_records_round_trips_canonically() {
    // The codec checks shape only. Which origin may follow which is replay's
    // rule, so a legacy record after attributed ones still round-trips here.
    let events = vec![
        semantic(1, SemanticOrigin::Legacy),
        event(2),
        semantic(
            3,
            SemanticOrigin::Request {
                request: "request:1".into(),
            },
        ),
        semantic(
            4,
            SemanticOrigin::PodOutput {
                request: "request:1".into(),
                pod: "pod-a".into(),
                level: VerificationLevel::SampleVerified,
            },
        ),
        semantic(
            5,
            SemanticOrigin::Host {
                principal: "operator".into(),
                verification: attestation(),
            },
        ),
        semantic(
            6,
            merge(
                MergeAuthorityRecord::Triage {
                    policy_version: "policy-3".into(),
                    score_bits: 0.5f32.to_bits(),
                },
                &[],
            ),
        ),
        semantic(
            7,
            merge(
                MergeAuthorityRecord::Reviewed {
                    reviewer: "bob".into(),
                },
                &["doc:a", "doc:b"],
            ),
        ),
        semantic(8, SemanticOrigin::Legacy),
    ];
    let bytes = encode_log(&events).unwrap();
    let verified = decode_log(&bytes).unwrap();
    assert_eq!(verified.events(), events.as_slice());
    assert_eq!(encode_log(verified.events()).unwrap(), bytes);

    // The durable path writes the same bytes and reads them back.
    let tmp = Temp::new();
    let mut ledger = FileLedger::create_new(tmp.path()).unwrap();
    for committed in &events {
        assert_eq!(
            ledger.append_durable(committed.event.clone()).unwrap(),
            committed.index
        );
    }
    drop(ledger);
    assert_eq!(std::fs::read(tmp.path()).unwrap(), bytes);
    assert_eq!(FileLedger::open(tmp.path()).unwrap().events(), events);
}

#[test]
fn a_record_above_the_bound_is_refused_before_framing() {
    // The largest delta whose tag-8 record still fits: the tag, both
    // revisions and the delta's length prefix take 21 bytes.
    let fits = MAX_RECORD_BYTES - 21;
    let legacy = LedgerEvent::SemanticDeltaCommitted {
        base_revision: Revision(0),
        revision: Revision(1),
        encoded_delta: vec![0; fits],
        origin: SemanticOrigin::Legacy,
    };
    check_encodable(&legacy).unwrap();
    // The same delta with an origin no longer fits.
    let attributed = LedgerEvent::SemanticDeltaCommitted {
        base_revision: Revision(0),
        revision: Revision(1),
        encoded_delta: vec![0; fits],
        origin: SemanticOrigin::Request {
            request: "request:1".into(),
        },
    };
    let error = check_encodable(&attributed).unwrap_err();
    assert_eq!(error.to_string(), "PTR_LOG_PAYLOAD_LIMIT");

    // An attestation outside its counts cannot be framed either, and is
    // refused rather than encoded into bytes no decoder reads back.
    let unattested = LedgerEvent::SemanticDeltaCommitted {
        base_revision: Revision(0),
        revision: Revision(1),
        encoded_delta: vec![0; 3],
        origin: SemanticOrigin::Host {
            principal: "operator".into(),
            verification: Attestation {
                verifiers: Vec::new(),
                ..attestation()
            },
        },
    };
    let error = check_encodable(&unattested).unwrap_err();
    assert_eq!(error.to_string(), "PTR_LEDGER_ATTESTATION_LIMIT");

    for (refused, code) in [
        (&attributed, "PTR_LOG_PAYLOAD_LIMIT"),
        (&unattested, "PTR_LEDGER_ATTESTATION_LIMIT"),
    ] {
        let committed = CommittedEvent {
            index: CommitIndex(1),
            event: refused.clone(),
        };
        assert_eq!(encode_log(&[committed]).unwrap_err().to_string(), code);

        let tmp = Temp::new();
        let bytes = encode_log(&[event(1)]).unwrap();
        let trusted = decode_log(&bytes).unwrap().anchor();
        let mut ledger = FileLedger::create_from_log(tmp.path(), &bytes, trusted).unwrap();
        let error = ledger.append_durable(refused.clone()).unwrap_err();
        assert_eq!(error.to_string(), code);
        // Refused before anything was written: the ledger still appends. The
        // bytes are read through the open ledger, because Windows refuses an
        // independent read of a file whose writer holds its lock.
        assert_eq!(ledger.anchor().unwrap(), trusted);
        assert_eq!(ledger.retained_bytes().unwrap(), bytes);
        assert_eq!(
            ledger.append_durable(event(2).event).unwrap(),
            CommitIndex(2)
        );
    }
    // The tag-8 record at the bound is framed and read back.
    let committed = CommittedEvent {
        index: CommitIndex(1),
        event: legacy,
    };
    let bytes = encode_log(std::slice::from_ref(&committed)).unwrap();
    assert_eq!(decode_log(&bytes).unwrap().events(), [committed]);
}
