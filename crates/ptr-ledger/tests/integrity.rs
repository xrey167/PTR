use ptr_ledger::{integrity::*, CommittedEvent, FileLedger, LedgerEvent};
use ptr_types::{CommitIndex, Generation};
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
            base_revision: ptr_types::Revision(0),
            revision: ptr_types::Revision(1),
            encoded_delta: vec![0; MAX_RECORD_BYTES],
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
