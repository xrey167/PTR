mod common;

use common::TempJournal;
use ptr_memory::{CorruptionReason, JournalError, JournalKind, JournalOptions, MemoryJournal};
use std::fs;

fn open(path: &std::path::Path) -> (MemoryJournal, ptr_memory::JournalRecovery) {
    MemoryJournal::open(path, JournalKind::Memory, JournalOptions::default()).unwrap()
}

#[test]
fn journal_round_trips_payloads_in_order() {
    let file = TempJournal::new("rt");
    let (mut journal, recovery) = open(&file.0);
    assert!(recovery.records.is_empty());
    let payloads: Vec<Vec<u8>> = (0..50_u8).map(|i| vec![i; usize::from(i)]).collect();
    for payload in &payloads {
        journal.append(payload).unwrap();
    }
    drop(journal);
    let (_, recovery) = open(&file.0);
    assert_eq!(recovery.records, payloads);
    assert_eq!(recovery.truncated_bytes, 0);
}

#[test]
fn every_torn_tail_length_recovers_the_intact_prefix() {
    let file = TempJournal::new("torn");
    let (mut journal, _) = open(&file.0);
    journal.append(b"first").unwrap();
    journal.append(b"second").unwrap();
    drop(journal);
    let full = fs::read(&file.0).unwrap();
    let boundary = 8 + (8 + 5); // header + first frame
    for cut in boundary..full.len() {
        fs::write(&file.0, &full[..cut]).unwrap();
        let (mut journal, recovery) = open(&file.0);
        assert_eq!(recovery.records, vec![b"first".to_vec()], "cut at {cut}");
        assert_eq!(
            recovery.truncated_bytes,
            (cut - boundary) as u64,
            "cut at {cut}"
        );
        // The journal is usable again and the torn bytes are gone.
        journal.append(b"third").unwrap();
        drop(journal);
        let (_, again) = open(&file.0);
        assert_eq!(again.records, vec![b"first".to_vec(), b"third".to_vec()]);
        assert_eq!(again.truncated_bytes, 0);
    }
}

#[test]
fn partially_written_header_is_recreated() {
    let file = TempJournal::new("hdr");
    fs::write(&file.0, b"PTM").unwrap();
    let (mut journal, recovery) = open(&file.0);
    assert!(recovery.records.is_empty());
    journal.append(b"x").unwrap();
    drop(journal);
    assert_eq!(open(&file.0).1.records, vec![b"x".to_vec()]);
}

#[test]
fn corrupt_last_frame_is_a_torn_write() {
    let file = TempJournal::new("lastcrc");
    let (mut journal, _) = open(&file.0);
    journal.append(b"keep").unwrap();
    journal.append(b"damaged").unwrap();
    drop(journal);
    let mut bytes = fs::read(&file.0).unwrap();
    let last = bytes.len() - 1;
    bytes[last] ^= 0xff;
    fs::write(&file.0, &bytes).unwrap();
    let (_, recovery) = open(&file.0);
    assert_eq!(recovery.records, vec![b"keep".to_vec()]);
    assert!(recovery.truncated_bytes > 0);
}

#[test]
fn mid_file_corruption_is_a_typed_error_and_leaves_the_file_alone() {
    let file = TempJournal::new("corrupt");
    let (mut journal, _) = open(&file.0);
    journal.append(b"aaaa").unwrap();
    journal.append(b"bbbb").unwrap();
    journal.append(b"cccc").unwrap();
    drop(journal);
    let mut bytes = fs::read(&file.0).unwrap();
    bytes[8 + 8 + 4 + 8] ^= 0x01; // first byte of the second payload
    fs::write(&file.0, &bytes).unwrap();
    let err =
        MemoryJournal::open(&file.0, JournalKind::Memory, JournalOptions::default()).unwrap_err();
    assert_eq!(
        err,
        JournalError::Corrupt {
            offset: 8 + 12,
            reason: CorruptionReason::ChecksumMismatch
        }
    );
    assert_eq!(
        fs::read(&file.0).unwrap(),
        bytes,
        "file must not be modified"
    );
}

#[test]
fn oversized_length_in_the_middle_is_corruption() {
    let file = TempJournal::new("oversize");
    let (mut journal, _) = open(&file.0);
    journal.append(b"aaaa").unwrap();
    journal.append(b"bbbb").unwrap();
    drop(journal);
    let mut bytes = fs::read(&file.0).unwrap();
    bytes[8..12].copy_from_slice(&u32::MAX.to_le_bytes());
    fs::write(&file.0, &bytes).unwrap();
    assert_eq!(
        MemoryJournal::open(&file.0, JournalKind::Memory, JournalOptions::default()).unwrap_err(),
        JournalError::Corrupt {
            offset: 8,
            reason: CorruptionReason::OversizedFrame
        }
    );
}

#[test]
fn rewrite_compacts_and_survives_reopen() {
    let file = TempJournal::new("compact");
    let (mut journal, _) = open(&file.0);
    for i in 0..100_u8 {
        journal.append(&[i; 32]).unwrap();
    }
    let before = fs::metadata(&file.0).unwrap().len();
    journal.rewrite(&[b"snapshot".to_vec()]).unwrap();
    let after = fs::metadata(&file.0).unwrap().len();
    assert!(after < before);
    journal.append(b"post").unwrap();
    drop(journal);
    assert_eq!(
        open(&file.0).1.records,
        vec![b"snapshot".to_vec(), b"post".to_vec()]
    );
}

#[test]
fn wrong_store_kind_is_refused() {
    let file = TempJournal::new("kind");
    drop(open(&file.0));
    assert!(matches!(
        MemoryJournal::open(&file.0, JournalKind::Knowledge, JournalOptions::default()),
        Err(JournalError::KindMismatch { .. })
    ));
}
