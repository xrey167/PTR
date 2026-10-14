mod common;

use std::fs::OpenOptions;
use std::io::Write;
use std::num::NonZeroU32;

use common::{keys, projection, telemetry, TempDir};
use ptr_events::{
    Authority, BackpressurePolicy, BusError, ClassPolicies, CorruptionKind, EventConsumer,
    EventProducer, FileEventLog, FileLogConfig, FsyncPolicy, Limit, Offset, PolicyViolation,
    RetentionPolicy,
};

fn config() -> FileLogConfig {
    FileLogConfig::new(1_000)
}

fn open(dir: &TempDir, config: FileLogConfig) -> FileEventLog {
    FileEventLog::open(dir.path(), config).unwrap()
}

fn fill(log: &mut FileEventLog, count: usize) {
    for i in 0..count {
        log.publish(projection(&format!("k{i:02}"))).unwrap();
    }
}

#[test]
fn records_keep_order_class_authority_and_content_across_reopen() {
    let dir = TempDir::new("order");
    {
        let mut log = open(&dir, config());
        log.register("c", Offset(0)).unwrap();
        assert_eq!(log.publish(projection("a")), Ok(Offset(1)));
        assert_eq!(log.publish(telemetry("b")), Ok(Offset(2)));
        assert_eq!(log.publish(projection("c")), Ok(Offset(3)));
    }
    let log = open(&dir, config());
    assert_eq!(log.recovery().records, 3);
    assert_eq!(log.recovery().torn_tail, None);
    let seen = log.poll("c", 10).unwrap();
    assert_eq!(keys(&seen), ["a", "b", "c"]);
    assert_eq!(seen[0].authority, Authority::Authoritative);
    assert_eq!(seen[1].authority, Authority::Diagnostic);
    assert_eq!(seen[1].payload, b"b");
    assert_eq!(log.end(), Offset(3));
}

#[test]
fn consumer_offsets_are_independent_and_survive_reopen() {
    let dir = TempDir::new("offsets");
    {
        let mut log = open(&dir, config());
        log.register("a", Offset(0)).unwrap();
        log.register("b", Offset(0)).unwrap();
        for key in ["1", "2", "3"] {
            log.publish(projection(key)).unwrap();
        }
        log.commit("a", Offset(3)).unwrap();
        log.commit("b", Offset(1)).unwrap();
        // A late, older commit does not rewind.
        log.commit("a", Offset(2)).unwrap();
        assert_eq!(log.committed("a"), Ok(Offset(3)));
    }
    let log = open(&dir, config());
    assert_eq!(log.committed("a"), Ok(Offset(3)));
    assert_eq!(log.committed("b"), Ok(Offset(1)));
    assert!(log.poll("a", 10).unwrap().is_empty());
    assert_eq!(keys(&log.poll("b", 10).unwrap()), ["2", "3"]);
    assert_eq!(log.lag("b"), Ok(2));
    assert!(!dir.path().join("offsets.bin.tmp").exists());
}

#[test]
fn ack_semantics_match_the_in_memory_bus() {
    let dir = TempDir::new("ack");
    let mut log = open(&dir, config());
    log.register("c", Offset(0)).unwrap();
    log.publish(projection("a")).unwrap();
    assert_eq!(log.poll("c", 10).unwrap().len(), 1);
    assert_eq!(log.poll("c", 10).unwrap().len(), 1);
    assert!(log.poll("c", 0).unwrap().is_empty());
    assert_eq!(
        log.commit("c", Offset(2)),
        Err(BusError::CommitBeyondEnd {
            requested: Offset(2),
            end: Offset(1)
        })
    );
    assert!(matches!(
        log.poll("nobody", 1),
        Err(BusError::UnknownConsumer { .. })
    ));
    assert!(matches!(
        log.register("x", Offset(9)),
        Err(BusError::CommitBeyondEnd { .. })
    ));
    log.commit("c", Offset(1)).unwrap();
    assert!(log.poll("c", 10).unwrap().is_empty());
}

#[test]
fn replay_after_restart_from_any_offset_at_or_above_low_water() {
    let dir = TempDir::new("replay");
    let mut cfg = config();
    cfg.retention = RetentionPolicy::with_max_records(100);
    {
        let mut log = open(&dir, cfg);
        log.register("c", Offset(0)).unwrap();
        for key in ["a", "b", "c", "d"] {
            log.publish(projection(key)).unwrap();
        }
        log.commit("c", Offset(4)).unwrap();
    }
    let log = open(&dir, cfg);
    assert_eq!(log.low_water(), Offset(0));
    assert_eq!(
        keys(&log.replay(Offset(0), 10).unwrap()),
        ["a", "b", "c", "d"]
    );
    assert_eq!(keys(&log.replay(Offset(2), 10).unwrap()), ["c", "d"]);
    assert_eq!(keys(&log.replay(Offset(1), 2).unwrap()), ["b", "c"]);
    assert!(log.replay(Offset(4), 10).unwrap().is_empty());
    assert!(matches!(
        log.replay(Offset(5), 10),
        Err(BusError::CommitBeyondEnd { .. })
    ));
}

#[test]
fn a_torn_tail_is_truncated_reported_and_the_log_stays_appendable() {
    let dir = TempDir::new("torn");
    {
        let mut log = open(&dir, config());
        log.register("c", Offset(0)).unwrap();
        log.publish(projection("a")).unwrap();
        log.publish(projection("b")).unwrap();
    }
    let segment = dir.segments().remove(0);
    let good_len = std::fs::metadata(&segment).unwrap().len();
    // A partial frame: a plausible header and part of the body.
    let mut file = OpenOptions::new().append(true).open(&segment).unwrap();
    file.write_all(&60u32.to_le_bytes()).unwrap();
    file.write_all(&0xDEAD_BEEFu32.to_le_bytes()).unwrap();
    file.write_all(&[7u8; 20]).unwrap();
    drop(file);

    let mut log = open(&dir, config());
    let torn = log.recovery().torn_tail.clone().unwrap();
    assert_eq!(torn.discarded_bytes, 28);
    assert_eq!(log.recovery().records, 2);
    assert_eq!(std::fs::metadata(&segment).unwrap().len(), good_len);
    assert_eq!(keys(&log.poll("c", 10).unwrap()), ["a", "b"]);
    assert_eq!(log.publish(projection("c")), Ok(Offset(3)));
    drop(log);
    let log = open(&dir, config());
    assert_eq!(log.recovery().torn_tail, None);
    assert_eq!(keys(&log.poll("c", 10).unwrap()), ["a", "b", "c"]);
}

#[test]
fn a_header_cut_short_is_also_a_torn_tail() {
    let dir = TempDir::new("torn-header");
    {
        let mut log = open(&dir, config());
        log.publish(projection("a")).unwrap();
    }
    let segment = dir.segments().remove(0);
    let mut file = OpenOptions::new().append(true).open(&segment).unwrap();
    file.write_all(&[1, 2, 3]).unwrap();
    drop(file);
    let log = open(&dir, config());
    assert_eq!(
        log.recovery().torn_tail.as_ref().unwrap().discarded_bytes,
        3
    );
    assert_eq!(log.end(), Offset(1));
}

#[test]
fn a_truncated_last_record_loses_only_that_record() {
    let dir = TempDir::new("cut");
    {
        let mut log = open(&dir, config());
        log.publish(projection("a")).unwrap();
        log.publish(projection("b")).unwrap();
    }
    let segment = dir.segments().remove(0);
    let len = std::fs::metadata(&segment).unwrap().len();
    let file = OpenOptions::new().write(true).open(&segment).unwrap();
    file.set_len(len - 3).unwrap();
    drop(file);
    let mut log = open(&dir, config());
    assert!(log.recovery().torn_tail.is_some());
    assert_eq!(log.end(), Offset(1));
    assert_eq!(log.publish(projection("b2")), Ok(Offset(2)));
}

#[test]
fn mid_file_corruption_is_a_typed_error_not_a_repair() {
    let dir = TempDir::new("corrupt");
    {
        let mut log = open(&dir, config());
        for key in ["a", "b", "c"] {
            log.publish(projection(key)).unwrap();
        }
    }
    let segment = dir.segments().remove(0);
    let mut bytes = std::fs::read(&segment).unwrap();
    // Flip a byte inside the first record's body (frame header is 8 bytes).
    bytes[40] ^= 0xFF;
    std::fs::write(&segment, &bytes).unwrap();
    let error = FileEventLog::open(dir.path(), config()).unwrap_err();
    assert!(matches!(
        error,
        BusError::Corrupt {
            kind: CorruptionKind::ChecksumMismatch,
            position: 0,
            ..
        }
    ));
    assert_eq!(error.code(), "PTR_BUS_CORRUPT");
    // The file was left untouched.
    assert_eq!(std::fs::read(&segment).unwrap(), bytes);
}

#[test]
fn a_damaged_non_final_segment_is_corruption_even_at_its_end() {
    let dir = TempDir::new("corrupt-old");
    let mut cfg = config();
    cfg.segment_max_bytes = 60;
    {
        let mut log = open(&dir, cfg);
        log.register("c", Offset(0)).unwrap();
        fill(&mut log, 3);
    }
    let segments = dir.segments();
    assert!(segments.len() >= 2);
    let len = std::fs::metadata(&segments[0]).unwrap().len();
    let file = OpenOptions::new().write(true).open(&segments[0]).unwrap();
    file.set_len(len - 2).unwrap();
    drop(file);
    assert!(matches!(
        FileEventLog::open(dir.path(), cfg),
        Err(BusError::Corrupt {
            kind: CorruptionKind::IncompleteFrame,
            ..
        })
    ));
}

#[test]
fn a_missing_middle_segment_is_detected() {
    let dir = TempDir::new("gap");
    let mut cfg = config();
    cfg.segment_max_bytes = 60;
    {
        let mut log = open(&dir, cfg);
        log.register("c", Offset(0)).unwrap();
        fill(&mut log, 4);
    }
    let segments = dir.segments();
    assert!(segments.len() >= 3);
    std::fs::remove_file(&segments[1]).unwrap();
    assert!(matches!(
        FileEventLog::open(dir.path(), cfg),
        Err(BusError::Corrupt {
            kind: CorruptionKind::SegmentGap { .. },
            ..
        })
    ));
}

#[test]
fn a_damaged_offsets_file_is_a_typed_error() {
    let dir = TempDir::new("badoffsets");
    {
        let mut log = open(&dir, config());
        log.register("c", Offset(0)).unwrap();
        log.publish(projection("a")).unwrap();
        log.commit("c", Offset(1)).unwrap();
    }
    let path = dir.path().join("offsets.bin");
    let mut bytes = std::fs::read(&path).unwrap();
    let middle = bytes.len() / 2;
    bytes[middle] ^= 1;
    std::fs::write(&path, bytes).unwrap();
    assert!(matches!(
        FileEventLog::open(dir.path(), config()),
        Err(BusError::Corrupt { .. })
    ));
}

#[test]
fn offsets_ahead_of_a_lost_tail_are_reported_not_clamped() {
    let dir = TempDir::new("ahead");
    {
        let mut log = open(&dir, config());
        log.register("c", Offset(0)).unwrap();
        log.publish(projection("a")).unwrap();
        log.publish(projection("b")).unwrap();
        log.commit("c", Offset(2)).unwrap();
    }
    let segment = dir.segments().remove(0);
    let len = std::fs::metadata(&segment).unwrap().len();
    let file = OpenOptions::new().write(true).open(&segment).unwrap();
    file.set_len(len - 1).unwrap();
    drop(file);
    assert_eq!(
        FileEventLog::open(dir.path(), config()).unwrap_err(),
        BusError::OffsetsAheadOfLog {
            consumer: "c".into(),
            committed: Offset(2),
            end: Offset(1)
        }
    );
}

#[test]
fn segments_roll_over_by_size_and_keep_ordering_across_files() {
    let dir = TempDir::new("rollover");
    let mut cfg = config();
    cfg.segment_max_bytes = 100;
    cfg.retention = RetentionPolicy::with_max_records(1_000);
    let mut log = open(&dir, cfg);
    log.register("c", Offset(0)).unwrap();
    let all: Vec<String> = (0..12).map(|i| format!("k{i:02}")).collect();
    fill(&mut log, 12);
    assert!(log.segment_count() >= 4, "{}", log.segment_count());
    assert_eq!(dir.segments().len(), log.segment_count());
    assert_eq!(keys(&log.poll("c", 100).unwrap()), all);
    drop(log);
    let log = open(&dir, cfg);
    assert_eq!(keys(&log.replay(Offset(0), 100).unwrap()), all);
    assert_eq!(keys(&log.replay(Offset(7), 100).unwrap()), all[7..]);
}

#[test]
fn acknowledged_segments_are_deleted_and_the_low_water_mark_advances() {
    let dir = TempDir::new("delete");
    let mut cfg = config();
    cfg.segment_max_bytes = 100;
    let mut log = open(&dir, cfg);
    log.register("c", Offset(0)).unwrap();
    fill(&mut log, 12);
    let before = log.segment_count();
    assert_eq!(log.low_water(), Offset(0));
    log.commit("c", Offset(12)).unwrap();
    // The newest segment is never deleted.
    assert_eq!(log.segment_count(), 1);
    assert_eq!(dir.segments().len(), 1);
    assert!(before > 1);
    let low = log.low_water();
    assert!(low.0 > 0 && low.0 < 12);
    assert_eq!(
        log.replay(Offset(0), 10),
        Err(BusError::Truncated {
            requested: Offset(0),
            low_water: low
        })
    );
    drop(log);
    let mut log = open(&dir, cfg);
    assert_eq!(log.low_water(), low);
    assert_eq!(log.publish(projection("next")), Ok(Offset(13)));
    assert_eq!(log.poll("c", 10).unwrap().len(), 1);
}

#[test]
fn unacknowledged_authoritative_segments_are_never_deleted() {
    let dir = TempDir::new("keep");
    let mut cfg = config();
    cfg.segment_max_bytes = 100;
    cfg.retention = RetentionPolicy {
        max_records: Some(0),
        max_bytes: Some(0),
        max_age_ticks: Some(0),
    };
    let mut log = open(&dir, cfg);
    log.register("slow", Offset(0)).unwrap();
    log.register("fast", Offset(0)).unwrap();
    fill(&mut log, 12);
    let segments = log.segment_count();
    log.commit("fast", Offset(12)).unwrap();
    log.advance_tick(1_000_000).unwrap();
    assert_eq!(log.segment_count(), segments);
    assert_eq!(log.low_water(), Offset(0));
    assert_eq!(log.poll("slow", 100).unwrap().len(), 12);
    // Only the part the slow consumer acknowledged can go.
    log.commit("slow", Offset(5)).unwrap();
    assert!(log.low_water().0 <= 5);
    assert_eq!(log.poll("slow", 100).unwrap().len(), 7);
}

#[test]
fn retention_by_count_decides_how_much_acked_history_stays() {
    let dir = TempDir::new("ret-count");
    let mut cfg = config();
    cfg.segment_max_bytes = 100;
    cfg.retention = RetentionPolicy::with_max_records(6);
    let mut log = open(&dir, cfg);
    log.register("c", Offset(0)).unwrap();
    fill(&mut log, 12);
    log.commit("c", Offset(12)).unwrap();
    let retained = 12 - log.low_water().0;
    // Whole segments go, so the window may overshoot by up to one segment.
    assert!(
        retained <= 6 + 3,
        "history stays near the window: {retained}"
    );
    assert!(retained > 0);
}

#[test]
fn retention_by_bytes_decides_how_much_acked_history_stays() {
    let dir = TempDir::new("ret-bytes");
    let mut cfg = config();
    cfg.segment_max_bytes = 100;
    cfg.retention = RetentionPolicy::with_max_bytes(150);
    let mut log = open(&dir, cfg);
    log.register("c", Offset(0)).unwrap();
    fill(&mut log, 12);
    let before = log.segment_count();
    log.commit("c", Offset(12)).unwrap();
    assert!(log.segment_count() >= 2 && log.segment_count() < before);
}

#[test]
fn retention_by_logical_age_waits_for_the_caller_clock() {
    let dir = TempDir::new("ret-age");
    let mut cfg = config();
    cfg.segment_max_bytes = 100;
    cfg.retention = RetentionPolicy {
        max_records: Some(1_000),
        ..RetentionPolicy::with_max_age_ticks(10)
    };
    let mut log = open(&dir, cfg);
    log.register("c", Offset(0)).unwrap();
    fill(&mut log, 12);
    log.commit("c", Offset(12)).unwrap();
    assert_eq!(log.low_water(), Offset(0));
    let early = log.advance_tick(10).unwrap();
    assert_eq!(early.removed_segments, 0);
    let late = log.advance_tick(11).unwrap();
    assert!(late.removed_segments > 0);
    assert_eq!(late.low_water, log.low_water());
    assert!(log.low_water().0 > 0);
    assert_eq!(
        log.advance_tick(1),
        Err(BusError::ClockRegression {
            current: 11,
            requested: 1
        })
    );
}

#[test]
fn the_clock_survives_reopen_through_the_newest_record() {
    let dir = TempDir::new("tick");
    {
        let mut log = open(&dir, config());
        log.advance_tick(42).unwrap();
        log.publish(projection("a")).unwrap();
    }
    let mut log = open(&dir, config());
    assert_eq!(log.tick(), 42);
    assert!(matches!(
        log.advance_tick(41),
        Err(BusError::ClockRegression { .. })
    ));
}

#[test]
fn backpressure_rejects_with_capacity_depth_and_the_blocking_consumer() {
    let dir = TempDir::new("full");
    let mut log = open(&dir, FileLogConfig::new(2));
    log.register("fast", Offset(0)).unwrap();
    log.register("slow", Offset(0)).unwrap();
    log.publish(telemetry("a")).unwrap();
    log.publish(projection("b")).unwrap();
    log.commit("fast", Offset(2)).unwrap();
    assert_eq!(
        log.publish(telemetry("c")),
        Err(BusError::Full {
            limit: Limit::Records,
            capacity: 2,
            depth: 2,
            blocked_by: Some("slow".into())
        })
    );
    assert_eq!(log.end(), Offset(2));
    assert_eq!(log.blocked_by().unwrap().consumer, "slow");
    assert_eq!(log.dropped("slow"), Ok(0));
    log.commit("slow", Offset(1)).unwrap();
    assert_eq!(log.publish(projection("c")), Ok(Offset(3)));
    assert_eq!(keys(&log.poll("slow", 10).unwrap()), ["b", "c"]);
}

#[test]
fn the_file_log_refuses_a_telemetry_dropping_policy() {
    let dir = TempDir::new("nodrop");
    let mut cfg = config();
    cfg.policies = ClassPolicies::new(
        BackpressurePolicy::Reject,
        BackpressurePolicy::DropOldestTelemetry,
    )
    .unwrap();
    assert_eq!(
        FileEventLog::open(dir.path(), cfg).unwrap_err(),
        BusError::InvalidPolicy {
            violation: PolicyViolation::DropUnsupportedByBackend
        }
    );
}

#[test]
fn fsync_policies_all_produce_a_recoverable_log() {
    for (label, fsync) in [
        ("always", FsyncPolicy::Always),
        (
            "every",
            FsyncPolicy::EveryRecords(NonZeroU32::new(2).unwrap()),
        ),
        ("never", FsyncPolicy::Never),
    ] {
        let dir = TempDir::new(label);
        let mut cfg = config();
        cfg.fsync = fsync;
        cfg.segment_max_bytes = 100;
        {
            let mut log = open(&dir, cfg);
            log.register("c", Offset(0)).unwrap();
            fill(&mut log, 6);
            log.sync().unwrap();
            log.commit("c", Offset(2)).unwrap();
        }
        let log = open(&dir, cfg);
        assert_eq!(log.end(), Offset(6), "{label}");
        assert_eq!(log.poll("c", 10).unwrap().len(), 4, "{label}");
    }
}

#[test]
fn oversized_records_and_foreign_files_are_handled() {
    let dir = TempDir::new("limits");
    let mut cfg = config();
    cfg.max_record_bytes = 8;
    let mut log = open(&dir, cfg);
    let mut big = projection("k");
    big.payload = vec![0; 9];
    assert!(matches!(
        log.publish(big),
        Err(BusError::FieldTooLong { len: 9, max: 8, .. })
    ));
    assert_eq!(log.end(), Offset(0));
    std::fs::write(dir.path().join("README.txt"), b"not a segment").unwrap();
    drop(log);
    let log = open(&dir, cfg);
    assert_eq!(log.recovery().segments, 0);
}
