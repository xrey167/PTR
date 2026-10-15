//! `ptrctl events` against logs written in the test with `ptr_events::FileEventLog`.

mod common;

use common::{code, ptrctl, stderr, stdout, TempDir};
use ptr_events::{
    EventClass, EventConsumer, EventProducer, FileEventLog, FileLogConfig, NewRecord, Offset,
};
use std::collections::BTreeMap;
use std::ffi::OsStr;
use std::path::{Path, PathBuf};

fn run(args: &[&OsStr]) -> std::process::Output {
    let mut all = vec![OsStr::new("events")];
    all.extend_from_slice(args);
    ptrctl(&all)
}

fn record(class: EventClass, topic: &str, payload: &[u8]) -> NewRecord {
    NewRecord {
        class,
        topic: topic.to_owned(),
        key: "capsule-1".to_owned(),
        payload: payload.to_vec(),
    }
}

/// Three records and one consumer ("proj") that has acknowledged the first.
fn write_log(dir: &Path) {
    let mut log = FileEventLog::open(dir, FileLogConfig::new(16)).expect("open");
    log.publish(record(EventClass::Projection, "ledger.commit", b"abcde"))
        .expect("publish");
    log.publish(record(EventClass::Telemetry, "exec.trace", b"xy"))
        .expect("publish");
    log.publish(record(EventClass::Projection, "ledger.commit", b""))
        .expect("publish");
    log.register("proj", Offset(0)).expect("register");
    log.commit("proj", Offset(1)).expect("commit");
}

/// Every file in `dir` with its bytes.
fn state(dir: &Path) -> BTreeMap<String, Vec<u8>> {
    std::fs::read_dir(dir)
        .expect("list")
        .map(|entry| {
            let entry = entry.expect("entry");
            (
                entry.file_name().to_string_lossy().into_owned(),
                std::fs::read(entry.path()).expect("read"),
            )
        })
        .collect()
}

fn segment(dir: &Path) -> PathBuf {
    std::fs::read_dir(dir)
        .expect("list")
        .map(|e| e.expect("entry").path())
        .find(|p| {
            p.file_name()
                .and_then(OsStr::to_str)
                .is_some_and(|n| n.starts_with("segment-"))
        })
        .expect("a segment file")
}

#[test]
fn lists_records_consumer_offsets_and_lag_without_touching_the_log() {
    let dir = TempDir::new("events-list");
    write_log(dir.path());
    // A stale temp file that `FileEventLog::open` would delete.
    dir.file("offsets.bin.tmp", b"stale");
    let before = state(dir.path());

    let output = run(&[
        dir.path().as_os_str(),
        OsStr::new("--consumer"),
        OsStr::new("proj"),
    ]);
    assert_eq!(code(&output), 0, "{}", stderr(&output));
    let text = stdout(&output);
    assert!(text.contains("records: 3"), "{text}");
    assert!(text.contains("end: 3"), "{text}");
    assert!(text.contains("consumer proj: committed 1 lag 2"), "{text}");
    assert!(
        text.contains("slowest consumer proj: committed 1 lag 2"),
        "{text}"
    );
    assert!(
        text.contains("1\tProjection\tAuthoritative\tledger.commit\t5\n"),
        "{text}"
    );
    assert!(
        text.contains("2\tTelemetry\tDiagnostic\texec.trace\t2\n"),
        "{text}"
    );
    assert!(
        text.contains("3\tProjection\tAuthoritative\tledger.commit\t0\n"),
        "{text}"
    );
    assert!(text.contains("listed 3 record(s)"), "{text}");

    assert_eq!(
        state(dir.path()),
        before,
        "the log directory must be unchanged"
    );
}

#[test]
fn from_and_limit_select_a_window_of_records() {
    let dir = TempDir::new("events-window");
    write_log(dir.path());
    let output = run(&[
        dir.path().as_os_str(),
        OsStr::new("--from"),
        OsStr::new("1"),
        OsStr::new("--limit"),
        OsStr::new("1"),
    ]);
    assert_eq!(code(&output), 0, "{}", stderr(&output));
    let text = stdout(&output);
    assert!(text.contains("\n2\tTelemetry"), "{text}");
    assert!(!text.contains("\n1\tProjection"), "{text}");
    assert!(!text.contains("\n3\tProjection"), "{text}");
    assert!(text.contains("listed 1 record(s)"), "{text}");
}

#[test]
fn torn_tail_is_refused_and_left_in_place() {
    let dir = TempDir::new("events-torn");
    write_log(dir.path());
    let path = segment(dir.path());
    let mut bytes = std::fs::read(&path).expect("read");
    bytes.extend_from_slice(&[1, 2, 3]);
    std::fs::write(&path, &bytes).expect("append partial frame");
    let before = state(dir.path());

    let output = run(&[dir.path().as_os_str()]);
    assert_eq!(code(&output), 1);
    assert!(stderr(&output).contains("torn tail"), "{}", stderr(&output));
    assert!(stdout(&output).is_empty());
    assert_eq!(
        state(dir.path()),
        before,
        "the torn tail must not be truncated"
    );
}

#[test]
fn damage_before_the_tail_is_reported_as_corruption() {
    let dir = TempDir::new("events-corrupt");
    write_log(dir.path());
    let path = segment(dir.path());
    let mut bytes = std::fs::read(&path).expect("read");
    // Inside the first record's frame, with later records behind it.
    bytes[20] ^= 0xff;
    std::fs::write(&path, &bytes).expect("damage");
    let before = state(dir.path());

    let output = run(&[dir.path().as_os_str()]);
    assert_eq!(code(&output), 1);
    assert!(
        stderr(&output).contains("not readable"),
        "{}",
        stderr(&output)
    );
    assert_eq!(state(dir.path()), before);
}

#[test]
fn missing_empty_and_unknown_inputs_exit_one_and_create_nothing() {
    let dir = TempDir::new("events-missing");
    let absent = dir.path().join("no-such-log");
    let output = run(&[absent.as_os_str()]);
    assert_eq!(code(&output), 1);
    assert!(
        !absent.exists(),
        "a missing log directory must not be created"
    );

    let empty = dir.path().join("empty");
    std::fs::create_dir(&empty).expect("mkdir");
    assert_eq!(code(&run(&[empty.as_os_str()])), 1);
    assert_eq!(std::fs::read_dir(&empty).expect("list").count(), 0);

    let log = dir.path().join("log");
    std::fs::create_dir(&log).expect("mkdir");
    write_log(&log);
    let output = run(&[
        log.as_os_str(),
        OsStr::new("--consumer"),
        OsStr::new("ghost"),
    ]);
    assert_eq!(code(&output), 1);
    assert!(stderr(&output).contains("ghost"), "{}", stderr(&output));

    let past_end = run(&[log.as_os_str(), OsStr::new("--from"), OsStr::new("99")]);
    assert_eq!(code(&past_end), 1);
}

#[test]
fn usage_errors_exit_two_and_help_exits_zero() {
    assert_eq!(code(&run(&[])), 2);
    assert_eq!(code(&run(&[OsStr::new("a"), OsStr::new("b")])), 2);
    assert_eq!(code(&run(&[OsStr::new("a"), OsStr::new("--from")])), 2);
    assert_eq!(
        code(&run(&[
            OsStr::new("a"),
            OsStr::new("--limit"),
            OsStr::new("-1")
        ])),
        2
    );
    assert_eq!(
        code(&run(&[
            OsStr::new("a"),
            OsStr::new("--limit"),
            OsStr::new("10001")
        ])),
        2
    );
    assert_eq!(code(&run(&[OsStr::new("a"), OsStr::new("--bogus")])), 2);
    let help = run(&[OsStr::new("--help")]);
    assert_eq!(code(&help), 0);
    let text = stdout(&help);
    assert!(text.contains("usage: ptrctl events"), "{text}");
    assert!(text.contains("snapshot"), "{text}");
    assert!(text.contains("exit codes:"), "{text}");
}

#[test]
fn control_characters_in_consumer_names_never_reach_the_terminal() {
    // A producer with write access to the log persists a consumer name carrying an OSC title
    // sequence; `ptrctl events` must print it escaped, both as the slowest consumer and when the
    // operator names it with --consumer.
    let hostile = "evil\u{1b}]0;pwned\u{7}\u{1b}[2J";
    let dir = TempDir::new("events-escape");
    {
        let mut log = FileEventLog::open(dir.path(), FileLogConfig::new(16)).expect("open");
        log.publish(record(EventClass::Projection, "ledger.commit", b"a"))
            .expect("publish");
        log.register(hostile, Offset(0)).expect("register");
    }

    for args in [
        vec![dir.path().as_os_str()],
        vec![
            dir.path().as_os_str(),
            OsStr::new("--consumer"),
            OsStr::new(hostile),
        ],
    ] {
        let output = run(&args);
        assert_eq!(code(&output), 0, "{}", stderr(&output));
        let text = stdout(&output);
        assert!(
            !text.contains(['\u{1b}', '\u{7}']),
            "raw control characters were printed: {text:?}"
        );
        assert!(text.contains("evil\\u{1b}]0;pwned\\u{7}"), "{text}");
    }
}
