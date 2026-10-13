mod common;

use common::{candidate, key_bytes};
use ptr_router::{
    ConstraintsOutcome, JsonLinesSink, MemorySink, RoutingRecord, RoutingTelemetry, TelemetryError,
    MAX_FIELD_BYTES,
};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU32, Ordering};

fn temp_file(tag: &str) -> PathBuf {
    static COUNTER: AtomicU32 = AtomicU32::new(0);
    std::env::temp_dir().join(format!(
        "ptr-router-telemetry-{}-{}-{}.jsonl",
        std::process::id(),
        tag,
        COUNTER.fetch_add(1, Ordering::SeqCst)
    ))
}

fn record(id: &str) -> RoutingRecord {
    RoutingRecord::from_candidate(
        &candidate(id, "zeta"),
        "zeta",
        ConstraintsOutcome::Satisfied {
            eligible: 3,
            feasible: 2,
        },
    )
    .unwrap()
}

fn contains_subslice(haystack: &[u8], needle: &[u8]) -> bool {
    haystack.windows(needle.len()).any(|w| w == needle)
}

#[test]
fn record_fields_come_from_the_checked_candidate() {
    let candidate = candidate("req-9", "zeta");
    let record =
        RoutingRecord::from_candidate(&candidate, "zeta", ConstraintsOutcome::NotApplied).unwrap();
    assert_eq!(record.decision_id(), "req-9");
    assert_eq!(&record.nonce(), candidate.nonce().as_bytes());
    assert_eq!(record.key_id(), candidate.key_id());
    assert_eq!(record.propensity(), 1.0);
    assert_eq!(record.chosen(), "zeta");
    assert_eq!(record.constraints(), ConstraintsOutcome::NotApplied);
}

#[test]
fn json_has_the_documented_shape() {
    let c = candidate("req-1", "zeta");
    let json = RoutingRecord::from_candidate(
        &c,
        "zeta",
        ConstraintsOutcome::Satisfied {
            eligible: 3,
            feasible: 2,
        },
    )
    .unwrap()
    .to_json();
    let nonce: String = c
        .nonce()
        .as_bytes()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    let key_id: String = c.key_id().iter().map(|b| format!("{b:02x}")).collect();
    assert_eq!(
        json,
        format!(
            "{{\"v\":1,\"decision_id\":\"req-1\",\"nonce\":\"{nonce}\",\"key_id\":\"{key_id}\",\
\"propensity\":1.0,\"chosen\":\"zeta\",\"constraints\":{{\"kind\":\"satisfied\",\
\"eligible\":3,\"feasible\":2}}}}"
        )
    );
    for (outcome, expected) in [
        (ConstraintsOutcome::NotApplied, "{\"kind\":\"not_applied\"}"),
        (
            ConstraintsOutcome::Infeasible { options: 4 },
            "{\"kind\":\"infeasible\",\"options\":4}",
        ),
    ] {
        let json = RoutingRecord::from_candidate(&c, "zeta", outcome)
            .unwrap()
            .to_json();
        assert!(
            json.ends_with(&format!(",\"constraints\":{expected}}}")),
            "{json}"
        );
    }
}

#[test]
fn strings_are_escaped_and_long_fields_refused() {
    let c = candidate("a\"b\\c\nd", "zeta");
    let json = RoutingRecord::from_candidate(&c, "x\ty", ConstraintsOutcome::NotApplied)
        .unwrap()
        .to_json();
    assert!(json.contains("\"decision_id\":\"a\\\"b\\\\c\\nd\""));
    assert!(json.contains("\"chosen\":\"x\\ty\""));
    assert!(!json.contains('\n'));

    let long = "x".repeat(MAX_FIELD_BYTES + 1);
    assert_eq!(
        RoutingRecord::from_candidate(&c, &long, ConstraintsOutcome::NotApplied),
        Err(TelemetryError::FieldTooLong { field: "chosen" })
    );
    let c = candidate(&long, "zeta");
    assert_eq!(
        RoutingRecord::from_candidate(&c, "z", ConstraintsOutcome::NotApplied),
        Err(TelemetryError::FieldTooLong {
            field: "decision_id"
        })
    );
    assert!(RoutingRecord::from_candidate(
        &candidate(&"y".repeat(MAX_FIELD_BYTES), "zeta"),
        &"x".repeat(MAX_FIELD_BYTES),
        ConstraintsOutcome::NotApplied
    )
    .is_ok());
}

#[test]
fn memory_sink_is_bounded_and_counts_rejections() {
    let sink = MemorySink::new(2);
    sink.record(&record("a")).unwrap();
    sink.record(&record("b")).unwrap();
    assert_eq!(sink.record(&record("c")), Err(TelemetryError::Full));
    assert_eq!(sink.record(&record("d")), Err(TelemetryError::Full));
    let kept = sink.records();
    assert_eq!(kept.len(), 2);
    assert_eq!(kept[0].decision_id(), "a");
    assert_eq!(kept[1].decision_id(), "b");
    assert_eq!(sink.rejected(), 2);
    assert_eq!(
        MemorySink::new(0).record(&record("a")),
        Err(TelemetryError::Full)
    );
}

#[test]
fn json_lines_sink_appends_one_line_per_record_and_never_the_key() {
    let path = temp_file("lines");
    let sink = JsonLinesSink::open(&path, 1 << 20).unwrap();
    for id in ["r1", "r2", "r3"] {
        sink.record(&record(id)).unwrap();
    }
    let bytes = std::fs::read(&path).unwrap();
    let text = String::from_utf8(bytes.clone()).unwrap();
    let lines: Vec<&str> = text.lines().collect();
    assert_eq!(lines.len(), 3);
    assert!(text.ends_with('\n'));
    for (line, id) in lines.iter().zip(["r1", "r2", "r3"]) {
        assert!(line.starts_with("{\"v\":1,"));
        assert!(line.contains(&format!("\"decision_id\":\"{id}\"")));
        assert!(line.ends_with('}'));
    }
    // Invariant 13: neither the raw key bytes nor their hex or decimal rendering appear.
    let key = key_bytes();
    assert!(!contains_subslice(&bytes, &key));
    assert!(!contains_subslice(&bytes, &key[..8]));
    let hex: String = key.iter().map(|b| format!("{b:02x}")).collect();
    assert!(!text.contains(&hex));
    assert!(!text.contains(&hex[..16]));
    // The non-secret key id is recorded.
    let key_id: String = record("r1")
        .key_id()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    assert!(text.contains(&key_id));
    std::fs::remove_file(&path).unwrap();
}

#[test]
fn json_lines_sink_is_append_only_across_opens_and_bounded_by_size() {
    let path = temp_file("bound");
    let one = record("r1").to_json().len() as u64 + 1;
    {
        let sink = JsonLinesSink::open(&path, one * 2).unwrap();
        sink.record(&record("r1")).unwrap();
    }
    let before = std::fs::read(&path).unwrap();
    {
        // The existing line counts against the bound; there is room for exactly one more.
        let sink = JsonLinesSink::open(&path, one * 2).unwrap();
        sink.record(&record("r2")).unwrap();
        assert_eq!(sink.record(&record("r3")), Err(TelemetryError::Full));
    }
    let after = std::fs::read(&path).unwrap();
    assert!(after.starts_with(&before));
    assert_eq!(after.iter().filter(|b| **b == b'\n').count(), 2);
    assert_eq!(after.len() as u64, one * 2);
    std::fs::remove_file(&path).unwrap();
}

#[test]
fn opening_a_path_that_cannot_be_created_is_a_typed_io_error() {
    let path = std::env::temp_dir()
        .join(format!("ptr-router-missing-dir-{}", std::process::id()))
        .join("x.jsonl");
    assert!(matches!(
        JsonLinesSink::open(&path, 10),
        Err(TelemetryError::Io { .. })
    ));
}

#[test]
fn sinks_are_usable_as_trait_objects() {
    let sinks: Vec<Box<dyn RoutingTelemetry>> = vec![Box::new(MemorySink::new(1))];
    for sink in &sinks {
        sink.record(&record("r")).unwrap();
    }
}
