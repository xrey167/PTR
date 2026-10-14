use std::sync::Arc;

use ptr_observe::{
    fields, FanoutSink, MemorySink, RedactingSink, RedactionPolicy, Redactor, TraceEvent,
    TraceLevel, TraceSink, TraceValue, REDACTED,
};
use ptr_types::{RequestId, Revision};

const PLAINTEXTS: [&str; 7] = [
    "hunter2-plaintext",
    "s3cr3t-cookie-value",
    "Bearer abcdefgh12345",
    "pa55word-in-url",
    "q8Zr3xK9LmP2vB7nT5yW1cD6hJ4sA0eFgUiOpXzRtYu",
    "0123456789abcdef0123456789abcdef",
    "sk-liveABCDEFGH12345",
];

fn secret_event() -> TraceEvent {
    TraceEvent::request(
        TraceLevel::Info,
        "login",
        RequestId::from("r1"),
        Revision(3),
    )
    .with_field("password", "hunter2-plaintext")
    .with_field("X-Api-Key", "s3cr3t-cookie-value")
    .with_field("note", "Bearer abcdefgh12345")
    .with_field("target", "postgres://app:pa55word-in-url@db:5432/ptr")
    .with_field("blob", "q8Zr3xK9LmP2vB7nT5yW1cD6hJ4sA0eFgUiOpXzRtYu")
    .with_field("digest", "0123456789abcdef0123456789abcdef")
    .with_field("opaque", "sk-liveABCDEFGH12345")
    .with_field(fields::OUTCOME, "ok")
}

#[test]
fn plaintext_never_reaches_inner_sink_even_through_debug() {
    let sink = RedactingSink::new(MemorySink::new(8));
    sink.emit(&secret_event()).expect("emit");

    let seen = sink.inner().events();
    assert_eq!(seen.len(), 1);
    let debug = format!("{seen:?} {sink:?}");
    for plaintext in PLAINTEXTS {
        assert!(!debug.contains(plaintext), "leaked {plaintext}");
    }
    let event = &seen[0];
    assert_eq!(
        event.fields["password"],
        TraceValue::String(REDACTED.into())
    );
    assert_eq!(
        event.fields["X-Api-Key"],
        TraceValue::String(REDACTED.into())
    );
    assert_eq!(
        event.fields[fields::OUTCOME],
        TraceValue::String("ok".into())
    );
    assert_eq!(
        event.fields[fields::REQUEST_ID],
        TraceValue::RequestId(RequestId::from("r1"))
    );
    assert_eq!(sink.redacted_values(), 7);
}

#[test]
fn redaction_applies_before_every_fanout_branch() {
    let first = Arc::new(MemorySink::new(4));
    let second = Arc::new(MemorySink::new(4));
    let fanout = FanoutSink::new()
        .with_sink(first.clone())
        .with_sink(second.clone());
    let sink = RedactingSink::new(fanout);
    sink.emit(&secret_event()).expect("emit");
    for memory in [first, second] {
        let debug = format!("{:?}", memory.events());
        for plaintext in PLAINTEXTS {
            assert!(!debug.contains(plaintext), "leaked {plaintext}");
        }
    }
}

#[test]
fn secret_in_event_name_and_non_string_value_is_redacted() {
    let sink = RedactingSink::new(MemorySink::new(4));
    let event = TraceEvent::new(TraceLevel::Warn, "failed Bearer abcdefghijkl")
        .with_field("token_count", 99u64)
        .with_field("cookie", true);
    sink.emit(&event).expect("emit");
    let seen = &sink.inner().events()[0];
    assert_eq!(seen.name, REDACTED);
    assert_eq!(
        seen.fields["token_count"],
        TraceValue::String(REDACTED.into())
    );
    assert_eq!(seen.fields["cookie"], TraceValue::String(REDACTED.into()));
}

#[test]
fn custom_policy_allow_list_keeps_value_and_original_event_is_untouched() {
    let policy = RedactionPolicy::default().with_allow_key("token_count");
    let sink = RedactingSink::with_redactor(MemorySink::new(4), Redactor::new(policy));
    let event = TraceEvent::new(TraceLevel::Info, "usage").with_field("token_count", 99u64);
    sink.emit(&event).expect("emit");
    assert_eq!(
        sink.inner().events()[0].fields["token_count"],
        TraceValue::U64(99)
    );
    assert_eq!(sink.redacted_values(), 0);
    assert_eq!(event.fields["token_count"], TraceValue::U64(99));
}

#[test]
fn inner_sink_errors_propagate() {
    struct Down;
    impl TraceSink for Down {
        fn emit(&self, _event: &TraceEvent) -> Result<(), ptr_observe::TraceError> {
            Err(ptr_observe::TraceError::SinkUnavailable {
                sink: "down".into(),
            })
        }
    }
    let sink = RedactingSink::new(Down);
    assert_eq!(
        sink.emit(&secret_event()).expect_err("down").code(),
        "sink_unavailable"
    );
}

const TOKEN_REQUEST_IDS: [&str; 2] = [
    "Bearer abcdef0123456789",
    "q8Zr3xK9LmP2vB7nT5yW1cD6hJ4sA0eFgUiOpXzRtYu",
];

#[test]
fn secret_looking_request_id_never_reaches_inner_sink() {
    for token in TOKEN_REQUEST_IDS {
        let sink = RedactingSink::new(MemorySink::new(4));
        let event = TraceEvent::request(
            TraceLevel::Info,
            "login",
            RequestId::from(token),
            Revision(1),
        );
        sink.emit(&event).expect("emit");

        let seen = sink.inner().events();
        assert!(!format!("{seen:?}").contains(token), "leaked {token}");
        assert_eq!(
            seen[0].fields[fields::REQUEST_ID],
            TraceValue::String(REDACTED.into())
        );
        assert_eq!(sink.redacted_values(), 1);
    }
}

#[test]
fn ordinary_request_id_is_preserved_as_typed_value() {
    let sink = RedactingSink::new(MemorySink::new(4));
    let event = TraceEvent::request(
        TraceLevel::Info,
        "ok",
        RequestId::from("req-42"),
        Revision(1),
    );
    sink.emit(&event).expect("emit");
    assert_eq!(
        sink.inner().events()[0].fields[fields::REQUEST_ID],
        TraceValue::RequestId(RequestId::from("req-42"))
    );
    assert_eq!(sink.redacted_values(), 0);
}

#[test]
fn tracer_truncation_cannot_hide_secret_keys_or_values_from_export() {
    use ptr_observe::{
        ManualClock, OtlpFileConfig, OtlpJsonFileExporter, Tracer, TracerConfig, TracerLimits,
    };

    let tracer = Tracer::new(
        TracerConfig {
            id_seed: 1,
            limits: TracerLimits {
                max_value_len: 4,
                ..TracerLimits::default()
            },
        },
        Arc::new(ManualClock::new(0)),
    );
    let long_secret = "q8Zr3xK9LmP2vB7nT5yW1cD6hJ4sA0eFgUiOpXzRtYu";
    let guard = tracer.root_span("root").expect("open");
    guard.set_field("password", "1234");
    guard.set_field("note", long_secret);
    guard.event(
        TraceEvent::new(TraceLevel::Info, "evt")
            .with_field("api_key", "9876")
            .with_field("token", "5555"),
    );
    guard.event(TraceEvent::new(TraceLevel::Info, long_secret));
    guard.finish();
    let spans = tracer.drain_finished();

    let exporter = OtlpJsonFileExporter::new("unused.jsonl", OtlpFileConfig::default());
    let rendered = exporter.render(&spans);
    for plaintext in ["1234", "9876", "5555", "q8Zr", long_secret] {
        assert!(
            !rendered.contains(plaintext),
            "leaked {plaintext}: {rendered}"
        );
    }
    let debug = format!("{spans:?}");
    for plaintext in ["1234", "9876", "5555", "q8Zr"] {
        assert!(!debug.contains(plaintext), "tracer retained {plaintext}");
    }
}
