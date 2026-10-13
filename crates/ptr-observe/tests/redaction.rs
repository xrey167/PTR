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
