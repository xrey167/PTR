use ptr_observe::{fields, NoopTraceSink, TraceEvent, TraceLevel, TraceSink, TraceValue};
use ptr_types::{Generation, RequestId, Revision};

#[test]
fn trace_event_carries_typed_expected_and_actual_values() {
    let event = TraceEvent::request(
        TraceLevel::Warn,
        "generation_mismatch",
        RequestId::from("r1"),
        Revision(9),
    )
    .with_field(fields::ERROR_CODE, TraceValue::String("stale_generation".into()))
    .with_field(fields::EXPECTED, TraceValue::Generation(Generation(8)))
    .with_field(fields::ACTUAL, TraceValue::Generation(Generation(7)));

    assert_eq!(
        event.fields.get(fields::EXPECTED),
        Some(&TraceValue::Generation(Generation(8)))
    );
    assert_eq!(
        event.fields.get(fields::ACTUAL),
        Some(&TraceValue::Generation(Generation(7)))
    );
    NoopTraceSink.emit(&event).expect("noop sink is infallible");
}
