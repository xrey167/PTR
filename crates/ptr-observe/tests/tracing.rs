mod common;

use ptr_observe::{fields, NoopTraceSink, TraceSink, TraceValue};
use ptr_types::Generation;

#[test]
fn trace_event_carries_typed_expected_and_actual_values() {
    let event = common::generation_mismatch_event();

    assert_eq!(
        event.fields.get(fields::EXPECTED),
        Some(&TraceValue::Generation(Generation(8)))
    );
    assert_eq!(
        event.fields.get(fields::ACTUAL),
        Some(&TraceValue::Generation(Generation(7)))
    );
    NoopTraceSink
        .emit(&event)
        .expect("noop trace sink must accept a valid PTR trace event");
}
