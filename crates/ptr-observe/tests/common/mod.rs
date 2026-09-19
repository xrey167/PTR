use ptr_observe::{fields, TraceEvent, TraceLevel, TraceValue};
use ptr_types::{Generation, RequestId, Revision};

pub fn generation_mismatch_event() -> TraceEvent {
    TraceEvent::request(
        TraceLevel::Warn,
        "generation_mismatch",
        RequestId::from("r1"),
        Revision(9),
    )
    .with_field(
        fields::ERROR_CODE,
        TraceValue::String("stale_generation".into()),
    )
    .with_field(fields::EXPECTED, TraceValue::Generation(Generation(8)))
    .with_field(fields::ACTUAL, TraceValue::Generation(Generation(7)))
}
