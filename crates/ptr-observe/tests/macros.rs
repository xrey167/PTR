use ptr_observe::{fields, trace_event, TraceLevel, TraceValue};
use ptr_types::{Generation, RequestId, Revision};

#[test]
fn exported_trace_event_macro_is_hygienic_and_builds_typed_fields() {
    let event = trace_event!(
        TraceLevel::Warn,
        "macro.generation_mismatch",
        fields::REQUEST_ID => RequestId::from("r-macro"),
        fields::REVISION => Revision(12),
        fields::EXPECTED => Generation(8),
        fields::ACTUAL => Generation(7),
        fields::ERROR_CODE => "stale_generation",
    );

    assert_eq!(event.name, "macro.generation_mismatch");
    assert_eq!(
        event.fields.get(fields::REQUEST_ID),
        Some(&TraceValue::RequestId(RequestId::from("r-macro")))
    );
    assert_eq!(
        event.fields.get(fields::EXPECTED),
        Some(&TraceValue::Generation(Generation(8)))
    );
    assert_eq!(
        event.fields.get(fields::ERROR_CODE),
        Some(&TraceValue::String("stale_generation".into()))
    );
}

#[test]
fn trace_value_conversion_macro_generated_impls_support_into() {
    assert_eq!(TraceValue::from(7_u64), TraceValue::U64(7));
    assert_eq!(TraceValue::from(true), TraceValue::Bool(true));
    assert_eq!(
        TraceValue::from(Generation(3)),
        TraceValue::Generation(Generation(3))
    );
}
