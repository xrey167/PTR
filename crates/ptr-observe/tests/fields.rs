mod common;

use ptr_observe::fields;

#[test]
fn tracing_fields_are_namespaced_and_present_on_fixture() {
    let event = common::generation_mismatch_event();

    assert!(fields::REQUEST_ID.starts_with("ptr."));
    assert!(fields::REVISION.starts_with("ptr."));
    assert!(event.fields.contains_key(fields::REQUEST_ID));
    assert!(event.fields.contains_key(fields::REVISION));
    assert!(event.fields.contains_key(fields::ERROR_CODE));
}
