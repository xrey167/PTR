use ptr_observe::fields;
#[test]
fn tracing_fields_are_namespaced() {
    assert!(fields::REQUEST_ID.starts_with("ptr."));
}
