mod common;

#[test]
fn reference_implementor_executes_through_public_trait_contract() {
    let service = common::reference_service();
    let result = service
        .execute(common::valid_request("hello"))
        .expect("valid reference request must execute successfully");

    assert_eq!(result.value, "hello");
    assert_eq!(result.backend, "reference");
}
