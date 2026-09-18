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


#[test]
fn execute_all_accepts_into_iterator_and_collects_results() {
    let service = common::reference_service();
    let requests = ["a", "b", "c"]
        .into_iter()
        .map(common::valid_request);

    let results = service
        .execute_all(requests)
        .expect("all valid iterator requests must execute");

    assert_eq!(
        results.into_iter().map(|result| result.value).collect::<Vec<_>>(),
        vec!["a", "b", "c"]
    );
}

#[test]
fn backend_name_iterator_supports_closure_based_filtering() {
    let service = common::reference_service();

    let names = service
        .matching_backend_names(|name| name.starts_with("ref"))
        .collect::<Vec<_>>();

    assert_eq!(names, vec!["reference"]);
}


#[test]
fn closure_filter_supports_fn_mut_state_capture() {
    let service = common::reference_service();
    let mut visited = 0_usize;

    let names = service
        .matching_backend_names(|name| {
            visited += 1;
            name == "reference"
        })
        .collect::<Vec<_>>();

    assert_eq!(visited, 1);
    assert_eq!(names, vec!["reference"]);
}
