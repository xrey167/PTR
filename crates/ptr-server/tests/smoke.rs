use ptr_server::ApiRequest;

#[test]
fn api_request_keeps_id() {
    let request = ApiRequest {
        id: "r".into(),
        text: "hello".into(),
    };
    assert_eq!(request.id, "r");
}
