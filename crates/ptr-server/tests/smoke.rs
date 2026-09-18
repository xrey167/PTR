use ptr_server::ApiRequest;
use ptr_types::RequestId;
#[test]
fn api_request_keeps_id() {
    let r = ApiRequest {
        id: RequestId::from("r"),
        text: "hello".into(),
    };
    assert_eq!(r.id.to_string(), "r");
}
