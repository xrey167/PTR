use ptr_model_api::{ModelEvent, ModelRequest};
use ptr_types::{RequestId, Revision};
#[test]
fn request_and_event_contracts_construct() {
    let r = ModelRequest {
        request_id: RequestId::from("r"),
        revision: Revision(1),
        raw_text: "x".into(),
    };
    assert_eq!(r.revision, Revision(1));
    assert!(matches!(ModelEvent::Finished, ModelEvent::Finished));
}
