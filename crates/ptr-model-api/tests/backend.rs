use ptr_model_api::{InferenceBackend, ModelEvent, ModelRequest, ReferenceEchoBackend};
use ptr_types::{RequestId, Revision};

#[test]
fn reference_backend_obeys_request_event_contract() {
    let backend = ReferenceEchoBackend;
    let events = backend
        .infer(&ModelRequest {
            request_id: RequestId::from("r1"),
            revision: Revision(2),
            raw_text: "hello".into(),
        })
        .unwrap();
    assert_eq!(
        events,
        vec![ModelEvent::Token("hello".into()), ModelEvent::Finished]
    );
}
