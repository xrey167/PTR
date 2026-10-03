use ptr_model_api::{ModelEvent, ModelRequest};
use ptr_types::{ReasoningOperator, RequestId, Revision, SemanticContext};

#[test]
fn request_and_event_contracts_construct() {
    let request = ModelRequest {
        request_id: RequestId::from("r"),
        semantic_context: SemanticContext {
            revision: Revision(1),
            entries: vec![],
        },
        raw_text: "x".into(),
    };
    assert_eq!(request.revision(), Revision(1));
    assert!(matches!(ModelEvent::Finished, ModelEvent::Finished));
}

#[test]
fn operator_requested_uses_shared_typed_reasoning_operator() {
    let event = ModelEvent::OperatorRequested {
        operator: ReasoningOperator::Probabilistic,
    };

    assert_eq!(
        event,
        ModelEvent::OperatorRequested {
            operator: ReasoningOperator::Probabilistic,
        }
    );
}
