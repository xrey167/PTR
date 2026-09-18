use ptr_protocol::generated::podwire::{PodCall, TypedPayload};
use ptr_protocol::{CallFrame, ProtocolError};

fn valid() -> PodCall {
    PodCall {
        session_id: "s".into(),
        call_id: "c".into(),
        capability: "predict".into(),
        generation: Some(1),
        revision: 2,
        payload: Some(TypedPayload {
            type_id: "Input".into(),
            payload: vec![1],
        }),
    }
}

#[test]
fn malformed_podwire_call_is_rejected() {
    let mut call = valid();
    call.capability.clear();
    assert_eq!(
        CallFrame::try_from(call),
        Err(ProtocolError::MissingCapability)
    );

    let mut call = valid();
    call.payload = None;
    assert_eq!(
        CallFrame::try_from(call),
        Err(ProtocolError::MissingPayload)
    );

    let parsed = CallFrame::try_from(valid()).unwrap();
    assert_eq!(parsed.call_id, "c");
}
