use prost::Message;
use ptr_protocol::generated::podwire::PodCall;

#[test]
fn generated_podwire_roundtrips() {
    let call = PodCall {
        call_id: "c1".into(),
        capability: "predict".into(),
        generation: Some(7),
        revision: 11,
        payload: None,
    };
    let bytes = call.encode_to_vec();
    let decoded = PodCall::decode(bytes.as_slice()).unwrap();
    assert_eq!(decoded.call_id, "c1");
    assert_eq!(decoded.generation, Some(7));
}
