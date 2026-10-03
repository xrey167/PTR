use std::fmt::Debug;

use prost::Message;
use ptr_protocol::generated::events::RuntimeEvent;
use ptr_protocol::generated::model::{ModelEvent, ModelRequest};
use ptr_protocol::generated::podwire::{PodCall, PodError, PodReturn, Revoke, TypedPayload};
use ptr_protocol::generated::raft::RaftEnvelope;

// These bytes are written from the .proto field numbers and wire types, rather
// than produced by the encoder under test, so paired codec regressions fail too.
const CALL_WIRE: &[u8] = &[
    0x12, 2, b'c', b'1', // call_id, field 2
    0x1a, 7, b'p', b'r', b'e', b'd', b'i', b'c', b't', // capability, field 3
    0x20, 7, // generation, field 4
    0x28, 11, // revision, field 5
    0x32, 8, 0x0a, 1, b'T', 0x12, 3, 0, 0x80, 0xff, // payload, field 6
];

fn call_fixture() -> PodCall {
    PodCall {
        call_id: "c1".into(),
        capability: "predict".into(),
        generation: Some(7),
        revision: 11,
        payload: Some(TypedPayload {
            type_id: "T".into(),
            payload: vec![0, 0x80, 0xff],
        }),
    }
}

#[track_caller]
fn assert_roundtrip<M: Message + Default + PartialEq + Debug>(message: M) -> Vec<u8> {
    let wire = message.encode_to_vec();
    assert_eq!(message.encoded_len(), wire.len());
    assert_eq!(M::decode(wire.as_slice()).expect("decode message"), message);
    wire
}

#[track_caller]
fn assert_wire<M: Message + Default + PartialEq + Debug>(message: M, wire: &[u8]) {
    assert_eq!(assert_roundtrip(message), wire);
}

#[test]
fn generated_podwire_roundtrips() {
    assert_roundtrip(call_fixture());
}

#[test]
fn podcall_preserves_schema_wire_bytes() {
    assert_wire(call_fixture(), CALL_WIRE);
}

#[test]
fn optional_generation_distinguishes_absent_from_explicit_zero() {
    for (generation, wire) in [(None, &[][..]), (Some(0), &[0x20, 0][..])] {
        assert_wire(
            PodCall {
                generation,
                ..Default::default()
            },
            wire,
        );
    }
}

#[test]
fn unsigned_fields_preserve_varint_boundaries() {
    for (value, varint) in [
        (0, &b"\x00"[..]),
        (127, &b"\x7f"[..]),
        (128, &b"\x80\x01"[..]),
        (16_383, &b"\xff\x7f"[..]),
        (16_384, &b"\x80\x80\x01"[..]),
        (u64::MAX, &b"\xff\xff\xff\xff\xff\xff\xff\xff\xff\x01"[..]),
    ] {
        let mut wire = vec![0x20];
        wire.extend_from_slice(varint);
        // Non-optional proto3 scalars omit their zero default.
        if value != 0 {
            wire.push(0x28);
            wire.extend_from_slice(varint);
        }
        assert_wire(
            PodCall {
                generation: Some(value),
                revision: value,
                ..Default::default()
            },
            &wire,
        );
    }
}

#[test]
fn payload_distinguishes_absent_from_present_empty_message() {
    for (payload, wire) in [
        (None, &[][..]),
        (Some(TypedPayload::default()), &[0x32, 0][..]),
    ] {
        assert_wire(
            PodCall {
                payload,
                ..Default::default()
            },
            wire,
        );
    }
}

#[test]
fn unicode_and_binary_payloads_survive_length_boundaries() {
    for length in [0, 1, 127, 128, 255, 256] {
        let call = PodCall {
            call_id: "呼び出し\0".into(),
            capability: "prédire".into(),
            generation: Some(7),
            revision: 11,
            payload: Some(TypedPayload {
                type_id: "入力".into(),
                payload: (0..=255).cycle().take(length).collect(),
            }),
        };
        assert_roundtrip(call);
    }
}

#[test]
fn unknown_and_reserved_fields_do_not_change_call_contents() {
    // Reserved session_id plus unknown varint, fixed64, bytes and fixed32 fields.
    let unknown = [
        0x0a, 3, b'o', b'l', b'd', 0x50, 1, 0x59, 1, 2, 3, 4, 5, 6, 7, 8, 0x62, 2, 0xff, 0, 0x6d,
        1, 2, 3, 4,
    ];
    for wire in [
        [unknown.as_slice(), CALL_WIRE].concat(),
        [CALL_WIRE, unknown.as_slice()].concat(),
    ] {
        let decoded = PodCall::decode(wire.as_slice()).expect("skip unknown fields");
        assert_eq!(decoded, call_fixture());
        assert_eq!(decoded.encode_to_vec(), CALL_WIRE);
    }
}

#[test]
fn repeated_scalar_fields_use_the_last_value_including_zero() {
    let mut wire = CALL_WIRE.to_vec();
    wire.extend_from_slice(&[0x12, 1, b'd', 0x20, 0, 0x28, 0]);
    assert_eq!(
        PodCall::decode(wire.as_slice()).expect("decode repeated scalar fields"),
        PodCall {
            call_id: "d".into(),
            generation: Some(0),
            revision: 0,
            ..call_fixture()
        }
    );
}

#[test]
fn repeated_payload_messages_merge_their_fields() {
    // Each occurrence supplies one field of the same singular nested message.
    let wire = b"\x32\x03\x0a\x01T\x32\x05\x12\x03\x00\x80\xff";
    assert_eq!(
        PodCall::decode(wire.as_slice())
            .expect("merge payload occurrences")
            .payload,
        call_fixture().payload
    );
}

#[test]
fn fields_decode_independently_of_wire_order() {
    // Reverse both the outer field order and the nested payload field order.
    let wire = b"\x32\x08\x12\x03\x00\x80\xff\x0a\x01T\x28\x0b\x20\x07\x1a\x07predict\x12\x02c1";
    let decoded = PodCall::decode(wire.as_slice()).expect("decode reordered fields");
    assert_eq!(decoded, call_fixture());
    assert_eq!(decoded.encode_to_vec(), CALL_WIRE);
}

#[test]
fn repeated_payload_bytes_replace_instead_of_append() {
    for bytes in [&b"\x01\x02"[..], &b""[..]] {
        // A second payload message supplies only its bytes field, including an
        // explicit empty value. The omitted type_id must survive the merge.
        let mut wire = CALL_WIRE.to_vec();
        wire.extend_from_slice(&[0x32, (bytes.len() + 2) as u8, 0x12, bytes.len() as u8]);
        wire.extend_from_slice(bytes);
        assert_eq!(
            PodCall::decode(wire.as_slice()).expect("replace singular payload bytes"),
            PodCall {
                payload: Some(TypedPayload {
                    type_id: "T".into(),
                    payload: bytes.to_vec(),
                }),
                ..call_fixture()
            }
        );
    }
}

#[test]
fn empty_repeated_payload_does_not_erase_existing_fields() {
    let wire = [CALL_WIRE, b"\x32\x00"].concat();
    assert_eq!(
        PodCall::decode(wire.as_slice()).expect("merge empty payload message"),
        call_fixture()
    );
}

#[test]
fn unknown_nested_fields_cannot_modify_outer_call_fields() {
    // Fields 4 and 6 belong to PodCall, but are unknown inside TypedPayload.
    // Their values must be skipped within the nested message's boundary.
    let wire = [CALL_WIRE, b"\x32\x07\x20\x00\x32\x03\x00\x80\xff"].concat();
    let decoded = PodCall::decode(wire.as_slice()).expect("skip nested unknown fields");
    assert_eq!(decoded, call_fixture());
    assert_eq!(decoded.encode_to_vec(), CALL_WIRE);
}

#[test]
fn generated_messages_accept_empty_input_and_omit_default_fields() {
    assert_wire(PodCall::default(), b"");
    assert_wire(TypedPayload::default(), b"");
    assert_wire(PodReturn::default(), b"");
    assert_wire(PodError::default(), b"");
    assert_wire(Revoke::default(), b"");
    assert_wire(ModelRequest::default(), b"");
    assert_wire(ModelEvent::default(), b"");
    assert_wire(RuntimeEvent::default(), b"");
    assert_wire(RaftEnvelope::default(), b"");
}

#[test]
fn malformed_calls_are_rejected_without_panicking() {
    let cases: &[(&str, &[u8])] = &[
        ("zero field number", b"\x00"),
        ("invalid wire type", b"\x16"),
        ("string with varint wire type", b"\x10\x01"),
        ("generation with bytes wire type", b"\x22\x00"),
        ("revision with fixed32 wire type", b"\x2d\x00\x00\x00\x00"),
        ("payload with varint wire type", b"\x30\x00"),
        ("truncated field key", b"\x80"),
        ("truncated string length", b"\x12\x80"),
        ("truncated string", b"\x12\x02c"),
        ("invalid UTF-8", b"\x12\x01\xff"),
        ("truncated generation", b"\x20\x80"),
        (
            "overflowing generation",
            b"\x20\xff\xff\xff\xff\xff\xff\xff\xff\xff\x02",
        ),
        ("truncated nested payload", b"\x32\x03\x0a\x02T"),
        ("invalid nested UTF-8", b"\x32\x03\x0a\x01\xff"),
        ("truncated nested bytes", b"\x32\x03\x12\x02\xff"),
        ("nested bytes with varint wire type", b"\x32\x02\x10\x00"),
        // The byte following the nested message cannot satisfy its inner length.
        (
            "nested bytes exceeding message boundary",
            b"\x32\x02\x12\x01\x00",
        ),
        ("truncated unknown field", b"\x62\x02x"),
        (
            "truncated unknown fixed64",
            b"\x59\x00\x00\x00\x00\x00\x00\x00",
        ),
        ("truncated unknown fixed32", b"\x6d\x00\x00\x00"),
        ("truncated unknown varint", b"\x50\x80"),
    ];
    for (name, wire) in cases {
        assert!(PodCall::decode(*wire).is_err(), "accepted {name}");
    }
}

#[test]
fn podwire_responses_and_revocation_preserve_schema_wire_bytes() {
    assert_wire(
        PodReturn {
            call_id: "c".into(),
            payload: Some(TypedPayload {
                type_id: "T".into(),
                payload: vec![0xff],
            }),
        },
        b"\x0a\x01c\x12\x06\x0a\x01T\x12\x01\xff",
    );
    assert_wire(
        PodError {
            call_id: "c".into(),
            code: "E".into(),
            message: "bad".into(),
        },
        b"\x0a\x01c\x12\x01E\x1a\x03bad",
    );
    assert_wire(
        Revoke {
            subject: "s".into(),
            generation: 128,
        },
        b"\x0a\x01s\x10\x80\x01",
    );
}

#[test]
fn model_messages_preserve_schema_wire_bytes() {
    assert_wire(
        ModelRequest {
            request_id: "r".into(),
            revision: 128,
            raw_text: "hi".into(),
        },
        b"\x0a\x01r\x10\x80\x01\x1a\x02hi",
    );
    assert_wire(
        ModelEvent {
            kind: "k".into(),
            payload: vec![0, 0xff],
        },
        b"\x0a\x01k\x12\x02\x00\xff",
    );
}

#[test]
fn runtime_events_and_raft_envelopes_preserve_schema_wire_bytes() {
    assert_wire(
        RuntimeEvent {
            sequence: 128,
            kind: "k".into(),
            payload: vec![0, 0xff],
        },
        b"\x08\x80\x01\x12\x01k\x1a\x02\x00\xff",
    );
    assert_wire(
        RaftEnvelope {
            from: 1,
            to: 128,
            raft_message: vec![0, 0xff],
        },
        b"\x08\x01\x10\x80\x01\x1a\x02\x00\xff",
    );
}
