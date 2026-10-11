//! The chunk protocol through the public API and the real request frame codec: a frame larger
//! than one PodWire body travels as signed chunks and arrives bit for bit, and the receiver
//! answers an attack with a typed refusal.

use ed25519_dalek::SigningKey;
use ptr_podwire::{
    chunk_frame, decode_request, encode_request, AssemblyPolicy, FrameAssembler, FrameError,
    FrameMeta, Offered, PodRequest, Rejection, MAX_BODY_BYTES,
};
use ptr_protocol::activation::{
    chunk_payload, frame_digest, ActivationFrame, BoundaryDtype, ChainElement, UpstreamChain,
    ACTIVATION_CHUNK_TYPE,
};
use ptr_protocol::TypedPayload;
use ptr_types::{CapabilityId, TypeId};

const PRINCIPAL: &str = "upstream";

fn policy() -> AssemblyPolicy {
    AssemblyPolicy {
        reorder_window: 4,
        max_incomplete_frames_per_peer: 8,
        max_chunks_in_flight: 32,
        max_frame_tombstones_per_principal: 64,
        max_frame_deadline_lifetime_ms: 5_000,
        max_clock_skew_ms: 1_000,
        replay_window_ms: 10_000,
        max_frame_bytes: 32 * 1024 * 1024,
        max_total_buffered_bytes: 64 * 1024 * 1024,
    }
}

fn element(name: &str) -> ChainElement {
    ChainElement {
        identity: name.to_owned(),
        generation: 1,
        digest: [1; 32],
        fence_epoch: 0,
    }
}

fn big_frame() -> ActivationFrame {
    // 1024 x 1024 F32 = 4 MiB: one body is not enough even before the header.
    let (tokens, width) = (1024u32, 1024u32);
    ActivationFrame {
        tokens,
        width,
        dtype: BoundaryDtype::F32,
        positions: (0..tokens).collect(),
        upstream_chain: UpstreamChain {
            stage: element("stage"),
            base_model: element("model"),
            adapters: vec![],
            experts: vec![],
        },
        placement_epoch: 3,
        tensor: (0..tokens as usize * width as usize * 4)
            .map(|index| (index.wrapping_mul(2654435761) >> 7) as u8)
            .collect(),
    }
}

fn over_the_wire(payload: &TypedPayload) -> Result<TypedPayload, FrameError> {
    let request = PodRequest {
        addressed_to: "downstream".to_owned(),
        request_id: 1,
        capability: CapabilityId::from("split-activation"),
        expect_protocol: 1,
        payload: payload.clone(),
    };
    decode_request(&encode_request(&request)?).map(|decoded| decoded.payload)
}

#[test]
fn a_frame_over_one_body_is_refused_whole_and_arrives_bit_for_bit_in_chunks() {
    let key = SigningKey::from_bytes(&[5; 32]);
    let mut assembler = FrameAssembler::new(policy()).unwrap();
    assembler.register_principal(PRINCIPAL, key.verifying_key(), "stage");
    let frame = big_frame();
    let sent = frame.to_bytes();
    assert!(sent.len() > MAX_BODY_BYTES);

    let whole = TypedPayload {
        type_id: TypeId::from(ACTIVATION_CHUNK_TYPE),
        bytes: sent.clone(),
    };
    assert!(
        over_the_wire(&whole).is_err(),
        "PodWire refuses a body over its bound"
    );

    let chunks = chunk_frame(
        &key,
        &frame,
        FrameMeta {
            frame_id: 7,
            parent_operation_id: 1,
            signed_absolute_deadline_ms: 3_000,
        },
        1_500_000,
    )
    .unwrap();
    assert_eq!(chunks.len(), 3);
    let mut delivered = None;
    for index in [1usize, 0, 2] {
        let arrived = over_the_wire(&chunk_payload(&chunks[index])).unwrap();
        if let Offered::Delivered(done) = assembler.offer(0, PRINCIPAL, 1, &arrived).unwrap() {
            delivered = Some(done);
        }
    }
    let delivered = delivered.expect("the last chunk delivers the frame");
    assert_eq!(delivered.bytes, sent);
    assert_eq!(delivered.frame, frame);
    assert_eq!(delivered.frame_digest, frame_digest(&sent));
    assert_eq!(assembler.buffered_bytes(), 0);
}

#[test]
fn a_replay_after_completion_and_a_forged_signature_are_typed_refusals() {
    let key = SigningKey::from_bytes(&[5; 32]);
    let mut assembler = FrameAssembler::new(policy()).unwrap();
    assembler.register_principal(PRINCIPAL, key.verifying_key(), "stage");
    let frame = big_frame();
    let meta = FrameMeta {
        frame_id: 9,
        parent_operation_id: 2,
        signed_absolute_deadline_ms: 3_000,
    };
    let chunks = chunk_frame(&key, &frame, meta, 1_500_000).unwrap();
    for chunk in &chunks {
        assembler
            .offer(0, PRINCIPAL, 1, &chunk_payload(chunk))
            .unwrap();
    }
    // Under any session, at any request id: the tuple is tombstoned until its deadline + skew.
    for session in [1, 2] {
        assert_eq!(
            assembler.offer(10, PRINCIPAL, session, &chunk_payload(&chunks[0])),
            Err(Rejection::FrameReplayed)
        );
    }
    // After that the signed deadline refuses it.
    assert_eq!(
        assembler.offer(4_001, PRINCIPAL, 1, &chunk_payload(&chunks[0])),
        Err(Rejection::Expired)
    );

    let other = SigningKey::from_bytes(&[6; 32]);
    let forged = chunk_frame(
        &other,
        &frame,
        FrameMeta {
            frame_id: 10,
            ..meta
        },
        1_500_000,
    )
    .unwrap();
    assert_eq!(
        assembler.offer(0, PRINCIPAL, 1, &chunk_payload(&forged[0])),
        Err(Rejection::BadSignature)
    );
    let kv_page = TypedPayload {
        type_id: TypeId::from("ptr.kv.page.v1"),
        bytes: vec![0; 16],
    };
    // A stranger is turned away before its bytes are looked at; a registered principal
    // sending a type off the allowlist gets the allowlist's answer.
    assert_eq!(
        assembler.offer(0, "nobody", 1, &kv_page),
        Err(Rejection::UnknownPrincipal)
    );
    assert_eq!(
        assembler.offer(0, PRINCIPAL, 1, &kv_page),
        Err(Rejection::NotAllowlisted("ptr.kv.page.v1".to_owned()))
    );
}
