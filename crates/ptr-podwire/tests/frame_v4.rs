//! The V4 frame layout: an absolute deadline travels with the address binding.
//!
//! No socket and no feature flag. What a deadline means once it arrives is the
//! endpoint's business and is tested in `tests/wire.rs`; what is tested here is
//! that the bytes carry it faithfully, that V3 frames stay readable, and that a
//! deadline can only shrink as a request is relayed.
use ptr_podwire::{
    decode_answer_v2, decode_request_v2, encode_answer_v2, encode_request_v2, FrameError,
    PodAddressBinding, PodAnswerV2, PodOutcome, PodRequestV2, RefusalCode, FORMAT_V3, FORMAT_V4,
};
use ptr_protocol::TypedPayload;
use ptr_types::{
    ArtifactId, CapabilityId, Generation, PodAddress, PodRevisionAddress, Revision, SessionId,
    Timestamp, TypeId,
};

fn binding(deadline: Option<(u64, u64)>) -> PodAddressBinding {
    PodAddressBinding {
        source: None,
        destination: PodRevisionAddress {
            address: PodAddress::new("alpha".into(), "test".into(), "summarizer".into()).unwrap(),
            semantic_revision: [3; 32],
            generation: Generation(2),
        },
        trace_id: "frame-v4".into(),
        hop_limit: 4,
        visited: Vec::new(),
        protocol: None,
        mesh: None,
        deadline: deadline.map(|(deadline, _)| Timestamp(deadline)),
        origin_ts: deadline.map(|(_, origin)| Timestamp(origin)),
    }
}

fn request(address: PodAddressBinding) -> PodRequestV2 {
    PodRequestV2 {
        address: Some(address),
        execution_manifest: Some([4; 32]),
        addressed_to: "key-of-the-host".into(),
        request_id: 9,
        artifact_id: ArtifactId::from("artifact"),
        manifest_hash: [5; 32],
        generation: Generation(2),
        revision: Revision(7),
        placement_epoch: None,
        fencing_token: None,
        identity_digest: [6; 32],
        session_id: SessionId::from("session"),
        policy_revision: Revision(1),
        capability: CapabilityId::from("Summarize<Document>"),
        expect_protocol: 1,
        payload: TypedPayload {
            type_id: TypeId::from("Document"),
            bytes: b"a document".to_vec(),
        },
    }
}

fn answer(address: Option<PodAddressBinding>) -> PodAnswerV2 {
    PodAnswerV2 {
        address,
        execution_manifest: Some([4; 32]),
        responder: "key-of-the-host".into(),
        request_id: 9,
        request_digest: [8; 32],
        generation: Generation(2),
        revision: Revision(7),
        placement_epoch: None,
        fencing_token: None,
        identity_digest: [6; 32],
        session_id: SessionId::from("session"),
        policy_revision: Revision(1),
        outcome: PodOutcome::Refused {
            code: RefusalCode::Unavailable,
        },
    }
}

fn format_of(frame: &[u8]) -> u16 {
    u16::from_le_bytes([frame[8], frame[9]])
}

#[test]
fn a_request_with_a_deadline_round_trips_in_the_v4_layout() {
    let asked = request(binding(Some((1_800_000_000_000, 1_799_999_995_000))));
    let frame = encode_request_v2(&asked).unwrap();
    assert_eq!(format_of(&frame), FORMAT_V4);
    assert_eq!(decode_request_v2(&frame).unwrap(), asked);
}

#[test]
fn an_answer_echoing_the_deadline_round_trips_in_the_v4_layout() {
    let answered = answer(Some(binding(Some((1_800_000_000_000, 1_799_999_995_000)))));
    let frame = encode_answer_v2(&answered).unwrap();
    assert_eq!(format_of(&frame), FORMAT_V4);
    assert_eq!(decode_answer_v2(&frame).unwrap(), answered);
}

#[test]
fn a_request_without_a_deadline_is_still_written_and_read_as_v3() {
    let asked = request(binding(None));
    let frame = encode_request_v2(&asked).unwrap();
    assert_eq!(
        format_of(&frame),
        FORMAT_V3,
        "a V3 peer keeps reading what a V4 build writes when no deadline is set"
    );
    let decoded = decode_request_v2(&frame).unwrap();
    assert_eq!(decoded, asked);
    assert_eq!(decoded.address.unwrap().deadline, None);
}

#[test]
fn a_v3_frame_has_no_deadline_to_read_and_a_v4_frame_is_longer_by_exactly_two_words() {
    let v3 = encode_request_v2(&request(binding(None))).unwrap();
    let v4 = encode_request_v2(&request(binding(Some((20, 10))))).unwrap();
    assert_eq!(v4.len(), v3.len() + 16);
}

#[test]
fn every_prefix_of_a_v4_frame_is_refused() {
    let frame = encode_request_v2(&request(binding(Some((20, 10))))).unwrap();
    for end in 0..frame.len() {
        assert!(
            decode_request_v2(&frame[..end]).is_err(),
            "a prefix of {end} bytes decoded"
        );
    }
}

#[test]
fn a_deadline_that_is_not_after_its_origin_is_refused_on_write_and_on_read() {
    for (deadline, origin) in [(10, 10), (9, 10), (0, 0)] {
        let asked = request(binding(Some((deadline, origin))));
        assert_eq!(
            encode_request_v2(&asked),
            Err(FrameError::InvalidField { field: "deadline" })
        );
    }
    // A frame composed by hand with the two words swapped is refused by the reader.
    let mut frame = encode_request_v2(&request(binding(Some((20, 10))))).unwrap();
    let words = frame
        .windows(16)
        .position(|window| window[..8] == 20u64.to_le_bytes() && window[8..] == 10u64.to_le_bytes())
        .unwrap();
    frame[words..words + 8].copy_from_slice(&10u64.to_le_bytes());
    frame[words + 8..words + 16].copy_from_slice(&20u64.to_le_bytes());
    assert_eq!(
        decode_request_v2(&frame),
        Err(FrameError::InvalidField { field: "deadline" })
    );
}

#[test]
fn a_deadline_without_an_origin_or_the_reverse_is_not_a_binding() {
    let mut half = binding(None);
    half.deadline = Some(Timestamp(20));
    assert_eq!(
        half.validate(),
        Err(FrameError::InvalidField { field: "deadline" })
    );
    let mut other = binding(None);
    other.origin_ts = Some(Timestamp(10));
    assert_eq!(
        other.validate(),
        Err(FrameError::InvalidField { field: "deadline" })
    );
}

#[test]
fn a_deadline_shrinks_across_hops_and_never_grows() {
    let origin = binding(Some((1_000, 100)));
    // A relay asking for more time than the caller allowed gets the caller's.
    assert_eq!(origin.narrowed_deadline(Timestamp(5_000)), Timestamp(1_000));
    // A relay asking for less keeps its tighter bound.
    assert_eq!(origin.narrowed_deadline(Timestamp(600)), Timestamp(600));

    let mut tighter = origin.clone();
    tighter.deadline = Some(origin.narrowed_deadline(Timestamp(600)));
    assert_eq!(tighter.ensure_deadline_within(&origin), Ok(()));

    let mut looser = origin.clone();
    looser.deadline = Some(Timestamp(1_001));
    assert_eq!(
        looser.ensure_deadline_within(&origin),
        Err(FrameError::InvalidField { field: "deadline" })
    );

    let mut dropped = origin.clone();
    dropped.deadline = None;
    dropped.origin_ts = None;
    assert_eq!(
        dropped.ensure_deadline_within(&origin),
        Err(FrameError::InvalidField { field: "deadline" }),
        "a relay cannot shed the deadline it was handed"
    );
}

#[test]
fn a_binding_with_no_deadline_adopts_the_one_a_hop_requests() {
    let open = binding(None);
    assert_eq!(open.narrowed_deadline(Timestamp(700)), Timestamp(700));
}
