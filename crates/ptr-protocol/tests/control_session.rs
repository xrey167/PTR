//! End-to-end tests of the generic control layer over the in-memory loopback transport,
//! using only the public API.

use ptr_protocol::control::{decode, encode, ControlFrame, IdempotencyKey};
use ptr_protocol::debug_json::{render_control_frame, render_negotiated};
use ptr_protocol::negotiation::{
    negotiate, Feature, FeatureSet, NegotiationError, ProtocolVersion, VersionOffer,
};
use ptr_protocol::session::{
    DriveError, FrameTransport, LoopbackTransport, ProtocolSession, SessionConfig, SessionError,
    SessionEvent, SessionState, TransportError, MAX_UNSENT_FRAMES,
};
use ptr_protocol::TypedPayload;
use ptr_types::{CapabilityId, TypeId};

fn offer() -> VersionOffer {
    VersionOffer {
        supported: vec![ProtocolVersion::new(1, 0, 0), ProtocolVersion::new(1, 1, 0)],
        minimum: ProtocolVersion::new(1, 0, 0),
        features: FeatureSet::all(),
        required: FeatureSet::of(&[Feature::Idempotency]),
    }
}

fn session(name: &str, capability: &str) -> ProtocolSession {
    let mut config = SessionConfig::new(name, offer(), 42);
    config.capabilities = vec![CapabilityId::from(capability)];
    ProtocolSession::new(config).unwrap()
}

fn payload(bytes: &[u8]) -> TypedPayload {
    TypedPayload {
        type_id: TypeId::from("Text"),
        bytes: bytes.to_vec(),
    }
}

fn key(text: &str) -> IdempotencyKey {
    IdempotencyKey::new(text).unwrap()
}

/// Handshakes two sessions over a loopback pair.
fn handshake() -> (
    (ProtocolSession, LoopbackTransport),
    (ProtocolSession, LoopbackTransport),
) {
    let (mut ta, mut tb) = LoopbackTransport::pair(16);
    let mut a = session("node-a", "predict");
    let mut b = session("node-b", "embed");
    ta.send(&a.start().unwrap()).unwrap();
    assert_eq!(a.state(), SessionState::HelloSent);
    // b answers Hello with its own Hello and Decl; a then sends its Decl; b receives it.
    let events = b.pump(&mut tb).unwrap();
    assert!(matches!(events[0], SessionEvent::Negotiated(_)));
    a.pump(&mut ta).unwrap();
    b.pump(&mut tb).unwrap();
    assert_eq!(a.state(), SessionState::Negotiated);
    assert_eq!(b.state(), SessionState::Negotiated);
    ((a, ta), (b, tb))
}

#[test]
fn handshake_negotiates_the_highest_common_version_and_exchanges_declarations() {
    let ((a, _), (b, _)) = handshake();
    assert_eq!(a.negotiated(), b.negotiated());
    let negotiated = a.negotiated().unwrap();
    assert_eq!(negotiated.version, ProtocolVersion::new(1, 1, 0));
    assert!(negotiated.features.contains(Feature::Drain));
    assert_eq!(a.peer_node_id(), Some("node-b"));
    assert_eq!(b.peer_node_id(), Some("node-a"));
    assert_eq!(a.peer_nonce(), Some(42));
    assert_eq!(a.peer_capabilities(), [CapabilityId::from("embed")]);
    assert_eq!(b.peer_capabilities(), [CapabilityId::from("predict")]);
}

#[test]
fn calls_retries_cancel_and_close_end_to_end() {
    let ((mut a, mut ta), (mut b, mut tb)) = handshake();
    let mut executions: Vec<Vec<u8>> = Vec::new();
    let mut handler = |p: &TypedPayload| {
        executions.push(p.bytes.clone());
        let mut bytes = b"echo:".to_vec();
        bytes.extend_from_slice(&p.bytes);
        TypedPayload {
            type_id: p.type_id.clone(),
            bytes,
        }
    };

    // Three calls, the third cancelled before b serves anything.
    for (k, text) in [("k1", "one"), ("k2", "two"), ("k3", "three")] {
        let frame = a.call(key(k), payload(text.as_bytes())).unwrap();
        ta.send(&frame).unwrap();
    }
    ta.send(&a.cancel(3, "not needed").unwrap()).unwrap();
    let events = b.pump(&mut tb).unwrap();
    assert_eq!(
        events,
        vec![
            SessionEvent::CallQueued { seq: 1 },
            SessionEvent::CallQueued { seq: 2 },
            SessionEvent::CallQueued { seq: 3 },
            SessionEvent::CallCancelled { seq: 3 },
        ]
    );
    assert_eq!(b.serve_all(&mut tb, &mut handler).unwrap(), 2);

    let events = a.pump(&mut ta).unwrap();
    assert_eq!(
        events,
        vec![
            SessionEvent::CancelAcknowledged { seq: 3 },
            SessionEvent::Replied {
                seq: 1,
                payload: payload(b"echo:one")
            },
            SessionEvent::Replied {
                seq: 2,
                payload: payload(b"echo:two")
            },
        ]
    );
    assert_eq!(a.outstanding_calls(), 0);

    // A fourth call whose reply is lost on the way back; the caller retries.
    ta.send(&a.call(key("k4"), payload(b"four")).unwrap())
        .unwrap();
    b.pump(&mut tb).unwrap();
    b.serve_all(&mut tb, &mut handler).unwrap();
    let lost = ta.recv().unwrap().unwrap();
    assert!(matches!(lost, ControlFrame::Reply { seq: 4, .. }));
    assert_eq!(a.outstanding_calls(), 1);
    ta.send(&a.retry(4).unwrap()).unwrap();
    let events = b.pump(&mut tb).unwrap();
    assert_eq!(events, vec![SessionEvent::DuplicateServed { seq: 4 }]);
    assert_eq!(b.serve_all(&mut tb, &mut handler).unwrap(), 0);
    let events = a.pump(&mut ta).unwrap();
    assert_eq!(
        events,
        vec![SessionEvent::Replied {
            seq: 4,
            payload: payload(b"echo:four")
        }]
    );

    // The handler ran once per distinct key: one, two and four; never three, never twice.
    assert_eq!(
        executions,
        vec![b"one".to_vec(), b"two".to_vec(), b"four".to_vec()]
    );

    // Reusing a completed key for another payload is rejected and executes nothing. The
    // frame is injected raw because a well-behaved caller session would not send it.
    let clash = ControlFrame::Call {
        seq: 5,
        idempotency_key: key("k1"),
        payload: payload(b"changed"),
    };
    ta.send(&clash).unwrap();
    assert_eq!(
        b.pump(&mut tb),
        Err(DriveError::Session(SessionError::KeyReused {
            key: key("k1")
        }))
    );
    assert_eq!(executions.len(), 3);

    // Close: a says Bye, b answers Bye, both end Closed with nothing in flight.
    let mut a2 = a;
    assert_eq!(a2.outstanding_calls(), 0);
    ta.send(&a2.close("done").unwrap()).unwrap();
    assert_eq!(a2.state(), SessionState::Closing);
    b.pump(&mut tb).unwrap();
    a2.pump(&mut ta).unwrap();
    assert_eq!(a2.state(), SessionState::Closed);
    assert_eq!(b.state(), SessionState::Closed);
    assert!(matches!(
        a2.call(key("after"), payload(b"x")),
        Err(SessionError::IllegalTransition {
            state: SessionState::Closed,
            ..
        })
    ));
}

#[test]
fn incompatible_peers_fail_the_handshake_with_a_typed_reason_and_a_bye() {
    let (mut ta, mut tb) = LoopbackTransport::pair(8);
    let mut a = session("node-a", "predict");
    let v2 = VersionOffer {
        supported: vec![ProtocolVersion::new(2, 0, 0)],
        minimum: ProtocolVersion::new(2, 0, 0),
        features: FeatureSet::EMPTY,
        required: FeatureSet::EMPTY,
    };
    let mut b = ProtocolSession::new(SessionConfig::new("node-b", v2, 1)).unwrap();
    ta.send(&a.start().unwrap()).unwrap();
    let error = b.pump(&mut tb).unwrap_err();
    assert!(matches!(
        error,
        DriveError::Session(SessionError::Negotiation(
            NegotiationError::MajorMismatch { .. }
        ))
    ));
    assert_eq!(b.state(), SessionState::Closed);
    // The rejected end told the other why, and the other end is not left negotiated.
    assert!(matches!(ta.recv().unwrap(), Some(ControlFrame::Bye { .. })));
}

#[test]
fn garbage_on_the_wire_is_a_typed_transport_error_not_a_panic() {
    let (mut ta, mut tb) = LoopbackTransport::pair(4);
    let mut b = session("node-b", "embed");
    ta.send_raw(b"not a frame at all".to_vec()).unwrap();
    assert!(matches!(b.pump(&mut tb), Err(DriveError::Transport(_))));
}

#[test]
fn codec_round_trip_and_debug_rendering_agree_on_a_handshake_frame() {
    let hello = ControlFrame::Hello {
        node_id: "node-a".into(),
        offer: offer(),
        nonce: 1,
    };
    let decoded = decode(&encode(&hello).unwrap()).unwrap();
    assert_eq!(render_control_frame(&decoded), render_control_frame(&hello));
    let session = negotiate(&offer(), &offer()).unwrap();
    assert_eq!(
        render_negotiated(&session),
        r#"{"version":"1.1.0","features":["idempotency","cancel","drain"]}"#
    );
}

#[test]
fn every_truncation_of_every_frame_kind_is_an_error() {
    let frames = [
        ControlFrame::Hello {
            node_id: "n".into(),
            offer: offer(),
            nonce: 3,
        },
        ControlFrame::Decl {
            capabilities: vec![CapabilityId::from("a")],
        },
        ControlFrame::Ack {
            seq: 1,
            idempotency_key: key("k"),
        },
        ControlFrame::Cancel {
            seq: 1,
            reason: "r".into(),
        },
        ControlFrame::Bye { reason: "r".into() },
        ControlFrame::Call {
            seq: 1,
            idempotency_key: key("k"),
            payload: payload(b"abc"),
        },
        ControlFrame::Reply {
            seq: 1,
            payload: payload(b"abc"),
        },
    ];
    for frame in frames {
        let bytes = encode(&frame).unwrap();
        for cut in 0..bytes.len() {
            assert!(decode(&bytes[..cut]).is_err());
        }
        assert_eq!(decode(&bytes), Ok(frame));
    }
}

/// A Reply that is late rather than lost: the caller retries, the peer answers the retry from
/// its completed cache, and both Replies arrive. The second one must not abort the session.
#[test]
fn duplicate_reply_after_retry_with_delayed_original_is_ignored_and_counted() {
    let ((mut a, mut ta), (mut b, mut tb)) = handshake();
    let mut handler = |p: &TypedPayload| payload(&[b"echo:".as_slice(), &p.bytes].concat());

    ta.send(&a.call(key("k"), payload(b"one")).unwrap())
        .unwrap();
    b.pump(&mut tb).unwrap();
    b.serve_all(&mut tb, &mut handler).unwrap();
    // The original Reply is delayed: it sits unread in a's inbound queue. The caller retries
    // meanwhile and the peer answers the retry from its cache.
    ta.send(&a.retry(1).unwrap()).unwrap();
    assert_eq!(
        b.pump(&mut tb).unwrap(),
        vec![SessionEvent::DuplicateServed { seq: 1 }]
    );

    // Both Replies now reach the caller in one pump.
    let events = a.pump(&mut ta).unwrap();
    assert_eq!(
        events,
        vec![
            SessionEvent::Replied {
                seq: 1,
                payload: payload(b"echo:one")
            },
            SessionEvent::DuplicateIgnored { seq: 1 },
        ]
    );
    assert_eq!(a.outstanding_calls(), 0);
    assert_eq!(a.duplicate_replies_ignored(), 1);

    // A recently completed seq answered with a different payload, and a seq never sent, stay
    // typed errors.
    tb.send(&ControlFrame::Reply {
        seq: 1,
        payload: payload(b"echo:other"),
    })
    .unwrap();
    assert_eq!(
        a.pump(&mut ta),
        Err(DriveError::Session(SessionError::UnknownSeq { seq: 1 }))
    );
    tb.send(&ControlFrame::Reply {
        seq: 77,
        payload: payload(b"x"),
    })
    .unwrap();
    assert_eq!(
        a.pump(&mut ta),
        Err(DriveError::Session(SessionError::UnknownSeq { seq: 77 }))
    );
}

#[test]
fn unencodable_calls_are_refused_without_reserving_window_slots() {
    use ptr_protocol::control::{CodecError, MAX_FRAME_BYTES};
    use ptr_protocol::session::DEFAULT_WINDOW;

    let ((mut a, mut ta), (mut b, mut tb)) = handshake();
    let empty_type = TypedPayload {
        type_id: TypeId::from(""),
        bytes: b"x".to_vec(),
    };
    let long_type = TypedPayload {
        type_id: TypeId::from("t".repeat(4096).as_str()),
        bytes: b"x".to_vec(),
    };
    let big_body = payload(&vec![0u8; MAX_FRAME_BYTES]);
    let invalid = [empty_type, long_type, big_body];

    for n in 0..=DEFAULT_WINDOW {
        let result = a.call(key(&format!("bad-{n}")), invalid[n % invalid.len()].clone());
        assert!(
            matches!(result, Err(SessionError::Encode(_))),
            "call {n} must be refused as unencodable, got {result:?}"
        );
        assert_eq!(a.outstanding_calls(), 0);
    }
    assert!(matches!(
        a.call(key("bad-x"), payload(&vec![0u8; MAX_FRAME_BYTES])),
        Err(SessionError::Encode(CodecError::FrameTooLarge { .. }))
    ));

    // A valid call afterwards still works end to end and gets sequence number 1.
    let frame = a.call(key("good"), payload(b"hello")).unwrap();
    assert!(matches!(frame, ControlFrame::Call { seq: 1, .. }));
    ta.send(&frame).unwrap();
    b.pump(&mut tb).unwrap();
    b.serve_all(&mut tb, &mut |p: &TypedPayload| {
        payload(&[b"echo:", &p.bytes[..]].concat())
    })
    .unwrap();
    let events = a.pump(&mut ta).unwrap();
    assert_eq!(
        events,
        vec![SessionEvent::Replied {
            seq: 1,
            payload: payload(b"echo:hello")
        }]
    );
    assert_eq!(a.outstanding_calls(), 0);
}

#[test]
fn unencodable_cancel_close_and_reply_do_not_mutate_session_state() {
    let ((mut a, mut ta), (mut b, mut tb)) = handshake();
    let long = "r".repeat(2048);

    // Cancel with an oversized reason is refused and the call is not marked cancelling.
    let frame = a.call(key("k1"), payload(b"one")).unwrap();
    assert!(matches!(
        a.cancel(1, long.clone()),
        Err(SessionError::Encode(_))
    ));
    // An Ack for a call that was never cancelled stays an AckMismatch.
    ta.send(&frame).unwrap();
    b.pump(&mut tb).unwrap();
    tb.send(&ControlFrame::Ack {
        seq: 1,
        idempotency_key: key("k1"),
    })
    .unwrap();
    assert_eq!(
        a.pump(&mut ta),
        Err(DriveError::Session(SessionError::AckMismatch { seq: 1 }))
    );

    // Close with an oversized reason is refused and the session stays Negotiated.
    let ((mut c, _tc), _) = handshake();
    assert!(matches!(c.close(long), Err(SessionError::Encode(_))));
    assert_eq!(c.state(), SessionState::Negotiated);
    assert!(c.close("done").is_ok());

    // A handler reply that cannot be encoded is an error and is not cached for retries.
    let ((mut d, mut td), (mut e, mut te)) = handshake();
    td.send(&d.call(key("k2"), payload(b"two")).unwrap())
        .unwrap();
    e.pump(&mut te).unwrap();
    let mut bad_handler = |_: &TypedPayload| TypedPayload {
        type_id: TypeId::from(""),
        bytes: Vec::new(),
    };
    assert!(matches!(
        e.serve_next(&mut bad_handler),
        Err(SessionError::Encode(_))
    ));
    assert_eq!(e.queued_calls(), 0);
    assert_eq!(e.completed_keys(), 0);
}

fn bad_configs() -> Vec<(&'static str, SessionConfig)> {
    let empty_id = SessionConfig::new("", offer(), 1);
    let long_id = SessionConfig::new(
        "n".repeat(ptr_protocol::control::MAX_STRING_BYTES + 1),
        offer(),
        1,
    );
    let mut malformed = offer();
    malformed.supported.clear();
    let bad_offer = SessionConfig::new("node-bad", malformed, 1);
    let mut many_caps = SessionConfig::new("node-caps", offer(), 1);
    many_caps.capabilities =
        vec![CapabilityId::from("c"); ptr_protocol::control::MAX_CAPABILITIES + 1];
    vec![
        ("empty node id", empty_id),
        ("oversized node id", long_id),
        ("malformed offer", bad_offer),
        ("oversized capability list", many_caps),
    ]
}

#[test]
fn start_with_an_unencodable_config_is_refused_and_leaves_the_session_idle() {
    for (name, config) in bad_configs() {
        let mut bad = ProtocolSession::new(config).unwrap();
        assert!(
            matches!(bad.start(), Err(SessionError::Encode(_))),
            "{name}: start must be refused with Encode"
        );
        assert_eq!(bad.state(), SessionState::Idle, "{name}");
        // Still Idle, so it is refused the same way again rather than IllegalTransition.
        assert!(
            matches!(bad.start(), Err(SessionError::Encode(_))),
            "{name}"
        );
    }
    // A session with a fixed config starts and negotiates over loopback.
    let ((a, _ta), (b, _tb)) = handshake();
    assert_eq!(a.state(), SessionState::Negotiated);
    assert_eq!(b.state(), SessionState::Negotiated);
}

#[test]
fn responding_to_a_hello_with_an_unencodable_config_does_not_negotiate() {
    let mut peer = session("node-a", "predict");
    let hello = peer.start().unwrap();
    for (name, config) in bad_configs() {
        let mut bad = ProtocolSession::new(config).unwrap();
        if name == "malformed offer" {
            // The negotiation itself refuses an invalid local offer: a typed handshake
            // failure that closes with a farewell, not an unsendable reply.
            assert!(matches!(
                bad.on_frame(hello.clone()),
                Err(SessionError::Negotiation(_))
            ));
            assert_eq!(bad.state(), SessionState::Closed);
            assert!(bad.take_farewell().is_some());
            continue;
        }
        assert!(
            matches!(bad.on_frame(hello.clone()), Err(SessionError::Encode(_))),
            "{name}: reply to Hello must be refused with Encode"
        );
        assert_eq!(bad.state(), SessionState::Idle, "{name}");
        assert!(bad.negotiated().is_none(), "{name}");
        assert!(bad.peer_node_id().is_none(), "{name}");
    }
}

/// Handshakes over a pair whose queues hold only `capacity` frames.
fn handshake_with_capacity(
    capacity: usize,
) -> (
    (ProtocolSession, LoopbackTransport),
    (ProtocolSession, LoopbackTransport),
) {
    let (mut ta, mut tb) = LoopbackTransport::pair(capacity);
    let mut a = session("node-a", "predict");
    let mut b = session("node-b", "embed");
    ta.send(&a.start().unwrap()).unwrap();
    b.pump(&mut tb).unwrap();
    a.pump(&mut ta).unwrap();
    b.pump(&mut tb).unwrap();
    ((a, ta), (b, tb))
}

/// Fills b's outgoing queue with harmless Decl frames, so b's next send fails.
fn fill(tb: &mut LoopbackTransport, capacity: usize) {
    for _ in 0..capacity {
        tb.send(&ControlFrame::Decl {
            capabilities: Vec::new(),
        })
        .unwrap();
    }
}

#[test]
fn cancel_ack_that_could_not_be_sent_is_delivered_after_the_peer_drains_and_retries() {
    let ((mut a, mut ta), (mut b, mut tb)) = handshake_with_capacity(2);
    ta.send(&a.call(key("k"), payload(b"x")).unwrap()).unwrap();
    ta.send(&a.cancel(1, "no longer needed").unwrap()).unwrap();
    fill(&mut tb, 2);
    // The Ack cannot be queued: the removal of the call has already been committed.
    assert!(matches!(
        b.pump(&mut tb),
        Err(DriveError::Transport(TransportError::Full { .. }))
    ));
    assert_eq!(b.queued_calls(), 0);
    // The peer drains its inbox and retries the Cancel.
    a.pump(&mut ta).unwrap();
    assert_eq!(a.outstanding_calls(), 1);
    ta.send(&a.cancel(1, "retry").unwrap()).unwrap();
    b.pump(&mut tb).unwrap();
    let events = a.pump(&mut ta).unwrap();
    assert_eq!(events, vec![SessionEvent::CancelAcknowledged { seq: 1 }]);
    assert_eq!(a.outstanding_calls(), 0);
    assert_eq!(b.unsent_frames(), 0);
}

#[test]
fn bye_reply_that_could_not_be_sent_is_delivered_later() {
    let ((mut a, mut ta), (mut b, mut tb)) = handshake_with_capacity(2);
    ta.send(&a.close("done").unwrap()).unwrap();
    fill(&mut tb, 2);
    assert!(matches!(
        b.pump(&mut tb),
        Err(DriveError::Transport(TransportError::Full { .. }))
    ));
    // Drain the filler by hand: Decl is not legal once a is Closing.
    while ta.recv().unwrap().is_some() {}
    // Nothing new arrives; the next pump still sends the held Bye.
    b.pump(&mut tb).unwrap();
    a.pump(&mut ta).unwrap();
    assert_eq!(a.state(), SessionState::Closed);
    assert_eq!(b.state(), SessionState::Closed);
}

#[test]
fn held_frames_stay_bounded_under_repeated_send_failures() {
    let ((mut a, mut ta), (mut b, mut tb)) = handshake_with_capacity(2);
    ta.send(&a.call(key("k"), payload(b"x")).unwrap()).unwrap();
    ta.send(&a.cancel(1, "c").unwrap()).unwrap();
    fill(&mut tb, 2);
    for _ in 0..10 {
        assert!(b.pump(&mut tb).is_err());
        assert!(b.unsent_frames() <= MAX_UNSENT_FRAMES);
    }
    // The held Ack blocks further receiving, so nothing more accumulates.
    assert_eq!(b.unsent_frames(), 1);
    drop(a);
}

#[test]
fn responder_hello_and_decl_survive_a_full_queue() {
    let (mut ta, mut tb) = LoopbackTransport::pair(1);
    let mut a = session("node-a", "predict");
    let mut b = session("node-b", "embed");
    ta.send(&a.start().unwrap()).unwrap();
    // The Hello fits, the Decl does not; b is already Negotiated.
    assert!(b.pump(&mut tb).is_err());
    assert_eq!(b.unsent_frames(), 1);
    a.pump(&mut ta).unwrap();
    b.pump(&mut tb).unwrap();
    assert_eq!(b.unsent_frames(), 0);
    a.pump(&mut ta).unwrap();
    assert_eq!(a.state(), SessionState::Negotiated);
}

#[test]
fn reply_refused_by_serve_all_is_held_and_sent_once_after_the_peer_drains() {
    let ((mut a, mut ta), (mut b, mut tb)) = handshake_with_capacity(2);
    let mut executions = 0;
    let mut handler = |p: &TypedPayload| {
        executions += 1;
        p.clone()
    };
    ta.send(&a.call(key("k"), payload(b"x")).unwrap()).unwrap();
    ta.send(&a.call(key("k2"), payload(b"y")).unwrap()).unwrap();
    b.pump(&mut tb).unwrap();
    fill(&mut tb, 2);
    // The first Reply is refused; serving stops there and the second call stays queued.
    assert!(matches!(
        b.serve_all(&mut tb, &mut handler),
        Err(DriveError::Transport(TransportError::Full { .. }))
    ));
    assert_eq!(b.unsent_frames(), 1);
    assert_eq!(b.queued_calls(), 1);
    // Still full: nothing is served and nothing more is held.
    assert!(b.serve_all(&mut tb, &mut handler).is_err());
    assert_eq!(b.unsent_frames(), 1);
    assert_eq!(b.queued_calls(), 1);
    // The peer drains its inbox; retrying the drive operation delivers the held Reply first.
    a.pump(&mut ta).unwrap();
    assert_eq!(b.serve_all(&mut tb, &mut handler).unwrap(), 1);
    assert_eq!(b.unsent_frames(), 0);
    let events = a.pump(&mut ta).unwrap();
    let replies: Vec<u64> = events
        .iter()
        .filter_map(|event| match event {
            SessionEvent::Replied { seq, .. } => Some(*seq),
            _ => None,
        })
        .collect();
    assert_eq!(replies, vec![1, 2]);
    assert_eq!(a.outstanding_calls(), 0);
}
