//! The serve loop: responder handshake, then calls until the peer says Bye.
//!
//! Queued calls run only once the transport has no further complete frame buffered (or the
//! inbound window is full). A `Cancel` that reaches the worker in the same read as its `Call`
//! therefore finds the call still queued and drops it; a `Cancel` that arrives after the
//! call ran is ignored by the session. Single-threaded: one call executes at a time.

use std::io::{Read, Write};

use ptr_pods::PodRegistry;
use ptr_protocol::negotiation::{Feature, FeatureSet, ProtocolVersion, VersionOffer};
use ptr_protocol::session::{
    ProtocolSession, SessionConfig, SessionError, SessionEvent, SessionState,
};
use ptr_protocol::TypedPayload;
use ptr_types::CapabilityId;

use crate::pods::{dispatch, BUILTIN};
use crate::transport::{StdioError, StdioTransport};

/// Node id the worker announces in its Hello.
pub const WORKER_NODE_ID: &str = "ptr-worker";

/// Why serving ended abnormally.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ServeError {
    Transport(StdioError),
    Session(SessionError),
    /// The peer's stream ended without a Bye (`during_handshake` when no Hello arrived).
    PeerVanished {
        during_handshake: bool,
    },
    BadConfig(SessionError),
}

impl std::fmt::Display for ServeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for ServeError {}

/// What a clean session did.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ServeSummary {
    pub peer_node: Option<String>,
    pub calls_executed: u64,
    pub duplicates_served: u64,
    pub calls_cancelled: u64,
}

/// The offer the worker makes: versions 1.0.0 and 1.1.0, all features, idempotency required.
pub fn worker_offer() -> VersionOffer {
    VersionOffer {
        supported: vec![ProtocolVersion::new(1, 0, 0), ProtocolVersion::new(1, 1, 0)],
        minimum: ProtocolVersion::new(1, 0, 0),
        features: FeatureSet::all(),
        required: FeatureSet::of(&[Feature::Idempotency]),
    }
}

/// Session bounds, in calls.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ServeLimits {
    pub window: usize,
    pub completed_capacity: usize,
}

impl Default for ServeLimits {
    fn default() -> Self {
        Self {
            window: ptr_protocol::session::DEFAULT_WINDOW,
            completed_capacity: ptr_protocol::session::DEFAULT_COMPLETED_CAPACITY,
        }
    }
}

/// Serves one session on `transport` until Bye (`Ok`) or failure.
pub fn serve<R: Read, W: Write>(
    transport: &mut StdioTransport<R, W>,
    registry: &PodRegistry,
    limits: ServeLimits,
    nonce: u64,
) -> Result<ServeSummary, ServeError> {
    let mut config = SessionConfig::new(WORKER_NODE_ID, worker_offer(), nonce);
    config.capabilities = BUILTIN
        .iter()
        .map(|pod| CapabilityId::from(pod.name))
        .collect();
    config.window = limits.window;
    config.completed_capacity = limits.completed_capacity;
    let mut session = ProtocolSession::new(config).map_err(ServeError::BadConfig)?;
    let mut summary = ServeSummary::default();
    let mut handler =
        |payload: &TypedPayload| -> TypedPayload { dispatch(registry, payload).to_payload() };

    while session.state() != SessionState::Closed {
        let frame = match transport.read_frame().map_err(ServeError::Transport)? {
            Some(frame) => frame,
            None => {
                return Err(ServeError::PeerVanished {
                    during_handshake: session.state() == SessionState::Idle,
                })
            }
        };
        match session.on_frame(frame) {
            Ok(handled) => {
                for out in &handled.send {
                    transport.write_frame(out).map_err(ServeError::Transport)?;
                }
                match handled.event {
                    SessionEvent::DuplicateServed { .. } => summary.duplicates_served += 1,
                    SessionEvent::CallCancelled { .. } => summary.calls_cancelled += 1,
                    SessionEvent::Negotiated(_)
                    | SessionEvent::PeerDeclared { .. }
                    | SessionEvent::CallQueued { .. }
                    | SessionEvent::DuplicateIgnored { .. }
                    | SessionEvent::Replied { .. }
                    | SessionEvent::CancelIgnored { .. }
                    | SessionEvent::CancelAcknowledged { .. }
                    | SessionEvent::PeerClosing { .. } => {}
                }
            }
            Err(error) => {
                if let Some(bye) = session.take_farewell() {
                    // Best effort: the session error is the cause worth reporting.
                    let _ = transport.write_frame(&bye);
                }
                return Err(ServeError::Session(error));
            }
        }
        if session.state() == SessionState::Closed {
            break;
        }
        if !transport.has_buffered_frame() || session.queued_calls() >= limits.window {
            // Serving the last call of a draining session closes it; stop there.
            while session.state() != SessionState::Closed {
                let Some(reply) = session
                    .serve_next(&mut handler)
                    .map_err(ServeError::Session)?
                else {
                    break;
                };
                summary.calls_executed += 1;
                transport
                    .write_frame(&reply)
                    .map_err(ServeError::Transport)?;
            }
        }
    }
    summary.peer_node = session.peer_node_id().map(str::to_owned);
    Ok(summary)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pods::builtin_registry;
    use crate::wire::WorkerCall;
    use crate::wire::WorkerReply;
    use ptr_protocol::control::{ControlFrame, IdempotencyKey};
    use std::io::Cursor;

    fn client_hello() -> ControlFrame {
        ControlFrame::Hello {
            node_id: "test-client".into(),
            offer: worker_offer(),
            nonce: 1,
        }
    }

    fn call(seq: u64, key: &str, pod: &str, input: &[u8]) -> ControlFrame {
        ControlFrame::Call {
            seq,
            idempotency_key: IdempotencyKey::new(key).unwrap(),
            payload: WorkerCall {
                pod: pod.into(),
                input: input.to_vec(),
            }
            .to_payload()
            .unwrap(),
        }
    }

    /// Runs the worker on a scripted client byte stream; returns the result and its frames.
    fn run(script: &[ControlFrame]) -> (Result<ServeSummary, ServeError>, Vec<ControlFrame>) {
        let mut input = StdioTransport::new(std::io::empty(), Vec::new());
        for frame in script {
            input.write_frame(frame).unwrap();
        }
        let bytes = input.into_parts().1;
        let mut t = StdioTransport::new(Cursor::new(bytes), Vec::new());
        let registry = builtin_registry().unwrap();
        let result = serve(&mut t, &registry, ServeLimits::default(), 9);
        let out = t.into_parts().1;
        let mut reader = StdioTransport::new(Cursor::new(out), Vec::new());
        let mut frames = Vec::new();
        while let Some(frame) = reader.read_frame().unwrap() {
            frames.push(frame);
        }
        (result, frames)
    }

    fn reply_of(frame: &ControlFrame) -> WorkerReply {
        match frame {
            ControlFrame::Reply { payload, .. } => WorkerReply::from_payload(payload).unwrap(),
            other => panic!("expected a reply, got {other:?}"),
        }
    }

    #[test]
    fn scripted_session_serves_dedups_cancels_and_drains() {
        let script = [
            client_hello(),
            call(1, "a", "counter", b""),
            call(1, "a", "counter", b""),
            call(2, "b", "echo", b"x"),
            ControlFrame::Cancel {
                seq: 2,
                reason: "no".into(),
            },
            ControlFrame::Bye {
                reason: "done".into(),
            },
        ];
        let (result, frames) = run(&script);
        let summary = result.unwrap();
        assert_eq!(summary.calls_executed, 1);
        assert_eq!(summary.calls_cancelled, 1);
        assert_eq!(summary.peer_node.as_deref(), Some("test-client"));
        // Hello, Decl, Ack(2), Bye, Reply(1): the duplicate was ignored while still queued and
        // the queued call drained after the peer's Bye.
        assert!(matches!(frames[0], ControlFrame::Hello { .. }));
        assert!(matches!(frames[1], ControlFrame::Decl { .. }));
        assert!(matches!(frames[2], ControlFrame::Ack { seq: 2, .. }));
        assert!(matches!(frames[3], ControlFrame::Bye { .. }));
        assert_eq!(reply_of(&frames[4]), WorkerReply::Ok(b"1".to_vec()));
        assert_eq!(frames.len(), 5);
    }

    #[test]
    fn eof_without_bye_and_before_hello_are_typed() {
        let (result, _) = run(&[client_hello()]);
        assert_eq!(
            result,
            Err(ServeError::PeerVanished {
                during_handshake: false
            })
        );
        let (result, frames) = run(&[]);
        assert_eq!(
            result,
            Err(ServeError::PeerVanished {
                during_handshake: true
            })
        );
        assert!(frames.is_empty());
    }

    #[test]
    fn session_violations_fail_with_a_farewell() {
        let (result, frames) = run(&[call(1, "a", "echo", b"")]);
        assert!(matches!(
            result,
            Err(ServeError::Session(SessionError::IllegalTransition { .. }))
        ));
        assert!(frames.is_empty());
        let (result, frames) = run(&[ControlFrame::Hello {
            node_id: WORKER_NODE_ID.into(),
            offer: worker_offer(),
            nonce: 1,
        }]);
        assert_eq!(result, Err(ServeError::Session(SessionError::SelfConnect)));
        assert!(matches!(frames.last(), Some(ControlFrame::Bye { .. })));
    }
}
