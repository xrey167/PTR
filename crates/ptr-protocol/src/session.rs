//! A session state machine that speaks [`ControlFrame`]s, and an in-memory transport to run
//! two of them against each other.
//!
//! This is the generic control layer. It is not what `ptr-podwire` runs: that crate has its
//! own bounded wire format. Nothing here crosses a real network; [`LoopbackTransport`] is
//! two bounded in-memory queues that carry the encoded bytes of the frames, so the codec is
//! exercised on every hop.
//!
//! State machine (`Idle -> HelloSent -> Negotiated -> Closing -> Closed`):
//! the end that calls [`ProtocolSession::start`] goes `Idle -> HelloSent` and becomes
//! `Negotiated` on the peer's Hello; the end that receives a Hello while `Idle` negotiates,
//! answers with its own Hello and Decl and is `Negotiated` at once. Any frame that is not
//! legal in the current state is an [`SessionError::IllegalTransition`].
//!
//! Bounds (Invariant 8): at most `window` outstanding calls in each direction, at most
//! `completed_capacity` remembered idempotency keys (least recently used is evicted, after
//! which a retry of that key would execute again), at most `completed_capacity` remembered
//! answered sequence numbers (oldest evicted), at most `capacity` frames per queue, and at
//! most [`MAX_UNSENT_FRAMES`] answers held by `pump` after a refused send.

use std::collections::{BTreeMap, VecDeque};
use std::fmt;
use std::sync::{Arc, Mutex, PoisonError};

use ptr_types::CapabilityId;

use crate::control::{
    decode, encode, payload_digest, CodecError, ControlFrame, FrameKind, IdempotencyKey,
};
use crate::negotiation::{
    CompatibilityMatrix, Feature, NegotiatedSession, NegotiationError, VersionOffer,
};
use crate::TypedPayload;

pub const DEFAULT_WINDOW: usize = 8;
pub const DEFAULT_COMPLETED_CAPACITY: usize = 64;
/// Hard maximum for [`SessionConfig::window`]; larger values are refused with
/// [`SessionError::WindowTooLarge`] so a host cannot turn the bounded call state into an
/// effectively unbounded one.
pub const MAX_WINDOW: usize = 1024;
/// Hard maximum for [`SessionConfig::completed_capacity`]; larger values are refused with
/// [`SessionError::CompletedCapacityTooLarge`].
pub const MAX_COMPLETED_CAPACITY: usize = 65_536;
/// Hard maximum frames per direction of a [`LoopbackTransport`]; a larger `capacity` passed to
/// [`LoopbackTransport::pair`] is clamped to this.
pub const MAX_TRANSPORT_CAPACITY: usize = 65_536;
/// Most frames [`ProtocolSession::pump`] holds after a failed send: one applied frame produces
/// at most two (Hello and Decl) and `pump` stops receiving until they are out.
pub const MAX_UNSENT_FRAMES: usize = 2;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SessionState {
    Idle,
    HelloSent,
    Negotiated,
    Closing,
    Closed,
}

/// Everything one end of a session is configured with.
#[derive(Clone, Debug)]
pub struct SessionConfig {
    pub node_id: String,
    pub offer: VersionOffer,
    pub matrix: CompatibilityMatrix,
    pub nonce: u64,
    /// Capabilities this end announces in its Decl.
    pub capabilities: Vec<CapabilityId>,
    /// Most outstanding calls in each direction.
    pub window: usize,
    /// Most completed idempotency keys remembered.
    pub completed_capacity: usize,
}

impl SessionConfig {
    pub fn new(node_id: impl Into<String>, offer: VersionOffer, nonce: u64) -> Self {
        Self {
            node_id: node_id.into(),
            offer,
            matrix: CompatibilityMatrix::default(),
            nonce,
            capabilities: Vec::new(),
            window: DEFAULT_WINDOW,
            completed_capacity: DEFAULT_COMPLETED_CAPACITY,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SessionError {
    ZeroWindow,
    ZeroCompletedCapacity,
    /// `window` exceeds [`MAX_WINDOW`].
    WindowTooLarge {
        requested: usize,
        max: usize,
    },
    /// `completed_capacity` exceeds [`MAX_COMPLETED_CAPACITY`].
    CompletedCapacityTooLarge {
        requested: usize,
        max: usize,
    },
    /// `frame` is not legal while the session is in `state` (sent or received).
    IllegalTransition {
        state: SessionState,
        frame: FrameKind,
    },
    Negotiation(NegotiationError),
    /// The peer's Hello names the same node id as ours.
    SelfConnect,
    FeatureNotNegotiated(Feature),
    /// Too many outstanding calls of our own.
    WindowFull {
        limit: usize,
    },
    /// The peer sent more outstanding calls than the window allows.
    InboundWindowExceeded {
        limit: usize,
    },
    /// An idempotency key was reused with a different payload.
    KeyReused {
        key: IdempotencyKey,
    },
    /// The key is already in flight under a different sequence number.
    KeyInFlight {
        key: IdempotencyKey,
        seq: u64,
    },
    /// A sequence number that is already in flight.
    DuplicateSeq {
        seq: u64,
    },
    UnknownSeq {
        seq: u64,
    },
    /// An Ack for a call that was not being cancelled, or with a different key.
    AckMismatch {
        seq: u64,
    },
    /// The frame this call would produce cannot be encoded (empty or oversized field, frame
    /// over the cap). Refused before any session state changed.
    Encode(CodecError),
}

/// Refuses a frame the codec would refuse, using the codec itself so the limits live in one
/// place. Called before a session method changes any state.
fn ensure_encodable(frame: &ControlFrame) -> Result<(), SessionError> {
    encode(frame).map(drop).map_err(SessionError::Encode)
}

impl fmt::Display for SessionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for SessionError {}

/// What a received frame did to the session.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SessionEvent {
    Negotiated(NegotiatedSession),
    PeerDeclared {
        capabilities: usize,
    },
    /// A call was accepted and waits for [`ProtocolSession::serve_next`].
    CallQueued {
        seq: u64,
    },
    /// The same call arrived again while still queued (the one answer will serve both), or a
    /// Reply repeated the answer to a call that already completed (counted by
    /// [`ProtocolSession::duplicate_replies_ignored`]).
    DuplicateIgnored {
        seq: u64,
    },
    /// A retried call was answered from the completed cache without executing again.
    DuplicateServed {
        seq: u64,
    },
    /// A reply to one of our calls.
    Replied {
        seq: u64,
        payload: TypedPayload,
    },
    /// The peer cancelled a queued call; it will not execute.
    CallCancelled {
        seq: u64,
    },
    /// A Cancel for a call that is not queued (already answered, or never seen).
    CancelIgnored {
        seq: u64,
    },
    /// The peer confirmed our cancel; the call will not be answered.
    CancelAcknowledged {
        seq: u64,
    },
    PeerClosing {
        reason: String,
    },
}

/// Result of [`ProtocolSession::on_frame`]: frames to send back and what happened.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Handled {
    pub send: Vec<ControlFrame>,
    pub event: SessionEvent,
}

#[derive(Clone, Debug)]
struct Outbound {
    key: IdempotencyKey,
    digest: [u8; 32],
    frame: ControlFrame,
    cancelling: bool,
}

#[derive(Clone, Debug)]
struct Inbound {
    seq: u64,
    key: IdempotencyKey,
    digest: [u8; 32],
    payload: TypedPayload,
}

#[derive(Clone, Debug)]
struct Completed {
    key: IdempotencyKey,
    digest: [u8; 32],
    reply: TypedPayload,
}

/// One end of a control session.
#[derive(Debug)]
pub struct ProtocolSession {
    config: SessionConfig,
    state: SessionState,
    negotiated: Option<NegotiatedSession>,
    peer_node: Option<String>,
    peer_nonce: Option<u64>,
    peer_capabilities: Vec<CapabilityId>,
    next_seq: u64,
    outbound: BTreeMap<u64, Outbound>,
    pending: VecDeque<Inbound>,
    /// Front is the least recently used.
    completed: VecDeque<Completed>,
    /// Outbound calls answered recently: sequence number and Reply payload digest, bounded by
    /// `completed_capacity`. A repeated Reply (late original plus the answer to a retry) is
    /// recognised here instead of being an unknown sequence number. Front is the oldest.
    answered: VecDeque<(u64, [u8; 32])>,
    duplicate_replies: usize,
    /// Frames `pump` applied but could not send; sent before anything else is received.
    unsent: VecDeque<ControlFrame>,
    bye_sent: bool,
    bye_received: bool,
    farewell: Option<ControlFrame>,
}

impl ProtocolSession {
    pub fn new(config: SessionConfig) -> Result<Self, SessionError> {
        if config.window == 0 {
            return Err(SessionError::ZeroWindow);
        }
        if config.completed_capacity == 0 {
            return Err(SessionError::ZeroCompletedCapacity);
        }
        if config.window > MAX_WINDOW {
            return Err(SessionError::WindowTooLarge {
                requested: config.window,
                max: MAX_WINDOW,
            });
        }
        if config.completed_capacity > MAX_COMPLETED_CAPACITY {
            return Err(SessionError::CompletedCapacityTooLarge {
                requested: config.completed_capacity,
                max: MAX_COMPLETED_CAPACITY,
            });
        }
        Ok(Self {
            config,
            state: SessionState::Idle,
            negotiated: None,
            peer_node: None,
            peer_nonce: None,
            peer_capabilities: Vec::new(),
            next_seq: 1,
            outbound: BTreeMap::new(),
            pending: VecDeque::new(),
            completed: VecDeque::new(),
            answered: VecDeque::new(),
            duplicate_replies: 0,
            unsent: VecDeque::new(),
            bye_sent: false,
            bye_received: false,
            farewell: None,
        })
    }

    pub fn state(&self) -> SessionState {
        self.state
    }

    pub fn negotiated(&self) -> Option<&NegotiatedSession> {
        self.negotiated.as_ref()
    }

    pub fn peer_node_id(&self) -> Option<&str> {
        self.peer_node.as_deref()
    }

    pub fn peer_nonce(&self) -> Option<u64> {
        self.peer_nonce
    }

    pub fn peer_capabilities(&self) -> &[CapabilityId] {
        &self.peer_capabilities
    }

    /// Calls we sent that have no reply (or cancel acknowledgement) yet.
    pub fn outstanding_calls(&self) -> usize {
        self.outbound.len()
    }

    /// Calls the peer sent that have not been served yet.
    pub fn queued_calls(&self) -> usize {
        self.pending.len()
    }

    /// Frames produced by an applied frame that [`ProtocolSession::pump`] could not send and
    /// will send first on its next call (at most [`MAX_UNSENT_FRAMES`]).
    pub fn unsent_frames(&self) -> usize {
        self.unsent.len()
    }

    /// Idempotency keys currently remembered.
    pub fn completed_keys(&self) -> usize {
        self.completed.len()
    }

    /// Replies dropped because they repeated the answer to a call that already completed.
    pub fn duplicate_replies_ignored(&self) -> usize {
        self.duplicate_replies
    }

    /// The Bye to send after a negotiation failure closed the session, once.
    pub fn take_farewell(&mut self) -> Option<ControlFrame> {
        self.farewell.take()
    }

    /// Opens the session from this end: `Idle -> HelloSent`.
    pub fn start(&mut self) -> Result<ControlFrame, SessionError> {
        self.require(&[SessionState::Idle], FrameKind::Hello)?;
        // The Hello and the Decl that follows it both come from the config: refuse an
        // unencodable one now, while the session is still Idle, rather than hand back a frame
        // the caller cannot send after the state has moved on.
        self.ensure_handshake_encodable()?;
        self.state = SessionState::HelloSent;
        Ok(self.hello())
    }

    /// Sends a call with a caller-chosen idempotency key.
    pub fn call(
        &mut self,
        key: IdempotencyKey,
        payload: TypedPayload,
    ) -> Result<ControlFrame, SessionError> {
        self.require(&[SessionState::Negotiated], FrameKind::Call)?;
        self.require_feature(Feature::Idempotency)?;
        if self.outbound.len() >= self.config.window {
            return Err(SessionError::WindowFull {
                limit: self.config.window,
            });
        }
        let digest = payload_digest(&payload);
        if let Some((seq, existing)) = self.outbound.iter().find(|(_, o)| o.key == key) {
            return Err(if existing.digest == digest {
                SessionError::KeyInFlight { key, seq: *seq }
            } else {
                SessionError::KeyReused { key }
            });
        }
        let seq = self.next_seq;
        let frame = ControlFrame::Call {
            seq,
            idempotency_key: key.clone(),
            payload,
        };
        ensure_encodable(&frame)?;
        self.next_seq = self.next_seq.saturating_add(1);
        self.outbound.insert(
            seq,
            Outbound {
                key,
                digest,
                frame: frame.clone(),
                cancelling: false,
            },
        );
        Ok(frame)
    }

    /// The original Call frame of an outstanding call, to send again (same sequence number
    /// and key). Safe by construction: the peer answers a duplicate from its cache.
    pub fn retry(&self, seq: u64) -> Result<ControlFrame, SessionError> {
        self.require(&[SessionState::Negotiated], FrameKind::Call)?;
        self.outbound
            .get(&seq)
            .map(|call| call.frame.clone())
            .ok_or(SessionError::UnknownSeq { seq })
    }

    /// Asks the peer to drop the outstanding call `seq`.
    pub fn cancel(
        &mut self,
        seq: u64,
        reason: impl Into<String>,
    ) -> Result<ControlFrame, SessionError> {
        self.require(
            &[SessionState::Negotiated, SessionState::Closing],
            FrameKind::Cancel,
        )?;
        self.require_feature(Feature::Cancel)?;
        let frame = ControlFrame::Cancel {
            seq,
            reason: reason.into(),
        };
        let call = self
            .outbound
            .get_mut(&seq)
            .ok_or(SessionError::UnknownSeq { seq })?;
        ensure_encodable(&frame)?;
        call.cancelling = true;
        Ok(frame)
    }

    /// Starts closing: `Negotiated -> Closing`. New calls are refused from here on; with the
    /// drain feature, the session closes once both Byes are exchanged and nothing is in
    /// flight in either direction.
    pub fn close(&mut self, reason: impl Into<String>) -> Result<ControlFrame, SessionError> {
        self.require(&[SessionState::Negotiated], FrameKind::Bye)?;
        let frame = ControlFrame::Bye {
            reason: reason.into(),
        };
        ensure_encodable(&frame)?;
        self.state = SessionState::Closing;
        self.bye_sent = true;
        Ok(frame)
    }

    /// Runs the handler on the oldest queued call and returns its Reply, remembering the
    /// result under the call's idempotency key.
    pub fn serve_next<H>(&mut self, handler: &mut H) -> Result<Option<ControlFrame>, SessionError>
    where
        H: FnMut(&TypedPayload) -> TypedPayload,
    {
        self.require(
            &[SessionState::Negotiated, SessionState::Closing],
            FrameKind::Reply,
        )?;
        let Some(call) = self.pending.pop_front() else {
            return Ok(None);
        };
        let reply = handler(&call.payload);
        let frame = ControlFrame::Reply {
            seq: call.seq,
            payload: reply.clone(),
        };
        // A reply the transport cannot encode is never cached: a retry of the call would
        // otherwise be answered with the same unsendable reply forever. The call is dropped
        // from the queue (the handler's fault), so the inbound window slot is not leaked.
        if let Err(error) = ensure_encodable(&frame) {
            self.close_if_drained();
            return Err(error);
        }
        if self.completed.len() >= self.config.completed_capacity {
            self.completed.pop_front();
        }
        self.completed.push_back(Completed {
            key: call.key,
            digest: call.digest,
            reply,
        });
        self.close_if_drained();
        Ok(Some(frame))
    }

    /// Applies one received frame.
    pub fn on_frame(&mut self, frame: ControlFrame) -> Result<Handled, SessionError> {
        let kind = frame.kind();
        let allowed: &[FrameKind] = match self.state {
            SessionState::Idle | SessionState::HelloSent => &[FrameKind::Hello],
            SessionState::Negotiated => &[
                FrameKind::Decl,
                FrameKind::Ack,
                FrameKind::Cancel,
                FrameKind::Bye,
                FrameKind::Call,
                FrameKind::Reply,
            ],
            SessionState::Closing => &[
                FrameKind::Ack,
                FrameKind::Cancel,
                FrameKind::Bye,
                FrameKind::Reply,
            ],
            SessionState::Closed => &[],
        };
        if !allowed.contains(&kind) {
            return Err(SessionError::IllegalTransition {
                state: self.state,
                frame: kind,
            });
        }
        match frame {
            ControlFrame::Hello {
                node_id,
                offer,
                nonce,
            } => self.on_hello(node_id, &offer, nonce),
            ControlFrame::Decl { capabilities } => {
                let count = capabilities.len();
                self.peer_capabilities = capabilities;
                Ok(Handled {
                    send: Vec::new(),
                    event: SessionEvent::PeerDeclared {
                        capabilities: count,
                    },
                })
            }
            ControlFrame::Call {
                seq,
                idempotency_key,
                payload,
            } => self.on_call(seq, idempotency_key, payload),
            ControlFrame::Reply { seq, payload } => {
                let digest = payload_digest(&payload);
                if self.outbound.remove(&seq).is_none() {
                    if self.answered.contains(&(seq, digest)) {
                        self.duplicate_replies = self.duplicate_replies.saturating_add(1);
                        return Ok(Handled {
                            send: Vec::new(),
                            event: SessionEvent::DuplicateIgnored { seq },
                        });
                    }
                    return Err(SessionError::UnknownSeq { seq });
                }
                if self.answered.len() >= self.config.completed_capacity {
                    self.answered.pop_front();
                }
                self.answered.push_back((seq, digest));
                self.close_if_drained();
                Ok(Handled {
                    send: Vec::new(),
                    event: SessionEvent::Replied { seq, payload },
                })
            }
            ControlFrame::Cancel { seq, .. } => {
                self.require_feature(Feature::Cancel)?;
                let Some(at) = self.pending.iter().position(|call| call.seq == seq) else {
                    return Ok(Handled {
                        send: Vec::new(),
                        event: SessionEvent::CancelIgnored { seq },
                    });
                };
                let Some(call) = self.pending.remove(at) else {
                    return Err(SessionError::UnknownSeq { seq });
                };
                self.close_if_drained();
                Ok(Handled {
                    send: vec![ControlFrame::Ack {
                        seq,
                        idempotency_key: call.key,
                    }],
                    event: SessionEvent::CallCancelled { seq },
                })
            }
            ControlFrame::Ack {
                seq,
                idempotency_key,
            } => {
                let call = self
                    .outbound
                    .get(&seq)
                    .ok_or(SessionError::UnknownSeq { seq })?;
                if !call.cancelling || call.key != idempotency_key {
                    return Err(SessionError::AckMismatch { seq });
                }
                self.outbound.remove(&seq);
                self.close_if_drained();
                Ok(Handled {
                    send: Vec::new(),
                    event: SessionEvent::CancelAcknowledged { seq },
                })
            }
            ControlFrame::Bye { reason } => {
                if self.bye_received {
                    return Err(SessionError::IllegalTransition {
                        state: self.state,
                        frame: FrameKind::Bye,
                    });
                }
                self.bye_received = true;
                self.state = SessionState::Closing;
                let mut send = Vec::new();
                if !self.bye_sent {
                    self.bye_sent = true;
                    send.push(ControlFrame::Bye {
                        reason: "peer closing".to_owned(),
                    });
                }
                self.close_if_drained();
                Ok(Handled {
                    send,
                    event: SessionEvent::PeerClosing { reason },
                })
            }
        }
    }

    fn on_hello(
        &mut self,
        node_id: String,
        offer: &VersionOffer,
        nonce: u64,
    ) -> Result<Handled, SessionError> {
        let responder = self.state == SessionState::Idle;
        if node_id == self.config.node_id {
            return Err(self.fail_handshake(SessionError::SelfConnect));
        }
        let negotiated = match self.config.matrix.negotiate(&self.config.offer, offer) {
            Ok(negotiated) => negotiated,
            Err(error) => return Err(self.fail_handshake(SessionError::Negotiation(error))),
        };
        // Answering needs this end's Hello and Decl on the wire; check before any state moves.
        self.ensure_handshake_encodable()?;
        self.negotiated = Some(negotiated);
        self.peer_node = Some(node_id);
        self.peer_nonce = Some(nonce);
        self.state = SessionState::Negotiated;
        let mut send = Vec::new();
        if responder {
            send.push(self.hello());
        }
        send.push(ControlFrame::Decl {
            capabilities: self.config.capabilities.clone(),
        });
        Ok(Handled {
            send,
            event: SessionEvent::Negotiated(negotiated),
        })
    }

    fn fail_handshake(&mut self, error: SessionError) -> SessionError {
        self.state = SessionState::Closed;
        self.farewell = Some(ControlFrame::Bye {
            reason: error.to_string(),
        });
        error
    }

    fn on_call(
        &mut self,
        seq: u64,
        key: IdempotencyKey,
        payload: TypedPayload,
    ) -> Result<Handled, SessionError> {
        self.require_feature(Feature::Idempotency)?;
        let digest = payload_digest(&payload);
        if let Some(at) = self.completed.iter().position(|done| done.key == key) {
            if self.completed[at].digest != digest {
                return Err(SessionError::KeyReused { key });
            }
            // Refresh recency, then answer without executing.
            let Some(done) = self.completed.remove(at) else {
                return Err(SessionError::UnknownSeq { seq });
            };
            let reply = done.reply.clone();
            self.completed.push_back(done);
            return Ok(Handled {
                send: vec![ControlFrame::Reply {
                    seq,
                    payload: reply,
                }],
                event: SessionEvent::DuplicateServed { seq },
            });
        }
        if let Some(queued) = self.pending.iter().find(|call| call.key == key) {
            if queued.digest != digest {
                return Err(SessionError::KeyReused { key });
            }
            if queued.seq != seq {
                return Err(SessionError::KeyInFlight {
                    key,
                    seq: queued.seq,
                });
            }
            return Ok(Handled {
                send: Vec::new(),
                event: SessionEvent::DuplicateIgnored { seq },
            });
        }
        if self.pending.iter().any(|call| call.seq == seq) {
            return Err(SessionError::DuplicateSeq { seq });
        }
        if self.pending.len() >= self.config.window {
            return Err(SessionError::InboundWindowExceeded {
                limit: self.config.window,
            });
        }
        self.pending.push_back(Inbound {
            seq,
            key,
            digest,
            payload,
        });
        Ok(Handled {
            send: Vec::new(),
            event: SessionEvent::CallQueued { seq },
        })
    }

    /// The two frames this end's config produces during the handshake must be encodable.
    fn ensure_handshake_encodable(&self) -> Result<(), SessionError> {
        ensure_encodable(&self.hello())?;
        ensure_encodable(&ControlFrame::Decl {
            capabilities: self.config.capabilities.clone(),
        })
    }

    fn hello(&self) -> ControlFrame {
        ControlFrame::Hello {
            node_id: self.config.node_id.clone(),
            offer: self.config.offer.clone(),
            nonce: self.config.nonce,
        }
    }

    fn require(&self, allowed: &[SessionState], frame: FrameKind) -> Result<(), SessionError> {
        if allowed.contains(&self.state) {
            Ok(())
        } else {
            Err(SessionError::IllegalTransition {
                state: self.state,
                frame,
            })
        }
    }

    fn require_feature(&self, feature: Feature) -> Result<(), SessionError> {
        match self.negotiated {
            Some(negotiated) if negotiated.features.contains(feature) => Ok(()),
            _ => Err(SessionError::FeatureNotNegotiated(feature)),
        }
    }

    /// `Closing -> Closed` once both Byes are out and, if draining, nothing is in flight.
    fn close_if_drained(&mut self) {
        if self.state != SessionState::Closing || !self.bye_sent || !self.bye_received {
            return;
        }
        let draining = self
            .negotiated
            .is_some_and(|negotiated| negotiated.features.contains(Feature::Drain));
        if draining && !(self.pending.is_empty() && self.outbound.is_empty()) {
            return;
        }
        self.pending.clear();
        self.outbound.clear();
        self.state = SessionState::Closed;
    }

    /// Receives every frame waiting on `transport`, applies it and sends the answers.
    /// Stops at the first session error, after sending any farewell.
    ///
    /// Applying a frame commits state (a cancelled call is gone, the Bye is marked sent, the
    /// session is Negotiated) before its answers are sent. An answer the transport refuses is
    /// therefore kept (at most [`MAX_UNSENT_FRAMES`]) and sent first on the next call, and
    /// nothing more is received until it is out, so the peer never has to retry a frame
    /// whose effect this end already took.
    pub fn pump<T: FrameTransport>(
        &mut self,
        transport: &mut T,
    ) -> Result<Vec<SessionEvent>, DriveError> {
        self.send_all(transport, Vec::new())?;
        let mut events = Vec::new();
        while let Some(frame) = transport.recv().map_err(DriveError::Transport)? {
            match self.on_frame(frame) {
                Ok(handled) => {
                    self.send_all(transport, handled.send)?;
                    events.push(handled.event);
                }
                Err(error) => {
                    if let Some(bye) = self.take_farewell() {
                        self.send_all(transport, vec![bye])?;
                    }
                    return Err(DriveError::Session(error));
                }
            }
        }
        Ok(events)
    }

    /// Sends the held frames, then `frames`; on a refusal keeps every frame not yet sent.
    fn send_all<T: FrameTransport>(
        &mut self,
        transport: &mut T,
        frames: Vec<ControlFrame>,
    ) -> Result<(), DriveError> {
        self.unsent.extend(frames);
        while let Some(frame) = self.unsent.front() {
            transport.send(frame).map_err(DriveError::Transport)?;
            self.unsent.pop_front();
        }
        Ok(())
    }

    /// Serves every queued call and sends the Replies; returns how many were served.
    pub fn serve_all<T, H>(
        &mut self,
        transport: &mut T,
        handler: &mut H,
    ) -> Result<usize, DriveError>
    where
        T: FrameTransport,
        H: FnMut(&TypedPayload) -> TypedPayload,
    {
        let mut served = 0;
        while let Some(reply) = self.serve_next(handler).map_err(DriveError::Session)? {
            transport.send(&reply).map_err(DriveError::Transport)?;
            served += 1;
        }
        Ok(served)
    }
}

/// Failure of [`ProtocolSession::pump`] or [`ProtocolSession::serve_all`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum DriveError {
    Session(SessionError),
    Transport(TransportError),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TransportError {
    /// The outgoing queue is at capacity.
    Full {
        capacity: usize,
    },
    Encode(CodecError),
    /// The queue held bytes that are not a valid frame.
    Decode(CodecError),
}

/// A way to move frames; the only implementation here is [`LoopbackTransport`].
pub trait FrameTransport {
    fn send(&mut self, frame: &ControlFrame) -> Result<(), TransportError>;
    fn recv(&mut self) -> Result<Option<ControlFrame>, TransportError>;
}

type Queue = Arc<Mutex<VecDeque<Vec<u8>>>>;

/// One end of an in-memory duplex link: two bounded queues of encoded frames.
#[derive(Debug)]
pub struct LoopbackTransport {
    outgoing: Queue,
    incoming: Queue,
    capacity: usize,
}

impl LoopbackTransport {
    /// Two connected ends; each direction holds at most `capacity` frames, with `capacity`
    /// clamped to [`MAX_TRANSPORT_CAPACITY`].
    pub fn pair(capacity: usize) -> (Self, Self) {
        let capacity = capacity.min(MAX_TRANSPORT_CAPACITY);
        let a_to_b: Queue = Arc::default();
        let b_to_a: Queue = Arc::default();
        (
            Self {
                outgoing: Arc::clone(&a_to_b),
                incoming: Arc::clone(&b_to_a),
                capacity,
            },
            Self {
                outgoing: b_to_a,
                incoming: a_to_b,
                capacity,
            },
        )
    }

    /// Frames waiting to be received at this end.
    pub fn inbox_len(&self) -> usize {
        self.incoming
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .len()
    }

    /// Puts raw bytes on this end's outgoing queue (for tests that inject garbage).
    pub fn send_raw(&mut self, bytes: Vec<u8>) -> Result<(), TransportError> {
        let mut queue = self.outgoing.lock().unwrap_or_else(PoisonError::into_inner);
        if queue.len() >= self.capacity {
            return Err(TransportError::Full {
                capacity: self.capacity,
            });
        }
        queue.push_back(bytes);
        Ok(())
    }
}

impl FrameTransport for LoopbackTransport {
    fn send(&mut self, frame: &ControlFrame) -> Result<(), TransportError> {
        let bytes = encode(frame).map_err(TransportError::Encode)?;
        self.send_raw(bytes)
    }

    fn recv(&mut self) -> Result<Option<ControlFrame>, TransportError> {
        let bytes = self
            .incoming
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .pop_front();
        bytes
            .map(|bytes| decode(&bytes).map_err(TransportError::Decode))
            .transpose()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::negotiation::{FeatureSet, ProtocolVersion};
    use ptr_types::TypeId;

    fn offer(minor: u16) -> VersionOffer {
        VersionOffer {
            supported: vec![ProtocolVersion::new(1, minor, 0)],
            minimum: ProtocolVersion::new(1, 0, 0),
            features: FeatureSet::all(),
            required: FeatureSet::of(&[Feature::Idempotency]),
        }
    }

    fn session(name: &str, minor: u16) -> ProtocolSession {
        ProtocolSession::new(SessionConfig::new(name, offer(minor), 7)).unwrap()
    }

    fn payload(bytes: &[u8]) -> TypedPayload {
        TypedPayload {
            type_id: TypeId::from("T"),
            bytes: bytes.to_vec(),
        }
    }

    fn key(text: &str) -> IdempotencyKey {
        IdempotencyKey::new(text).unwrap()
    }

    /// Hands each frame to `b` directly, returning what `b` sends back.
    fn deliver(b: &mut ProtocolSession, frames: Vec<ControlFrame>) -> Vec<ControlFrame> {
        let mut out = Vec::new();
        for frame in frames {
            out.extend(b.on_frame(frame).unwrap().send);
        }
        out
    }

    fn connected(minor_a: u16, minor_b: u16) -> (ProtocolSession, ProtocolSession) {
        let mut a = session("a", minor_a);
        let mut b = session("b", minor_b);
        let hello = a.start().unwrap();
        let back = deliver(&mut b, vec![hello]);
        let tail = deliver(&mut a, back);
        deliver(&mut b, tail);
        assert_eq!(a.state(), SessionState::Negotiated);
        assert_eq!(b.state(), SessionState::Negotiated);
        (a, b)
    }

    fn echo(p: &TypedPayload) -> TypedPayload {
        p.clone()
    }

    #[test]
    fn zero_bounds_are_refused() {
        let mut config = SessionConfig::new("a", offer(1), 1);
        config.window = 0;
        assert_eq!(
            ProtocolSession::new(config).unwrap_err(),
            SessionError::ZeroWindow
        );
        let mut config = SessionConfig::new("a", offer(1), 1);
        config.completed_capacity = 0;
        assert_eq!(
            ProtocolSession::new(config).unwrap_err(),
            SessionError::ZeroCompletedCapacity
        );
    }

    #[test]
    fn oversize_bounds_are_refused_and_the_maximum_is_accepted() {
        for window in [usize::MAX, MAX_WINDOW + 1] {
            let mut config = SessionConfig::new("a", offer(1), 1);
            config.window = window;
            assert_eq!(
                ProtocolSession::new(config).unwrap_err(),
                SessionError::WindowTooLarge {
                    requested: window,
                    max: MAX_WINDOW
                }
            );
        }
        for cap in [usize::MAX, MAX_COMPLETED_CAPACITY + 1] {
            let mut config = SessionConfig::new("a", offer(1), 1);
            config.completed_capacity = cap;
            assert_eq!(
                ProtocolSession::new(config).unwrap_err(),
                SessionError::CompletedCapacityTooLarge {
                    requested: cap,
                    max: MAX_COMPLETED_CAPACITY
                }
            );
        }
        let mut config = SessionConfig::new("a", offer(1), 1);
        config.window = MAX_WINDOW;
        config.completed_capacity = MAX_COMPLETED_CAPACITY;
        assert!(ProtocolSession::new(config).is_ok());
    }

    #[test]
    fn transport_capacity_is_clamped() {
        let (mut left, _right) = LoopbackTransport::pair(usize::MAX);
        let bye = ControlFrame::Bye { reason: "x".into() };
        for _ in 0..MAX_TRANSPORT_CAPACITY {
            left.send(&bye).unwrap();
        }
        assert_eq!(
            left.send(&bye),
            Err(TransportError::Full {
                capacity: MAX_TRANSPORT_CAPACITY
            })
        );
    }

    #[test]
    fn frames_are_illegal_outside_their_state() {
        let mut idle = session("a", 1);
        assert_eq!(
            idle.call(key("k"), payload(b"x")).unwrap_err(),
            SessionError::IllegalTransition {
                state: SessionState::Idle,
                frame: FrameKind::Call
            }
        );
        assert_eq!(
            idle.on_frame(ControlFrame::Bye { reason: "x".into() })
                .unwrap_err(),
            SessionError::IllegalTransition {
                state: SessionState::Idle,
                frame: FrameKind::Bye
            }
        );
        assert!(idle.close("x").is_err());
        idle.start().unwrap();
        assert_eq!(
            idle.start().unwrap_err(),
            SessionError::IllegalTransition {
                state: SessionState::HelloSent,
                frame: FrameKind::Hello
            }
        );

        let (mut a, _b) = connected(1, 1);
        let hello = a.hello();
        assert_eq!(
            a.on_frame(hello).unwrap_err(),
            SessionError::IllegalTransition {
                state: SessionState::Negotiated,
                frame: FrameKind::Hello
            }
        );
    }

    #[test]
    fn negotiation_failure_closes_and_leaves_a_farewell() {
        let mut a = session("a", 1);
        let mut b = ProtocolSession::new(SessionConfig::new(
            "b",
            VersionOffer {
                supported: vec![ProtocolVersion::new(2, 0, 0)],
                minimum: ProtocolVersion::new(2, 0, 0),
                features: FeatureSet::EMPTY,
                required: FeatureSet::EMPTY,
            },
            1,
        ))
        .unwrap();
        let hello = a.start().unwrap();
        let error = b.on_frame(hello).unwrap_err();
        assert!(matches!(
            error,
            SessionError::Negotiation(NegotiationError::MajorMismatch { .. })
        ));
        assert_eq!(b.state(), SessionState::Closed);
        assert!(matches!(b.take_farewell(), Some(ControlFrame::Bye { .. })));
        assert!(b.take_farewell().is_none());
    }

    #[test]
    fn connecting_to_yourself_is_refused() {
        let mut a = session("same", 1);
        let mut b = session("same", 1);
        let hello = a.start().unwrap();
        assert_eq!(b.on_frame(hello).unwrap_err(), SessionError::SelfConnect);
    }

    #[test]
    fn duplicate_call_is_served_from_cache_without_executing() {
        let (mut a, mut b) = connected(1, 1);
        let call = a.call(key("k"), payload(b"x")).unwrap();
        b.on_frame(call.clone()).unwrap();
        let mut runs = 0;
        let mut handler = |p: &TypedPayload| {
            runs += 1;
            p.clone()
        };
        let reply = b.serve_next(&mut handler).unwrap().unwrap();
        let again = b.on_frame(call).unwrap();
        assert_eq!(again.event, SessionEvent::DuplicateServed { seq: 1 });
        assert_eq!(again.send, vec![reply]);
        assert_eq!(runs, 1);
    }

    #[test]
    fn key_reused_with_another_payload_is_rejected_when_completed_and_queued() {
        let (mut a, mut b) = connected(1, 1);
        let first = a.call(key("k"), payload(b"x")).unwrap();
        b.on_frame(first).unwrap();
        let clash = ControlFrame::Call {
            seq: 2,
            idempotency_key: key("k"),
            payload: payload(b"different"),
        };
        assert_eq!(
            b.on_frame(clash.clone()).unwrap_err(),
            SessionError::KeyReused { key: key("k") }
        );
        b.serve_next(&mut echo).unwrap();
        assert_eq!(
            b.on_frame(clash).unwrap_err(),
            SessionError::KeyReused { key: key("k") }
        );
    }

    #[test]
    fn same_key_queued_under_another_seq_and_duplicate_seq_are_typed() {
        let (_a, mut b) = connected(1, 1);
        let call = |seq, k: &str, bytes: &[u8]| ControlFrame::Call {
            seq,
            idempotency_key: key(k),
            payload: payload(bytes),
        };
        b.on_frame(call(1, "k", b"x")).unwrap();
        assert_eq!(
            b.on_frame(call(1, "k", b"x")).unwrap().event,
            SessionEvent::DuplicateIgnored { seq: 1 }
        );
        assert_eq!(
            b.on_frame(call(2, "k", b"x")).unwrap_err(),
            SessionError::KeyInFlight {
                key: key("k"),
                seq: 1
            }
        );
        assert_eq!(
            b.on_frame(call(1, "other", b"y")).unwrap_err(),
            SessionError::DuplicateSeq { seq: 1 }
        );
    }

    #[test]
    fn windows_bound_both_directions() {
        let mut config_a = SessionConfig::new("a", offer(1), 1);
        config_a.window = 2;
        let mut config_b = SessionConfig::new("b", offer(1), 1);
        config_b.window = 2;
        let mut a = ProtocolSession::new(config_a).unwrap();
        let mut b = ProtocolSession::new(config_b).unwrap();
        let hello = a.start().unwrap();
        let back = deliver(&mut b, vec![hello]);
        deliver(&mut a, back);

        let c1 = a.call(key("1"), payload(b"1")).unwrap();
        let c2 = a.call(key("2"), payload(b"2")).unwrap();
        assert_eq!(
            a.call(key("3"), payload(b"3")).unwrap_err(),
            SessionError::WindowFull { limit: 2 }
        );
        b.on_frame(c1).unwrap();
        b.on_frame(c2).unwrap();
        let overflow = ControlFrame::Call {
            seq: 3,
            idempotency_key: key("3"),
            payload: payload(b"3"),
        };
        assert_eq!(
            b.on_frame(overflow).unwrap_err(),
            SessionError::InboundWindowExceeded { limit: 2 }
        );
        assert_eq!(b.queued_calls(), 2);
    }

    #[test]
    fn answered_record_is_bounded_and_forgets_oldest() {
        let mut config = SessionConfig::new("a", offer(1), 7);
        config.completed_capacity = 2;
        let mut a = ProtocolSession::new(config).unwrap();
        let mut b = session("b", 1);
        let hello = a.start().unwrap();
        let back = deliver(&mut b, vec![hello]);
        let tail = deliver(&mut a, back);
        deliver(&mut b, tail);
        for n in 0..3u8 {
            let call = a.call(key(&format!("k{n}")), payload(&[n])).unwrap();
            let reply = deliver(&mut b, vec![call]);
            assert!(reply.is_empty());
            let reply = b.serve_next(&mut echo).unwrap().unwrap();
            a.on_frame(reply).unwrap();
        }
        let again = |seq: u64, n: u8| ControlFrame::Reply {
            seq,
            payload: payload(&[n]),
        };
        // Seqs 2 and 3 are remembered; seq 1 was evicted and is unknown again.
        assert_eq!(
            a.on_frame(again(3, 2)).unwrap().event,
            SessionEvent::DuplicateIgnored { seq: 3 }
        );
        assert_eq!(
            a.on_frame(again(1, 0)).unwrap_err(),
            SessionError::UnknownSeq { seq: 1 }
        );
        assert_eq!(a.duplicate_replies_ignored(), 1);
    }

    #[test]
    fn completed_cache_is_lru_bounded() {
        let mut config = SessionConfig::new("b", offer(1), 1);
        config.completed_capacity = 2;
        let mut b = ProtocolSession::new(config).unwrap();
        let mut a = session("a", 1);
        let hello = a.start().unwrap();
        let back = b.on_frame(hello).unwrap().send;
        deliver(&mut a, back);

        let run = |b: &mut ProtocolSession, seq, k: &str| {
            b.on_frame(ControlFrame::Call {
                seq,
                idempotency_key: key(k),
                payload: payload(k.as_bytes()),
            })
            .unwrap();
            b.serve_next(&mut echo).unwrap();
        };
        run(&mut b, 1, "k1");
        run(&mut b, 2, "k2");
        // Touch k1 so k2 becomes least recently used, then add k3.
        let touch = b
            .on_frame(ControlFrame::Call {
                seq: 3,
                idempotency_key: key("k1"),
                payload: payload(b"k1"),
            })
            .unwrap();
        assert_eq!(touch.event, SessionEvent::DuplicateServed { seq: 3 });
        run(&mut b, 4, "k3");
        assert_eq!(b.completed_keys(), 2);
        // k1 survived, k2 was evicted: a retry of k2 queues again.
        let k1 = b
            .on_frame(ControlFrame::Call {
                seq: 5,
                idempotency_key: key("k1"),
                payload: payload(b"k1"),
            })
            .unwrap();
        assert_eq!(k1.event, SessionEvent::DuplicateServed { seq: 5 });
        let k2 = b
            .on_frame(ControlFrame::Call {
                seq: 6,
                idempotency_key: key("k2"),
                payload: payload(b"k2"),
            })
            .unwrap();
        assert_eq!(k2.event, SessionEvent::CallQueued { seq: 6 });
    }

    #[test]
    fn cancel_removes_a_queued_call_and_late_cancel_is_ignored() {
        let (mut a, mut b) = connected(1, 1);
        let call = a.call(key("k"), payload(b"x")).unwrap();
        b.on_frame(call).unwrap();
        let cancel = a.cancel(1, "no longer needed").unwrap();
        let handled = b.on_frame(cancel.clone()).unwrap();
        assert_eq!(handled.event, SessionEvent::CallCancelled { seq: 1 });
        assert_eq!(b.queued_calls(), 0);
        assert!(b.serve_next(&mut echo).unwrap().is_none());
        let acked = a.on_frame(handled.send[0].clone()).unwrap();
        assert_eq!(acked.event, SessionEvent::CancelAcknowledged { seq: 1 });
        assert_eq!(a.outstanding_calls(), 0);
        // The same Cancel again finds nothing queued.
        assert_eq!(
            b.on_frame(cancel).unwrap().event,
            SessionEvent::CancelIgnored { seq: 1 }
        );
        // Cancelling an unknown call locally is typed.
        assert_eq!(
            a.cancel(42, "x").unwrap_err(),
            SessionError::UnknownSeq { seq: 42 }
        );
    }

    #[test]
    fn unsolicited_ack_and_unknown_reply_are_typed() {
        let (mut a, _b) = connected(1, 1);
        a.call(key("k"), payload(b"x")).unwrap();
        let ack = ControlFrame::Ack {
            seq: 1,
            idempotency_key: key("k"),
        };
        assert_eq!(
            a.on_frame(ack).unwrap_err(),
            SessionError::AckMismatch { seq: 1 }
        );
        assert_eq!(
            a.on_frame(ControlFrame::Reply {
                seq: 9,
                payload: payload(b"")
            })
            .unwrap_err(),
            SessionError::UnknownSeq { seq: 9 }
        );
        a.cancel(1, "x").unwrap();
        let wrong_key = ControlFrame::Ack {
            seq: 1,
            idempotency_key: key("not-k"),
        };
        assert_eq!(
            a.on_frame(wrong_key).unwrap_err(),
            SessionError::AckMismatch { seq: 1 }
        );
    }

    #[test]
    fn features_gate_cancel_and_drain_follows_version() {
        // Version 1.0 on one side: drain is not usable, cancel still is.
        let (mut a, mut b) = connected(1, 0);
        let features = a.negotiated().unwrap().features;
        assert!(features.contains(Feature::Cancel));
        assert!(!features.contains(Feature::Drain));

        // Without drain, Byes close at once even with a call still queued.
        let call = a.call(key("k"), payload(b"x")).unwrap();
        b.on_frame(call).unwrap();
        let bye = a.close("done").unwrap();
        let answer = b.on_frame(bye).unwrap();
        assert_eq!(b.state(), SessionState::Closed);
        assert_eq!(b.queued_calls(), 0);
        a.on_frame(answer.send[0].clone()).unwrap();
        assert_eq!(a.state(), SessionState::Closed);
    }

    #[test]
    fn missing_cancel_feature_is_typed() {
        let no_cancel = VersionOffer {
            supported: vec![ProtocolVersion::new(1, 1, 0)],
            minimum: ProtocolVersion::new(1, 0, 0),
            features: FeatureSet::of(&[Feature::Idempotency]),
            required: FeatureSet::EMPTY,
        };
        let mut a = ProtocolSession::new(SessionConfig::new("a", no_cancel, 1)).unwrap();
        let mut b = session("b", 1);
        let hello = a.start().unwrap();
        let back = deliver(&mut b, vec![hello]);
        deliver(&mut a, back);
        a.call(key("k"), payload(b"x")).unwrap();
        assert_eq!(
            a.cancel(1, "x").unwrap_err(),
            SessionError::FeatureNotNegotiated(Feature::Cancel)
        );
    }

    #[test]
    fn bye_drains_in_flight_work_before_closing() {
        let (mut a, mut b) = connected(1, 1);
        assert!(a.negotiated().unwrap().features.contains(Feature::Drain));
        let call = a.call(key("k"), payload(b"x")).unwrap();
        b.on_frame(call).unwrap();
        let bye = a.close("done").unwrap();
        assert_eq!(a.state(), SessionState::Closing);
        // New calls are refused while closing, on both ends.
        assert!(a.call(key("late"), payload(b"y")).is_err());
        let answer = b.on_frame(bye).unwrap();
        assert_eq!(b.state(), SessionState::Closing);
        assert_eq!(
            b.on_frame(ControlFrame::Call {
                seq: 2,
                idempotency_key: key("late"),
                payload: payload(b"y"),
            })
            .unwrap_err(),
            SessionError::IllegalTransition {
                state: SessionState::Closing,
                frame: FrameKind::Call
            }
        );
        // b's Bye reaches a, but a still waits for its reply.
        a.on_frame(answer.send[0].clone()).unwrap();
        assert_eq!(a.state(), SessionState::Closing);
        let reply = b.serve_next(&mut echo).unwrap().unwrap();
        assert_eq!(b.state(), SessionState::Closed);
        a.on_frame(reply).unwrap();
        assert_eq!(a.state(), SessionState::Closed);
        // A second Bye, or anything else, after Closed is illegal.
        assert!(matches!(
            a.on_frame(ControlFrame::Bye { reason: "x".into() }),
            Err(SessionError::IllegalTransition {
                state: SessionState::Closed,
                ..
            })
        ));
    }

    #[test]
    fn loopback_is_bounded_and_reports_garbage() {
        let (mut left, mut right) = LoopbackTransport::pair(2);
        let bye = ControlFrame::Bye { reason: "x".into() };
        left.send(&bye).unwrap();
        left.send(&bye).unwrap();
        assert_eq!(left.send(&bye), Err(TransportError::Full { capacity: 2 }));
        assert_eq!(right.inbox_len(), 2);
        assert_eq!(right.recv().unwrap(), Some(bye.clone()));
        assert_eq!(right.recv().unwrap(), Some(bye));
        assert_eq!(right.recv().unwrap(), None);
        left.send_raw(vec![1, 2, 3]).unwrap();
        assert_eq!(
            right.recv(),
            Err(TransportError::Decode(CodecError::Truncated))
        );
    }
}
