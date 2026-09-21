//! An execution gateway and a requester, both over `ALPN_EXEC`.
//!
//! Neither of these is a daemon. `serve_once` accepts one connection and answers
//! it; `request` sends one request and checks one receipt. A caller decides when
//! either happens, which is what keeps the tests free of sleeps and keeps "how long
//! do we wait" and "how often do we accept" where they can be decided deliberately.
use crate::frame::{
    decode_receipt, decode_request, encode_receipt, encode_request, request_digest, FrameError,
    RefusalCode, WireOutcome, WireReceipt, WireRequest, MAX_BODY_BYTES, MAX_FRAME_BYTES,
};
use ptr_net::{EndpointAddr, IrohTransport, NodeIdentity, ALPN_EXEC};
use ptr_runtime::execution::{ExecutionError, ExecutionSession};
use ptr_runtime::PtrRuntime;
use ptr_types::NodeId;
use std::collections::{BTreeMap, VecDeque};
use std::sync::Mutex;
use std::time::Duration;

/// How long one exchange waits for an answer before giving up.
///
/// A default rather than a rule: the right deadline depends on the deployment, and
/// what matters is that there is one. Without it a host that stopped answering holds
/// a requester open for as long as the transport allows.
pub const DEFAULT_REQUEST_TIMEOUT: Duration = Duration::from_secs(5);

/// How long a permit prepared for a wire request stays valid.
///
/// Short on purpose: the permit exists for the length of one request, and a permit
/// that outlived the connection that caused it would be authority nobody is holding.
pub const DEFAULT_PERMIT_TTL: Duration = Duration::from_secs(30);

/// How many recent request ids one peer's replay window holds.
pub const MAX_REPLAY_WINDOW: usize = 256;

/// How many peers a host keeps a window for.
///
/// The window only ever holds peers the policy admitted, so this is a bound on a
/// bound rather than the mechanism. What happens when it is reached is written down
/// in [`ExecutionHost::serve_once`] rather than left to be discovered.
pub const MAX_TRACKED_PEERS: usize = 1024;

/// Why a wire operation was refused, at the resolution the *local* side gets.
///
/// This is deliberately richer than what crosses the wire: a host's operator is
/// entitled to know which authority refused a request, and the peer is not.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum WireError {
    /// The transport refused, failed, or did not answer in time.
    Transport(String),
    /// The frame was malformed.
    Frame(FrameError),
    /// The request named another endpoint's key.
    Misaddressed {
        addressed_to: String,
        expected: String,
    },
    /// This request id is already in this peer's window.
    Replayed { request_id: u64 },
    /// The runtime refused. The reason stays here and does not cross the wire.
    Runtime(ExecutionError),
    /// A receipt whose author is not the peer the connection authenticated.
    ///
    /// The case this layer exists for on the answering side: a receipt is a claim
    /// that can outlive its connection, so one that names somebody else is refused
    /// rather than filed.
    ForgedReceipt {
        authenticated: String,
        claimed: String,
    },
    /// A receipt answering a different request id.
    WrongRequest { expected: u64, answered: u64 },
    /// A receipt whose digest is not the request that was sent. A well-formed answer
    /// to a question nobody asked here.
    UnboundReceipt,
}

impl WireError {
    /// Stable diagnostic code for this refusal.
    pub fn code(&self) -> &'static str {
        match self {
            Self::Transport(_) => "PTR_EXECW_TRANSPORT",
            Self::Frame(_) => "PTR_EXECW_FRAME",
            Self::Misaddressed { .. } => "PTR_EXECW_MISADDRESSED",
            Self::Replayed { .. } => "PTR_EXECW_REPLAYED",
            Self::Runtime(_) => "PTR_EXECW_REFUSED",
            Self::ForgedReceipt { .. } => "PTR_EXECW_FORGED_RECEIPT",
            Self::WrongRequest { .. } => "PTR_EXECW_WRONG_REQUEST",
            Self::UnboundReceipt => "PTR_EXECW_UNBOUND_RECEIPT",
        }
    }
}

impl From<FrameError> for WireError {
    /// Carry a framing refusal through unchanged.
    fn from(error: FrameError) -> Self {
        Self::Frame(error)
    }
}

impl std::fmt::Display for WireError {
    /// Render the stable refusal code.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.code())
    }
}

impl std::error::Error for WireError {}

/// What one served request did, for the host's own caller.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Serviced {
    /// The peer the connection authenticated. Not a value from the payload — there
    /// is no such value.
    pub peer: NodeIdentity,
    /// The request id, when the frame decoded far enough to have one.
    pub request_id: Option<u64>,
    /// What the receipt told the peer.
    pub reported: WireOutcome,
    /// The local reason behind a refusal, which the receipt does not carry.
    pub refused: Option<WireError>,
}

/// Which request ids a peer has already spent.
#[derive(Default)]
struct PeerWindow {
    ids: VecDeque<u64>,
    last_used: u64,
}

/// A bounded per-peer replay window.
///
/// Bounded and in memory, which is exactly as much as it claims: it refuses a frame
/// sent twice cheaply, it does not survive a restart, and an id that has left the
/// window can be spent again. What makes a retry apply once is the runtime's
/// at-most-once key, recorded in the ledger. Two mechanisms, and the cheap one never
/// stands in for the durable one.
#[derive(Default)]
struct ReplayWindow {
    peers: BTreeMap<String, PeerWindow>,
    clock: u64,
}

impl ReplayWindow {
    /// Record `request_id` for `peer`, or report that it was already spent.
    fn spend(&mut self, peer: &str, request_id: u64) -> bool {
        self.clock += 1;
        let clock = self.clock;
        if !self.peers.contains_key(peer) && self.peers.len() >= MAX_TRACKED_PEERS {
            // Drop the least recently used peer's window rather than refusing a
            // request from an admitted peer. Availability is the thing a host owes;
            // what is lost is the cheap half of replay protection for whoever has
            // been quiet longest, and the durable half is the at-most-once key.
            if let Some(stalest) = self
                .peers
                .iter()
                .min_by_key(|(_, window)| window.last_used)
                .map(|(key, _)| key.clone())
            {
                self.peers.remove(&stalest);
            }
        }
        let window = self.peers.entry(peer.to_owned()).or_default();
        window.last_used = clock;
        if window.ids.contains(&request_id) {
            return false;
        }
        if window.ids.len() >= MAX_REPLAY_WINDOW {
            window.ids.pop_front();
        }
        window.ids.push_back(request_id);
        true
    }
}

/// A runtime reachable over `ALPN_EXEC`.
///
/// The host owns the runtime because a request must not be able to reach one through
/// anything but this path. What a peer may ask for is the runtime's admission policy,
/// installed by the host before any of this, and re-derived on every request.
pub struct ExecutionHost {
    runtime: Mutex<PtrRuntime>,
    transport: IrohTransport,
    window: Mutex<ReplayWindow>,
    permit_ttl: Duration,
}

impl ExecutionHost {
    /// Bind an endpoint for `ALPN_EXEC` around a runtime the host has already
    /// configured, admission policy included.
    pub async fn bind(runtime: PtrRuntime) -> Result<Self, WireError> {
        let transport = IrohTransport::bind(&[ALPN_EXEC])
            .await
            .map_err(WireError::Transport)?;
        Ok(Self {
            runtime: Mutex::new(runtime),
            transport,
            window: Mutex::new(ReplayWindow::default()),
            permit_ttl: DEFAULT_PERMIT_TTL,
        })
    }

    /// The identity a requester must address.
    pub fn identity(&self) -> NodeIdentity {
        self.transport.identity()
    }

    /// Where requesters should send.
    pub fn address(&self) -> EndpointAddr {
        self.transport.direct_addr()
    }

    /// How long a permit prepared for one request stays valid.
    pub fn with_permit_ttl(&mut self, ttl: Duration) {
        self.permit_ttl = ttl;
    }

    /// The runtime, for the host's own trusted use.
    ///
    /// Panics if a previous holder panicked: a runtime whose commit sequence was
    /// interrupted is not a runtime to keep serving from.
    pub fn runtime(&self) -> std::sync::MutexGuard<'_, PtrRuntime> {
        self.runtime.lock().expect("runtime lock is not poisoned")
    }

    /// Accept one connection, serve what it carries, and answer with a receipt.
    ///
    /// Every request is answered, including every refused one: leaving a peer hanging
    /// makes this side's refusal the peer's problem. The receipt is bound to the bytes
    /// that actually arrived, so even a request this build cannot parse gets an answer
    /// its sender can check against what it sent.
    ///
    /// The order below is the protocol, and it is an order rather than a set:
    ///
    /// 1. The peer is the **authenticated connection**. Nothing in the payload names
    ///    a sender, and nothing here reads one.
    /// 2. The frame must decode. A refusal at this point spends nothing: the bytes
    ///    were never this runtime's request.
    /// 3. The request must be addressed to this endpoint's key. A request legitimate
    ///    at another runtime is not legitimate here.
    /// 4. The peer must be admitted by policy — as a peer, not as an authority for
    ///    anything in particular. `admit_peer` names no principal and picks no grant;
    ///    both come from the policy.
    /// 5. Only now is the request id spent, so every admitted peer's requests are
    ///    treated alike whatever the runtime then decides. An admitted peer can learn
    ///    from a replay that its own key is admitted here, which is a fact about
    ///    itself; the alternative is a window anyone can fill.
    /// 6. The runtime prepares and dispatches. Every refusal from here on crosses the
    ///    wire as one code, so a requester cannot tell a grant it does not have from a
    ///    verification that failed.
    /// 7. The session is revoked, because it existed for one request.
    pub async fn serve_once(&self) -> Result<Serviced, WireError> {
        let incoming = self
            .transport
            .accept_once(MAX_FRAME_BYTES)
            .await
            .map_err(WireError::Transport)?;

        // (1) The sender is the connection. There is no payload field to prefer over
        // it, which is the shape of the format rather than a check performed here.
        let peer = incoming.peer.clone();
        let digest = request_digest(&incoming.payload);
        let (request_id, outcome, refused) = self.decide(&peer, &incoming.payload);

        let receipt = WireReceipt {
            responder: self.identity().public_key,
            request_id: request_id.unwrap_or(0),
            request_digest: digest,
            outcome: outcome.clone(),
        };
        let frame = encode_receipt(&receipt)?;
        incoming
            .respond(&frame)
            .await
            .map_err(WireError::Transport)?;

        Ok(Serviced {
            peer,
            request_id,
            reported: outcome,
            refused,
        })
    }

    /// Everything between the bytes arriving and the receipt going out.
    ///
    /// Split out so that answering the peer is unconditional: a refusal decided here
    /// still becomes a receipt rather than a dropped connection.
    fn decide(
        &self,
        peer: &NodeIdentity,
        payload: &[u8],
    ) -> (Option<u64>, WireOutcome, Option<WireError>) {
        // (2) The frame must decode.
        let request = match decode_request(payload) {
            Ok(request) => request,
            Err(error) => {
                return (
                    None,
                    WireOutcome::Refused {
                        code: RefusalCode::Malformed,
                    },
                    Some(WireError::Frame(error)),
                )
            }
        };
        let request_id = Some(request.request_id);

        // (3) Addressed here, or nowhere.
        let expected = self.identity().public_key;
        if request.addressed_to != expected {
            return (
                request_id,
                WireOutcome::Refused {
                    code: RefusalCode::Misaddressed,
                },
                Some(WireError::Misaddressed {
                    addressed_to: request.addressed_to,
                    expected,
                }),
            );
        }

        let mut runtime = self.runtime();

        // (4) Admitted as a peer. The principal and the grants are the policy's.
        let session = match runtime.admit_peer(&NodeId(peer.public_key.clone())) {
            Ok(session) => session,
            Err(error) => {
                return (
                    request_id,
                    refused_by_runtime(&error),
                    Some(WireError::Runtime(error)),
                )
            }
        };

        // (5) The request id is spent by every admitted peer's request alike.
        let fresh = self
            .window
            .lock()
            .expect("replay window lock is not poisoned")
            .spend(&peer.public_key, request.request_id);
        if !fresh {
            let _ = runtime.revoke_execution_session(&session);
            return (
                request_id,
                WireOutcome::Refused {
                    code: RefusalCode::Replayed,
                },
                Some(WireError::Replayed {
                    request_id: request.request_id,
                }),
            );
        }

        // (6) The runtime decides. Every refusal from here is one code on the wire.
        let outcome = self.dispatch(&mut runtime, &session, &request);

        // (7) One request, one session.
        let _ = runtime.revoke_execution_session(&session);
        match outcome {
            Ok(reported) => (request_id, reported, None),
            Err(error) => {
                let reported = refused_by_runtime(&error);
                (request_id, reported, Some(WireError::Runtime(error)))
            }
        }
    }

    /// Prepare and dispatch one request.
    fn dispatch(
        &self,
        runtime: &mut PtrRuntime,
        session: &ExecutionSession,
        request: &WireRequest,
    ) -> Result<WireOutcome, ExecutionError> {
        let permit = match &request.once_key {
            Some(key) => runtime.prepare_execution_once(
                session,
                &request.project,
                &request.action,
                self.permit_ttl,
                key.clone(),
            )?,
            None => runtime.prepare_execution(
                session,
                &request.project,
                &request.action,
                self.permit_ttl,
            )?,
        };
        let response = runtime.execute_prepared(session, permit)?;
        // An answer that does not fit in a frame is not a refusal: the effect
        // applied. Saying "refused" here would invite a retry of something that
        // already happened.
        if response.len() > MAX_BODY_BYTES {
            return Ok(WireOutcome::AppliedWithoutResponse);
        }
        Ok(WireOutcome::Applied { response })
    }

    /// Close the endpoint.
    pub async fn close(&self) {
        self.transport.close().await;
    }
}

/// Map a runtime refusal onto what a requester is told.
///
/// Matched **exhaustively** on purpose. A catch-all would quietly report a refusal a
/// later build invents as "nothing was attempted", and if that new refusal happened
/// to mean the opposite the requester would retry an effect that had already applied.
/// Adding a variant to [`ExecutionError`] must break this build instead.
fn refused_by_runtime(error: &ExecutionError) -> WireOutcome {
    match error {
        // The effect may have applied and this runtime does not know. The fence is
        // standing on the host; telling the requester "refused" would throw it away
        // at the last step.
        ExecutionError::Executor(_) => WireOutcome::Uncertain,
        // It applied under this key and the response is gone.
        ExecutionError::ResponseNotRetained { .. } => WireOutcome::AppliedWithoutResponse,
        // Everything else: nothing was attempted, and the reason stays here.
        ExecutionError::InvalidSession
        | ExecutionError::InvalidGrant
        | ExecutionError::DuplicateScope
        | ExecutionError::CapacityExceeded
        | ExecutionError::CounterExhausted
        | ExecutionError::InvalidTtl
        | ExecutionError::ForeignRuntime
        | ExecutionError::SessionClosed
        | ExecutionError::SessionMismatch
        | ExecutionError::Expired
        | ExecutionError::StalePermit
        | ExecutionError::ScopeDenied
        | ExecutionError::ProjectMismatch
        | ExecutionError::RuntimeFenced
        | ExecutionError::AmbiguousOutcome { .. }
        | ExecutionError::UnknownAttempt { .. }
        | ExecutionError::InvalidKey
        | ExecutionError::InvalidEvidence
        | ExecutionError::UnknownPeer { .. }
        | ExecutionError::PeerNotAdmitted { .. }
        | ExecutionError::DuplicatePeerEntry { .. }
        | ExecutionError::DispatchMismatch { .. }
        | ExecutionError::NotDetached { .. }
        | ExecutionError::Audit(_)
        | ExecutionError::AuthorizationDenied(_)
        | ExecutionError::VerificationRejected { .. }
        | ExecutionError::HardFinding => WireOutcome::Refused {
            code: RefusalCode::Runtime,
        },
    }
}

/// A requester with an authenticated endpoint.
pub struct ExecutionClient {
    transport: IrohTransport,
    request_timeout: Duration,
}

impl ExecutionClient {
    /// Bind an endpoint for `ALPN_EXEC`.
    pub async fn bind() -> Result<Self, WireError> {
        let transport = IrohTransport::bind(&[ALPN_EXEC])
            .await
            .map_err(WireError::Transport)?;
        Ok(Self {
            transport,
            request_timeout: DEFAULT_REQUEST_TIMEOUT,
        })
    }

    /// The identity a host will authenticate for this requester.
    pub fn identity(&self) -> NodeIdentity {
        self.transport.identity()
    }

    /// How long to wait for a host before giving up.
    pub fn with_request_timeout(&mut self, timeout: Duration) {
        self.request_timeout = timeout;
    }

    /// Send one request and return the receipt, or refuse it.
    ///
    /// Three checks, in this order. The **author** first: a receipt from an endpoint
    /// other than the one the connection authenticated is refused before anything in
    /// it is read, because nothing in it is worth reading. Then the request id, then
    /// the digest of the exact bytes that were sent — so a receipt cannot be presented
    /// as the answer to a request it does not belong to.
    ///
    /// What this does **not** establish: a receipt is not signed. Inside this call the
    /// connection vouches for its author; once the bytes are stored or forwarded,
    /// nothing does. A receipt in a file is hearsay, and this crate refuses to pretend
    /// otherwise by offering no way to verify one.
    pub async fn request(
        &self,
        host: EndpointAddr,
        request: &WireRequest,
    ) -> Result<WireReceipt, WireError> {
        let authenticated = host.id.to_string();
        let frame = encode_request(request)?;
        let digest = request_digest(&frame);

        let response = tokio::time::timeout(
            self.request_timeout,
            self.transport
                .request(host, ALPN_EXEC, &frame, MAX_FRAME_BYTES),
        )
        .await
        .map_err(|_| WireError::Transport("the host did not answer in time".to_owned()))?
        .map_err(WireError::Transport)?;

        let receipt = decode_receipt(&response)?;
        if receipt.responder != authenticated {
            return Err(WireError::ForgedReceipt {
                authenticated,
                claimed: receipt.responder,
            });
        }
        if receipt.request_id != request.request_id {
            return Err(WireError::WrongRequest {
                expected: request.request_id,
                answered: receipt.request_id,
            });
        }
        if receipt.request_digest != digest {
            return Err(WireError::UnboundReceipt);
        }
        Ok(receipt)
    }

    /// Close the endpoint.
    pub async fn close(&self) {
        self.transport.close().await;
    }
}
