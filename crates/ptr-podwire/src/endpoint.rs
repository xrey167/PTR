//! A Pod gateway and a requester, both over `ALPN_PODWIRE`.
//!
//! Neither is a daemon. `serve_once` accepts one connection and answers it;
//! `request` sends one request and checks one answer. A caller decides when either
//! happens, which keeps the tests free of sleeps and keeps "how long do we wait"
//! and "how often do we accept" where they can be decided deliberately.
use crate::access::{answer, PodAccessPolicy};
use crate::frame::{
    decode_answer, decode_request, encode_answer, encode_request, request_digest, FrameError,
    PodAnswer, PodOutcome, PodRequest, RefusalCode, MAX_FRAME_BYTES,
};
use ptr_net::{EndpointAddr, IrohTransport, NodeIdentity, ALPN_PODWIRE};
use ptr_pods::PodRegistry;
use ptr_protocol::TypedPayload;
use ptr_types::NodeId;
use ptr_verifier::Verifier;
use std::time::Duration;

/// How long one exchange waits for an answer before giving up.
///
/// A default rather than a rule: the right deadline depends on the deployment, and
/// what matters is that there is one. Without it a host that stopped answering
/// holds a requester open for as long as the transport allows.
pub const DEFAULT_REQUEST_TIMEOUT: Duration = Duration::from_secs(5);

/// Why a pod-wire operation was refused, at the resolution the *local* side gets.
///
/// Richer than what crosses the wire, in one direction only: a host's operator is
/// entitled to know which check refused a request, and the peer is not.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PodWireError {
    /// The transport refused, failed, or did not answer in time.
    Transport(String),
    /// The frame was malformed.
    Frame(FrameError),
    /// The request named another endpoint's key.
    Misaddressed {
        addressed_to: String,
        expected: String,
    },
    /// The access decision refused. Which refusal it was stays here *and* crosses
    /// the wire, because every code this protocol sends is already a fact about the
    /// requester or its own scope — the three that would be an oracle were collapsed
    /// into one before they ever reached this type.
    Refused(RefusalCode),
    /// An answer whose author is not the peer the connection authenticated.
    ///
    /// The case this layer exists for on the requesting side: an answer can be
    /// stored and forwarded, so one that names somebody else is refused rather than
    /// read.
    ForgedAnswer {
        authenticated: String,
        claimed: String,
    },
    /// An answer to a different request id.
    WrongRequest { expected: u64, answered: u64 },
    /// An answer whose digest is not the request that was sent. A well-formed reply
    /// to a question nobody asked here.
    UnboundAnswer,
}

impl PodWireError {
    /// Stable diagnostic code for this refusal.
    pub fn code(&self) -> &'static str {
        match self {
            Self::Transport(_) => "PTR_PODW_TRANSPORT",
            Self::Frame(_) => "PTR_PODW_FRAME",
            Self::Misaddressed { .. } => "PTR_PODW_MISADDRESSED",
            Self::Refused(code) => code.code(),
            Self::ForgedAnswer { .. } => "PTR_PODW_FORGED_ANSWER",
            Self::WrongRequest { .. } => "PTR_PODW_WRONG_REQUEST",
            Self::UnboundAnswer => "PTR_PODW_UNBOUND_ANSWER",
        }
    }
}

impl From<FrameError> for PodWireError {
    /// Carry a framing refusal through unchanged.
    fn from(error: FrameError) -> Self {
        Self::Frame(error)
    }
}

impl std::fmt::Display for PodWireError {
    /// Render the stable refusal code.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.code())
    }
}

impl std::error::Error for PodWireError {}

/// What one served request did, for the host's own caller.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Served {
    /// The peer the connection authenticated. Not a value from the payload — there
    /// is no such value.
    pub peer: NodeIdentity,
    /// The request id, when the frame decoded far enough to have one.
    pub request_id: Option<u64>,
    /// What the answer told the peer.
    pub reported: PodOutcome,
    /// The local reason behind a refusal.
    pub refused: Option<PodWireError>,
}

/// Pods reachable over `ALPN_PODWIRE`.
///
/// The host owns the registry, the policy and the verifier. What it does **not**
/// own is a runtime or a ledger, and that absence is the protocol's strongest
/// claim: a Pod invoked through this endpoint cannot write to the host's history,
/// because this endpoint has no history to write to. The in-process loop promotes
/// a Pod's output into the semantic state of the model run that asked for it;
/// there is no such run behind a request that arrived over a wire, and inventing
/// one would let a peer write to a host's semantic store through a Read-only door.
pub struct PodHost {
    registry: PodRegistry,
    policy: PodAccessPolicy,
    verifier: Box<dyn Verifier<TypedPayload> + Send + Sync>,
    transport: IrohTransport,
}

impl PodHost {
    /// Bind an endpoint for `ALPN_PODWIRE` around a registry, a policy and the
    /// verifier every answer must pass before it leaves.
    pub async fn bind<V>(
        registry: PodRegistry,
        policy: PodAccessPolicy,
        verifier: V,
    ) -> Result<Self, PodWireError>
    where
        V: Verifier<TypedPayload> + Send + Sync + 'static,
    {
        let transport = IrohTransport::bind(&[ALPN_PODWIRE])
            .await
            .map_err(PodWireError::Transport)?;
        Ok(Self {
            registry,
            policy,
            verifier: Box::new(verifier),
            transport,
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

    /// The policy, for the host's own trusted use — admitting and withdrawing
    /// peers between requests.
    pub fn policy_mut(&mut self) -> &mut PodAccessPolicy {
        &mut self.policy
    }

    /// Accept one connection, serve what it carries, and answer.
    ///
    /// Every request is answered, including every refused one: leaving a peer
    /// hanging makes this side's refusal the peer's problem. The answer is bound to
    /// the bytes that actually arrived, so even a request this build cannot parse
    /// gets a reply its sender can check against what it sent.
    ///
    /// The order is the protocol:
    ///
    /// 1. The peer is the **authenticated connection**. Nothing in the payload
    ///    names a sender, and nothing here reads one.
    /// 2. The frame must decode.
    /// 3. The request must be addressed to this endpoint's key. A request
    ///    legitimate at another host is not legitimate here.
    /// 4. [`answer`] decides the rest, from the policy and the registry.
    pub async fn serve_once(&self) -> Result<Served, PodWireError> {
        let incoming = self
            .transport
            .accept_once(MAX_FRAME_BYTES)
            .await
            .map_err(PodWireError::Transport)?;

        // (1) The sender is the connection. There is no payload field to prefer
        // over it, which is the shape of the format rather than a check made here.
        let peer = incoming.peer.clone();
        let digest = request_digest(&incoming.payload);
        let (request_id, outcome, mut refused) = self.decide(&peer, &incoming.payload);

        let mut answered = PodAnswer {
            responder: self.identity().public_key,
            request_id: request_id.unwrap_or(0),
            request_digest: digest,
            outcome,
        };

        // An answer that will not fit in a frame is still an answer. Pre-checking
        // one field would have missed the others — a Pod's output *type* is bounded
        // too — and every miss leaves a peer holding an open connection because this
        // side could not phrase its reply. So the bound is enforced where it is
        // known: by trying, and refusing when the attempt fails.
        //
        // Nothing applied, so there is nothing for the requester to reconcile; it is
        // told the answer exists and does not fit, which it can act on.
        let frame = match encode_answer(&answered) {
            Ok(frame) => frame,
            Err(error) => {
                answered.outcome = PodOutcome::Refused {
                    code: RefusalCode::AnswerTooLarge,
                };
                refused = Some(PodWireError::Frame(error));
                encode_answer(&answered)?
            }
        };
        incoming
            .respond(&frame)
            .await
            .map_err(PodWireError::Transport)?;

        Ok(Served {
            peer,
            request_id,
            reported: answered.outcome,
            refused,
        })
    }

    /// Everything between the bytes arriving and the answer going out.
    ///
    /// Split out so that answering the peer is unconditional: a refusal decided
    /// here still becomes an answer rather than a dropped connection.
    fn decide(
        &self,
        peer: &NodeIdentity,
        payload: &[u8],
    ) -> (Option<u64>, PodOutcome, Option<PodWireError>) {
        // (2) The frame must decode.
        let request = match decode_request(payload) {
            Ok(request) => request,
            Err(error) => {
                return (
                    None,
                    PodOutcome::Refused {
                        code: RefusalCode::Malformed,
                    },
                    Some(PodWireError::Frame(error)),
                )
            }
        };
        let request_id = Some(request.request_id);

        // (3) Addressed here, or nowhere.
        let expected = self.identity().public_key;
        if request.addressed_to != expected {
            return (
                request_id,
                PodOutcome::Refused {
                    code: RefusalCode::Misaddressed,
                },
                Some(PodWireError::Misaddressed {
                    addressed_to: request.addressed_to,
                    expected,
                }),
            );
        }

        // (4) The access decision. The peer is the connection's, never the frame's.
        match answer(
            &self.policy,
            &self.registry,
            self.verifier.as_ref(),
            &NodeId(peer.public_key.clone()),
            &request.capability,
            request.expect_protocol,
            request.payload,
        ) {
            Ok(output) => (request_id, PodOutcome::Answered { output }, None),
            Err(code) => (
                request_id,
                PodOutcome::Refused { code },
                Some(PodWireError::Refused(code)),
            ),
        }
    }

    /// Close the endpoint.
    pub async fn close(&self) {
        self.transport.close().await;
    }
}

/// A requester with an authenticated endpoint.
pub struct PodClient {
    transport: IrohTransport,
    request_timeout: Duration,
}

impl PodClient {
    /// Bind an endpoint for `ALPN_PODWIRE`.
    pub async fn bind() -> Result<Self, PodWireError> {
        let transport = IrohTransport::bind(&[ALPN_PODWIRE])
            .await
            .map_err(PodWireError::Transport)?;
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

    /// Send one request and return the answer, or refuse it.
    ///
    /// Three checks, in this order. The **author** first: an answer from an endpoint
    /// other than the one the connection authenticated is refused before anything in
    /// it is read, because nothing in it is worth reading. Then the request id, then
    /// the digest of the exact bytes that were sent.
    ///
    /// What this does **not** establish: an answer is not signed. Inside this call
    /// the connection vouches for its author; once the bytes are stored or
    /// forwarded, nothing does.
    pub async fn request(
        &self,
        host: EndpointAddr,
        request: &PodRequest,
    ) -> Result<PodAnswer, PodWireError> {
        let authenticated = host.id.to_string();
        let frame = encode_request(request)?;
        let digest = request_digest(&frame);

        let response = tokio::time::timeout(
            self.request_timeout,
            self.transport
                .request(host, ALPN_PODWIRE, &frame, MAX_FRAME_BYTES),
        )
        .await
        .map_err(|_| PodWireError::Transport("the host did not answer in time".to_owned()))?
        .map_err(PodWireError::Transport)?;

        let answered = decode_answer(&response)?;
        if answered.responder != authenticated {
            return Err(PodWireError::ForgedAnswer {
                authenticated,
                claimed: answered.responder,
            });
        }
        if answered.request_id != request.request_id {
            return Err(PodWireError::WrongRequest {
                expected: request.request_id,
                answered: answered.request_id,
            });
        }
        if answered.request_digest != digest {
            return Err(PodWireError::UnboundAnswer);
        }
        Ok(answered)
    }

    /// Close the endpoint.
    pub async fn close(&self) {
        self.transport.close().await;
    }
}
