//! The wire format for one Pod invocation and the answer to it.
//!
//! Two shapes, two magics, decided before a version is looked at. A request asks
//! for a Pod's answer and an answer reports one; reading either as the other is
//! refused by the first eight bytes.
//!
//! **A request names no sender, no session and no project.** Each of the three is
//! absent for its own reason rather than by symmetry with `ptr-execwire`:
//!
//! - *No sender.* Who is asking is the authenticated connection's answer. There is
//!   nothing to forge because there is nothing to write.
//! - *No session.* A session is a capability the host holds, not a string a caller
//!   presents. `proto/podwire.proto` once required a `session_id` in exactly this
//!   position, which is the shape this format refuses.
//! - *No project.* This is the one that is specific to Pod access. Resolution
//!   inside a project is the *only* thing standing between a request and a Pod's
//!   data — `PodManifest` says so in its own doc comment — so a requester that
//!   named its project would be choosing which project's Pods it reaches. The
//!   project comes from host policy, keyed by the authenticated peer.
//!
//! What a request does name is whom it is *for*, so a request composed for one
//! host cannot be relayed to another and be accepted as meant for it.
//!
//! Every field is length-prefixed and bounded, every enumeration crosses as an
//! explicit code from a two-way table rather than a cast of a discriminant, and a
//! frame that disagrees with itself is refused whole.
//!
//! **An answer is signed** (D2 in issue #23) by the responder's endpoint key, in
//! every layout. What the signature attests, and what it does not, is written on
//! [`verify_answer`]; the bytes it covers are written on [`ANSWER_DOMAIN`].
use ptr_net::{NodeSigner, SigningDomain, SIGNATURE_BYTES};
use ptr_pods::ProtocolBinding;
use ptr_protocol::TypedPayload;
use ptr_types::{
    ArtifactId, CapabilityId, Generation, IdentityContext, MeshEndpointBinding, MeshRouteKind,
    NodeId, PeerId, PodAddress, PodRevisionAddress, Revision, SessionId, TraceId, TypeId,
};
use sha2::{Digest, Sha256};
use std::fmt;

/// Marks a request. Foreign bytes are refused before anything is interpreted.
const REQUEST_MAGIC: &[u8; 8] = b"PTRPWREQ";

/// Marks an answer.
const ANSWER_MAGIC: &[u8; 8] = b"PTRPWANS";

/// Domain separation for a request digest, so that a digest of these bytes cannot
/// collide with a digest taken over anything else in the system — the execution
/// wire's request digest included.
const DIGEST_DOMAIN: &[u8] = b"PTRPW001-REQUEST";

/// The only frame layout this build writes and reads.
pub const FORMAT_V1: u16 = 1;

/// Generation-bound frame layout. V1 remains wire-compatible and is limited to
/// legacy Pure/Read calls; V2 binds the artifact, manifest, generation and
/// semantic revision before a peer can invoke a Pod.
pub const FORMAT_V2: u16 = 2;
/// Address-bound extension of V2. FORMAT_V2 remains readable for legacy peers.
pub const FORMAT_V3: u16 = 3;

/// The signed answer to a [`FORMAT_V1`] request: the V1 answer's fields followed by
/// a signature.
///
/// From here on answers are numbered apart from requests. A request carries nothing
/// a signature changes, so requests keep [`FORMAT_V1`], [`FORMAT_V2`] and
/// [`FORMAT_V3`]; an *answer* in any of those three layouts is the unsigned one
/// earlier builds wrote, and is refused by name ([`FrameError::UnsignedAnswer`]).
pub const SIGNED_ANSWER_V1: u16 = 4;

/// The signed answer to a [`FORMAT_V2`] request.
pub const SIGNED_ANSWER_V2: u16 = 5;

/// The signed answer to a [`FORMAT_V3`] request, carrying the address binding.
pub const SIGNED_ANSWER_V3: u16 = 6;

/// The domain an answer signature is made in, for all three signed layouts.
///
/// The signed message is this tag followed by **every byte of the answer frame
/// before the signature**: the magic, the layout version — which tells the three
/// layouts apart, so one cannot be re-read as another — and every field the layout
/// carries, in frame order. For V3 that is the address binding and execution
/// manifest; for V2 and V3 the generation, revision, state binding, identity digest,
/// session id and policy revision; for every layout the `responder`, `request_id`,
/// `request_digest`, the outcome code and the outcome's own fields (the output type
/// and bytes, or the refusal code). The frame *is* the message, so there is no
/// second encoding of the same fields to drift from the first.
///
/// Distinct from the execution wire's domain, so a receipt is never read as an
/// answer or an answer as a receipt even by a verifier that skipped the magic.
pub const ANSWER_DOMAIN: SigningDomain = SigningDomain::new(b"ptr-pod-answer/v1\0");

/// Bound on one frame in either direction. A peer does not get to decide how much
/// a receiver reads.
pub const MAX_FRAME_BYTES: usize = 8 * 1024 * 1024;

/// Bound on one named field.
pub const MAX_FIELD_BYTES: usize = 4 * 1024;

/// Bound on a payload in either direction.
pub const MAX_BODY_BYTES: usize = 4 * 1024 * 1024;

/// One Pod invocation, asked across a network boundary.
///
/// Decoding one of these resolves nothing and authorizes nothing. Which Pod — if
/// any — answers it is decided from the host's policy and registry, never from
/// this struct.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PodRequest {
    /// The public key of the endpoint the requester believes it is addressing.
    ///
    /// Checked by the receiver against its own key. Without it, a request that was
    /// legitimate at one host could be relayed to another and be indistinguishable
    /// from one composed for it.
    pub addressed_to: String,
    /// A correlator for this request, so an answer can be checked against the
    /// question.
    ///
    /// Not a replay nonce. This protocol has no replay window, and that is a
    /// property rather than an omission: a window exists to stop an *effect*
    /// applying twice, and every Pod reachable here is Pure or Read by
    /// construction. A frame sent twice costs the host the work twice and can do
    /// nothing else.
    pub request_id: u64,
    /// Which capability is wanted.
    pub capability: CapabilityId,
    /// The protocol version the requester composed its payload for.
    ///
    /// `PodManifest` declares one and, before this wire existed, nothing anywhere
    /// read it: in one process the caller and the Pod are built together. Across a
    /// boundary they are not, so a payload composed for one version of a Pod must
    /// not be quietly handed to another.
    pub expect_protocol: u32,
    /// The typed payload. Its `type_id` is half the resolution key, which is why
    /// there is no separate input-type field to disagree with it.
    pub payload: TypedPayload,
}

/// What a host reports back about one request.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PodAnswer {
    /// The public key of the endpoint that produced this answer.
    ///
    /// Checked by the requester against the peer the connection authenticated. An
    /// answer read from anywhere else has no such check available to it, which is
    /// the whole of what this field is worth.
    pub responder: String,
    /// The `request_id` this answers.
    pub request_id: u64,
    /// A digest over the exact bytes that arrived, so an answer is bound to one
    /// request and cannot be presented as the answer to another.
    pub request_digest: [u8; 32],
    /// What happened.
    pub outcome: PodOutcome,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PodRequestV2 {
    pub address: Option<PodAddressBinding>,
    /// V3-only binding to the immutable execution context.
    pub execution_manifest: Option<[u8; 32]>,
    pub addressed_to: String,
    pub request_id: u64,
    pub artifact_id: ArtifactId,
    pub manifest_hash: [u8; 32],
    pub generation: Generation,
    pub revision: Revision,
    /// Stateful calls set both fields; pure/read V2 calls may leave both absent.
    pub placement_epoch: Option<u64>,
    pub fencing_token: Option<u128>,
    pub identity_digest: [u8; 32],
    pub session_id: SessionId,
    pub policy_revision: Revision,
    pub capability: CapabilityId,
    pub expect_protocol: u32,
    pub payload: TypedPayload,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PodAnswerV2 {
    pub address: Option<PodAddressBinding>,
    /// V3-only echo of the request's execution-manifest binding.
    pub execution_manifest: Option<[u8; 32]>,
    pub responder: String,
    pub request_id: u64,
    pub request_digest: [u8; 32],
    pub generation: Generation,
    pub revision: Revision,
    pub placement_epoch: Option<u64>,
    pub fencing_token: Option<u128>,
    pub identity_digest: [u8; 32],
    pub session_id: SessionId,
    pub policy_revision: Revision,
    pub outcome: PodOutcome,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PodAddressBinding {
    pub source: Option<PodAddress>,
    pub destination: PodRevisionAddress,
    pub trace_id: TraceId,
    pub hop_limit: u16,
    pub visited: Vec<PodAddress>,
    pub protocol: Option<ProtocolBinding>,
    pub mesh: Option<MeshEndpointBinding>,
}

impl PodAddressBinding {
    pub fn placeholder() -> Self {
        Self {
            source: None,
            destination: PodRevisionAddress {
                address: PodAddress {
                    project: "malformed".into(),
                    namespace: "malformed".into(),
                    pod_id: "malformed".into(),
                },
                semantic_revision: [1; 32],
                generation: Generation(0),
            },
            trace_id: "malformed".into(),
            hop_limit: 1,
            visited: Vec::new(),
            protocol: None,
            mesh: None,
        }
    }

    pub fn validate(&self) -> Result<(), FrameError> {
        self.destination
            .validate()
            .map_err(|_| FrameError::InvalidField {
                field: "destination",
            })?;
        if self.trace_id.0.is_empty() {
            return Err(FrameError::InvalidField { field: "trace_id" });
        }
        if self.hop_limit == 0 {
            return Err(FrameError::InvalidField { field: "hop_limit" });
        }
        if self.visited.len() > 64 {
            return Err(FrameError::InvalidField { field: "visited" });
        }
        for address in &self.visited {
            address
                .validate()
                .map_err(|_| FrameError::InvalidField { field: "visited" })?;
            if *address == self.destination.address {
                return Err(FrameError::InvalidField { field: "visited" });
            }
        }
        let mut unique = self.visited.clone();
        unique.sort();
        unique.dedup();
        if unique.len() != self.visited.len() {
            return Err(FrameError::InvalidField { field: "visited" });
        }
        if let Some(source) = &self.source {
            source
                .validate()
                .map_err(|_| FrameError::InvalidField { field: "source" })?;
            if source.project != self.destination.address.project
                || source.namespace != self.destination.address.namespace
            {
                return Err(FrameError::InvalidField {
                    field: "address_scope",
                });
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PodAdmissionBinding {
    pub identity_digest: [u8; 32],
    pub session_id: SessionId,
    pub policy_revision: Revision,
}

impl PodAdmissionBinding {
    pub fn from_identity(
        identity: &IdentityContext,
        policy_revision: Revision,
    ) -> Result<Self, FrameError> {
        if !identity.has_valid_digest() {
            return Err(FrameError::InvalidField {
                field: "identity_digest",
            });
        }
        Ok(Self {
            identity_digest: identity.identity_digest,
            session_id: identity.session_id.clone(),
            policy_revision,
        })
    }
}

/// The outcome an answer reports.
///
/// **Two, where the execution wire has four.** That difference is the protocol's
/// central claim rather than a simplification of it. `ALPN_EXEC` needs "applied
/// without a response" and "uncertain" because an effect may have happened and the
/// host may not know. Nothing reachable here has an effect, so a refusal here
/// always means nothing happened, and there is no state for a requester to be
/// uncertain about.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PodOutcome {
    /// A Pod answered, and its output passed the host's verifier.
    Answered {
        /// What the Pod produced.
        output: TypedPayload,
    },
    /// No output. Nothing was written anywhere, because nothing on this wire
    /// writes.
    Refused {
        /// Why, at the resolution the wire allows.
        code: RefusalCode,
    },
}

/// Why a request was refused.
///
/// Where this stops, and why it stops there, is the part worth reading.
///
/// [`RefusalCode::Unavailable`] is **one code for three situations**: this peer's
/// scope does not cover the capability, no Pod in this peer's project serves it,
/// and a Pod serving it belongs to another project. Telling them apart would
/// answer, from outside a project, whether a given Pod exists inside it — the same
/// oracle `PodUnavailable` already closes in one process, and an isolation
/// boundary that answers questions about its far side is porous without ever
/// breaking.
///
/// The rest are each a fact about the requester itself or about its own granted
/// scope, and stay distinct because a requester that cannot tell them apart cannot
/// act on any of them: how it framed its bytes, which key it addressed, whether its
/// own key is admitted here at all, whether the Pod it may reach wants the other
/// protocol, whether it composed for the wrong protocol version, and whether the
/// Pod ran and failed as against ran and was rejected.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RefusalCode {
    /// The request did not decode.
    Malformed,
    /// The request named another endpoint's key.
    Misaddressed,
    /// This peer has no entry in the host's Pod access policy. A fact about the
    /// requester's own key and about nothing else.
    NotAdmitted,
    /// Nothing in this peer's scope answers this capability and payload type. One
    /// code for out of scope, not registered, and registered in another project.
    Unavailable,
    /// A Pod answers it and it is not Pure or Read. Effects cross a boundary over
    /// `ALPN_EXEC`, where an attempt is committed before the effect and a receipt
    /// can say "uncertain"; this protocol can do neither and must not pretend.
    RequiresActionBoundary,
    /// The Pod that serves this capability declares another protocol version.
    ///
    /// Which version it does declare is not reported. Finding a Pod's version is
    /// discovery, and this wire has none.
    ProtocolMismatch,
    /// The Pod ran and returned an error. Its message stays on the host: a Pod's
    /// internal diagnostics are the host's, not the requester's.
    PodFailed,
    /// The Pod ran and its output did not pass the host's verifier. Distinct from
    /// [`Self::PodFailed`] because the two are actionable in opposite directions:
    /// one says the input was not usable, the other says the host would not vouch
    /// for the answer.
    Unverified,
    /// The Pod ran, the host would vouch for the answer, and the answer does not
    /// fit in a frame.
    ///
    /// Its own code rather than [`Self::Unavailable`], which means *nothing in your
    /// scope answers this* and would be false here: something did. The requester can
    /// act on this one — ask for less — and could act on nothing if it were folded
    /// in. The execution wire's counterpart is `AppliedWithoutResponse`, and the
    /// difference is the whole difference between the two protocols: there the
    /// effect happened and the bytes are gone, here nothing happened at all.
    AnswerTooLarge,
}

impl RefusalCode {
    /// Stable diagnostic code, for a log rather than for the wire.
    pub fn code(self) -> &'static str {
        match self {
            Self::Malformed => "PTR_PODW_MALFORMED",
            Self::Misaddressed => "PTR_PODW_MISADDRESSED",
            Self::NotAdmitted => "PTR_PODW_NOT_ADMITTED",
            Self::Unavailable => "PTR_PODW_UNAVAILABLE",
            Self::RequiresActionBoundary => "PTR_PODW_REQUIRES_ACTION_BOUNDARY",
            Self::ProtocolMismatch => "PTR_PODW_PROTOCOL_MISMATCH",
            Self::PodFailed => "PTR_PODW_POD_FAILED",
            Self::Unverified => "PTR_PODW_UNVERIFIED",
            Self::AnswerTooLarge => "PTR_PODW_ANSWER_TOO_LARGE",
        }
    }

    /// Every refusal this build can send, so a test can hold the table to account.
    pub const ALL: [Self; 9] = [
        Self::Malformed,
        Self::Misaddressed,
        Self::NotAdmitted,
        Self::Unavailable,
        Self::RequiresActionBoundary,
        Self::ProtocolMismatch,
        Self::PodFailed,
        Self::Unverified,
        Self::AnswerTooLarge,
    ];
}

impl fmt::Display for RefusalCode {
    /// Render the stable refusal code.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.code())
    }
}

/// Why a frame was refused.
///
/// None of these yields a partial frame. A request assembled from the fields that
/// happened to parse is a request nobody composed.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum FrameError {
    /// The bytes do not begin with this frame kind's magic.
    WrongKind,
    /// A layout this build does not know, rather than one parsed anyway.
    UnknownFormat { format: u16 },
    /// The frame ends inside a field.
    Truncated { field: &'static str },
    /// Bytes after the last field: the writer and the reader disagree.
    TrailingBytes { extra: usize },
    /// The frame, or one field in it, is over its bound.
    TooLarge {
        field: &'static str,
        bytes: usize,
        limit: usize,
    },
    /// A code this build has no meaning for. Never guessed at, because a code is
    /// the wire's name for a decision and half a decision is not one.
    UnknownCode { field: &'static str, code: u64 },
    /// A field that should be UTF-8 and is not.
    NotUtf8 { field: &'static str },
    /// A typed field failed local admission validation.
    InvalidField { field: &'static str },
    /// An answer in one of the unsigned layouts earlier builds wrote.
    ///
    /// Its own refusal rather than [`Self::UnknownFormat`], because this build knows
    /// those layouts and refuses them on purpose: an unsigned answer is exactly the
    /// hearsay a signature exists to replace, and reading one alongside signed ones
    /// would let a stripped signature pass as an old peer.
    UnsignedAnswer { format: u16 },
}

impl FrameError {
    /// Stable diagnostic code for this refusal.
    pub fn code(&self) -> &'static str {
        match self {
            Self::WrongKind => "PTR_PODW_WRONG_KIND",
            Self::UnknownFormat { .. } => "PTR_PODW_UNKNOWN_FORMAT",
            Self::Truncated { .. } => "PTR_PODW_TRUNCATED",
            Self::TrailingBytes { .. } => "PTR_PODW_TRAILING_BYTES",
            Self::TooLarge { .. } => "PTR_PODW_FRAME_TOO_LARGE",
            Self::UnknownCode { .. } => "PTR_PODW_UNKNOWN_CODE",
            Self::NotUtf8 { .. } => "PTR_PODW_NOT_UTF8",
            Self::InvalidField { .. } => "PTR_PODW_INVALID_FIELD",
            Self::UnsignedAnswer { .. } => "PTR_PODW_UNSIGNED_ANSWER",
        }
    }
}

impl fmt::Display for FrameError {
    /// Render the stable refusal code.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.code())
    }
}

impl std::error::Error for FrameError {}

/// Why an answer did not verify.
///
/// Three refusals, checked in this order: the bytes must be a well-formed signed
/// answer, it must name the responder it is being verified against, and the
/// signature must be that responder's over the frame.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum AnswerError {
    /// The bytes are not a well-formed signed answer.
    Frame(FrameError),
    /// The answer names an author other than the one it is verified against.
    ///
    /// Checked before the signature: a valid signature by the wrong author attests
    /// nothing about the right one.
    WrongResponder { expected: String, claimed: String },
    /// The signature is not the named responder's over these bytes.
    BadSignature,
}

impl AnswerError {
    /// Stable diagnostic code for this refusal.
    pub fn code(&self) -> &'static str {
        match self {
            Self::Frame(error) => error.code(),
            Self::WrongResponder { .. } => "PTR_PODW_WRONG_RESPONDER",
            Self::BadSignature => "PTR_PODW_BAD_SIGNATURE",
        }
    }
}

impl From<FrameError> for AnswerError {
    /// Carry a framing refusal through unchanged.
    fn from(error: FrameError) -> Self {
        Self::Frame(error)
    }
}

impl fmt::Display for AnswerError {
    /// Render the stable refusal code.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.code())
    }
}

impl std::error::Error for AnswerError {}

/// The digest an answer is bound to: taken over the bytes that actually arrived.
///
/// Over the received bytes rather than over a decoded request, so that even a
/// request this build cannot parse still gets an answer its sender can check.
pub fn request_digest(bytes: &[u8]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(DIGEST_DOMAIN);
    hasher.update((bytes.len() as u64).to_le_bytes());
    hasher.update(bytes);
    hasher.finalize().into()
}

/// The wire's name for an outcome kind.
fn outcome_code(outcome: &PodOutcome) -> u8 {
    match outcome {
        PodOutcome::Answered { .. } => 1,
        PodOutcome::Refused { .. } => 2,
    }
}

/// The wire's name for a refusal.
///
/// An explicit table in both directions. A discriminant is an implementation
/// detail of this build and a wire code is a promise to every other one.
fn refusal_code(code: RefusalCode) -> u16 {
    match code {
        RefusalCode::Malformed => 1,
        RefusalCode::Misaddressed => 2,
        RefusalCode::NotAdmitted => 3,
        RefusalCode::Unavailable => 4,
        RefusalCode::RequiresActionBoundary => 5,
        RefusalCode::ProtocolMismatch => 6,
        RefusalCode::PodFailed => 7,
        RefusalCode::Unverified => 8,
        RefusalCode::AnswerTooLarge => 9,
    }
}

/// Read a refusal, or refuse a code this build has no meaning for.
fn refusal_from_code(code: u16) -> Result<RefusalCode, FrameError> {
    match code {
        1 => Ok(RefusalCode::Malformed),
        2 => Ok(RefusalCode::Misaddressed),
        3 => Ok(RefusalCode::NotAdmitted),
        4 => Ok(RefusalCode::Unavailable),
        5 => Ok(RefusalCode::RequiresActionBoundary),
        6 => Ok(RefusalCode::ProtocolMismatch),
        7 => Ok(RefusalCode::PodFailed),
        8 => Ok(RefusalCode::Unverified),
        9 => Ok(RefusalCode::AnswerTooLarge),
        other => Err(FrameError::UnknownCode {
            field: "refusal",
            code: u64::from(other),
        }),
    }
}

/// Encode a request.
pub fn encode_request(request: &PodRequest) -> Result<Vec<u8>, FrameError> {
    let mut frame = Vec::with_capacity(128 + request.payload.bytes.len());
    frame.extend_from_slice(REQUEST_MAGIC);
    frame.extend_from_slice(&FORMAT_V1.to_le_bytes());
    put_str(&mut frame, "addressed_to", &request.addressed_to)?;
    frame.extend_from_slice(&request.request_id.to_le_bytes());
    put_str(&mut frame, "capability", &request.capability.0)?;
    frame.extend_from_slice(&request.expect_protocol.to_le_bytes());
    put_str(&mut frame, "input_type", &request.payload.type_id.0)?;
    put_bytes(
        &mut frame,
        "payload",
        &request.payload.bytes,
        MAX_BODY_BYTES,
    )?;
    bound_frame(frame)
}

/// Decode a request, or refuse it.
pub fn decode_request(bytes: &[u8]) -> Result<PodRequest, FrameError> {
    let mut at = open(bytes, REQUEST_MAGIC)?;
    let addressed_to = take_str(bytes, &mut at, "addressed_to")?;
    let request_id = take_u64(bytes, &mut at, "request_id")?;
    let capability = CapabilityId(take_str(bytes, &mut at, "capability")?);
    let expect_protocol = take_u32(bytes, &mut at, "expect_protocol")?;
    let type_id = TypeId(take_str(bytes, &mut at, "input_type")?);
    let payload = take_bytes(bytes, &mut at, "payload", MAX_BODY_BYTES)?;
    close(bytes, at)?;
    Ok(PodRequest {
        addressed_to,
        request_id,
        capability,
        expect_protocol,
        payload: TypedPayload {
            type_id,
            bytes: payload,
        },
    })
}

/// Encode an answer to a V1 request and sign it.
///
/// The signer is whoever is answering; nothing here checks that it is the key the
/// answer's `responder` names. An answer that names one key and is signed by another
/// is refused by every verifier, and making it impossible to write would only make
/// that refusal impossible to test.
pub fn encode_answer(answer: &PodAnswer, signer: &NodeSigner) -> Result<Vec<u8>, FrameError> {
    let mut frame = Vec::with_capacity(128 + SIGNATURE_BYTES);
    frame.extend_from_slice(ANSWER_MAGIC);
    frame.extend_from_slice(&SIGNED_ANSWER_V1.to_le_bytes());
    put_str(&mut frame, "responder", &answer.responder)?;
    frame.extend_from_slice(&answer.request_id.to_le_bytes());
    frame.extend_from_slice(&answer.request_digest);
    frame.push(outcome_code(&answer.outcome));
    match &answer.outcome {
        PodOutcome::Answered { output } => {
            put_str(&mut frame, "output_type", &output.type_id.0)?;
            put_bytes(&mut frame, "output", &output.bytes, MAX_BODY_BYTES)?;
        }
        PodOutcome::Refused { code } => {
            frame.extend_from_slice(&refusal_code(*code).to_le_bytes());
        }
    }
    sign_frame(frame, signer)
}

/// Append the signature over everything written so far, then bound the result.
fn sign_frame(mut frame: Vec<u8>, signer: &NodeSigner) -> Result<Vec<u8>, FrameError> {
    let signature = signer.sign(ANSWER_DOMAIN, &frame);
    frame.extend_from_slice(&signature);
    bound_frame(frame)
}

/// Check what [`verify_answer`] and [`verify_answer_v2`] share: the author named in
/// the frame, then the signature over the bytes it covers.
fn verify_signed(
    claimed: &str,
    signed: &[u8],
    signature: &[u8; SIGNATURE_BYTES],
    responder: &NodeId,
) -> Result<(), AnswerError> {
    if claimed != responder.0 {
        return Err(AnswerError::WrongResponder {
            expected: responder.0.clone(),
            claimed: claimed.to_owned(),
        });
    }
    if !ptr_net::verify(&responder.0, ANSWER_DOMAIN, signed, signature) {
        return Err(AnswerError::BadSignature);
    }
    Ok(())
}

/// Verify an answer to a V1 request as having been written by `responder`, and
/// return it.
///
/// The only way to read a V1 answer out of bytes, on purpose: an answer that has
/// not been verified is not one this crate hands anybody. It needs no connection, so
/// a stored answer is checked exactly as one that just arrived.
///
/// **What an answer that verifies attests:** that the host holding the endpoint key
/// `responder` stated outcome *O* — a Pod output its verifier passed, or a refusal
/// with its code — for request id *X*, bound to the request digest *D* of the exact
/// bytes it received (and, for V2/V3, for the generation, revision, state binding,
/// identity digest, session and policy revision, and address binding it echoes).
/// Every word of that is about the responder's own statement.
///
/// **What it does not attest:**
/// - that the output is correct. It is what the host's Pod produced and the host's
///   verifier passed; the signature makes that attributable, not true;
/// - anything about a principal other than the responder — not who asked (a request
///   names no sender), not which project the host resolved it in, not anything
///   about another host;
/// - when it was made: nothing in it is a timestamp;
/// - that `responder` is still trusted. A signature outlives its key's admission,
///   and whether an answer from a withdrawn peer still counts is the verifier's
///   policy. A rotated key is a new identity, so an answer signed by the old key
///   verifies under the old key and under nothing else.
pub fn verify_answer(bytes: &[u8], responder: &NodeId) -> Result<PodAnswer, AnswerError> {
    let (answer, signed, signature) = parse_answer(bytes)?;
    verify_signed(&answer.responder, signed, &signature, responder)?;
    Ok(answer)
}

/// Verify an answer to a V2 or V3 request as having been written by `responder`,
/// and return it. Attests exactly what [`verify_answer`] says, and no more.
pub fn verify_answer_v2(bytes: &[u8], responder: &NodeId) -> Result<PodAnswerV2, AnswerError> {
    let (answer, signed, signature) = parse_answer_v2(bytes)?;
    verify_signed(&answer.responder, signed, &signature, responder)?;
    Ok(answer)
}

/// A signed answer's fields, the bytes its signature covers, and the signature —
/// read without checking it.
type Parsed<'a, T> = (T, &'a [u8], [u8; SIGNATURE_BYTES]);

/// Read the trailing signature and close the frame.
fn take_signature<'a, T>(
    bytes: &'a [u8],
    mut at: usize,
    answer: T,
) -> Result<Parsed<'a, T>, FrameError> {
    let signed = &bytes[..at];
    let signature = take_fixed::<SIGNATURE_BYTES>(bytes, &mut at, "signature")?;
    close(bytes, at)?;
    Ok((answer, signed, signature))
}

pub fn encode_request_v2(request: &PodRequestV2) -> Result<Vec<u8>, FrameError> {
    let mut frame = Vec::with_capacity(192 + request.payload.bytes.len());
    frame.extend_from_slice(REQUEST_MAGIC);
    let format = if request.address.is_some() {
        FORMAT_V3
    } else {
        FORMAT_V2
    };
    frame.extend_from_slice(&format.to_le_bytes());
    if let Some(address) = &request.address {
        put_address_binding(&mut frame, address)?;
        frame.extend_from_slice(&request.execution_manifest.ok_or(FrameError::InvalidField {
            field: "execution_manifest",
        })?);
    } else if request.execution_manifest.is_some() {
        return Err(FrameError::InvalidField {
            field: "execution_manifest",
        });
    }
    put_str(&mut frame, "addressed_to", &request.addressed_to)?;
    frame.extend_from_slice(&request.request_id.to_le_bytes());
    put_str(&mut frame, "artifact_id", &request.artifact_id.0)?;
    frame.extend_from_slice(&request.manifest_hash);
    frame.extend_from_slice(&request.generation.0.to_le_bytes());
    frame.extend_from_slice(&request.revision.0.to_le_bytes());
    put_state_binding(&mut frame, request.placement_epoch, request.fencing_token)?;
    frame.extend_from_slice(&request.identity_digest);
    put_str(&mut frame, "session_id", &request.session_id.0)?;
    frame.extend_from_slice(&request.policy_revision.0.to_le_bytes());
    put_str(&mut frame, "capability", &request.capability.0)?;
    frame.extend_from_slice(&request.expect_protocol.to_le_bytes());
    put_str(&mut frame, "input_type", &request.payload.type_id.0)?;
    put_bytes(
        &mut frame,
        "payload",
        &request.payload.bytes,
        MAX_BODY_BYTES,
    )?;
    bound_frame(frame)
}

pub fn decode_request_v2(bytes: &[u8]) -> Result<PodRequestV2, FrameError> {
    let (mut at, has_address) = open_v2_compatible(bytes, REQUEST_MAGIC)?;
    let address = has_address
        .then(|| take_address_binding(bytes, &mut at))
        .transpose()?;
    let execution_manifest = has_address
        .then(|| take_digest(bytes, &mut at))
        .transpose()?;
    let addressed_to = take_str(bytes, &mut at, "addressed_to")?;
    let request_id = take_u64(bytes, &mut at, "request_id")?;
    let artifact_id = ArtifactId(take_str(bytes, &mut at, "artifact_id")?);
    let manifest_hash = take_digest(bytes, &mut at)?;
    let generation = Generation(take_u64(bytes, &mut at, "generation")?);
    let revision = Revision(take_u64(bytes, &mut at, "revision")?);
    let (placement_epoch, fencing_token) = take_state_binding(bytes, &mut at)?;
    let identity_digest = take_digest(bytes, &mut at)?;
    let session_id = SessionId(take_str(bytes, &mut at, "session_id")?);
    let policy_revision = Revision(take_u64(bytes, &mut at, "policy_revision")?);
    let capability = CapabilityId(take_str(bytes, &mut at, "capability")?);
    let expect_protocol = take_u32(bytes, &mut at, "expect_protocol")?;
    let type_id = TypeId(take_str(bytes, &mut at, "input_type")?);
    let payload = take_bytes(bytes, &mut at, "payload", MAX_BODY_BYTES)?;
    close(bytes, at)?;
    Ok(PodRequestV2 {
        address,
        execution_manifest,
        addressed_to,
        request_id,
        artifact_id,
        manifest_hash,
        generation,
        revision,
        placement_epoch,
        fencing_token,
        identity_digest,
        session_id,
        policy_revision,
        capability,
        expect_protocol,
        payload: TypedPayload {
            type_id,
            bytes: payload,
        },
    })
}

/// Encode an answer to a V2 or V3 request and sign it. See [`encode_answer`] on who
/// the signer is.
pub fn encode_answer_v2(answer: &PodAnswerV2, signer: &NodeSigner) -> Result<Vec<u8>, FrameError> {
    let mut frame = Vec::with_capacity(192 + SIGNATURE_BYTES);
    frame.extend_from_slice(ANSWER_MAGIC);
    let format = if answer.address.is_some() {
        SIGNED_ANSWER_V3
    } else {
        SIGNED_ANSWER_V2
    };
    frame.extend_from_slice(&format.to_le_bytes());
    if let Some(address) = &answer.address {
        put_address_binding(&mut frame, address)?;
        frame.extend_from_slice(&answer.execution_manifest.ok_or(FrameError::InvalidField {
            field: "execution_manifest",
        })?);
    } else if answer.execution_manifest.is_some() {
        return Err(FrameError::InvalidField {
            field: "execution_manifest",
        });
    }
    put_str(&mut frame, "responder", &answer.responder)?;
    frame.extend_from_slice(&answer.request_id.to_le_bytes());
    frame.extend_from_slice(&answer.request_digest);
    frame.extend_from_slice(&answer.generation.0.to_le_bytes());
    frame.extend_from_slice(&answer.revision.0.to_le_bytes());
    put_state_binding(&mut frame, answer.placement_epoch, answer.fencing_token)?;
    frame.extend_from_slice(&answer.identity_digest);
    put_str(&mut frame, "session_id", &answer.session_id.0)?;
    frame.extend_from_slice(&answer.policy_revision.0.to_le_bytes());
    frame.push(outcome_code(&answer.outcome));
    match &answer.outcome {
        PodOutcome::Answered { output } => {
            put_str(&mut frame, "output_type", &output.type_id.0)?;
            put_bytes(&mut frame, "output", &output.bytes, MAX_BODY_BYTES)?;
        }
        PodOutcome::Refused { code } => frame.extend_from_slice(&refusal_code(*code).to_le_bytes()),
    }
    sign_frame(frame, signer)
}

/// Read a signed V2/V3 answer without verifying it. Private: see
/// [`verify_answer_v2`].
fn parse_answer_v2(bytes: &[u8]) -> Result<Parsed<'_, PodAnswerV2>, FrameError> {
    let mut at = open_answer(bytes, &[SIGNED_ANSWER_V2, SIGNED_ANSWER_V3])?;
    let has_address = u16::from_le_bytes([bytes[8], bytes[9]]) == SIGNED_ANSWER_V3;
    let address = has_address
        .then(|| take_address_binding(bytes, &mut at))
        .transpose()?;
    let execution_manifest = has_address
        .then(|| take_digest(bytes, &mut at))
        .transpose()?;
    let responder = take_str(bytes, &mut at, "responder")?;
    let request_id = take_u64(bytes, &mut at, "request_id")?;
    let request_digest = take_digest(bytes, &mut at)?;
    let generation = Generation(take_u64(bytes, &mut at, "generation")?);
    let revision = Revision(take_u64(bytes, &mut at, "revision")?);
    let (placement_epoch, fencing_token) = take_state_binding(bytes, &mut at)?;
    let identity_digest = take_digest(bytes, &mut at)?;
    let session_id = SessionId(take_str(bytes, &mut at, "session_id")?);
    let policy_revision = Revision(take_u64(bytes, &mut at, "policy_revision")?);
    let outcome = match take_u8(bytes, &mut at, "outcome")? {
        1 => PodOutcome::Answered {
            output: TypedPayload {
                type_id: TypeId(take_str(bytes, &mut at, "output_type")?),
                bytes: take_bytes(bytes, &mut at, "output", MAX_BODY_BYTES)?,
            },
        },
        2 => PodOutcome::Refused {
            code: refusal_from_code(take_u16(bytes, &mut at, "refusal")?)?,
        },
        other => {
            return Err(FrameError::UnknownCode {
                field: "outcome",
                code: u64::from(other),
            })
        }
    };
    take_signature(
        bytes,
        at,
        PodAnswerV2 {
            address,
            execution_manifest,
            responder,
            request_id,
            request_digest,
            generation,
            revision,
            placement_epoch,
            fencing_token,
            identity_digest,
            session_id,
            policy_revision,
            outcome,
        },
    )
}

/// Read a signed V1 answer without verifying it. Private: see [`verify_answer`].
fn parse_answer(bytes: &[u8]) -> Result<Parsed<'_, PodAnswer>, FrameError> {
    let mut at = open_answer(bytes, &[SIGNED_ANSWER_V1])?;
    let responder = take_str(bytes, &mut at, "responder")?;
    let request_id = take_u64(bytes, &mut at, "request_id")?;
    let request_digest = take_digest(bytes, &mut at)?;
    let outcome = match take_u8(bytes, &mut at, "outcome")? {
        1 => PodOutcome::Answered {
            output: TypedPayload {
                type_id: TypeId(take_str(bytes, &mut at, "output_type")?),
                bytes: take_bytes(bytes, &mut at, "output", MAX_BODY_BYTES)?,
            },
        },
        2 => PodOutcome::Refused {
            code: refusal_from_code(take_u16(bytes, &mut at, "refusal")?)?,
        },
        other => {
            return Err(FrameError::UnknownCode {
                field: "outcome",
                code: u64::from(other),
            })
        }
    };
    take_signature(
        bytes,
        at,
        PodAnswer {
            responder,
            request_id,
            request_digest,
            outcome,
        },
    )
}

/// Refuse a frame this build built over the bound, rather than sending it.
fn bound_frame(frame: Vec<u8>) -> Result<Vec<u8>, FrameError> {
    if frame.len() > MAX_FRAME_BYTES {
        return Err(FrameError::TooLarge {
            field: "frame",
            bytes: frame.len(),
            limit: MAX_FRAME_BYTES,
        });
    }
    Ok(frame)
}

/// Check the bound, the magic and the layout version, and return where the fields
/// begin.
fn open(bytes: &[u8], magic: &[u8; 8]) -> Result<usize, FrameError> {
    open_format(bytes, magic, FORMAT_V1)
}

fn open_format(bytes: &[u8], magic: &[u8; 8], expected: u16) -> Result<usize, FrameError> {
    if bytes.len() > MAX_FRAME_BYTES {
        return Err(FrameError::TooLarge {
            field: "frame",
            bytes: bytes.len(),
            limit: MAX_FRAME_BYTES,
        });
    }
    if bytes.len() < magic.len() || &bytes[..magic.len()] != magic {
        return Err(FrameError::WrongKind);
    }
    let mut at = magic.len();
    let format = take_u16(bytes, &mut at, "format")?;
    if format != expected {
        return Err(FrameError::UnknownFormat { format });
    }
    Ok(at)
}

/// Check an answer's bound, magic and layout version against the signed layouts a
/// caller reads. The unsigned layouts are refused by name; a signed layout the
/// caller does not read is an unknown one, as it always was.
fn open_answer(bytes: &[u8], signed: &[u16]) -> Result<usize, FrameError> {
    if bytes.len() > MAX_FRAME_BYTES {
        return Err(FrameError::TooLarge {
            field: "frame",
            bytes: bytes.len(),
            limit: MAX_FRAME_BYTES,
        });
    }
    if bytes.len() < ANSWER_MAGIC.len() || &bytes[..ANSWER_MAGIC.len()] != ANSWER_MAGIC {
        return Err(FrameError::WrongKind);
    }
    let mut at = ANSWER_MAGIC.len();
    let format = take_u16(bytes, &mut at, "format")?;
    if signed.contains(&format) {
        return Ok(at);
    }
    match format {
        FORMAT_V1 | FORMAT_V2 | FORMAT_V3 => Err(FrameError::UnsignedAnswer { format }),
        other => Err(FrameError::UnknownFormat { format: other }),
    }
}

fn open_v2_compatible(bytes: &[u8], magic: &[u8; 8]) -> Result<(usize, bool), FrameError> {
    if bytes.len() > MAX_FRAME_BYTES {
        return Err(FrameError::TooLarge {
            field: "frame",
            bytes: bytes.len(),
            limit: MAX_FRAME_BYTES,
        });
    }
    if bytes.len() < magic.len() || &bytes[..magic.len()] != magic {
        return Err(FrameError::WrongKind);
    }
    let mut at = magic.len();
    let format = take_u16(bytes, &mut at, "format")?;
    match format {
        FORMAT_V2 => Ok((at, false)),
        FORMAT_V3 => Ok((at, true)),
        other => Err(FrameError::UnknownFormat { format: other }),
    }
}

/// Refuse bytes after the last field.
fn close(bytes: &[u8], at: usize) -> Result<(), FrameError> {
    if at != bytes.len() {
        return Err(FrameError::TrailingBytes {
            extra: bytes.len() - at,
        });
    }
    Ok(())
}

/// Write a bounded, length-prefixed string.
fn put_str(frame: &mut Vec<u8>, field: &'static str, value: &str) -> Result<(), FrameError> {
    put_bytes(frame, field, value.as_bytes(), MAX_FIELD_BYTES)
}

/// Write a bounded, length-prefixed byte string.
fn put_bytes(
    frame: &mut Vec<u8>,
    field: &'static str,
    value: &[u8],
    limit: usize,
) -> Result<(), FrameError> {
    if value.len() > limit {
        return Err(FrameError::TooLarge {
            field,
            bytes: value.len(),
            limit,
        });
    }
    frame.extend_from_slice(&(value.len() as u32).to_le_bytes());
    frame.extend_from_slice(value);
    Ok(())
}

fn put_state_binding(
    frame: &mut Vec<u8>,
    placement_epoch: Option<u64>,
    fencing_token: Option<u128>,
) -> Result<(), FrameError> {
    match (placement_epoch, fencing_token) {
        (None, None) => frame.push(0),
        (Some(epoch), Some(token)) => {
            frame.push(1);
            frame.extend_from_slice(&epoch.to_le_bytes());
            frame.extend_from_slice(&token.to_le_bytes());
        }
        _ => {
            return Err(FrameError::UnknownCode {
                field: "state_binding",
                code: 2,
            })
        }
    }
    Ok(())
}

fn put_address_binding(frame: &mut Vec<u8>, binding: &PodAddressBinding) -> Result<(), FrameError> {
    binding.validate()?;
    match &binding.source {
        Some(source) => {
            frame.push(1);
            put_address(frame, source)?;
        }
        None => frame.push(0),
    }
    put_address(frame, &binding.destination.address)?;
    frame.extend_from_slice(&binding.destination.semantic_revision);
    frame.extend_from_slice(&binding.destination.generation.0.to_le_bytes());
    put_str(frame, "trace_id", &binding.trace_id.0)?;
    frame.extend_from_slice(&binding.hop_limit.to_le_bytes());
    match &binding.protocol {
        Some(protocol) => {
            frame.push(1);
            put_protocol(frame, protocol)?;
        }
        None => frame.push(0),
    }
    match &binding.mesh {
        Some(mesh) => {
            mesh.validate()
                .map_err(|_| FrameError::InvalidField { field: "mesh" })?;
            frame.push(1);
            put_str(frame, "mesh_network", &mesh.network_id.0)?;
            put_str(frame, "mesh_peer", &mesh.peer_id.0)?;
            frame.push(match mesh.route {
                MeshRouteKind::Direct => 0,
                MeshRouteKind::Relay => 1,
            });
            match &mesh.relay_id {
                Some(relay) => {
                    frame.push(1);
                    put_str(frame, "mesh_relay", &relay.0)?;
                }
                None => frame.push(0),
            }
            frame.extend_from_slice(&mesh.membership_generation.0.to_le_bytes());
            frame.extend_from_slice(&mesh.membership_revision.0.to_le_bytes());
        }
        None => frame.push(0),
    }
    frame.extend_from_slice(&(binding.visited.len() as u16).to_le_bytes());
    for address in &binding.visited {
        put_address(frame, address)?;
    }
    Ok(())
}

fn take_address_binding(bytes: &[u8], at: &mut usize) -> Result<PodAddressBinding, FrameError> {
    let source = match take_u8(bytes, at, "source")? {
        0 => None,
        1 => Some(take_address(bytes, at, "source")?),
        code => {
            return Err(FrameError::UnknownCode {
                field: "source",
                code: u64::from(code),
            })
        }
    };
    let address = take_address(bytes, at, "destination")?;
    let semantic_revision = take_digest(bytes, at)?;
    let generation = Generation(take_u64(bytes, at, "generation")?);
    let trace_id = TraceId(take_str(bytes, at, "trace_id")?);
    let hop_limit = take_u16(bytes, at, "hop_limit")?;
    let protocol = match take_u8(bytes, at, "protocol")? {
        0 => None,
        1 => Some(take_protocol(bytes, at)?),
        code => {
            return Err(FrameError::UnknownCode {
                field: "protocol",
                code: u64::from(code),
            })
        }
    };
    let mesh = match take_u8(bytes, at, "mesh")? {
        0 => None,
        1 => {
            let network_id = ptr_types::NetworkId(take_str(bytes, at, "mesh_network")?);
            let peer_id = PeerId(take_str(bytes, at, "mesh_peer")?);
            let route = match take_u8(bytes, at, "mesh_route")? {
                0 => MeshRouteKind::Direct,
                1 => MeshRouteKind::Relay,
                code => {
                    return Err(FrameError::UnknownCode {
                        field: "mesh_route",
                        code: u64::from(code),
                    })
                }
            };
            let relay_id = match take_u8(bytes, at, "mesh_relay")? {
                0 => None,
                1 => Some(PeerId(take_str(bytes, at, "mesh_relay")?)),
                code => {
                    return Err(FrameError::UnknownCode {
                        field: "mesh_relay",
                        code: u64::from(code),
                    })
                }
            };
            Some(MeshEndpointBinding {
                network_id,
                peer_id,
                route,
                relay_id,
                membership_generation: Generation(take_u64(bytes, at, "mesh_generation")?),
                membership_revision: Revision(take_u64(bytes, at, "mesh_revision")?),
            })
        }
        code => {
            return Err(FrameError::UnknownCode {
                field: "mesh",
                code: u64::from(code),
            })
        }
    };
    let visited_len = take_u16(bytes, at, "visited_len")? as usize;
    if visited_len > 64 {
        return Err(FrameError::InvalidField { field: "visited" });
    }
    let mut visited = Vec::with_capacity(visited_len);
    for _ in 0..visited_len {
        visited.push(take_address(bytes, at, "visited")?);
    }
    let binding = PodAddressBinding {
        source,
        destination: PodRevisionAddress {
            address,
            semantic_revision,
            generation,
        },
        trace_id,
        hop_limit,
        visited,
        protocol,
        mesh,
    };
    binding.validate()?;
    Ok(binding)
}

fn put_protocol(frame: &mut Vec<u8>, protocol: &ProtocolBinding) -> Result<(), FrameError> {
    let name = match protocol {
        ProtocolBinding::InProcess => "in-process",
        ProtocolBinding::LocalWorker => "local-worker",
        ProtocolBinding::IrohQuic => "iroh-quic",
        ProtocolBinding::Tcp => "tcp",
        ProtocolBinding::Udp => "udp",
        ProtocolBinding::Ssh => "ssh",
        ProtocolBinding::Wireguard => "wireguard",
        ProtocolBinding::Mqtt => "mqtt",
        ProtocolBinding::Http => "http",
        ProtocolBinding::WebSocket => "websocket",
        ProtocolBinding::Grpc => "grpc",
        ProtocolBinding::Mcp => "mcp",
        ProtocolBinding::Cuda => "cuda",
        ProtocolBinding::Moq => "moq",
        ProtocolBinding::Rfb => "rfb",
        ProtocolBinding::Ipc => "ipc",
        ProtocolBinding::Mavlink => "mavlink",
        ProtocolBinding::Zmtp => "zmtp",
        ProtocolBinding::Socks5 => "socks5",
        ProtocolBinding::HttpConnect => "http-connect",
        ProtocolBinding::WebRtc => "webrtc",
        ProtocolBinding::JsonRpc => "json-rpc",
        ProtocolBinding::Custom(value) => value,
    };
    put_str(frame, "protocol", name)
}

fn take_protocol(bytes: &[u8], at: &mut usize) -> Result<ProtocolBinding, FrameError> {
    let value = take_str(bytes, at, "protocol")?;
    Ok(match value.as_str() {
        "in-process" => ProtocolBinding::InProcess,
        "local-worker" => ProtocolBinding::LocalWorker,
        "iroh-quic" => ProtocolBinding::IrohQuic,
        "tcp" => ProtocolBinding::Tcp,
        "udp" => ProtocolBinding::Udp,
        "ssh" => ProtocolBinding::Ssh,
        "wireguard" => ProtocolBinding::Wireguard,
        "mqtt" => ProtocolBinding::Mqtt,
        "http" => ProtocolBinding::Http,
        "websocket" => ProtocolBinding::WebSocket,
        "grpc" => ProtocolBinding::Grpc,
        "mcp" => ProtocolBinding::Mcp,
        "cuda" => ProtocolBinding::Cuda,
        "moq" => ProtocolBinding::Moq,
        "rfb" => ProtocolBinding::Rfb,
        "ipc" => ProtocolBinding::Ipc,
        "mavlink" => ProtocolBinding::Mavlink,
        "zmtp" => ProtocolBinding::Zmtp,
        "socks5" => ProtocolBinding::Socks5,
        "http-connect" => ProtocolBinding::HttpConnect,
        "webrtc" => ProtocolBinding::WebRtc,
        "json-rpc" => ProtocolBinding::JsonRpc,
        value if !value.is_empty() => ProtocolBinding::Custom(value.to_owned()),
        _ => return Err(FrameError::InvalidField { field: "protocol" }),
    })
}

fn put_address(frame: &mut Vec<u8>, address: &PodAddress) -> Result<(), FrameError> {
    put_str(frame, "project", &address.project.0)?;
    put_str(frame, "namespace", &address.namespace.0)?;
    put_str(frame, "pod_id", &address.pod_id.0)?;
    Ok(())
}

fn take_address(
    bytes: &[u8],
    at: &mut usize,
    field: &'static str,
) -> Result<PodAddress, FrameError> {
    let address = PodAddress {
        project: ptr_types::ProjectId(take_str(bytes, at, field)?),
        namespace: ptr_types::NamespaceId(take_str(bytes, at, field)?),
        pod_id: ptr_types::PodId(take_str(bytes, at, field)?),
    };
    address
        .validate()
        .map_err(|_| FrameError::InvalidField { field })?;
    Ok(address)
}

fn take_state_binding(
    bytes: &[u8],
    at: &mut usize,
) -> Result<(Option<u64>, Option<u128>), FrameError> {
    match take_u8(bytes, at, "state_binding")? {
        0 => Ok((None, None)),
        1 => {
            let epoch = take_u64(bytes, at, "placement_epoch")?;
            let token = u128::from_le_bytes(take_fixed::<16>(bytes, at, "fencing_token")?);
            Ok((Some(epoch), Some(token)))
        }
        code => Err(FrameError::UnknownCode {
            field: "state_binding",
            code: u64::from(code),
        }),
    }
}

/// Read a bounded, length-prefixed byte string.
///
/// The length is a claim: one past the bound is refused by the bound rather than by
/// reserving for it.
fn take_bytes(
    bytes: &[u8],
    at: &mut usize,
    field: &'static str,
    limit: usize,
) -> Result<Vec<u8>, FrameError> {
    let length = take_u32(bytes, at, field)? as usize;
    if length > limit {
        return Err(FrameError::TooLarge {
            field,
            bytes: length,
            limit,
        });
    }
    let end = at
        .checked_add(length)
        .ok_or(FrameError::Truncated { field })?;
    let slice = bytes.get(*at..end).ok_or(FrameError::Truncated { field })?;
    *at = end;
    Ok(slice.to_vec())
}

/// Read a bounded, length-prefixed string.
fn take_str(bytes: &[u8], at: &mut usize, field: &'static str) -> Result<String, FrameError> {
    let raw = take_bytes(bytes, at, field, MAX_FIELD_BYTES)?;
    String::from_utf8(raw).map_err(|_| FrameError::NotUtf8 { field })
}

/// Read a digest.
fn take_digest(bytes: &[u8], at: &mut usize) -> Result<[u8; 32], FrameError> {
    let field = "request_digest";
    let slice = bytes
        .get(*at..at.checked_add(32).ok_or(FrameError::Truncated { field })?)
        .ok_or(FrameError::Truncated { field })?;
    *at += 32;
    let mut digest = [0_u8; 32];
    digest.copy_from_slice(slice);
    Ok(digest)
}

/// Read a `u8`.
fn take_u8(bytes: &[u8], at: &mut usize, field: &'static str) -> Result<u8, FrameError> {
    let value = *bytes.get(*at).ok_or(FrameError::Truncated { field })?;
    *at += 1;
    Ok(value)
}

/// Read a little-endian `u16`.
fn take_u16(bytes: &[u8], at: &mut usize, field: &'static str) -> Result<u16, FrameError> {
    Ok(u16::from_le_bytes(take_fixed::<2>(bytes, at, field)?))
}

/// Read a little-endian `u32`.
fn take_u32(bytes: &[u8], at: &mut usize, field: &'static str) -> Result<u32, FrameError> {
    Ok(u32::from_le_bytes(take_fixed::<4>(bytes, at, field)?))
}

/// Read a little-endian `u64`.
fn take_u64(bytes: &[u8], at: &mut usize, field: &'static str) -> Result<u64, FrameError> {
    Ok(u64::from_le_bytes(take_fixed::<8>(bytes, at, field)?))
}

/// Read a fixed-width field.
fn take_fixed<const N: usize>(
    bytes: &[u8],
    at: &mut usize,
    field: &'static str,
) -> Result<[u8; N], FrameError> {
    let end = at.checked_add(N).ok_or(FrameError::Truncated { field })?;
    let slice = bytes.get(*at..end).ok_or(FrameError::Truncated { field })?;
    *at = end;
    let mut value = [0_u8; N];
    value.copy_from_slice(slice);
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A request with every field set to something distinguishable.
    fn request() -> PodRequest {
        PodRequest {
            addressed_to: "key-of-the-host".to_owned(),
            request_id: 7,
            capability: CapabilityId::from("summarize"),
            expect_protocol: 1,
            payload: TypedPayload {
                type_id: TypeId::from("Document"),
                bytes: b"a document to summarize".to_vec(),
            },
        }
    }

    /// The host's endpoint key.
    fn signer() -> NodeSigner {
        NodeSigner::from_test_seed([5; 32])
    }

    /// Somebody else's.
    fn stranger() -> NodeSigner {
        NodeSigner::from_test_seed([6; 32])
    }

    /// The id a verifier expects an answer from.
    fn responder() -> NodeId {
        NodeId(signer().public_key().to_owned())
    }

    fn answer() -> PodAnswer {
        PodAnswer {
            responder: signer().public_key().to_owned(),
            request_id: 7,
            request_digest: request_digest(b"whatever arrived"),
            outcome: PodOutcome::Answered {
                output: TypedPayload {
                    type_id: TypeId::from("Summary"),
                    bytes: b"summarized".to_vec(),
                },
            },
        }
    }

    #[test]
    fn a_request_and_an_answer_round_trip() {
        let request = request();
        let frame = encode_request(&request).unwrap();
        assert_eq!(decode_request(&frame).unwrap(), request);

        let answer = answer();
        let frame = encode_answer(&answer, &signer()).unwrap();
        assert_eq!(verify_answer(&frame, &responder()).unwrap(), answer);

        // Every refusal is an answer, not only the happy one. A round trip covering
        // one outcome would leave the eight that carry the protocol's decisions
        // untested.
        let mut outcomes = vec![PodOutcome::Answered {
            output: TypedPayload {
                type_id: TypeId::from("Empty"),
                bytes: Vec::new(),
            },
        }];
        outcomes.extend(
            RefusalCode::ALL
                .into_iter()
                .map(|code| PodOutcome::Refused { code }),
        );
        for outcome in outcomes {
            let answered = PodAnswer {
                outcome,
                ..answer.clone()
            };
            let frame = encode_answer(&answered, &signer()).unwrap();
            assert_eq!(verify_answer(&frame, &responder()).unwrap(), answered);
        }
    }

    #[test]
    fn v2_round_trip_binds_artifact_manifest_generation_and_revision() {
        let request = PodRequestV2 {
            address: Some(PodAddressBinding {
                source: None,
                destination: ptr_types::PodRevisionAddress {
                    address: ptr_types::PodAddress::new(
                        "project".into(),
                        "namespace".into(),
                        "pod".into(),
                    )
                    .unwrap(),
                    semantic_revision: [6; 32],
                    generation: Generation(3),
                },
                trace_id: "frame-test".into(),
                hop_limit: 8,
                visited: Vec::new(),
                protocol: Some(ProtocolBinding::Ssh),
                mesh: None,
            }),
            execution_manifest: Some([9; 32]),
            addressed_to: "key-of-the-host".into(),
            request_id: 42,
            artifact_id: ArtifactId::from("model-artifact"),
            manifest_hash: [7; 32],
            generation: Generation(3),
            revision: Revision(11),
            placement_epoch: Some(4),
            fencing_token: Some(99),
            identity_digest: [8; 32],
            session_id: SessionId::from("session-1"),
            policy_revision: Revision(5),
            capability: CapabilityId::from("summarize"),
            expect_protocol: 2,
            payload: TypedPayload {
                type_id: TypeId::from("Document"),
                bytes: b"input".to_vec(),
            },
        };
        let frame = encode_request_v2(&request).unwrap();
        assert_eq!(decode_request_v2(&frame).unwrap(), request);

        let answer = PodAnswerV2 {
            address: request.address.clone(),
            execution_manifest: request.execution_manifest,
            responder: signer().public_key().into(),
            request_id: 42,
            request_digest: request_digest(&frame),
            generation: Generation(3),
            revision: Revision(11),
            placement_epoch: Some(4),
            fencing_token: Some(99),
            identity_digest: [8; 32],
            session_id: SessionId::from("session-1"),
            policy_revision: Revision(5),
            outcome: PodOutcome::Answered {
                output: TypedPayload {
                    type_id: TypeId::from("Summary"),
                    bytes: b"output".to_vec(),
                },
            },
        };
        let answer_frame = encode_answer_v2(&answer, &signer()).unwrap();
        assert_eq!(
            u16::from_le_bytes([answer_frame[8], answer_frame[9]]),
            SIGNED_ANSWER_V3
        );
        assert_eq!(
            verify_answer_v2(&answer_frame, &responder()).unwrap(),
            answer
        );
    }

    #[test]
    fn legacy_v2_round_trip_remains_available_without_address_binding() {
        let mut request = PodRequestV2 {
            address: Some(PodAddressBinding {
                source: None,
                destination: ptr_types::PodRevisionAddress {
                    address: ptr_types::PodAddress::new(
                        "project".into(),
                        "namespace".into(),
                        "pod".into(),
                    )
                    .unwrap(),
                    semantic_revision: [6; 32],
                    generation: Generation(3),
                },
                trace_id: "frame-test".into(),
                hop_limit: 8,
                visited: Vec::new(),
                protocol: None,
                mesh: None,
            }),
            execution_manifest: Some([9; 32]),
            addressed_to: "key-of-the-host".into(),
            request_id: 42,
            artifact_id: ArtifactId::from("model-artifact"),
            manifest_hash: [7; 32],
            generation: Generation(3),
            revision: Revision(11),
            placement_epoch: Some(4),
            fencing_token: Some(99),
            identity_digest: [8; 32],
            session_id: SessionId::from("session-1"),
            policy_revision: Revision(5),
            capability: CapabilityId::from("summarize"),
            expect_protocol: 2,
            payload: TypedPayload {
                type_id: TypeId::from("Document"),
                bytes: b"input".to_vec(),
            },
        };
        request.address = None;
        request.execution_manifest = None;
        let frame = encode_request_v2(&request).unwrap();
        assert_eq!(u16::from_le_bytes([frame[8], frame[9]]), FORMAT_V2);
        assert_eq!(decode_request_v2(&frame).unwrap(), request);

        let answer = PodAnswerV2 {
            address: None,
            execution_manifest: None,
            responder: signer().public_key().into(),
            request_id: request.request_id,
            request_digest: request_digest(&frame),
            generation: request.generation,
            revision: request.revision,
            placement_epoch: request.placement_epoch,
            fencing_token: request.fencing_token,
            identity_digest: request.identity_digest,
            session_id: request.session_id.clone(),
            policy_revision: request.policy_revision,
            outcome: PodOutcome::Refused {
                code: RefusalCode::Unavailable,
            },
        };
        let answer_frame = encode_answer_v2(&answer, &signer()).unwrap();
        assert_eq!(
            u16::from_le_bytes([answer_frame[8], answer_frame[9]]),
            SIGNED_ANSWER_V2
        );
        assert_eq!(
            verify_answer_v2(&answer_frame, &responder()).unwrap(),
            answer
        );
    }

    #[test]
    fn an_outcome_kind_this_build_has_no_meaning_for_is_refused() {
        // Refused rather than read as the nearest kind this build knows. An outcome
        // a later build invents must not silently become "answered" with whatever
        // bytes follow.
        let answer = answer();
        let frame = encode_answer(&answer, &signer()).unwrap();
        let kind_at =
            frame.len() - SIGNATURE_BYTES - 1 - (4 + b"Summary".len()) - (4 + b"summarized".len());
        assert_eq!(
            frame[kind_at], 1,
            "the outcome kind is where it is expected"
        );
        let mut unknown = frame.clone();
        unknown[kind_at] = 9;
        assert_eq!(
            verify_answer(&unknown, &responder()).expect_err("an outcome kind from a later build"),
            AnswerError::Frame(FrameError::UnknownCode {
                field: "outcome",
                code: 9
            })
        );
    }

    #[test]
    fn the_protocol_version_a_requester_composed_for_survives_the_wire() {
        // It is the one field that exists to be compared against a Pod's manifest,
        // so a version that did not survive encoding would make every comparison
        // pass or every one fail without either being about the Pod.
        for expect_protocol in [0, 1, 2, u32::MAX] {
            let composed = PodRequest {
                expect_protocol,
                ..request()
            };
            let frame = encode_request(&composed).unwrap();
            assert_eq!(
                decode_request(&frame).unwrap().expect_protocol,
                expect_protocol
            );
        }
        // And two versions are two different frames rather than one.
        assert_ne!(
            encode_request(&PodRequest {
                expect_protocol: 1,
                ..request()
            })
            .unwrap(),
            encode_request(&PodRequest {
                expect_protocol: 2,
                ..request()
            })
            .unwrap()
        );
    }

    #[test]
    fn an_answer_is_never_read_as_a_request_and_a_request_never_as_an_answer() {
        // Decided by the magic, before a version or a field is looked at.
        let request = encode_request(&request()).unwrap();
        let answer = encode_answer(&answer(), &signer()).unwrap();
        assert_eq!(
            verify_answer(&request, &responder()).expect_err("a request is not an answer"),
            AnswerError::Frame(FrameError::WrongKind)
        );
        assert_eq!(
            decode_request(&answer).expect_err("an answer is not a request"),
            FrameError::WrongKind
        );
        // Nor is the execution wire's request, which is the neighbour this one is
        // most likely to be confused with: two protocols whose frames parsed as each
        // other would make the two ALPNs decorative.
        assert_eq!(
            decode_request(b"PTREXREQ\x01\x00").expect_err("the execution wire's frame"),
            FrameError::WrongKind
        );
    }

    #[test]
    fn every_prefix_of_each_frame_is_refused() {
        for frame in [
            encode_request(&request()).unwrap(),
            encode_answer(&answer(), &signer()).unwrap(),
        ] {
            let is_request = decode_request(&frame).is_ok();
            for length in 0..frame.len() {
                let error = if is_request {
                    decode_request(&frame[..length]).expect_err("a prefix is not a request")
                } else {
                    match verify_answer(&frame[..length], &responder()) {
                        Err(AnswerError::Frame(error)) => error,
                        other => panic!("a prefix of {length} is not an answer: {other:?}"),
                    }
                };
                assert!(
                    matches!(error, FrameError::WrongKind | FrameError::Truncated { .. }),
                    "prefix of {length}: {error:?}"
                );
            }
        }
    }

    #[test]
    fn no_single_bit_change_yields_the_original_request() {
        let original = request();
        let frame = encode_request(&original).unwrap();
        for index in 0..frame.len() {
            for bit in 0..8 {
                let mut damaged = frame.clone();
                damaged[index] ^= 1 << bit;
                if let Ok(decoded) = decode_request(&damaged) {
                    assert_ne!(
                        decoded, original,
                        "byte {index} bit {bit} decoded to the original request"
                    );
                }
            }
        }
        // The control: the frame this was damaged from does decode, so the assertion
        // above is about the damage rather than about a frame nothing can read.
        assert_eq!(decode_request(&frame).unwrap(), original);
    }

    #[test]
    fn trailing_bytes_and_a_layout_this_build_does_not_know_are_each_refused() {
        let mut trailing = encode_request(&request()).unwrap();
        trailing.extend_from_slice(&[0, 0, 0]);
        assert_eq!(
            decode_request(&trailing).expect_err("bytes after the request"),
            FrameError::TrailingBytes { extra: 3 }
        );

        let mut future = encode_request(&request()).unwrap();
        future[8..10].copy_from_slice(&2_u16.to_le_bytes());
        assert_eq!(
            decode_request(&future).expect_err("a layout this build does not know"),
            FrameError::UnknownFormat { format: 2 }
        );

        let mut future_answer = encode_answer(&answer(), &signer()).unwrap();
        future_answer[8..10].copy_from_slice(&9_u16.to_le_bytes());
        assert_eq!(
            verify_answer(&future_answer, &responder())
                .expect_err("an answer layout from elsewhere"),
            AnswerError::Frame(FrameError::UnknownFormat { format: 9 })
        );
    }

    #[test]
    fn a_length_a_peer_invented_is_refused_by_the_bound_rather_than_reserved_for() {
        // The length is a claim, not an instruction. The first field of a request is
        // a string, so its length prefix sits immediately after the layout version.
        let mut lying = encode_request(&request()).unwrap();
        let claim = (MAX_FIELD_BYTES + 1) as u32;
        lying[10..14].copy_from_slice(&claim.to_le_bytes());
        assert_eq!(
            decode_request(&lying).expect_err("a field larger than its bound"),
            FrameError::TooLarge {
                field: "addressed_to",
                bytes: MAX_FIELD_BYTES + 1,
                limit: MAX_FIELD_BYTES,
            }
        );

        // A plausible length with nothing behind it runs out of bytes instead.
        let mut short = encode_request(&request()).unwrap();
        short[10..14].copy_from_slice(&4096_u32.to_le_bytes());
        assert_eq!(
            decode_request(&short).expect_err("four thousand bytes are not there"),
            FrameError::Truncated {
                field: "addressed_to"
            }
        );
    }

    #[test]
    fn a_frame_over_the_bound_is_refused_before_it_is_parsed() {
        let oversize = vec![0_u8; MAX_FRAME_BYTES + 1];
        let answer_error = match verify_answer(&oversize, &responder()) {
            Err(AnswerError::Frame(error)) => error,
            other => panic!("over the bound: {other:?}"),
        };
        for error in [
            decode_request(&oversize).expect_err("over the bound"),
            answer_error,
        ] {
            assert_eq!(
                error,
                FrameError::TooLarge {
                    field: "frame",
                    bytes: MAX_FRAME_BYTES + 1,
                    limit: MAX_FRAME_BYTES,
                }
            );
        }

        // And this build refuses to *send* one, in both directions, rather than
        // handing a peer bytes it is required to refuse.
        let mut huge = request();
        huge.payload.bytes = vec![0; MAX_BODY_BYTES + 1];
        assert_eq!(
            encode_request(&huge).expect_err("a payload over its bound"),
            FrameError::TooLarge {
                field: "payload",
                bytes: MAX_BODY_BYTES + 1,
                limit: MAX_BODY_BYTES,
            }
        );
        let huge_answer = PodAnswer {
            outcome: PodOutcome::Answered {
                output: TypedPayload {
                    type_id: TypeId::from("Summary"),
                    bytes: vec![0; MAX_BODY_BYTES + 1],
                },
            },
            ..answer()
        };
        assert_eq!(
            encode_answer(&huge_answer, &signer()).expect_err("an output over its bound"),
            FrameError::TooLarge {
                field: "output",
                bytes: MAX_BODY_BYTES + 1,
                limit: MAX_BODY_BYTES,
            }
        );
    }

    #[test]
    fn every_refusal_crosses_the_wire_as_its_own_code_and_an_unknown_one_is_refused() {
        let mut codes: Vec<u16> = RefusalCode::ALL.iter().copied().map(refusal_code).collect();
        let total = codes.len();
        codes.sort_unstable();
        codes.dedup();
        assert_eq!(codes.len(), total, "two refusals share a wire code");
        for refusal in RefusalCode::ALL {
            assert_eq!(refusal_from_code(refusal_code(refusal)).unwrap(), refusal);
        }
        // Zero is deliberately not a code: a zeroed byte is what a truncated or
        // padded frame looks like, and it must not mean a refusal.
        assert_eq!(
            refusal_from_code(0).expect_err("zero is not a refusal"),
            FrameError::UnknownCode {
                field: "refusal",
                code: 0
            }
        );
        assert!(
            refusal_from_code(10).is_err(),
            "a refusal code from a later build"
        );
    }

    #[test]
    fn a_digest_is_over_the_bytes_that_arrived() {
        let frame = encode_request(&request()).unwrap();
        assert_eq!(request_digest(&frame), request_digest(&frame.clone()));

        let mut other = frame.clone();
        let last = other.len() - 1;
        other[last] ^= 1;
        assert_ne!(
            request_digest(&frame),
            request_digest(&other),
            "one bit changes the digest"
        );

        // Taken over bytes rather than over a decoded request, so a request this
        // build cannot parse still gets an answer its sender can check.
        let unparseable = b"not a request at all";
        assert!(decode_request(unparseable).is_err());
        assert_ne!(request_digest(unparseable), [0; 32]);

        // Length is part of the material, so a shorter frame is not a prefix of a
        // longer one's digest material.
        assert_ne!(request_digest(b"ab"), request_digest(b"abc"));
    }

    #[test]
    fn this_wire_s_digest_is_not_the_execution_wire_s_digest_of_the_same_bytes() {
        // Both domains are constants in two crates, so the separation is asserted
        // rather than assumed: the pod wire's domain string is written here, and the
        // execution wire's is written out in full so that a later edit to either has
        // to pass this test rather than quietly collide.
        let same_bytes = b"identical bytes on two wires";
        let mut execution = Sha256::new();
        execution.update(b"PTREXW01-REQUEST");
        execution.update((same_bytes.len() as u64).to_le_bytes());
        execution.update(same_bytes);
        let execution: [u8; 32] = execution.finalize().into();
        assert_ne!(request_digest(same_bytes), execution);
    }

    #[test]
    fn each_refusal_carries_its_own_diagnostic_code() {
        let frame_errors = [
            FrameError::WrongKind,
            FrameError::UnknownFormat { format: 2 },
            FrameError::Truncated { field: "x" },
            FrameError::TrailingBytes { extra: 1 },
            FrameError::TooLarge {
                field: "x",
                bytes: 1,
                limit: 0,
            },
            FrameError::UnknownCode {
                field: "x",
                code: 0,
            },
            FrameError::NotUtf8 { field: "x" },
            FrameError::InvalidField { field: "x" },
            FrameError::UnsignedAnswer { format: 1 },
        ];
        let mut codes: Vec<&str> = frame_errors.iter().map(FrameError::code).collect();
        codes.extend(RefusalCode::ALL.iter().map(|code| code.code()));
        codes.extend(
            [
                AnswerError::WrongResponder {
                    expected: String::new(),
                    claimed: String::new(),
                },
                AnswerError::BadSignature,
            ]
            .iter()
            .map(AnswerError::code),
        );
        let total = codes.len();
        codes.sort_unstable();
        codes.dedup();
        assert_eq!(codes.len(), total, "two refusals share a diagnostic code");
    }

    /// A V2 answer with every optional binding present, so every echoed field is in
    /// the signed bytes.
    fn bound_answer() -> PodAnswerV2 {
        PodAnswerV2 {
            address: Some(PodAddressBinding {
                source: None,
                destination: ptr_types::PodRevisionAddress {
                    address: ptr_types::PodAddress::new(
                        "project".into(),
                        "namespace".into(),
                        "pod".into(),
                    )
                    .unwrap(),
                    semantic_revision: [6; 32],
                    generation: Generation(3),
                },
                trace_id: "frame-test".into(),
                hop_limit: 8,
                visited: Vec::new(),
                protocol: Some(ProtocolBinding::Ssh),
                mesh: None,
            }),
            execution_manifest: Some([9; 32]),
            responder: signer().public_key().into(),
            request_id: 42,
            request_digest: request_digest(b"the request"),
            generation: Generation(3),
            revision: Revision(11),
            placement_epoch: Some(4),
            fencing_token: Some(99),
            identity_digest: [8; 32],
            session_id: SessionId::from("session-1"),
            policy_revision: Revision(5),
            outcome: PodOutcome::Answered {
                output: TypedPayload {
                    type_id: TypeId::from("Summary"),
                    bytes: b"output".to_vec(),
                },
            },
        }
    }

    /// `changed`, encoded, carrying the signature `original` was sent with: one field
    /// edited after signing, exactly as a stored answer would be.
    fn spliced(mut changed: Vec<u8>, original: &[u8]) -> Vec<u8> {
        changed.truncate(changed.len() - SIGNATURE_BYTES);
        changed.extend_from_slice(&original[original.len() - SIGNATURE_BYTES..]);
        changed
    }

    #[test]
    fn a_stored_answer_verifies_offline_and_a_change_to_any_field_is_refused() {
        // The control first: stored bytes, untouched, verify with nothing but the
        // responder's id.
        let stored = encode_answer(&answer(), &signer()).unwrap();
        assert_eq!(verify_answer(&stored, &responder()), Ok(answer()));
        let edits = [
            (
                "request_id",
                PodAnswer {
                    request_id: 8,
                    ..answer()
                },
            ),
            (
                "request_digest",
                PodAnswer {
                    request_digest: request_digest(b"another request"),
                    ..answer()
                },
            ),
            (
                "output",
                PodAnswer {
                    outcome: PodOutcome::Answered {
                        output: TypedPayload {
                            type_id: TypeId::from("Summary"),
                            bytes: b"summarizeD".to_vec(),
                        },
                    },
                    ..answer()
                },
            ),
            (
                "output type",
                PodAnswer {
                    outcome: PodOutcome::Answered {
                        output: TypedPayload {
                            type_id: TypeId::from("Summarx"),
                            bytes: b"summarized".to_vec(),
                        },
                    },
                    ..answer()
                },
            ),
            (
                "outcome",
                PodAnswer {
                    outcome: PodOutcome::Refused {
                        code: RefusalCode::Unavailable,
                    },
                    ..answer()
                },
            ),
        ];
        for (field, changed) in edits {
            let forged = spliced(encode_answer(&changed, &signer()).unwrap(), &stored);
            assert_eq!(
                verify_answer(&forged, &responder()),
                Err(AnswerError::BadSignature),
                "{field} edited after signing"
            );
        }

        // Every field a V2/V3 answer echoes is signed too: each edited after signing
        // is refused, and the untouched answer is the control.
        let stored_v2 = encode_answer_v2(&bound_answer(), &signer()).unwrap();
        assert_eq!(
            verify_answer_v2(&stored_v2, &responder()),
            Ok(bound_answer())
        );
        let mut edits_v2: Vec<(&str, PodAnswerV2)> = Vec::new();
        let mut edit = |field, change: fn(&mut PodAnswerV2)| {
            let mut answer = bound_answer();
            change(&mut answer);
            edits_v2.push((field, answer));
        };
        edit("request_id", |a| a.request_id += 1);
        edit("request_digest", |a| a.request_digest[0] ^= 1);
        edit("generation", |a| a.generation = Generation(4));
        edit("revision", |a| a.revision = Revision(12));
        edit("placement_epoch", |a| a.placement_epoch = Some(5));
        edit("fencing_token", |a| a.fencing_token = Some(100));
        edit("identity_digest", |a| a.identity_digest[0] ^= 1);
        edit("session_id", |a| {
            a.session_id = SessionId::from("session-2")
        });
        edit("policy_revision", |a| a.policy_revision = Revision(6));
        edit("execution_manifest", |a| {
            a.execution_manifest = Some([1; 32])
        });
        edit("address hop_limit", |a| {
            a.address.as_mut().unwrap().hop_limit = 7;
        });
        edit("address trace_id", |a| {
            a.address.as_mut().unwrap().trace_id = "frame-tesT".into();
        });
        edit("outcome", |a| {
            a.outcome = PodOutcome::Refused {
                code: RefusalCode::PodFailed,
            };
        });
        for (field, changed) in edits_v2 {
            let forged = spliced(encode_answer_v2(&changed, &signer()).unwrap(), &stored_v2);
            assert_eq!(
                verify_answer_v2(&forged, &responder()),
                Err(AnswerError::BadSignature),
                "{field} edited after signing"
            );
        }

        // And every single bit of the stored V1 frame, signature included.
        for index in 0..stored.len() {
            for bit in 0..8 {
                let mut damaged = stored.clone();
                damaged[index] ^= 1 << bit;
                assert!(
                    verify_answer(&damaged, &responder()).is_err(),
                    "V1 byte {index} bit {bit} still verifies"
                );
            }
        }
        // Every byte of the larger V3 frame, one bit each, rotating through the bit
        // positions: the same coverage of fields at an eighth of the verifications.
        for index in 0..stored_v2.len() {
            let mut damaged = stored_v2.clone();
            damaged[index] ^= 1 << (index % 8);
            assert!(
                verify_answer_v2(&damaged, &responder()).is_err(),
                "V3 byte {index} still verifies"
            );
        }
    }

    #[test]
    fn an_answer_signed_by_another_key_or_readdressed_to_another_responder_is_refused() {
        let stored = encode_answer(&answer(), &signer()).unwrap();
        let other = NodeId(stranger().public_key().to_owned());

        // Names the right responder, signed by somebody else.
        assert_eq!(
            verify_answer(
                &encode_answer(&answer(), &stranger()).unwrap(),
                &responder()
            ),
            Err(AnswerError::BadSignature)
        );
        // Presented as another responder's: refused on the name.
        assert_eq!(
            verify_answer(&stored, &other),
            Err(AnswerError::WrongResponder {
                expected: other.0.clone(),
                claimed: responder().0,
            })
        );
        // Re-addressed to the other key with the original signature.
        let readdressed = PodAnswer {
            responder: other.0.clone(),
            ..answer()
        };
        let forged = spliced(encode_answer(&readdressed, &signer()).unwrap(), &stored);
        assert_eq!(
            verify_answer(&forged, &other),
            Err(AnswerError::BadSignature)
        );
        // A V1 answer is not a V2 answer even with its signature intact: the layout
        // version is in the signed bytes and each reader takes only its own.
        assert_eq!(
            verify_answer_v2(&stored, &responder()),
            Err(AnswerError::Frame(FrameError::UnknownFormat {
                format: SIGNED_ANSWER_V1
            }))
        );
        // The control.
        assert_eq!(verify_answer(&stored, &responder()), Ok(answer()));
    }

    #[test]
    fn an_unsigned_answer_from_an_earlier_build_is_refused_by_version() {
        // Exactly the layouts earlier builds wrote: the same fields under the old
        // version number and no signature.
        let signed_v1 = encode_answer(&answer(), &signer()).unwrap();
        let mut unsigned_v1 = signed_v1[..signed_v1.len() - SIGNATURE_BYTES].to_vec();
        unsigned_v1[8..10].copy_from_slice(&FORMAT_V1.to_le_bytes());
        assert_eq!(
            verify_answer(&unsigned_v1, &responder()),
            Err(AnswerError::Frame(FrameError::UnsignedAnswer {
                format: FORMAT_V1
            }))
        );

        let mut legacy = bound_answer();
        legacy.address = None;
        legacy.execution_manifest = None;
        for (answer, old) in [(bound_answer(), FORMAT_V3), (legacy, FORMAT_V2)] {
            let signed = encode_answer_v2(&answer, &signer()).unwrap();
            let mut unsigned = signed[..signed.len() - SIGNATURE_BYTES].to_vec();
            unsigned[8..10].copy_from_slice(&old.to_le_bytes());
            assert_eq!(
                verify_answer_v2(&unsigned, &responder()),
                Err(AnswerError::Frame(FrameError::UnsignedAnswer {
                    format: old
                }))
            );
            // The control: the frame it was made from verifies.
            assert_eq!(verify_answer_v2(&signed, &responder()), Ok(answer));
        }
        assert_eq!(verify_answer(&signed_v1, &responder()), Ok(answer()));
    }

    #[test]
    fn a_field_that_is_not_utf8_is_refused_rather_than_replaced() {
        // Lossy decoding would turn a capability nobody wrote into one that might
        // match a scope.
        let mut frame = encode_request(&request()).unwrap();
        let at = frame
            .windows(9)
            .position(|window| window == b"summarize")
            .expect("the capability is in the frame");
        frame[at] = 0xff;
        assert_eq!(
            decode_request(&frame).expect_err("invalid UTF-8 in a field"),
            FrameError::NotUtf8 {
                field: "capability"
            }
        );
    }
}
