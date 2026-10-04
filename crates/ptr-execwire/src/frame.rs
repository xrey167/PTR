//! The wire format for one execution request and the receipt that answers it.
//!
//! Two shapes, two magics. A receipt is never read as a request and a request
//! never as a receipt, decided before a version is even looked at, because the
//! two mean opposite things: one asks for an effect and the other reports one.
//!
//! **A request carries no field naming its sender.** There is nothing to forge
//! because there is nothing to write: who is asking is the authenticated
//! connection's answer, and the principal an action is audited under comes from
//! host policy. What a request does name is whom it is *for*, so a request
//! captured from one runtime cannot be replayed at another and be accepted as
//! meant for it.
//!
//! Every field is length-prefixed and bounded, every enumeration crosses the wire
//! as an explicit code from a two-way table rather than a cast of a discriminant,
//! and a frame that disagrees with itself is refused whole. A partially read
//! request would be an action nobody composed.
//!
//! **A receipt is signed** (D2 in issue #23) by the responder's endpoint key, which
//! is the key its `responder` field names and the key a connection to it
//! authenticates. What the signature attests, and what it does not, is written on
//! [`verify_receipt`]; the bytes it covers are written on [`RECEIPT_DOMAIN`].
use ptr_core::action_head::ActionIr;
use ptr_net::{NodeSigner, SigningDomain, SIGNATURE_BYTES};
use ptr_types::{CapabilityId, Effect, Generation, NodeId, ProjectId, Revision, TypeId};
use sha2::{Digest, Sha256};
use std::fmt;

/// Marks a request. Foreign bytes are refused before anything is interpreted.
const REQUEST_MAGIC: &[u8; 8] = b"PTREXREQ";

/// Marks a receipt.
const RECEIPT_MAGIC: &[u8; 8] = b"PTREXRCP";

/// Domain separation for a request digest, so that a digest of these bytes cannot
/// collide with a digest taken over anything else in the system.
const DIGEST_DOMAIN: &[u8] = b"PTREXW01-REQUEST";

/// The only request layout this build writes and reads.
///
/// It is also the layout of the *unsigned* receipt earlier builds wrote. A receipt
/// carrying it is refused by name ([`FrameError::UnsignedReceipt`]) rather than as
/// an unknown layout: this build knows exactly what it is, and that is why it is
/// refused.
pub const FORMAT_V1: u16 = 1;

/// The only receipt layout this build writes and reads: the V1 fields followed by
/// a signature.
///
/// Requests and receipts are versioned independently from here on. A request
/// carries nothing a signature would change, so its layout stays where it was.
pub const RECEIPT_FORMAT_V2: u16 = 2;

/// The domain a receipt signature is made in.
///
/// The signed message is this tag followed by **every byte of the receipt frame
/// before the signature**: the magic, the layout version, the length-prefixed
/// `responder`, `request_id`, `request_digest`, the outcome code and the outcome's
/// own fields (the response, or the refusal code). Every field the frame carries is
/// covered because the frame *is* the message — there is no second encoding of the
/// same fields to drift from the first, and a field added to a later layout is
/// signed without anyone remembering to sign it.
///
/// No position in the host's ledger is signed, because none is carried: a receipt
/// deliberately names none (see [`WireOutcome`]). Signing one in would hand every
/// requester a measure of every other principal's traffic.
pub const RECEIPT_DOMAIN: SigningDomain = SigningDomain::new(b"ptr-exec-receipt/v2\0");

/// Bound on one frame in either direction. A peer does not get to decide how much
/// a receiver reads.
pub const MAX_FRAME_BYTES: usize = 8 * 1024 * 1024;

/// Bound on one identifier-shaped field.
pub const MAX_FIELD_BYTES: usize = 64 * 1024;

/// Bound on an action's payload, and on a response carried back.
pub const MAX_BODY_BYTES: usize = 1024 * 1024;

/// An action requested across a network boundary.
///
/// Decoding one of these authorizes nothing. It is a request, not a permit: the
/// only type in this system that carries execution authority is constructed inside
/// the runtime and cannot be built from bytes.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WireRequest {
    /// The public key of the endpoint the requester believes it is addressing.
    ///
    /// Checked by the receiver against its own key. Without it, a request that was
    /// legitimate at one runtime could be relayed to another and be indistinguishable
    /// from one composed for it.
    pub addressed_to: String,
    /// A nonce for this request, unique per requester within the receiver's replay
    /// window. It is not a retry key: retrying the same intent means a new
    /// `request_id` and the same `once_key`.
    pub request_id: u64,
    /// The project the action belongs to. Which project a grant covers is the
    /// grant's business, so naming it here cannot widen anything.
    pub project: ProjectId,
    /// The action itself.
    pub action: ActionIr,
    /// An at-most-once key, when the requester means "this one again" rather than
    /// "do it once more".
    pub once_key: Option<String>,
}

/// What a runtime reports back about one request.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WireReceipt {
    /// The public key of the endpoint that produced this receipt.
    ///
    /// A receipt is a claim that may outlive the connection that carried it, so it
    /// names its own author. The check that makes the name mean anything is the
    /// requester comparing it against the peer the connection authenticated — which
    /// a receipt read from a file rather than a connection cannot have.
    pub responder: String,
    /// The `request_id` this answers.
    pub request_id: u64,
    /// A digest over the exact bytes that arrived, so a receipt is bound to one
    /// request and cannot be presented as the answer to another.
    pub request_digest: [u8; 32],
    /// What happened.
    pub outcome: WireOutcome,
}

/// The outcome a receipt reports.
///
/// Four answers rather than two, because "it worked" and "it did not" are not the
/// only things that can be true of an effect. A requester that read an uncertain
/// outcome as a refusal and retried would apply it twice, which is the whole reason
/// the runtime keeps a fence; collapsing that onto the wire would throw the fence
/// away at the last step.
///
/// None of these names a position in the receiver's ledger. How far that ledger has
/// got is a fact about every other principal's activity as well, and a requester is
/// entitled to the outcome of its own request and not to a measure of the host's
/// traffic. An operator reconciles an uncertain attempt from the host's side; the
/// requester's handle on it is its own at-most-once key.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum WireOutcome {
    /// The effect applied and this is what it returned.
    Applied {
        /// What the adapter returned.
        response: Vec<u8>,
    },
    /// The effect applied and the response is not available: either the receiver no
    /// longer retains it, or it does not fit in a frame. From here those are the
    /// same fact — it applied, and the bytes are not coming.
    AppliedWithoutResponse,
    /// The effect may have applied and the receiver does not know. **Not** a
    /// refusal: a retry without an at-most-once key could apply it a second time.
    Uncertain,
    /// Nothing was attempted. The code says only as much as a requester is entitled
    /// to know, with one exception: at-most-once keys are one namespace per host
    /// runtime, so a `Runtime` refusal under a key can tell a requester that another
    /// principal spent it.
    Refused {
        /// Why, at the coarseness the wire allows.
        code: RefusalCode,
    },
}

/// Why a request was refused, at the resolution the wire deliberately stops at.
///
/// Every refusal that is about **authority** — a peer the host never admitted, an
/// action outside the grant, a project the grant does not cover, an expired permit,
/// a revoked generation, a verification that failed, a fenced runtime — is the one
/// `Runtime` code. A requester that could tell those apart would have an oracle
/// for the host's policy, which is the same reason a cross-project Pod request is
/// refused identically to a Pod that does not exist.
///
/// The others are not about authority and tell the requester nothing it did not
/// already know: how it framed its own bytes, which key it addressed, and which
/// `request_id` it chose.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RefusalCode {
    /// The runtime refused. One code for every reason of authority.
    Runtime,
    /// The request did not decode.
    Malformed,
    /// The request named another endpoint's key.
    Misaddressed,
    /// This `request_id` is already in the receiver's window for this peer.
    Replayed,
}

impl RefusalCode {
    /// Stable diagnostic code, for a log rather than for the wire.
    pub fn code(self) -> &'static str {
        match self {
            Self::Runtime => "PTR_EXECW_REFUSED",
            Self::Malformed => "PTR_EXECW_MALFORMED",
            Self::Misaddressed => "PTR_EXECW_MISADDRESSED",
            Self::Replayed => "PTR_EXECW_REPLAYED",
        }
    }

    /// Every refusal this build can send, so a test can hold the table to account.
    pub const ALL: [Self; 4] = [
        Self::Runtime,
        Self::Malformed,
        Self::Misaddressed,
        Self::Replayed,
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
/// None of these yields a partial frame. An action assembled from the fields that
/// happened to parse is an action nobody asked for.
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
    /// A receipt in the unsigned layout earlier builds wrote ([`FORMAT_V1`]).
    ///
    /// Its own refusal rather than [`Self::UnknownFormat`], because this build knows
    /// the layout and refuses it on purpose: an unsigned receipt is exactly the
    /// hearsay a signature exists to replace, and reading one alongside signed ones
    /// would let a stripped signature pass as an old peer.
    UnsignedReceipt,
}

impl FrameError {
    /// Stable diagnostic code for this refusal.
    pub fn code(&self) -> &'static str {
        match self {
            Self::WrongKind => "PTR_EXECW_WRONG_KIND",
            Self::UnknownFormat { .. } => "PTR_EXECW_UNKNOWN_FORMAT",
            Self::Truncated { .. } => "PTR_EXECW_TRUNCATED",
            Self::TrailingBytes { .. } => "PTR_EXECW_TRAILING_BYTES",
            Self::TooLarge { .. } => "PTR_EXECW_FRAME_TOO_LARGE",
            Self::UnknownCode { .. } => "PTR_EXECW_UNKNOWN_CODE",
            Self::NotUtf8 { .. } => "PTR_EXECW_NOT_UTF8",
            Self::UnsignedReceipt => "PTR_EXECW_UNSIGNED_RECEIPT",
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

/// Why a receipt did not verify.
///
/// Three refusals, checked in this order: the bytes must be a well-formed signed
/// receipt, it must name the responder it is being verified against, and the
/// signature must be that responder's over the frame.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ReceiptError {
    /// The bytes are not a well-formed signed receipt.
    Frame(FrameError),
    /// The receipt names an author other than the one it is verified against.
    ///
    /// Checked before the signature, because a receipt from somebody else is not
    /// worth verifying: a valid signature by the wrong author attests nothing about
    /// the right one.
    WrongResponder { expected: String, claimed: String },
    /// The signature is not the named responder's over these bytes.
    BadSignature,
}

impl ReceiptError {
    /// Stable diagnostic code for this refusal.
    pub fn code(&self) -> &'static str {
        match self {
            Self::Frame(error) => error.code(),
            Self::WrongResponder { .. } => "PTR_EXECW_WRONG_RESPONDER",
            Self::BadSignature => "PTR_EXECW_BAD_SIGNATURE",
        }
    }
}

impl From<FrameError> for ReceiptError {
    /// Carry a framing refusal through unchanged.
    fn from(error: FrameError) -> Self {
        Self::Frame(error)
    }
}

impl fmt::Display for ReceiptError {
    /// Render the stable refusal code.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.code())
    }
}

impl std::error::Error for ReceiptError {}

/// The digest a receipt is bound to: taken over the bytes that actually arrived.
///
/// Over the received bytes rather than over a decoded request, so that even a
/// request this build cannot parse still gets a receipt its sender can check.
pub fn request_digest(bytes: &[u8]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(DIGEST_DOMAIN);
    hasher.update((bytes.len() as u64).to_le_bytes());
    hasher.update(bytes);
    hasher.finalize().into()
}

/// The wire's name for an effect class.
///
/// An explicit table in both directions. A discriminant is an implementation
/// detail of this build and a wire code is a promise to every other one; reordering
/// the enum would silently redefine `Mutation` as `External` if the two were the
/// same number.
fn effect_code(effect: Effect) -> u8 {
    match effect {
        Effect::Pure => 1,
        Effect::Read => 2,
        Effect::Mutation => 3,
        Effect::External => 4,
        Effect::Irreversible => 5,
    }
}

/// Read an effect class, or refuse a code this build has no meaning for.
fn effect_from_code(code: u8) -> Result<Effect, FrameError> {
    match code {
        1 => Ok(Effect::Pure),
        2 => Ok(Effect::Read),
        3 => Ok(Effect::Mutation),
        4 => Ok(Effect::External),
        5 => Ok(Effect::Irreversible),
        other => Err(FrameError::UnknownCode {
            field: "effect",
            code: u64::from(other),
        }),
    }
}

/// The wire's name for an outcome kind.
fn outcome_code(outcome: &WireOutcome) -> u8 {
    match outcome {
        WireOutcome::Applied { .. } => 1,
        WireOutcome::AppliedWithoutResponse => 2,
        WireOutcome::Uncertain => 3,
        WireOutcome::Refused { .. } => 4,
    }
}

/// The wire's name for a refusal.
fn refusal_code(code: RefusalCode) -> u16 {
    match code {
        RefusalCode::Runtime => 1,
        RefusalCode::Malformed => 2,
        RefusalCode::Misaddressed => 3,
        RefusalCode::Replayed => 4,
    }
}

/// Read a refusal, or refuse a code this build has no meaning for.
fn refusal_from_code(code: u16) -> Result<RefusalCode, FrameError> {
    match code {
        1 => Ok(RefusalCode::Runtime),
        2 => Ok(RefusalCode::Malformed),
        3 => Ok(RefusalCode::Misaddressed),
        4 => Ok(RefusalCode::Replayed),
        other => Err(FrameError::UnknownCode {
            field: "refusal",
            code: u64::from(other),
        }),
    }
}

/// Encode a request.
pub fn encode_request(request: &WireRequest) -> Result<Vec<u8>, FrameError> {
    let mut frame = Vec::with_capacity(256 + request.action.payload.len());
    frame.extend_from_slice(REQUEST_MAGIC);
    frame.extend_from_slice(&FORMAT_V1.to_le_bytes());
    put_str(&mut frame, "addressed_to", &request.addressed_to)?;
    frame.extend_from_slice(&request.request_id.to_le_bytes());
    put_str(&mut frame, "project", &request.project.0)?;
    let action = &request.action;
    put_str(&mut frame, "operation", &action.operation)?;
    put_str(&mut frame, "target", &action.target)?;
    put_str(&mut frame, "capability", &action.capability.0)?;
    put_str(&mut frame, "input_type", &action.input_type.0)?;
    frame.push(effect_code(action.effect));
    frame.extend_from_slice(&action.generation.0.to_le_bytes());
    frame.extend_from_slice(&action.revision.0.to_le_bytes());
    put_bytes(&mut frame, "payload", &action.payload, MAX_BODY_BYTES)?;
    match &request.once_key {
        // Presence is its own byte rather than a zero length, because "no key" and
        // "the empty key" are different requests.
        None => frame.push(0),
        Some(key) => {
            frame.push(1);
            put_str(&mut frame, "once_key", key)?;
        }
    }
    bound_frame(frame)
}

/// Decode a request, or refuse it.
pub fn decode_request(bytes: &[u8]) -> Result<WireRequest, FrameError> {
    let mut at = open(bytes, REQUEST_MAGIC)?;
    let addressed_to = take_str(bytes, &mut at, "addressed_to")?;
    let request_id = take_u64(bytes, &mut at, "request_id")?;
    let project = ProjectId(take_str(bytes, &mut at, "project")?);
    let operation = take_str(bytes, &mut at, "operation")?;
    let target = take_str(bytes, &mut at, "target")?;
    let capability = CapabilityId(take_str(bytes, &mut at, "capability")?);
    let input_type = TypeId(take_str(bytes, &mut at, "input_type")?);
    let effect = effect_from_code(take_u8(bytes, &mut at, "effect")?)?;
    let generation = Generation(take_u64(bytes, &mut at, "generation")?);
    let revision = Revision(take_u64(bytes, &mut at, "revision")?);
    let payload = take_bytes(bytes, &mut at, "payload", MAX_BODY_BYTES)?;
    let once_key = match take_u8(bytes, &mut at, "once_key_present")? {
        0 => None,
        1 => Some(take_str(bytes, &mut at, "once_key")?),
        other => {
            return Err(FrameError::UnknownCode {
                field: "once_key_present",
                code: u64::from(other),
            })
        }
    };
    close(bytes, at)?;
    Ok(WireRequest {
        addressed_to,
        request_id,
        project,
        action: ActionIr {
            operation,
            target,
            capability,
            effect,
            input_type,
            generation,
            revision,
            payload,
        },
        once_key,
    })
}

/// Encode a receipt and sign it.
///
/// The signer is whoever is answering; nothing here checks that it is the key the
/// receipt's `responder` names. A receipt that names one key and is signed by
/// another is refused by every verifier, and making it impossible to write would
/// only make that refusal impossible to test.
pub fn encode_receipt(receipt: &WireReceipt, signer: &NodeSigner) -> Result<Vec<u8>, FrameError> {
    let mut frame = Vec::with_capacity(128 + SIGNATURE_BYTES);
    frame.extend_from_slice(RECEIPT_MAGIC);
    frame.extend_from_slice(&RECEIPT_FORMAT_V2.to_le_bytes());
    put_str(&mut frame, "responder", &receipt.responder)?;
    frame.extend_from_slice(&receipt.request_id.to_le_bytes());
    frame.extend_from_slice(&receipt.request_digest);
    frame.push(outcome_code(&receipt.outcome));
    match &receipt.outcome {
        WireOutcome::Applied { response } => {
            put_bytes(&mut frame, "response", response, MAX_BODY_BYTES)?;
        }
        // Two outcomes carry nothing beyond their own kind. That is the point of
        // them: the receiver is saying what it does not know.
        WireOutcome::AppliedWithoutResponse | WireOutcome::Uncertain => {}
        WireOutcome::Refused { code } => {
            frame.extend_from_slice(&refusal_code(*code).to_le_bytes());
        }
    }
    let signature = signer.sign(RECEIPT_DOMAIN, &frame);
    frame.extend_from_slice(&signature);
    bound_frame(frame)
}

/// Verify a receipt as having been written by `responder`, and return it.
///
/// This is the only way to read a receipt out of bytes, on purpose: a receipt that
/// has not been verified is not one this crate hands anybody. It needs no
/// connection, so a receipt stored for a year is checked exactly as one that
/// arrived a moment ago.
///
/// **What a receipt that verifies attests:** that the runtime holding the endpoint
/// key `responder` recorded outcome *O* for the request id *X* it was asked under,
/// bound to the request digest *D* — the digest of the exact bytes it received.
/// Every word of that is about the responder's own statement.
///
/// **What it does not attest:**
/// - that the effect happened. `Applied` is the runtime's record of what its adapter
///   reported, and the signature makes that record attributable, not true;
/// - anything about a principal other than the responder — not who sent the request
///   (a request names no sender), not which principal the host audited it under, not
///   anything about another runtime;
/// - when it was made, or that it is the only receipt for *X*: nothing in it is a
///   timestamp or a position in the host's ledger;
/// - that `responder` is still trusted. A signature outlives its key's admission,
///   and whether a receipt from a withdrawn peer still counts is the verifier's
///   policy. A rotated key is a new identity, so a receipt signed by the old key
///   verifies under the old key and under nothing else.
///
/// The checks run in the order they are worth doing: a frame that is not a signed
/// receipt is refused first, then a receipt naming another author, and only then
/// the signature.
pub fn verify_receipt(bytes: &[u8], responder: &NodeId) -> Result<WireReceipt, ReceiptError> {
    let (receipt, signed, signature) = parse_receipt(bytes)?;
    if receipt.responder != responder.0 {
        return Err(ReceiptError::WrongResponder {
            expected: responder.0.clone(),
            claimed: receipt.responder,
        });
    }
    if !ptr_net::verify(&responder.0, RECEIPT_DOMAIN, signed, &signature) {
        return Err(ReceiptError::BadSignature);
    }
    Ok(receipt)
}

/// A signed receipt's fields, the bytes its signature covers, and the signature —
/// read without checking it. Private: see [`verify_receipt`].
type ParsedReceipt<'a> = (WireReceipt, &'a [u8], [u8; SIGNATURE_BYTES]);

/// Read a signed receipt without verifying it.
fn parse_receipt(bytes: &[u8]) -> Result<ParsedReceipt<'_>, FrameError> {
    let mut at = open_receipt(bytes)?;
    let responder = take_str(bytes, &mut at, "responder")?;
    let request_id = take_u64(bytes, &mut at, "request_id")?;
    let request_digest = take_digest(bytes, &mut at)?;
    let outcome = match take_u8(bytes, &mut at, "outcome")? {
        1 => WireOutcome::Applied {
            response: take_bytes(bytes, &mut at, "response", MAX_BODY_BYTES)?,
        },
        2 => WireOutcome::AppliedWithoutResponse,
        3 => WireOutcome::Uncertain,
        4 => WireOutcome::Refused {
            code: refusal_from_code(take_u16(bytes, &mut at, "refusal")?)?,
        },
        other => {
            return Err(FrameError::UnknownCode {
                field: "outcome",
                code: u64::from(other),
            })
        }
    };
    let signed = &bytes[..at];
    let signature = take_fixed::<SIGNATURE_BYTES>(bytes, &mut at, "signature")?;
    close(bytes, at)?;
    Ok((
        WireReceipt {
            responder,
            request_id,
            request_digest,
            outcome,
        },
        signed,
        signature,
    ))
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
    let (at, format) = open_any(bytes, magic)?;
    if format != FORMAT_V1 {
        return Err(FrameError::UnknownFormat { format });
    }
    Ok(at)
}

/// Check a receipt's bound, magic and layout version. The unsigned layout is refused
/// by name.
fn open_receipt(bytes: &[u8]) -> Result<usize, FrameError> {
    let (at, format) = open_any(bytes, RECEIPT_MAGIC)?;
    match format {
        RECEIPT_FORMAT_V2 => Ok(at),
        FORMAT_V1 => Err(FrameError::UnsignedReceipt),
        other => Err(FrameError::UnknownFormat { format: other }),
    }
}

/// Check the bound and the magic, and read the layout version.
fn open_any(bytes: &[u8], magic: &[u8; 8]) -> Result<(usize, u16), FrameError> {
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
    Ok((at, format))
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
    let slice = take_fixed::<2>(bytes, at, field)?;
    Ok(u16::from_le_bytes(slice))
}

/// Read a little-endian `u32`.
fn take_u32(bytes: &[u8], at: &mut usize, field: &'static str) -> Result<u32, FrameError> {
    let slice = take_fixed::<4>(bytes, at, field)?;
    Ok(u32::from_le_bytes(slice))
}

/// Read a little-endian `u64`.
fn take_u64(bytes: &[u8], at: &mut usize, field: &'static str) -> Result<u64, FrameError> {
    let slice = take_fixed::<8>(bytes, at, field)?;
    Ok(u64::from_le_bytes(slice))
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
    fn request() -> WireRequest {
        WireRequest {
            addressed_to: "key-of-the-runtime".to_owned(),
            request_id: 7,
            project: ProjectId::from("p"),
            action: ActionIr {
                operation: "write".into(),
                target: "capsule:a".into(),
                capability: CapabilityId::from("file.write"),
                effect: Effect::Mutation,
                input_type: TypeId::from("Bytes"),
                generation: Generation(3),
                revision: Revision(9),
                payload: b"verified payload".to_vec(),
            },
            once_key: Some("k".to_owned()),
        }
    }

    /// The runtime's endpoint key.
    fn signer() -> NodeSigner {
        NodeSigner::from_test_seed([7; 32])
    }

    /// Somebody else's.
    fn stranger() -> NodeSigner {
        NodeSigner::from_test_seed([8; 32])
    }

    /// The id a verifier expects the receipt from.
    fn responder() -> NodeId {
        NodeId(signer().public_key().to_owned())
    }

    fn receipt() -> WireReceipt {
        WireReceipt {
            responder: signer().public_key().to_owned(),
            request_id: 7,
            request_digest: request_digest(b"whatever arrived"),
            outcome: WireOutcome::Applied {
                response: b"executed".to_vec(),
            },
        }
    }

    #[test]
    fn a_request_and_a_receipt_round_trip() {
        let request = request();
        let frame = encode_request(&request).unwrap();
        assert_eq!(decode_request(&frame).unwrap(), request);

        let receipt = receipt();
        let frame = encode_receipt(&receipt, &signer()).unwrap();
        assert_eq!(verify_receipt(&frame, &responder()).unwrap(), receipt);

        // Every outcome is a receipt, including the two that carry nothing and the
        // refusal. A round trip that only covered the happy answer would leave the
        // three that matter most untested.
        for outcome in [
            WireOutcome::Applied {
                response: Vec::new(),
            },
            WireOutcome::AppliedWithoutResponse,
            WireOutcome::Uncertain,
            WireOutcome::Refused {
                code: RefusalCode::Runtime,
            },
            WireOutcome::Refused {
                code: RefusalCode::Replayed,
            },
        ] {
            let answered = WireReceipt {
                outcome,
                ..receipt.clone()
            };
            let frame = encode_receipt(&answered, &signer()).unwrap();
            assert_eq!(verify_receipt(&frame, &responder()).unwrap(), answered);
        }
    }

    #[test]
    fn an_outcome_kind_this_build_has_no_meaning_for_is_refused() {
        // A receipt whose outcome this build cannot name is refused rather than read
        // as the nearest one it knows. Guessing here would turn an outcome a later
        // build invented — say, one that means "applied" — into "nothing happened".
        let frame = encode_receipt(&receipt(), &signer()).unwrap();
        let kind_at = frame.len() - SIGNATURE_BYTES - 1 - 4 - b"executed".len();
        assert_eq!(
            frame[kind_at], 1,
            "the outcome kind is where it is expected"
        );
        let mut unknown = frame.clone();
        unknown[kind_at] = 9;
        assert_eq!(
            verify_receipt(&unknown, &responder()).expect_err("an outcome kind from a later build"),
            ReceiptError::Frame(FrameError::UnknownCode {
                field: "outcome",
                code: 9
            })
        );
    }

    #[test]
    fn an_empty_once_key_is_not_no_key() {
        // Encoding presence as a zero length would make these the same request, and
        // they are not: one asks for at-most-once and the other does not.
        let without = WireRequest {
            once_key: None,
            ..request()
        };
        let empty = WireRequest {
            once_key: Some(String::new()),
            ..request()
        };
        let a = encode_request(&without).unwrap();
        let b = encode_request(&empty).unwrap();
        assert_ne!(a, b);
        assert_eq!(decode_request(&a).unwrap().once_key, None);
        assert_eq!(
            decode_request(&b).unwrap().once_key,
            Some(String::new()),
            "an empty key survives as an empty key"
        );
    }

    #[test]
    fn a_receipt_is_never_read_as_a_request_and_a_request_never_as_a_receipt() {
        // Decided by the magic, before a version or a field is looked at: the two
        // frames mean opposite things, so reading one as the other must not be a
        // matter of the fields happening not to line up.
        let request = encode_request(&request()).unwrap();
        let receipt = encode_receipt(&receipt(), &signer()).unwrap();
        assert_eq!(
            verify_receipt(&request, &responder()).expect_err("a request is not a receipt"),
            ReceiptError::Frame(FrameError::WrongKind)
        );
        assert_eq!(
            decode_request(&receipt).expect_err("a receipt is not a request"),
            FrameError::WrongKind
        );
        assert_eq!(
            decode_request(b"PTRRAFTW\x01\x00").expect_err("another protocol's frame"),
            FrameError::WrongKind
        );
    }

    #[test]
    fn every_prefix_of_each_frame_is_refused() {
        for frame in [
            encode_request(&request()).unwrap(),
            encode_receipt(&receipt(), &signer()).unwrap(),
        ] {
            let is_request = decode_request(&frame).is_ok();
            for length in 0..frame.len() {
                let error = if is_request {
                    decode_request(&frame[..length]).expect_err("a prefix is not a request")
                } else {
                    match verify_receipt(&frame[..length], &responder()) {
                        Err(ReceiptError::Frame(error)) => error,
                        other => panic!("a prefix of {length} is not a receipt: {other:?}"),
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

        let mut future_receipt = encode_receipt(&receipt(), &signer()).unwrap();
        future_receipt[8..10].copy_from_slice(&9_u16.to_le_bytes());
        assert_eq!(
            verify_receipt(&future_receipt, &responder())
                .expect_err("a receipt layout from elsewhere"),
            ReceiptError::Frame(FrameError::UnknownFormat { format: 9 })
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
        let receipt_error = match verify_receipt(&oversize, &responder()) {
            Err(ReceiptError::Frame(error)) => error,
            other => panic!("over the bound: {other:?}"),
        };
        for error in [
            decode_request(&oversize).expect_err("over the bound"),
            receipt_error,
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

        // And this build refuses to *send* one, rather than handing a peer bytes it
        // is required to refuse.
        let mut huge = request();
        huge.action.payload = vec![0; MAX_BODY_BYTES + 1];
        assert_eq!(
            encode_request(&huge).expect_err("a payload over its bound"),
            FrameError::TooLarge {
                field: "payload",
                bytes: MAX_BODY_BYTES + 1,
                limit: MAX_BODY_BYTES,
            }
        );
    }

    #[test]
    fn every_effect_crosses_the_wire_as_its_own_code_and_an_unknown_one_is_refused() {
        let effects = [
            Effect::Pure,
            Effect::Read,
            Effect::Mutation,
            Effect::External,
            Effect::Irreversible,
        ];
        let mut codes: Vec<u8> = effects.iter().copied().map(effect_code).collect();
        let total = codes.len();
        codes.sort_unstable();
        codes.dedup();
        assert_eq!(codes.len(), total, "two effects share a wire code");
        for effect in effects {
            assert_eq!(
                effect_from_code(effect_code(effect)).unwrap(),
                effect,
                "{effect:?} does not survive the table"
            );
        }
        // Zero is deliberately not a code: a zeroed byte is what a truncated or
        // padded frame looks like, and it must not mean an effect class.
        assert_eq!(
            effect_from_code(0).expect_err("zero is not an effect"),
            FrameError::UnknownCode {
                field: "effect",
                code: 0
            }
        );
        assert!(effect_from_code(6).is_err(), "a code from a later build");
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
        assert_eq!(
            refusal_from_code(0).expect_err("zero is not a refusal"),
            FrameError::UnknownCode {
                field: "refusal",
                code: 0
            }
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
        // build cannot parse still gets a receipt its sender can check.
        let unparseable = b"not a request at all";
        assert!(decode_request(unparseable).is_err());
        assert_ne!(request_digest(unparseable), [0; 32]);

        // Length is part of the material, so a shorter frame is not a prefix of a
        // longer one's digest material.
        assert_ne!(request_digest(b"ab"), request_digest(b"abc"));
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
            FrameError::UnsignedReceipt,
        ];
        let mut codes: Vec<&str> = frame_errors.iter().map(FrameError::code).collect();
        codes.extend(RefusalCode::ALL.iter().map(|code| code.code()));
        codes.extend(
            [
                ReceiptError::WrongResponder {
                    expected: String::new(),
                    claimed: String::new(),
                },
                ReceiptError::BadSignature,
            ]
            .iter()
            .map(ReceiptError::code),
        );
        let total = codes.len();
        codes.sort_unstable();
        codes.dedup();
        assert_eq!(codes.len(), total, "two refusals share a diagnostic code");
    }

    /// The bytes of `changed`, encoded, carrying the signature `original` was sent
    /// with: one field edited after signing, exactly as a stored receipt would be.
    fn with_original_signature(changed: &WireReceipt, original: &[u8]) -> Vec<u8> {
        let mut frame = encode_receipt(changed, &signer()).unwrap();
        frame.truncate(frame.len() - SIGNATURE_BYTES);
        frame.extend_from_slice(&original[original.len() - SIGNATURE_BYTES..]);
        frame
    }

    #[test]
    fn a_stored_receipt_verifies_offline_and_a_change_to_any_field_is_refused() {
        // The control first: the stored bytes, untouched, verify with nothing but the
        // responder's id — no connection, no transport, no clock.
        let stored = encode_receipt(&receipt(), &signer()).unwrap();
        assert_eq!(verify_receipt(&stored, &responder()).unwrap(), receipt());

        // Each field edited after signing, with the signature it was sent with. The
        // edits that keep a frame well formed are the ones a forger would make.
        let edits = [
            (
                "request_id",
                WireReceipt {
                    request_id: 8,
                    ..receipt()
                },
            ),
            (
                "request_digest",
                WireReceipt {
                    request_digest: request_digest(b"another request"),
                    ..receipt()
                },
            ),
            (
                "outcome",
                WireReceipt {
                    outcome: WireOutcome::Uncertain,
                    ..receipt()
                },
            ),
            (
                "response",
                WireReceipt {
                    outcome: WireOutcome::Applied {
                        response: b"executeD".to_vec(),
                    },
                    ..receipt()
                },
            ),
            (
                "refusal code",
                WireReceipt {
                    outcome: WireOutcome::Refused {
                        code: RefusalCode::Malformed,
                    },
                    ..receipt()
                },
            ),
        ];
        for (field, changed) in edits {
            let forged = with_original_signature(&changed, &stored);
            assert_eq!(
                verify_receipt(&forged, &responder()).expect_err(field),
                ReceiptError::BadSignature,
                "{field} edited after signing"
            );
        }
        // A refusal edited into another refusal is refused too, so the code a refusal
        // carries is signed and not only the fact of it.
        let refused = WireReceipt {
            outcome: WireOutcome::Refused {
                code: RefusalCode::Runtime,
            },
            ..receipt()
        };
        let stored_refusal = encode_receipt(&refused, &signer()).unwrap();
        assert!(verify_receipt(&stored_refusal, &responder()).is_ok());
        let replayed = WireReceipt {
            outcome: WireOutcome::Refused {
                code: RefusalCode::Replayed,
            },
            ..receipt()
        };
        assert_eq!(
            verify_receipt(
                &with_original_signature(&replayed, &stored_refusal),
                &responder()
            ),
            Err(ReceiptError::BadSignature)
        );

        // And every single bit of the stored frame, signature included: none yields a
        // receipt that verifies.
        for index in 0..stored.len() {
            for bit in 0..8 {
                let mut damaged = stored.clone();
                damaged[index] ^= 1 << bit;
                assert!(
                    verify_receipt(&damaged, &responder()).is_err(),
                    "byte {index} bit {bit} still verifies"
                );
            }
        }
    }

    #[test]
    fn a_receipt_signed_by_another_key_is_refused() {
        // Names the right responder and is signed by somebody else: the case a
        // receipt passed through an untrusted hand would be.
        let forged = encode_receipt(&receipt(), &stranger()).unwrap();
        assert_eq!(
            verify_receipt(&forged, &responder()),
            Err(ReceiptError::BadSignature)
        );
        // The control: the same receipt, signed by its responder.
        let honest = encode_receipt(&receipt(), &signer()).unwrap();
        assert_eq!(verify_receipt(&honest, &responder()), Ok(receipt()));
    }

    #[test]
    fn a_receipt_readdressed_to_another_responder_is_refused() {
        let stored = encode_receipt(&receipt(), &signer()).unwrap();
        let other = NodeId(stranger().public_key().to_owned());

        // Presented as somebody else's receipt: refused on the name, before the
        // signature is looked at.
        assert_eq!(
            verify_receipt(&stored, &other),
            Err(ReceiptError::WrongResponder {
                expected: other.0.clone(),
                claimed: responder().0,
            })
        );

        // The name rewritten to the other key, with the signature it was sent with:
        // the name now matches and the signature is not that key's.
        let readdressed = WireReceipt {
            responder: other.0.clone(),
            ..receipt()
        };
        assert_eq!(
            verify_receipt(&with_original_signature(&readdressed, &stored), &other),
            Err(ReceiptError::BadSignature)
        );

        // The other key re-signing the receipt as its own: valid under that key, and
        // refused as the original responder's — a signature makes a statement the
        // signer's, never somebody else's.
        let adopted = encode_receipt(&readdressed, &stranger()).unwrap();
        assert!(verify_receipt(&adopted, &other).is_ok());
        assert!(matches!(
            verify_receipt(&adopted, &responder()),
            Err(ReceiptError::WrongResponder { .. })
        ));
    }

    #[test]
    fn an_unsigned_receipt_from_an_earlier_build_is_refused_by_version() {
        // Exactly the layout earlier builds wrote: the V1 fields and no signature.
        let signed = encode_receipt(&receipt(), &signer()).unwrap();
        let mut unsigned = signed[..signed.len() - SIGNATURE_BYTES].to_vec();
        unsigned[8..10].copy_from_slice(&FORMAT_V1.to_le_bytes());
        assert_eq!(
            verify_receipt(&unsigned, &responder()),
            Err(ReceiptError::Frame(FrameError::UnsignedReceipt))
        );
        // A signature left on does not rescue the old layout: the version is read
        // before anything else, and a signed V1 frame is not a format.
        let mut relabelled = signed.clone();
        relabelled[8..10].copy_from_slice(&FORMAT_V1.to_le_bytes());
        assert_eq!(
            verify_receipt(&relabelled, &responder()),
            Err(ReceiptError::Frame(FrameError::UnsignedReceipt))
        );
        // The control: the frame both were made from verifies.
        assert_eq!(verify_receipt(&signed, &responder()), Ok(receipt()));
        // And requests keep the layout they had: nothing a signature changes is in one.
        let request = encode_request(&request()).unwrap();
        assert_eq!(&request[8..10], &FORMAT_V1.to_le_bytes());
    }

    #[test]
    fn a_field_that_is_not_utf8_is_refused_rather_than_replaced() {
        // Lossy decoding would turn a target nobody wrote into a target that might
        // match a grant.
        let mut frame = encode_request(&request()).unwrap();
        let at = frame
            .windows(9)
            .position(|window| window == b"capsule:a")
            .expect("the target is in the frame");
        frame[at] = 0xff;
        assert_eq!(
            decode_request(&frame).expect_err("invalid UTF-8 in a field"),
            FrameError::NotUtf8 { field: "target" }
        );
    }
}
