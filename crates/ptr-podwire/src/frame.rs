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
use ptr_protocol::TypedPayload;
use ptr_types::{CapabilityId, TypeId};
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

/// Encode an answer.
pub fn encode_answer(answer: &PodAnswer) -> Result<Vec<u8>, FrameError> {
    let mut frame = Vec::with_capacity(128);
    frame.extend_from_slice(ANSWER_MAGIC);
    frame.extend_from_slice(&FORMAT_V1.to_le_bytes());
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
    bound_frame(frame)
}

/// Decode an answer, or refuse it.
pub fn decode_answer(bytes: &[u8]) -> Result<PodAnswer, FrameError> {
    let mut at = open(bytes, ANSWER_MAGIC)?;
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
    close(bytes, at)?;
    Ok(PodAnswer {
        responder,
        request_id,
        request_digest,
        outcome,
    })
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
    if format != FORMAT_V1 {
        return Err(FrameError::UnknownFormat { format });
    }
    Ok(at)
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

    fn answer() -> PodAnswer {
        PodAnswer {
            responder: "key-of-the-host".to_owned(),
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
        let frame = encode_answer(&answer).unwrap();
        assert_eq!(decode_answer(&frame).unwrap(), answer);

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
            let frame = encode_answer(&answered).unwrap();
            assert_eq!(decode_answer(&frame).unwrap(), answered);
        }
    }

    #[test]
    fn an_outcome_kind_this_build_has_no_meaning_for_is_refused() {
        // Refused rather than read as the nearest kind this build knows. An outcome
        // a later build invents must not silently become "answered" with whatever
        // bytes follow.
        let answer = answer();
        let frame = encode_answer(&answer).unwrap();
        let kind_at = frame.len() - 1 - (4 + b"Summary".len()) - (4 + b"summarized".len());
        assert_eq!(
            frame[kind_at], 1,
            "the outcome kind is where it is expected"
        );
        let mut unknown = frame.clone();
        unknown[kind_at] = 9;
        assert_eq!(
            decode_answer(&unknown).expect_err("an outcome kind from a later build"),
            FrameError::UnknownCode {
                field: "outcome",
                code: 9
            }
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
        let answer = encode_answer(&answer()).unwrap();
        assert_eq!(
            decode_answer(&request).expect_err("a request is not an answer"),
            FrameError::WrongKind
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
            encode_answer(&answer()).unwrap(),
        ] {
            let is_request = decode_request(&frame).is_ok();
            for length in 0..frame.len() {
                let error = if is_request {
                    decode_request(&frame[..length]).expect_err("a prefix is not a request")
                } else {
                    decode_answer(&frame[..length]).expect_err("a prefix is not an answer")
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

        let mut future_answer = encode_answer(&answer()).unwrap();
        future_answer[8..10].copy_from_slice(&9_u16.to_le_bytes());
        assert_eq!(
            decode_answer(&future_answer).expect_err("an answer layout from elsewhere"),
            FrameError::UnknownFormat { format: 9 }
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
        for error in [
            decode_request(&oversize).expect_err("over the bound"),
            decode_answer(&oversize).expect_err("over the bound"),
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
            encode_answer(&huge_answer).expect_err("an output over its bound"),
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
        ];
        let mut codes: Vec<&str> = frame_errors.iter().map(FrameError::code).collect();
        codes.extend(RefusalCode::ALL.iter().map(|code| code.code()));
        let total = codes.len();
        codes.sort_unstable();
        codes.dedup();
        assert_eq!(codes.len(), total, "two refusals share a diagnostic code");
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
