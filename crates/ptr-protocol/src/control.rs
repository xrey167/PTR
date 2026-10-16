//! Control frames of the generic protocol session and their bounded binary codec.
//!
//! [`ControlFrame`] is a new enum beside the existing [`crate::Frame`]; it does not change
//! it. The codec is hand-written, versioned and bounded: every frame starts with a fixed
//! header, every field is length-prefixed, every size is capped before anything is
//! allocated, every failure is a typed [`CodecError`], and trailing bytes are rejected.
//!
//! Layout (all integers little-endian):
//!
//! ```text
//! magic "PTRC" | codec version u8 | frame tag u8 | body length u32 | body
//! ```
//!
//! Strings and byte strings are `u32 length | bytes`.

use std::fmt;

use ptr_types::{CapabilityId, TypeId};
use sha2::{Digest as _, Sha256};

use crate::negotiation::{
    FeatureSet, OfferFault, ProtocolVersion, VersionOffer, MAX_OFFER_VERSIONS,
};
use crate::TypedPayload;

pub const CODEC_MAGIC: [u8; 4] = *b"PTRC";
pub const CODEC_VERSION: u8 = 1;
/// Magic, codec version, frame tag and body length.
pub const HEADER_LEN: usize = 10;
/// Largest encoded frame, header included.
pub const MAX_FRAME_BYTES: usize = 1 << 20;
/// Longest string field (node id, reason, type id, capability).
pub const MAX_STRING_BYTES: usize = 1024;
/// Longest idempotency key.
pub const MAX_KEY_BYTES: usize = 128;
/// Most capabilities one `Decl` may list.
pub const MAX_CAPABILITIES: usize = 64;

const PAYLOAD_DOMAIN: &[u8] = b"ptr-protocol/control/payload/v1";

/// A caller-chosen key that makes a call safe to retry.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct IdempotencyKey(String);

/// Why a string is not an idempotency key.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum KeyError {
    Empty,
    TooLong { len: usize },
}

impl IdempotencyKey {
    pub fn new(key: impl Into<String>) -> Result<Self, KeyError> {
        let key = key.into();
        if key.is_empty() {
            return Err(KeyError::Empty);
        }
        if key.len() > MAX_KEY_BYTES {
            return Err(KeyError::TooLong { len: key.len() });
        }
        Ok(Self(key))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for IdempotencyKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

/// Digest of a payload's type id and bytes; what "same payload" means for a reused key.
pub fn payload_digest(payload: &TypedPayload) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(PAYLOAD_DOMAIN);
    hasher.update((payload.type_id.0.len() as u64).to_le_bytes());
    hasher.update(payload.type_id.0.as_bytes());
    hasher.update((payload.bytes.len() as u64).to_le_bytes());
    hasher.update(&payload.bytes);
    hasher.finalize().into()
}

/// The kind of a [`ControlFrame`], used in typed errors.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum FrameKind {
    Hello,
    Decl,
    Ack,
    Cancel,
    Bye,
    Call,
    Reply,
}

impl FrameKind {
    pub const fn name(self) -> &'static str {
        match self {
            Self::Hello => "HELLO",
            Self::Decl => "DECL",
            Self::Ack => "ACK",
            Self::Cancel => "CANCEL",
            Self::Bye => "BYE",
            Self::Call => "CALL",
            Self::Reply => "REPLY",
        }
    }

    const fn tag(self) -> u8 {
        match self {
            Self::Hello => 1,
            Self::Decl => 2,
            Self::Ack => 3,
            Self::Cancel => 4,
            Self::Bye => 5,
            Self::Call => 6,
            Self::Reply => 7,
        }
    }

    fn from_tag(tag: u8) -> Option<Self> {
        Some(match tag {
            1 => Self::Hello,
            2 => Self::Decl,
            3 => Self::Ack,
            4 => Self::Cancel,
            5 => Self::Bye,
            6 => Self::Call,
            7 => Self::Reply,
            _ => return None,
        })
    }
}

/// One frame of the generic control layer.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ControlFrame {
    /// Opens a session: who is speaking, what versions and features it offers.
    Hello {
        node_id: String,
        offer: VersionOffer,
        nonce: u64,
    },
    /// Declares the capabilities this end serves.
    Decl { capabilities: Vec<CapabilityId> },
    /// Acknowledges a cancelled call: it will not execute.
    Ack {
        seq: u64,
        idempotency_key: IdempotencyKey,
    },
    /// Asks the peer to drop the in-flight call `seq`.
    Cancel { seq: u64, reason: String },
    /// Closes the session; in-flight work drains if the drain feature is negotiated.
    Bye { reason: String },
    /// A call. Retrying it with the same key must not execute it twice.
    Call {
        seq: u64,
        idempotency_key: IdempotencyKey,
        payload: TypedPayload,
    },
    /// The answer to the call `seq`.
    Reply { seq: u64, payload: TypedPayload },
}

impl ControlFrame {
    pub fn kind(&self) -> FrameKind {
        match self {
            Self::Hello { .. } => FrameKind::Hello,
            Self::Decl { .. } => FrameKind::Decl,
            Self::Ack { .. } => FrameKind::Ack,
            Self::Cancel { .. } => FrameKind::Cancel,
            Self::Bye { .. } => FrameKind::Bye,
            Self::Call { .. } => FrameKind::Call,
            Self::Reply { .. } => FrameKind::Reply,
        }
    }
}

/// Every way encoding or decoding a [`ControlFrame`] can fail.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CodecError {
    /// The input ends before the frame does.
    Truncated,
    BadMagic,
    UnsupportedCodecVersion(u8),
    UnknownTag(u8),
    /// The frame (or its declared body) exceeds [`MAX_FRAME_BYTES`].
    FrameTooLarge {
        len: usize,
    },
    /// Bytes remain after the frame or after its last field.
    TrailingBytes,
    StringTooLong {
        field: &'static str,
        len: usize,
    },
    InvalidUtf8 {
        field: &'static str,
    },
    EmptyField(&'static str),
    TooManyItems {
        field: &'static str,
        len: usize,
    },
    InvalidKey(KeyError),
    InvalidOffer(OfferFault),
    /// A feature mask with a bit no known feature owns.
    UnknownFeatureBits(u32),
}

impl fmt::Display for CodecError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for CodecError {}

/// Encodes a frame, refusing anything that would not decode again.
pub fn encode(frame: &ControlFrame) -> Result<Vec<u8>, CodecError> {
    let mut body = Vec::new();
    match frame {
        ControlFrame::Hello {
            node_id,
            offer,
            nonce,
        } => {
            put_str(&mut body, "node_id", node_id, false)?;
            put_offer(&mut body, offer)?;
            body.extend_from_slice(&nonce.to_le_bytes());
        }
        ControlFrame::Decl { capabilities } => {
            if capabilities.len() > MAX_CAPABILITIES {
                return Err(CodecError::TooManyItems {
                    field: "capabilities",
                    len: capabilities.len(),
                });
            }
            body.extend_from_slice(&(capabilities.len() as u32).to_le_bytes());
            for capability in capabilities {
                put_str(&mut body, "capability", &capability.0, false)?;
            }
        }
        ControlFrame::Ack {
            seq,
            idempotency_key,
        } => {
            body.extend_from_slice(&seq.to_le_bytes());
            put_str(
                &mut body,
                "idempotency_key",
                idempotency_key.as_str(),
                false,
            )?;
        }
        ControlFrame::Cancel { seq, reason } => {
            body.extend_from_slice(&seq.to_le_bytes());
            put_str(&mut body, "reason", reason, true)?;
        }
        ControlFrame::Bye { reason } => put_str(&mut body, "reason", reason, true)?,
        ControlFrame::Call {
            seq,
            idempotency_key,
            payload,
        } => {
            body.extend_from_slice(&seq.to_le_bytes());
            put_str(
                &mut body,
                "idempotency_key",
                idempotency_key.as_str(),
                false,
            )?;
            put_payload(&mut body, payload)?;
        }
        ControlFrame::Reply { seq, payload } => {
            body.extend_from_slice(&seq.to_le_bytes());
            put_payload(&mut body, payload)?;
        }
    }
    let total = HEADER_LEN + body.len();
    if total > MAX_FRAME_BYTES {
        return Err(CodecError::FrameTooLarge { len: total });
    }
    let mut out = Vec::with_capacity(total);
    out.extend_from_slice(&CODEC_MAGIC);
    out.push(CODEC_VERSION);
    out.push(frame.kind().tag());
    out.extend_from_slice(&(body.len() as u32).to_le_bytes());
    out.extend_from_slice(&body);
    Ok(out)
}

/// Decodes exactly one frame from `input`. Never panics; every malformed input is a typed
/// error, and bytes after the frame are [`CodecError::TrailingBytes`].
pub fn decode(input: &[u8]) -> Result<ControlFrame, CodecError> {
    if input.len() > MAX_FRAME_BYTES {
        return Err(CodecError::FrameTooLarge { len: input.len() });
    }
    if input.len() < HEADER_LEN {
        return Err(CodecError::Truncated);
    }
    if input[..4] != CODEC_MAGIC {
        return Err(CodecError::BadMagic);
    }
    if input[4] != CODEC_VERSION {
        return Err(CodecError::UnsupportedCodecVersion(input[4]));
    }
    let kind = FrameKind::from_tag(input[5]).ok_or(CodecError::UnknownTag(input[5]))?;
    let body_len = u32::from_le_bytes([input[6], input[7], input[8], input[9]]) as usize;
    if body_len > MAX_FRAME_BYTES - HEADER_LEN {
        return Err(CodecError::FrameTooLarge {
            len: body_len.saturating_add(HEADER_LEN),
        });
    }
    let available = input.len() - HEADER_LEN;
    if available < body_len {
        return Err(CodecError::Truncated);
    }
    if available > body_len {
        return Err(CodecError::TrailingBytes);
    }
    let mut reader = Reader {
        rest: &input[HEADER_LEN..],
    };
    let frame = match kind {
        FrameKind::Hello => {
            let node_id = reader.string("node_id", false)?;
            let offer = reader.offer()?;
            let nonce = reader.u64()?;
            ControlFrame::Hello {
                node_id,
                offer,
                nonce,
            }
        }
        FrameKind::Decl => {
            let count = reader.u32()? as usize;
            if count > MAX_CAPABILITIES {
                return Err(CodecError::TooManyItems {
                    field: "capabilities",
                    len: count,
                });
            }
            let mut capabilities = Vec::with_capacity(count);
            for _ in 0..count {
                capabilities.push(CapabilityId(reader.string("capability", false)?));
            }
            ControlFrame::Decl { capabilities }
        }
        FrameKind::Ack => ControlFrame::Ack {
            seq: reader.u64()?,
            idempotency_key: reader.key()?,
        },
        FrameKind::Cancel => ControlFrame::Cancel {
            seq: reader.u64()?,
            reason: reader.string("reason", true)?,
        },
        FrameKind::Bye => ControlFrame::Bye {
            reason: reader.string("reason", true)?,
        },
        FrameKind::Call => ControlFrame::Call {
            seq: reader.u64()?,
            idempotency_key: reader.key()?,
            payload: reader.payload()?,
        },
        FrameKind::Reply => ControlFrame::Reply {
            seq: reader.u64()?,
            payload: reader.payload()?,
        },
    };
    if !reader.rest.is_empty() {
        return Err(CodecError::TrailingBytes);
    }
    Ok(frame)
}

fn put_str(
    out: &mut Vec<u8>,
    field: &'static str,
    value: &str,
    may_be_empty: bool,
) -> Result<(), CodecError> {
    if value.is_empty() && !may_be_empty {
        return Err(CodecError::EmptyField(field));
    }
    if value.len() > MAX_STRING_BYTES {
        return Err(CodecError::StringTooLong {
            field,
            len: value.len(),
        });
    }
    out.extend_from_slice(&(value.len() as u32).to_le_bytes());
    out.extend_from_slice(value.as_bytes());
    Ok(())
}

fn put_payload(out: &mut Vec<u8>, payload: &TypedPayload) -> Result<(), CodecError> {
    put_str(out, "type_id", &payload.type_id.0, false)?;
    if payload.bytes.len() > MAX_FRAME_BYTES {
        return Err(CodecError::FrameTooLarge {
            len: payload.bytes.len(),
        });
    }
    out.extend_from_slice(&(payload.bytes.len() as u32).to_le_bytes());
    out.extend_from_slice(&payload.bytes);
    Ok(())
}

fn put_offer(out: &mut Vec<u8>, offer: &VersionOffer) -> Result<(), CodecError> {
    offer.validate().map_err(CodecError::InvalidOffer)?;
    out.push(offer.supported.len() as u8);
    for version in &offer.supported {
        put_version(out, *version);
    }
    put_version(out, offer.minimum);
    out.extend_from_slice(&offer.features.bits().to_le_bytes());
    out.extend_from_slice(&offer.required.bits().to_le_bytes());
    Ok(())
}

fn put_version(out: &mut Vec<u8>, version: ProtocolVersion) {
    out.extend_from_slice(&version.major.to_le_bytes());
    out.extend_from_slice(&version.minor.to_le_bytes());
    out.extend_from_slice(&version.patch.to_le_bytes());
}

struct Reader<'a> {
    rest: &'a [u8],
}

impl<'a> Reader<'a> {
    fn take(&mut self, len: usize) -> Result<&'a [u8], CodecError> {
        if self.rest.len() < len {
            return Err(CodecError::Truncated);
        }
        let (head, tail) = self.rest.split_at(len);
        self.rest = tail;
        Ok(head)
    }

    fn u8(&mut self) -> Result<u8, CodecError> {
        Ok(self.take(1)?[0])
    }

    fn u16(&mut self) -> Result<u16, CodecError> {
        let b = self.take(2)?;
        Ok(u16::from_le_bytes([b[0], b[1]]))
    }

    fn u32(&mut self) -> Result<u32, CodecError> {
        let b = self.take(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn u64(&mut self) -> Result<u64, CodecError> {
        let b = self.take(8)?;
        let mut bytes = [0u8; 8];
        bytes.copy_from_slice(b);
        Ok(u64::from_le_bytes(bytes))
    }

    fn string(&mut self, field: &'static str, may_be_empty: bool) -> Result<String, CodecError> {
        let len = self.u32()? as usize;
        if len > MAX_STRING_BYTES {
            return Err(CodecError::StringTooLong { field, len });
        }
        if len == 0 && !may_be_empty {
            return Err(CodecError::EmptyField(field));
        }
        let bytes = self.take(len)?;
        std::str::from_utf8(bytes)
            .map(str::to_owned)
            .map_err(|_| CodecError::InvalidUtf8 { field })
    }

    fn key(&mut self) -> Result<IdempotencyKey, CodecError> {
        let len = self.u32()? as usize;
        if len > MAX_KEY_BYTES {
            return Err(CodecError::InvalidKey(KeyError::TooLong { len }));
        }
        let bytes = self.take(len)?;
        let text = std::str::from_utf8(bytes).map_err(|_| CodecError::InvalidUtf8 {
            field: "idempotency_key",
        })?;
        IdempotencyKey::new(text).map_err(CodecError::InvalidKey)
    }

    fn payload(&mut self) -> Result<TypedPayload, CodecError> {
        let type_id = TypeId(self.string("type_id", false)?);
        let len = self.u32()? as usize;
        let bytes = self.take(len)?.to_vec();
        Ok(TypedPayload { type_id, bytes })
    }

    fn version(&mut self) -> Result<ProtocolVersion, CodecError> {
        Ok(ProtocolVersion::new(self.u16()?, self.u16()?, self.u16()?))
    }

    fn features(&mut self) -> Result<FeatureSet, CodecError> {
        let bits = self.u32()?;
        FeatureSet::from_bits(bits).ok_or(CodecError::UnknownFeatureBits(bits))
    }

    fn offer(&mut self) -> Result<VersionOffer, CodecError> {
        let count = self.u8()? as usize;
        if count > MAX_OFFER_VERSIONS {
            return Err(CodecError::TooManyItems {
                field: "supported",
                len: count,
            });
        }
        let mut supported = Vec::with_capacity(count);
        for _ in 0..count {
            supported.push(self.version()?);
        }
        let offer = VersionOffer {
            supported,
            minimum: self.version()?,
            features: self.features()?,
            required: self.features()?,
        };
        offer.validate().map_err(CodecError::InvalidOffer)?;
        Ok(offer)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::negotiation::Feature;

    pub(crate) fn key(text: &str) -> IdempotencyKey {
        IdempotencyKey::new(text).unwrap()
    }

    fn payload(bytes: &[u8]) -> TypedPayload {
        TypedPayload {
            type_id: TypeId::from("Input"),
            bytes: bytes.to_vec(),
        }
    }

    pub(crate) fn sample_frames() -> Vec<ControlFrame> {
        vec![
            ControlFrame::Hello {
                node_id: "node-a".into(),
                offer: VersionOffer {
                    supported: vec![ProtocolVersion::new(1, 0, 0), ProtocolVersion::new(1, 1, 2)],
                    minimum: ProtocolVersion::new(1, 0, 0),
                    features: FeatureSet::all(),
                    required: FeatureSet::of(&[Feature::Idempotency]),
                },
                nonce: 0xDEAD_BEEF,
            },
            ControlFrame::Decl {
                capabilities: vec![CapabilityId::from("predict"), CapabilityId::from("embed")],
            },
            ControlFrame::Ack {
                seq: 9,
                idempotency_key: key("k-9"),
            },
            ControlFrame::Cancel {
                seq: 3,
                reason: "caller gave up".into(),
            },
            ControlFrame::Bye {
                reason: String::new(),
            },
            ControlFrame::Call {
                seq: 1,
                idempotency_key: key("k-1"),
                payload: payload(&[1, 2, 3]),
            },
            ControlFrame::Reply {
                seq: 1,
                payload: payload(&[]),
            },
        ]
    }

    #[test]
    fn every_frame_round_trips() {
        for frame in sample_frames() {
            let bytes = encode(&frame).unwrap();
            assert_eq!(decode(&bytes), Ok(frame));
        }
    }

    #[test]
    fn every_truncation_is_an_error_and_never_a_panic() {
        for frame in sample_frames() {
            let bytes = encode(&frame).unwrap();
            for cut in 0..bytes.len() {
                assert!(
                    decode(&bytes[..cut]).is_err(),
                    "{:?} cut at {cut} decoded",
                    frame.kind()
                );
            }
        }
    }

    #[test]
    fn trailing_bytes_are_rejected() {
        let mut bytes = encode(&sample_frames()[5]).unwrap();
        bytes.push(0);
        assert_eq!(decode(&bytes), Err(CodecError::TrailingBytes));
    }

    #[test]
    fn trailing_bytes_inside_the_declared_body_are_rejected() {
        let mut bytes = encode(&ControlFrame::Bye { reason: "x".into() }).unwrap();
        bytes.push(0);
        let new_len = (bytes.len() - HEADER_LEN) as u32;
        bytes[6..10].copy_from_slice(&new_len.to_le_bytes());
        assert_eq!(decode(&bytes), Err(CodecError::TrailingBytes));
    }

    #[test]
    fn header_faults_are_typed() {
        let good = encode(&ControlFrame::Bye { reason: "x".into() }).unwrap();
        let mut bad = good.clone();
        bad[0] = b'X';
        assert_eq!(decode(&bad), Err(CodecError::BadMagic));
        let mut bad = good.clone();
        bad[4] = 9;
        assert_eq!(decode(&bad), Err(CodecError::UnsupportedCodecVersion(9)));
        let mut bad = good.clone();
        bad[5] = 99;
        assert_eq!(decode(&bad), Err(CodecError::UnknownTag(99)));
        let mut bad = good.clone();
        bad[6..10].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(matches!(
            decode(&bad),
            Err(CodecError::FrameTooLarge { .. })
        ));
        assert!(matches!(
            decode(&vec![0u8; MAX_FRAME_BYTES + 1]),
            Err(CodecError::FrameTooLarge { .. })
        ));
    }

    #[test]
    fn field_faults_are_typed() {
        // A reason longer than the cap, declared inside an otherwise consistent frame.
        let mut body = Vec::new();
        body.extend_from_slice(&((MAX_STRING_BYTES + 1) as u32).to_le_bytes());
        body.extend(std::iter::repeat_n(b'a', MAX_STRING_BYTES + 1));
        assert!(matches!(
            decode(&frame_with_tag(5, &body)),
            Err(CodecError::StringTooLong {
                field: "reason",
                ..
            })
        ));

        let mut body = Vec::new();
        body.extend_from_slice(&2u32.to_le_bytes());
        body.extend_from_slice(&[0xFF, 0xFE]);
        assert_eq!(
            decode(&frame_with_tag(5, &body)),
            Err(CodecError::InvalidUtf8 { field: "reason" })
        );

        // Call with an empty idempotency key.
        let mut body = 1u64.to_le_bytes().to_vec();
        body.extend_from_slice(&0u32.to_le_bytes());
        assert_eq!(
            decode(&frame_with_tag(6, &body)),
            Err(CodecError::InvalidKey(KeyError::Empty))
        );

        // Decl claiming more capabilities than the cap.
        let body = ((MAX_CAPABILITIES + 1) as u32).to_le_bytes().to_vec();
        assert!(matches!(
            decode(&frame_with_tag(2, &body)),
            Err(CodecError::TooManyItems {
                field: "capabilities",
                ..
            })
        ));

        // A payload length that points past the frame.
        let mut body = 1u64.to_le_bytes().to_vec();
        body.extend_from_slice(&1u32.to_le_bytes());
        body.push(b'T');
        body.extend_from_slice(&1000u32.to_le_bytes());
        assert_eq!(
            decode(&frame_with_tag(7, &body)),
            Err(CodecError::Truncated)
        );
    }

    #[test]
    fn hello_with_unknown_feature_bits_or_bad_offer_is_rejected() {
        let hello = &sample_frames()[0];
        let good = encode(hello).unwrap();
        // The required-feature mask is the last 4 bytes before the 8-byte nonce.
        let mut bad = good.clone();
        let at = bad.len() - 8 - 4;
        bad[at..at + 4].copy_from_slice(&0x80u32.to_le_bytes());
        assert_eq!(decode(&bad), Err(CodecError::UnknownFeatureBits(0x80)));

        let unsatisfiable = ControlFrame::Hello {
            node_id: "n".into(),
            offer: VersionOffer {
                supported: vec![],
                minimum: ProtocolVersion::default(),
                features: FeatureSet::EMPTY,
                required: FeatureSet::EMPTY,
            },
            nonce: 0,
        };
        assert_eq!(
            encode(&unsatisfiable),
            Err(CodecError::InvalidOffer(OfferFault::NoVersions))
        );
    }

    #[test]
    fn encode_refuses_oversize_and_empty_fields() {
        let big = ControlFrame::Reply {
            seq: 1,
            payload: payload(&vec![0u8; MAX_FRAME_BYTES]),
        };
        assert!(matches!(
            encode(&big),
            Err(CodecError::FrameTooLarge { .. })
        ));
        let long = ControlFrame::Bye {
            reason: "x".repeat(MAX_STRING_BYTES + 1),
        };
        assert!(matches!(
            encode(&long),
            Err(CodecError::StringTooLong { .. })
        ));
        let empty_node = ControlFrame::Hello {
            node_id: String::new(),
            offer: VersionOffer {
                supported: vec![ProtocolVersion::new(1, 0, 0)],
                minimum: ProtocolVersion::new(1, 0, 0),
                features: FeatureSet::EMPTY,
                required: FeatureSet::EMPTY,
            },
            nonce: 0,
        };
        assert_eq!(encode(&empty_node), Err(CodecError::EmptyField("node_id")));
        let many = ControlFrame::Decl {
            capabilities: vec![CapabilityId::from("c"); MAX_CAPABILITIES + 1],
        };
        assert!(matches!(
            encode(&many),
            Err(CodecError::TooManyItems { .. })
        ));
        let bad_type = ControlFrame::Reply {
            seq: 1,
            payload: TypedPayload {
                type_id: TypeId::from(""),
                bytes: vec![],
            },
        };
        assert_eq!(encode(&bad_type), Err(CodecError::EmptyField("type_id")));
    }

    #[test]
    fn keys_are_bounded() {
        assert_eq!(IdempotencyKey::new(""), Err(KeyError::Empty));
        assert_eq!(
            IdempotencyKey::new("k".repeat(MAX_KEY_BYTES + 1)),
            Err(KeyError::TooLong {
                len: MAX_KEY_BYTES + 1
            })
        );
        assert!(IdempotencyKey::new("k".repeat(MAX_KEY_BYTES)).is_ok());
    }

    #[test]
    fn payload_digest_binds_type_and_bytes() {
        let a = payload(&[1, 2]);
        let mut other_type = a.clone();
        other_type.type_id = TypeId::from("Other");
        let mut other_bytes = a.clone();
        other_bytes.bytes.push(3);
        assert_eq!(payload_digest(&a), payload_digest(&a.clone()));
        assert_ne!(payload_digest(&a), payload_digest(&other_type));
        assert_ne!(payload_digest(&a), payload_digest(&other_bytes));
    }

    #[test]
    fn seeded_random_bytes_and_mutations_never_panic() {
        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for _ in 0..2000 {
            let len = (next() % 96) as usize;
            let bytes: Vec<u8> = (0..len).map(|_| next() as u8).collect();
            let _ = decode(&bytes);
            // Valid header, random body: reaches the field parsers.
            let mut framed = frame_with_tag((next() % 9) as u8, &bytes);
            let _ = decode(&framed);
            // Bit flips of a valid frame.
            for frame in sample_frames() {
                framed = encode(&frame).unwrap();
                let at = (next() as usize) % framed.len();
                framed[at] ^= 1 << (next() % 8);
                let _ = decode(&framed);
            }
        }
    }

    fn frame_with_tag(tag: u8, body: &[u8]) -> Vec<u8> {
        let mut out = CODEC_MAGIC.to_vec();
        out.push(CODEC_VERSION);
        out.push(tag);
        out.extend_from_slice(&(body.len() as u32).to_le_bytes());
        out.extend_from_slice(body);
        out
    }
}
