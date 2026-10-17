//! The data-plane calls a worker serves over the channel (doc 42 stage S1), and the key
//! handshake that precedes the control session. Shared by the worker and the host.
//!
//! The data-plane calls are ordinary `Call` payloads of the control session, with their own
//! type (`worker.data.v1`) next to the Pod calls (`worker.call.v1`):
//!
//! ```text
//! request (type "worker.data.v1"), first byte is the operation:
//!   0x01 produce : u32 BE template length | template handle (encoded, unsigned) | tensor bytes
//!   0x02 consume : flags u8 (bit 0: keep the mapping) | the admitted handle (encoded)
//!   0x03 probe   : nothing
//!   0x04 produce staged : the template (encoded, unsigned, its locator already the CudaIpc one
//!                         the host expects); descriptors: [staging buffer, sealed memfd with the
//!                         tensor]
//!   0x05 consume staged : the export record (36 bytes, see `WireExportRecord`) | the admitted
//!                         handle (encoded); descriptor: [staging buffer]
//! reply (type "worker.data.reply.v1"): status u8 (0 ok, 1 error) | body
//!   produce ok : the signed handle (encoded); the region, if any, is the frame's descriptor
//!                (a staged produce has none: the tensor is in the staging buffer)
//!   consume ok : content digest (32 bytes) | mapped length u64 BE
//!   probe ok   : u8 count | per entry: u8 name length | name | i32 BE errno
//!   error      : a short code
//! ```
//!
//! The key handshake uses three frames before the control session starts. They are not
//! `ControlFrame`s: the first carries a secret and must not be mistaken for session traffic.
//!
//! ```text
//! key       : "PTR-KEY1" | session u64 BE | u16 BE stage length | stage | secret key (32 bytes)
//! challenge : "PTR-CHAL1" | nonce (32 bytes)
//! response  : "PTR-SIG1" | Ed25519 signature (64 bytes) over
//!             KEY_DOMAIN | session u64 BE | u16 BE stage length | stage | nonce
//!   or, from a worker that cannot start (a GPU profile without a driver):
//!             "PTR-ERR1" | a short startup code (UTF-8, at most 64 bytes)
//! ```

use ptr_protocol::TypedPayload;
use ptr_types::TypeId;

pub const DATA_TYPE: &str = "worker.data.v1";
pub const DATA_REPLY_TYPE: &str = "worker.data.reply.v1";
pub const KEY_FRAME_MAGIC: &[u8] = b"PTR-KEY1";
pub const CHALLENGE_MAGIC: &[u8] = b"PTR-CHAL1";
pub const RESPONSE_MAGIC: &[u8] = b"PTR-SIG1";
pub const STARTUP_REFUSAL_MAGIC: &[u8] = b"PTR-ERR1";
/// Longest startup code a refusal carries.
pub const MAX_STARTUP_CODE_BYTES: usize = 64;
/// The domain the challenge signature is bound to.
pub const KEY_DOMAIN: &[u8] = b"ptr-worker.key-challenge.v1\0";
pub const NONCE_BYTES: usize = 32;
pub const MAX_STAGE_IDENTITY_BYTES: usize = 256;
/// Most entries and longest name of a probe report.
pub const MAX_PROBE_ENTRIES: usize = 32;

const OP_PRODUCE: u8 = 1;
const OP_CONSUME: u8 = 2;
const OP_PROBE: u8 = 3;
const OP_PRODUCE_STAGED: u8 = 4;
const OP_CONSUME_STAGED: u8 = 5;
const STATUS_OK: u8 = 0;
const STATUS_ERROR: u8 = 1;
const FLAG_RETAIN: u8 = 1;

/// Why a data-plane payload or a handshake frame is malformed.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DataWireError {
    WrongType,
    Truncated,
    UnknownOperation(u8),
    UnknownStatus(u8),
    BadMagic,
    StageTooLong,
    StageNotUtf8,
    EmptyStage,
}

impl std::fmt::Display for DataWireError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for DataWireError {}

/// Bytes of an encoded [`WireExportRecord`].
const WIRE_RECORD_BYTES: usize = 8 + 8 + 8 + 4 + 8;

/// The supervisor's export record, as the consumer of a staged handle receives it (the CUDA
/// import checks it against the handle; the fake import checks the same fields). It comes from
/// the host, which minted it; the worker never chooses it.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct WireExportRecord {
    pub producer_session: u64,
    pub consumer_session: u64,
    pub operation: u64,
    pub device: u32,
    pub size: u64,
}

impl WireExportRecord {
    fn encode_into(&self, bytes: &mut Vec<u8>) {
        bytes.extend_from_slice(&self.producer_session.to_be_bytes());
        bytes.extend_from_slice(&self.consumer_session.to_be_bytes());
        bytes.extend_from_slice(&self.operation.to_be_bytes());
        bytes.extend_from_slice(&self.device.to_be_bytes());
        bytes.extend_from_slice(&self.size.to_be_bytes());
    }

    fn decode(bytes: &[u8]) -> Result<Self, DataWireError> {
        let field = |range: std::ops::Range<usize>| -> Result<[u8; 8], DataWireError> {
            bytes
                .get(range)
                .and_then(|slice| slice.try_into().ok())
                .ok_or(DataWireError::Truncated)
        };
        let device: [u8; 4] = bytes
            .get(24..28)
            .and_then(|slice| slice.try_into().ok())
            .ok_or(DataWireError::Truncated)?;
        Ok(Self {
            producer_session: u64::from_be_bytes(field(0..8)?),
            consumer_session: u64::from_be_bytes(field(8..16)?),
            operation: u64::from_be_bytes(field(16..24)?),
            device: u32::from_be_bytes(device),
            size: u64::from_be_bytes(field(28..36)?),
        })
    }
}

/// A data-plane request.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum DataRequest {
    /// Build the tensor, sign the handle (the template with the digest and locator filled in).
    Produce { template: Vec<u8>, tensor: Vec<u8> },
    /// Map or import the admitted handle, verify it, answer. `retain` keeps the mapping alive
    /// after the answer: a consumer that lies about having let go, for tests of the quota rule.
    Consume { handle: Vec<u8>, retain: bool },
    /// Report what the worker's filter refuses.
    Probe,
    /// Write the tensor through the staging buffer descriptor and sign the handle (`CudaIpc`,
    /// doc 42 rule 3a). The tensor itself is the second descriptor of the call, a sealed
    /// `memfd` of exactly `byte_len` bytes, so a tensor is not bounded by the frame size.
    ProduceStaged { template: Vec<u8> },
    /// Import the staging buffer descriptor under the export record, copy the tensor into
    /// memory of the worker's own and verify the digest on that copy.
    ConsumeStaged {
        handle: Vec<u8>,
        record: WireExportRecord,
    },
}

impl DataRequest {
    pub fn to_payload(&self) -> TypedPayload {
        let mut bytes = Vec::new();
        match self {
            Self::Produce { template, tensor } => {
                bytes.push(OP_PRODUCE);
                bytes.extend_from_slice(&(template.len() as u32).to_be_bytes());
                bytes.extend_from_slice(template);
                bytes.extend_from_slice(tensor);
            }
            Self::Consume { handle, retain } => {
                bytes.push(OP_CONSUME);
                bytes.push(if *retain { FLAG_RETAIN } else { 0 });
                bytes.extend_from_slice(handle);
            }
            Self::Probe => bytes.push(OP_PROBE),
            Self::ProduceStaged { template } => {
                bytes.push(OP_PRODUCE_STAGED);
                bytes.extend_from_slice(template);
            }
            Self::ConsumeStaged { handle, record } => {
                bytes.push(OP_CONSUME_STAGED);
                record.encode_into(&mut bytes);
                bytes.extend_from_slice(handle);
            }
        }
        TypedPayload {
            type_id: TypeId::from(DATA_TYPE),
            bytes,
        }
    }

    pub fn from_payload(payload: &TypedPayload) -> Result<Self, DataWireError> {
        if payload.type_id.0 != DATA_TYPE {
            return Err(DataWireError::WrongType);
        }
        let (&operation, rest) = payload
            .bytes
            .split_first()
            .ok_or(DataWireError::Truncated)?;
        match operation {
            OP_PRODUCE => {
                let prefix: [u8; 4] = rest
                    .get(..4)
                    .and_then(|bytes| bytes.try_into().ok())
                    .ok_or(DataWireError::Truncated)?;
                let length = u32::from_be_bytes(prefix) as usize;
                let body = &rest[4..];
                let template = body.get(..length).ok_or(DataWireError::Truncated)?;
                Ok(Self::Produce {
                    template: template.to_vec(),
                    tensor: body[length..].to_vec(),
                })
            }
            OP_CONSUME => {
                let (&flags, handle) = rest.split_first().ok_or(DataWireError::Truncated)?;
                Ok(Self::Consume {
                    handle: handle.to_vec(),
                    retain: flags & FLAG_RETAIN != 0,
                })
            }
            OP_PROBE => Ok(Self::Probe),
            OP_PRODUCE_STAGED => Ok(Self::ProduceStaged {
                template: rest.to_vec(),
            }),
            OP_CONSUME_STAGED => {
                let record = WireExportRecord::decode(rest)?;
                Ok(Self::ConsumeStaged {
                    handle: rest.get(WIRE_RECORD_BYTES..).unwrap_or_default().to_vec(),
                    record,
                })
            }
            other => Err(DataWireError::UnknownOperation(other)),
        }
    }
}

/// The answer to a data-plane request.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum DataReply {
    Produced { handle: Vec<u8> },
    Consumed { digest: [u8; 32], mapped_len: u64 },
    Probed { calls: Vec<(String, i32)> },
    Refused { code: String },
}

impl DataReply {
    pub fn to_payload(&self) -> TypedPayload {
        let mut bytes = Vec::new();
        match self {
            Self::Produced { handle } => {
                bytes.push(STATUS_OK);
                bytes.push(OP_PRODUCE);
                bytes.extend_from_slice(handle);
            }
            Self::Consumed { digest, mapped_len } => {
                bytes.push(STATUS_OK);
                bytes.push(OP_CONSUME);
                bytes.extend_from_slice(digest);
                bytes.extend_from_slice(&mapped_len.to_be_bytes());
            }
            Self::Probed { calls } => {
                bytes.push(STATUS_OK);
                bytes.push(OP_PROBE);
                let count = calls.len().min(MAX_PROBE_ENTRIES);
                bytes.push(count as u8);
                for (name, errno) in calls.iter().take(count) {
                    let name = name.as_bytes();
                    let length = name.len().min(255);
                    bytes.push(length as u8);
                    bytes.extend_from_slice(&name[..length]);
                    bytes.extend_from_slice(&errno.to_be_bytes());
                }
            }
            Self::Refused { code } => {
                bytes.push(STATUS_ERROR);
                let mut end = code.len().min(crate::wire::MAX_ERROR_BYTES);
                while !code.is_char_boundary(end) {
                    end -= 1;
                }
                bytes.extend_from_slice(&code.as_bytes()[..end]);
            }
        }
        TypedPayload {
            type_id: TypeId::from(DATA_REPLY_TYPE),
            bytes,
        }
    }

    pub fn from_payload(payload: &TypedPayload) -> Result<Self, DataWireError> {
        if payload.type_id.0 != DATA_REPLY_TYPE {
            return Err(DataWireError::WrongType);
        }
        let (&status, rest) = payload
            .bytes
            .split_first()
            .ok_or(DataWireError::Truncated)?;
        match status {
            STATUS_ERROR => Ok(Self::Refused {
                code: String::from_utf8_lossy(rest).into_owned(),
            }),
            STATUS_OK => {
                let (&operation, body) = rest.split_first().ok_or(DataWireError::Truncated)?;
                match operation {
                    OP_PRODUCE => Ok(Self::Produced {
                        handle: body.to_vec(),
                    }),
                    OP_CONSUME => {
                        let digest: [u8; 32] = body
                            .get(..32)
                            .and_then(|bytes| bytes.try_into().ok())
                            .ok_or(DataWireError::Truncated)?;
                        let length: [u8; 8] = body
                            .get(32..40)
                            .and_then(|bytes| bytes.try_into().ok())
                            .ok_or(DataWireError::Truncated)?;
                        Ok(Self::Consumed {
                            digest,
                            mapped_len: u64::from_be_bytes(length),
                        })
                    }
                    OP_PROBE => {
                        let (&count, mut entries) =
                            body.split_first().ok_or(DataWireError::Truncated)?;
                        if usize::from(count) > MAX_PROBE_ENTRIES {
                            return Err(DataWireError::Truncated);
                        }
                        let mut calls = Vec::new();
                        for _ in 0..count {
                            let (&length, tail) =
                                entries.split_first().ok_or(DataWireError::Truncated)?;
                            let name = tail
                                .get(..usize::from(length))
                                .ok_or(DataWireError::Truncated)?;
                            let errno: [u8; 4] = tail
                                .get(usize::from(length)..usize::from(length) + 4)
                                .and_then(|bytes| bytes.try_into().ok())
                                .ok_or(DataWireError::Truncated)?;
                            calls.push((
                                String::from_utf8_lossy(name).into_owned(),
                                i32::from_be_bytes(errno),
                            ));
                            entries = &tail[usize::from(length) + 4..];
                        }
                        Ok(Self::Probed { calls })
                    }
                    other => Err(DataWireError::UnknownOperation(other)),
                }
            }
            other => Err(DataWireError::UnknownStatus(other)),
        }
    }
}

/// Overwrite `bytes` with zeros in a way the optimizer cannot drop (volatile writes), for
/// secret key material that is about to go out of scope. Best effort: copies the program made
/// earlier are not reached.
pub fn wipe(bytes: &mut [u8]) {
    for byte in bytes.iter_mut() {
        // SAFETY: `byte` is a valid, exclusive reference to one initialised byte.
        unsafe { std::ptr::write_volatile(byte, 0) };
    }
    std::sync::atomic::compiler_fence(std::sync::atomic::Ordering::SeqCst);
}

/// What the supervisor tells the worker first: its session, the one stage it may serve, and
/// the secret key bound to that stage. `Debug` shows no secret, and the secret is wiped when
/// the frame is dropped.
#[derive(Clone, Eq, PartialEq)]
pub struct KeyFrame {
    pub session: u64,
    pub stage_identity: String,
    pub secret: [u8; 32],
}

impl std::fmt::Debug for KeyFrame {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KeyFrame")
            .field("session", &self.session)
            .field("stage_identity", &self.stage_identity)
            .field("secret", &"<redacted>")
            .finish()
    }
}

impl Drop for KeyFrame {
    fn drop(&mut self) {
        wipe(&mut self.secret);
    }
}

impl KeyFrame {
    pub fn encode(&self) -> Result<Vec<u8>, DataWireError> {
        if self.stage_identity.is_empty() {
            return Err(DataWireError::EmptyStage);
        }
        if self.stage_identity.len() > MAX_STAGE_IDENTITY_BYTES {
            return Err(DataWireError::StageTooLong);
        }
        let mut bytes = Vec::new();
        bytes.extend_from_slice(KEY_FRAME_MAGIC);
        bytes.extend_from_slice(&self.session.to_be_bytes());
        bytes.extend_from_slice(&(self.stage_identity.len() as u16).to_be_bytes());
        bytes.extend_from_slice(self.stage_identity.as_bytes());
        bytes.extend_from_slice(&self.secret);
        Ok(bytes)
    }

    pub fn decode(bytes: &[u8]) -> Result<Self, DataWireError> {
        let rest = bytes
            .strip_prefix(KEY_FRAME_MAGIC)
            .ok_or(DataWireError::BadMagic)?;
        let session: [u8; 8] = rest
            .get(..8)
            .and_then(|bytes| bytes.try_into().ok())
            .ok_or(DataWireError::Truncated)?;
        let length: [u8; 2] = rest
            .get(8..10)
            .and_then(|bytes| bytes.try_into().ok())
            .ok_or(DataWireError::Truncated)?;
        let length = usize::from(u16::from_be_bytes(length));
        if length == 0 {
            return Err(DataWireError::EmptyStage);
        }
        if length > MAX_STAGE_IDENTITY_BYTES {
            return Err(DataWireError::StageTooLong);
        }
        let stage = rest.get(10..10 + length).ok_or(DataWireError::Truncated)?;
        let secret: [u8; 32] = rest
            .get(10 + length..10 + length + 32)
            .and_then(|bytes| bytes.try_into().ok())
            .ok_or(DataWireError::Truncated)?;
        if rest.len() != 10 + length + 32 {
            return Err(DataWireError::Truncated);
        }
        Ok(Self {
            session: u64::from_be_bytes(session),
            stage_identity: String::from_utf8(stage.to_vec())
                .map_err(|_| DataWireError::StageNotUtf8)?,
            secret,
        })
    }
}

pub fn encode_challenge(nonce: &[u8; NONCE_BYTES]) -> Vec<u8> {
    let mut bytes = CHALLENGE_MAGIC.to_vec();
    bytes.extend_from_slice(nonce);
    bytes
}

pub fn decode_challenge(bytes: &[u8]) -> Result<[u8; NONCE_BYTES], DataWireError> {
    let rest = bytes
        .strip_prefix(CHALLENGE_MAGIC)
        .ok_or(DataWireError::BadMagic)?;
    rest.try_into().map_err(|_| DataWireError::Truncated)
}

pub fn encode_response(signature: &[u8; 64]) -> Vec<u8> {
    let mut bytes = RESPONSE_MAGIC.to_vec();
    bytes.extend_from_slice(signature);
    bytes
}

/// What a worker answers to the challenge: the signature, or the code of why it cannot serve.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Handshake {
    Signed([u8; 64]),
    /// The worker refused to start (for example `gpu-no-driver`).
    Refused(String),
}

/// A refusal frame instead of a signature. The code is cut to [`MAX_STARTUP_CODE_BYTES`].
pub fn encode_startup_refusal(code: &str) -> Vec<u8> {
    let mut end = code.len().min(MAX_STARTUP_CODE_BYTES);
    while !code.is_char_boundary(end) {
        end -= 1;
    }
    let mut bytes = STARTUP_REFUSAL_MAGIC.to_vec();
    bytes.extend_from_slice(&code.as_bytes()[..end]);
    bytes
}

/// Decode the worker's answer to the challenge: a signature or a startup refusal.
pub fn decode_handshake(bytes: &[u8]) -> Result<Handshake, DataWireError> {
    if let Some(code) = bytes.strip_prefix(STARTUP_REFUSAL_MAGIC) {
        if code.len() > MAX_STARTUP_CODE_BYTES {
            return Err(DataWireError::Truncated);
        }
        return Ok(Handshake::Refused(
            String::from_utf8_lossy(code).into_owned(),
        ));
    }
    decode_response(bytes).map(Handshake::Signed)
}

pub fn decode_response(bytes: &[u8]) -> Result<[u8; 64], DataWireError> {
    let rest = bytes
        .strip_prefix(RESPONSE_MAGIC)
        .ok_or(DataWireError::BadMagic)?;
    rest.try_into().map_err(|_| DataWireError::Truncated)
}

/// The bytes the worker signs: the domain, the session, the stage identity the supervisor
/// bound the worker to (length-prefixed) and the nonce, so a signature proves possession of the
/// key for exactly this session and stage.
pub fn challenge_message(session: u64, stage: &str, nonce: &[u8; NONCE_BYTES]) -> Vec<u8> {
    let mut message = KEY_DOMAIN.to_vec();
    message.extend_from_slice(&session.to_be_bytes());
    message.extend_from_slice(&(stage.len() as u16).to_be_bytes());
    message.extend_from_slice(stage.as_bytes());
    message.extend_from_slice(nonce);
    message
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn requests_and_replies_round_trip() {
        for request in [
            DataRequest::Produce {
                template: vec![1, 2, 3],
                tensor: vec![4, 5],
            },
            DataRequest::Consume {
                handle: vec![9; 7],
                retain: true,
            },
            DataRequest::Probe,
            DataRequest::ProduceStaged {
                template: vec![8; 11],
            },
            DataRequest::ConsumeStaged {
                handle: vec![6; 5],
                record: WireExportRecord {
                    producer_session: 1,
                    consumer_session: 2,
                    operation: 3,
                    device: 4,
                    size: 8192,
                },
            },
        ] {
            assert_eq!(
                DataRequest::from_payload(&request.to_payload()),
                Ok(request)
            );
        }
        for reply in [
            DataReply::Produced { handle: vec![1; 5] },
            DataReply::Consumed {
                digest: [7; 32],
                mapped_len: 4096,
            },
            DataReply::Probed {
                calls: vec![("socket".into(), 1), ("fork".into(), 0)],
            },
            DataReply::Refused {
                code: "stage-not-bound".into(),
            },
        ] {
            assert_eq!(DataReply::from_payload(&reply.to_payload()), Ok(reply));
        }
    }

    #[test]
    fn malformed_payloads_are_typed_errors() {
        let mut payload = DataRequest::Probe.to_payload();
        payload.type_id = TypeId::from("other");
        assert_eq!(
            DataRequest::from_payload(&payload),
            Err(DataWireError::WrongType)
        );
        let mut payload = DataRequest::Probe.to_payload();
        payload.bytes = vec![0x55];
        assert_eq!(
            DataRequest::from_payload(&payload),
            Err(DataWireError::UnknownOperation(0x55))
        );
        payload.bytes = vec![OP_PRODUCE, 0, 0, 0, 9, 1];
        assert_eq!(
            DataRequest::from_payload(&payload),
            Err(DataWireError::Truncated)
        );
        payload.bytes = Vec::new();
        assert_eq!(
            DataRequest::from_payload(&payload),
            Err(DataWireError::Truncated)
        );
        // A staged consume shorter than its record is refused at every length.
        for length in 0..WIRE_RECORD_BYTES {
            payload.bytes = vec![OP_CONSUME_STAGED];
            payload.bytes.extend(std::iter::repeat_n(1u8, length));
            assert_eq!(
                DataRequest::from_payload(&payload),
                Err(DataWireError::Truncated),
                "length {length}"
            );
        }
        let mut reply = DataReply::Probed { calls: Vec::new() }.to_payload();
        reply.bytes = vec![STATUS_OK, OP_CONSUME, 1, 2];
        assert_eq!(
            DataReply::from_payload(&reply),
            Err(DataWireError::Truncated)
        );
        reply.bytes = vec![7];
        assert_eq!(
            DataReply::from_payload(&reply),
            Err(DataWireError::UnknownStatus(7))
        );
    }

    #[test]
    fn the_key_frame_and_the_challenge_round_trip_and_refuse_garbage() {
        let frame = KeyFrame {
            session: 0x0102_0304_0506_0708,
            stage_identity: "stage-a".to_owned(),
            secret: [3; 32],
        };
        assert_eq!(
            KeyFrame::decode(&frame.encode().unwrap()),
            Ok(frame.clone())
        );
        let mut bytes = frame.encode().unwrap();
        bytes.push(0);
        assert_eq!(KeyFrame::decode(&bytes), Err(DataWireError::Truncated));
        assert_eq!(KeyFrame::decode(b"nope"), Err(DataWireError::BadMagic));
        let empty = KeyFrame {
            stage_identity: String::new(),
            ..frame
        };
        assert_eq!(empty.encode(), Err(DataWireError::EmptyStage));
        let nonce = [9u8; NONCE_BYTES];
        assert_eq!(decode_challenge(&encode_challenge(&nonce)), Ok(nonce));
        assert_eq!(
            decode_challenge(b"PTR-CHAL1short"),
            Err(DataWireError::Truncated)
        );
        assert_eq!(decode_response(&encode_response(&[5; 64])), Ok([5; 64]));
        assert_eq!(decode_response(b"x"), Err(DataWireError::BadMagic));
        assert_eq!(
            decode_handshake(&encode_response(&[5; 64])),
            Ok(Handshake::Signed([5; 64]))
        );
        assert_eq!(
            decode_handshake(&encode_startup_refusal("gpu-no-driver")),
            Ok(Handshake::Refused("gpu-no-driver".to_owned()))
        );
        let long = encode_startup_refusal(&"x".repeat(500));
        assert_eq!(
            long.len(),
            STARTUP_REFUSAL_MAGIC.len() + MAX_STARTUP_CODE_BYTES
        );
        assert_eq!(decode_handshake(b"junk"), Err(DataWireError::BadMagic));
        assert!(challenge_message(1, "s", &nonce).starts_with(KEY_DOMAIN));
        // The stage is part of what is signed.
        assert_ne!(
            challenge_message(1, "a", &nonce),
            challenge_message(1, "b", &nonce)
        );
        assert_ne!(
            challenge_message(1, "a", &nonce),
            challenge_message(2, "a", &nonce)
        );
        assert!(!format!("{frame:?}").contains("secret: ["));
        assert!(format!("{frame:?}").contains("<redacted>"));
        let mut secret = [7u8; 32];
        wipe(&mut secret);
        assert_eq!(secret, [0u8; 32]);
    }
}
