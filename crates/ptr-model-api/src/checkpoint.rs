//! Opaque, versioned, digest-bound continuation handles.
//!
//! A [`CheckpointHandle`] seals a backend's exported state under a backend id and a format
//! version, bound by SHA-256. [`CheckpointHandle::verify`] checks, in this order: version,
//! backend id, digest. The digest is an integrity check against corruption and accidental
//! mix-ups, not authentication: it is unkeyed, so anyone who can write a handle can also
//! recompute its digest.
//!
//! [`CheckpointableBackend`] is a new trait with provided methods; the echo and scenario
//! backends implement it by exporting the continuation state of a [`ModelResumeRequest`]
//! (request id, revision, round and observation). The semantic context itself is not part
//! of the handle; it is supplied again on resume and must match the recorded revision.

use crate::{
    ModelError, ModelEvent, ModelObservation, ModelResumeRequest, ReferenceEchoBackend,
    ResumableInferenceBackend, ScenarioBackend,
};
use ptr_types::{Digest, RequestId, Revision, SemanticContext, TypeId};
use sha2::{Digest as _, Sha256};
use std::fmt;

/// Current handle format version.
pub const CHECKPOINT_VERSION: u32 = 1;
/// Largest state a handle may seal (Invariant 8).
pub const MAX_CHECKPOINT_STATE_BYTES: usize = 1 << 20;
const MAX_BACKEND_ID_BYTES: usize = 256;
const MAGIC: &[u8; 4] = b"PTCK";
const DOMAIN: &[u8] = b"ptr.model.checkpoint.v1\0";

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CheckpointError {
    EmptyBackendId,
    BackendIdTooLong {
        len: usize,
    },
    StateTooLarge {
        len: usize,
    },
    UnsupportedVersion {
        found: u32,
        supported: u32,
    },
    WrongBackend {
        expected: String,
        found: String,
    },
    DigestMismatch,
    Malformed(&'static str),
    RevisionMismatch {
        recorded: Revision,
        supplied: Revision,
    },
    Resume(ModelError),
}

impl fmt::Display for CheckpointError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyBackendId => write!(f, "checkpoint backend id is empty"),
            Self::BackendIdTooLong { len } => write!(f, "backend id of {len} bytes is too long"),
            Self::StateTooLarge { len } => {
                write!(f, "checkpoint state of {len} bytes is too large")
            }
            Self::UnsupportedVersion { found, supported } => {
                write!(
                    f,
                    "checkpoint version {found} unsupported (supported {supported})"
                )
            }
            Self::WrongBackend { expected, found } => {
                write!(f, "checkpoint is for backend {found}, not {expected}")
            }
            Self::DigestMismatch => write!(f, "checkpoint digest does not match its content"),
            Self::Malformed(why) => write!(f, "malformed checkpoint: {why}"),
            Self::RevisionMismatch { recorded, supplied } => write!(
                f,
                "checkpoint recorded revision {} but context is at {}",
                recorded.0, supplied.0
            ),
            Self::Resume(error) => write!(f, "resume failed: {}", error.0),
        }
    }
}

impl std::error::Error for CheckpointError {}

fn bind(version: u32, backend_id: &str, state: &[u8]) -> Digest {
    let mut hasher = Sha256::new();
    hasher.update(DOMAIN);
    hasher.update(version.to_le_bytes());
    hasher.update((backend_id.len() as u64).to_le_bytes());
    hasher.update(backend_id.as_bytes());
    hasher.update((state.len() as u64).to_le_bytes());
    hasher.update(state);
    hasher.finalize().into()
}

/// Sealed backend state. Fields are private; use [`CheckpointHandle::to_bytes`] /
/// [`CheckpointHandle::from_bytes`] to move it.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CheckpointHandle {
    version: u32,
    backend_id: String,
    state: Vec<u8>,
    digest: Digest,
}

impl CheckpointHandle {
    pub fn seal(backend_id: impl Into<String>, state: Vec<u8>) -> Result<Self, CheckpointError> {
        let backend_id = backend_id.into();
        if backend_id.is_empty() {
            return Err(CheckpointError::EmptyBackendId);
        }
        if backend_id.len() > MAX_BACKEND_ID_BYTES {
            return Err(CheckpointError::BackendIdTooLong {
                len: backend_id.len(),
            });
        }
        if state.len() > MAX_CHECKPOINT_STATE_BYTES {
            return Err(CheckpointError::StateTooLarge { len: state.len() });
        }
        let digest = bind(CHECKPOINT_VERSION, &backend_id, &state);
        Ok(Self {
            version: CHECKPOINT_VERSION,
            backend_id,
            state,
            digest,
        })
    }

    pub fn version(&self) -> u32 {
        self.version
    }

    pub fn backend_id(&self) -> &str {
        &self.backend_id
    }

    pub fn digest(&self) -> &Digest {
        &self.digest
    }

    /// Check version, then backend id, then digest; on success return the sealed state.
    pub fn verify(&self, expected_backend: &str) -> Result<&[u8], CheckpointError> {
        if self.version != CHECKPOINT_VERSION {
            return Err(CheckpointError::UnsupportedVersion {
                found: self.version,
                supported: CHECKPOINT_VERSION,
            });
        }
        if self.backend_id != expected_backend {
            return Err(CheckpointError::WrongBackend {
                expected: expected_backend.to_owned(),
                found: self.backend_id.clone(),
            });
        }
        if bind(self.version, &self.backend_id, &self.state) != self.digest {
            return Err(CheckpointError::DigestMismatch);
        }
        Ok(&self.state)
    }

    /// `PTCK | version u32 | id len u32 | id | state len u64 | state | digest`, little endian.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.state.len() + self.backend_id.len() + 52);
        out.extend_from_slice(MAGIC);
        out.extend_from_slice(&self.version.to_le_bytes());
        out.extend_from_slice(&(self.backend_id.len() as u32).to_le_bytes());
        out.extend_from_slice(self.backend_id.as_bytes());
        out.extend_from_slice(&(self.state.len() as u64).to_le_bytes());
        out.extend_from_slice(&self.state);
        out.extend_from_slice(&self.digest);
        out
    }

    /// Parse without verifying the digest; call [`CheckpointHandle::verify`] before use.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, CheckpointError> {
        let mut reader = Reader { bytes };
        if reader.take(4)? != MAGIC {
            return Err(CheckpointError::Malformed("bad magic"));
        }
        let version = reader.u32()?;
        let id_len = reader.u32()? as usize;
        if id_len == 0 || id_len > MAX_BACKEND_ID_BYTES {
            return Err(CheckpointError::Malformed("backend id length"));
        }
        let backend_id = String::from_utf8(reader.take(id_len)?.to_vec())
            .map_err(|_| CheckpointError::Malformed("backend id not utf-8"))?;
        let state_len = usize::try_from(reader.u64()?)
            .map_err(|_| CheckpointError::Malformed("state length"))?;
        if state_len > MAX_CHECKPOINT_STATE_BYTES {
            return Err(CheckpointError::StateTooLarge { len: state_len });
        }
        let state = reader.take(state_len)?.to_vec();
        let mut digest = [0u8; 32];
        digest.copy_from_slice(reader.take(32)?);
        if !reader.bytes.is_empty() {
            return Err(CheckpointError::Malformed("trailing bytes"));
        }
        Ok(Self {
            version,
            backend_id,
            state,
            digest,
        })
    }
}

struct Reader<'a> {
    bytes: &'a [u8],
}

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], CheckpointError> {
        if self.bytes.len() < n {
            return Err(CheckpointError::Malformed("truncated"));
        }
        let (head, tail) = self.bytes.split_at(n);
        self.bytes = tail;
        Ok(head)
    }

    fn u32(&mut self) -> Result<u32, CheckpointError> {
        let mut buf = [0u8; 4];
        buf.copy_from_slice(self.take(4)?);
        Ok(u32::from_le_bytes(buf))
    }

    fn u64(&mut self) -> Result<u64, CheckpointError> {
        let mut buf = [0u8; 8];
        buf.copy_from_slice(self.take(8)?);
        Ok(u64::from_le_bytes(buf))
    }

    fn blob(&mut self) -> Result<&'a [u8], CheckpointError> {
        let len =
            usize::try_from(self.u64()?).map_err(|_| CheckpointError::Malformed("field length"))?;
        self.take(len)
    }

    fn text(&mut self) -> Result<String, CheckpointError> {
        String::from_utf8(self.blob()?.to_vec())
            .map_err(|_| CheckpointError::Malformed("field not utf-8"))
    }
}

fn put_blob(out: &mut Vec<u8>, data: &[u8]) {
    out.extend_from_slice(&(data.len() as u64).to_le_bytes());
    out.extend_from_slice(data);
}

#[cfg(test)]
thread_local! {
    static ENCODE_CALLS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

fn encode_resume_state(request: &ModelResumeRequest) -> Vec<u8> {
    #[cfg(test)]
    ENCODE_CALLS.with(|c| c.set(c.get() + 1));
    let mut out = Vec::new();
    put_blob(&mut out, request.request_id.0.as_bytes());
    out.extend_from_slice(&request.semantic_context.revision.0.to_le_bytes());
    out.extend_from_slice(&request.round.to_le_bytes());
    out.extend_from_slice(&request.observation.revision.0.to_le_bytes());
    put_blob(&mut out, request.observation.source.as_bytes());
    put_blob(&mut out, request.observation.type_id.0.as_bytes());
    put_blob(&mut out, &request.observation.payload);
    out
}

/// Exact length `encode_resume_state` will produce, without materializing it. Keep the two
/// in lockstep (a unit test asserts equality). `None` on arithmetic overflow.
fn encoded_resume_state_len(request: &ModelResumeRequest) -> Option<usize> {
    // Fixed fields: context revision (8), round (4), observation revision (8).
    const FIXED: usize = 8 + 4 + 8;
    // Each blob carries a u64 length prefix.
    const PREFIX: usize = 8;
    [
        request.request_id.0.len(),
        request.observation.source.len(),
        request.observation.type_id.0.len(),
        request.observation.payload.len(),
    ]
    .into_iter()
    .try_fold(FIXED, |acc, len| acc.checked_add(PREFIX)?.checked_add(len))
}

fn decode_resume_state(
    state: &[u8],
    context: SemanticContext,
) -> Result<ModelResumeRequest, CheckpointError> {
    let mut reader = Reader { bytes: state };
    let request_id = RequestId(reader.text()?);
    let recorded = Revision(reader.u64()?);
    let round = reader.u32()?;
    let observation_revision = Revision(reader.u64()?);
    let source = reader.text()?;
    let type_id = TypeId(reader.text()?);
    let payload = reader.blob()?.to_vec();
    if !reader.bytes.is_empty() {
        return Err(CheckpointError::Malformed("trailing state bytes"));
    }
    if context.revision != recorded {
        return Err(CheckpointError::RevisionMismatch {
            recorded,
            supplied: context.revision,
        });
    }
    Ok(ModelResumeRequest {
        request_id,
        semantic_context: context,
        round,
        observation: ModelObservation {
            revision: observation_revision,
            source,
            type_id,
            payload,
        },
    })
}

/// Resumable backend whose continuation can be exported and re-imported as a
/// [`CheckpointHandle`]. All methods are provided, so implementing it is opt-in and does
/// not touch [`ResumableInferenceBackend`].
pub trait CheckpointableBackend: ResumableInferenceBackend {
    /// Seal the continuation state of `request` under this backend's name.
    fn export_checkpoint(
        &self,
        request: &ModelResumeRequest,
    ) -> Result<CheckpointHandle, CheckpointError> {
        match encoded_resume_state_len(request) {
            Some(len) if len <= MAX_CHECKPOINT_STATE_BYTES => {}
            Some(len) => return Err(CheckpointError::StateTooLarge { len }),
            None => return Err(CheckpointError::StateTooLarge { len: usize::MAX }),
        }
        CheckpointHandle::seal(self.name(), encode_resume_state(request))
    }

    /// Verify `handle` for this backend, rebuild the resume request with `context`, and
    /// resume. `context.revision` must equal the revision recorded at export.
    fn resume_from_checkpoint(
        &self,
        handle: &CheckpointHandle,
        context: SemanticContext,
    ) -> Result<Vec<ModelEvent>, CheckpointError> {
        let state = handle.verify(self.name())?;
        let request = decode_resume_state(state, context)?;
        self.resume(&request).map_err(CheckpointError::Resume)
    }
}

impl CheckpointableBackend for ReferenceEchoBackend {}
impl CheckpointableBackend for ScenarioBackend {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn handle_round_trips_through_bytes() {
        let handle = CheckpointHandle::seal("b", vec![1, 2, 3]).unwrap();
        let parsed = CheckpointHandle::from_bytes(&handle.to_bytes()).unwrap();
        assert_eq!(parsed, handle);
        assert_eq!(parsed.verify("b"), Ok(&[1u8, 2, 3][..]));
    }

    #[test]
    fn seal_validates_inputs() {
        assert_eq!(
            CheckpointHandle::seal("", vec![]),
            Err(CheckpointError::EmptyBackendId)
        );
        assert!(matches!(
            CheckpointHandle::seal("b", vec![0; MAX_CHECKPOINT_STATE_BYTES + 1]),
            Err(CheckpointError::StateTooLarge { .. })
        ));
        assert!(matches!(
            CheckpointHandle::seal("x".repeat(257), vec![]),
            Err(CheckpointError::BackendIdTooLong { .. })
        ));
    }

    fn request(id: &str, source: &str, type_id: &str, payload: Vec<u8>) -> ModelResumeRequest {
        ModelResumeRequest {
            request_id: RequestId(id.into()),
            semantic_context: SemanticContext {
                revision: Revision(1),
                entries: vec![],
            },
            round: 3,
            observation: ModelObservation {
                revision: Revision(2),
                source: source.into(),
                type_id: TypeId(type_id.into()),
                payload,
            },
        }
    }

    #[test]
    fn encoded_len_matches_encoding() {
        for r in [
            request("", "", "", vec![]),
            request("req-1", "tool", "ptr.t.v1", vec![1, 2, 3]),
            request(
                "r\u{e9}q-\u{1f600}",
                "s\u{fc}\u{4e2d}",
                "t\u{3b1}",
                vec![0; 1000],
            ),
        ] {
            assert_eq!(
                encoded_resume_state_len(&r),
                Some(encode_resume_state(&r).len())
            );
        }
    }

    #[test]
    fn oversized_request_is_rejected_before_encoding() {
        let backend = ReferenceEchoBackend;
        let big = request("r", "s", "t", vec![0; MAX_CHECKPOINT_STATE_BYTES + 1]);
        ENCODE_CALLS.with(|c| c.set(0));
        assert!(matches!(
            backend.export_checkpoint(&big),
            Err(CheckpointError::StateTooLarge { .. })
        ));
        assert_eq!(
            ENCODE_CALLS.with(std::cell::Cell::get),
            0,
            "encoded before the cap check"
        );
        let ok = request("r", "s", "t", vec![0; 16]);
        assert!(backend.export_checkpoint(&ok).is_ok());
        assert_eq!(ENCODE_CALLS.with(std::cell::Cell::get), 1);
    }

    #[test]
    fn declared_lengths_beyond_buffer_are_rejected_without_allocating() {
        let mut state = Vec::new();
        state.extend_from_slice(&u64::MAX.to_le_bytes());
        assert!(matches!(
            decode_resume_state(
                &state,
                SemanticContext {
                    revision: Revision(1),
                    entries: vec![]
                }
            ),
            Err(CheckpointError::Malformed(_))
        ));
        let mut wire = CheckpointHandle::seal("b", vec![1]).unwrap().to_bytes();
        let at = 4 + 4 + 4 + 1;
        wire[at..at + 8].copy_from_slice(&(MAX_CHECKPOINT_STATE_BYTES as u64).to_le_bytes());
        assert_eq!(
            CheckpointHandle::from_bytes(&wire),
            Err(CheckpointError::Malformed("truncated"))
        );
    }
}
