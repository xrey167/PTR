//! The payload schemas a split-pod (pipeline layer split) operation admits, and nothing else.
//!
//! `TypedPayload` is arbitrary bytes plus a caller-controlled `type_id`, so a deny-list of
//! KV type ids can be evaded by mislabelling (doc 36, P5a). A split operation therefore
//! admits **only** the schemas named in [`ALLOWLISTED_TYPES`], decodes them (typed,
//! schema-validated, bounded) and refuses any other `type_id`. The receiving side has no
//! path from these schemas to a KV page: an [`ActivationFrame`] decodes into a tensor of
//! activations and positions, and the only thing that can be done with it is to feed it to
//! compute.
//!
//! A frame is `tokens x width` activations in the boundary dtype (F32 only: the reference
//! path computes in F32 and a cast would hide a mismatch) plus the position of each row,
//! the [`UpstreamChain`] that produced them and the placement epoch the producing stage
//! held. One frame can exceed `MAX_BODY_BYTES` of PodWire, so it travels as
//! [`ActivationChunk`]s, each its own `TypedPayload`; every chunk repeats the digest of the
//! whole frame, the parent operation, the signed deadline and the chain, and carries a
//! signature over its envelope.
//!
//! The codecs are hand written, length-prefixed and bounded, every enumeration crosses as
//! an explicit code, and bytes that disagree with themselves are refused whole.

use ptr_types::TypeId;
use sha2::{Digest as _, Sha256};
use std::fmt;

use crate::TypedPayload;

/// Type id of one chunk of an activation frame.
pub const ACTIVATION_CHUNK_TYPE: &str = "ptr.split.activation-chunk.v1";
/// Type id of the policy payload of a split operation.
pub const SPLIT_POLICY_TYPE: &str = "ptr.split.policy.v1";
/// Every type id a split operation admits. Anything else is refused.
pub const ALLOWLISTED_TYPES: [&str; 2] = [ACTIVATION_CHUNK_TYPE, SPLIT_POLICY_TYPE];

/// Bound on the rows of one frame.
pub const MAX_FRAME_TOKENS: u32 = 4096;
/// Bound on the width of one row.
pub const MAX_FRAME_WIDTH: u32 = 8192;
/// Bound on the chunks of one frame.
pub const MAX_CHUNKS_PER_FRAME: u32 = 64;
/// Bound on the elements of one chain list (adapters, experts).
pub const MAX_CHAIN_ELEMENTS: usize = 16;
/// Bound on an identity string.
pub const MAX_IDENTITY_BYTES: usize = 256;
/// Bound on the bytes of one frame (tensor included).
pub const MAX_FRAME_BYTES: u64 = 64 * 1024 * 1024;

const FRAME_MAGIC: &[u8; 8] = b"PTRACT01";
const CHUNK_MAGIC: &[u8; 8] = b"PTRCHK01";
const POLICY_MAGIC: &[u8; 8] = b"PTRPOL01";
const FRAME_DOMAIN: &[u8] = b"ptr.split.frame.v1\0";
const CHUNK_DOMAIN: &[u8] = b"ptr.split.chunk.v1\0";
const ENVELOPE_DOMAIN: &[u8] = b"ptr.split.envelope.v1\0";
const POLICY_DOMAIN: &[u8] = b"ptr.split.policy-envelope.v1\0";

/// The dtype of the boundary tensor. F32 only: the reference path is F32.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BoundaryDtype {
    F32,
}

impl BoundaryDtype {
    /// Bytes per element.
    pub const fn size(self) -> u64 {
        match self {
            Self::F32 => 4,
        }
    }

    const fn code(self) -> u8 {
        match self {
            Self::F32 => 1,
        }
    }

    fn from_code(code: u8) -> Result<Self, SchemaError> {
        match code {
            1 => Ok(Self::F32),
            other => Err(SchemaError::UnknownDtype(other)),
        }
    }
}

/// One generation-bearing dependency of the upstream layer range: identity, generation,
/// digest and, for an expert, the fence epoch it was computed under (zero for the others).
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ChainElement {
    pub identity: String,
    pub generation: u64,
    pub digest: [u8; 32],
    pub fence_epoch: u64,
}

/// What the activations of a frame depend on upstream: the stage pod that produced them
/// (identity, generation, manifest digest), the base model, and every adapter and expert
/// of the upstream layer range. Downstream admission revalidates every element.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct UpstreamChain {
    pub stage: ChainElement,
    pub base_model: ChainElement,
    pub adapters: Vec<ChainElement>,
    pub experts: Vec<ChainElement>,
}

/// One frame of activations crossing between the stages of a pipeline split.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ActivationFrame {
    pub tokens: u32,
    pub width: u32,
    pub dtype: BoundaryDtype,
    /// The sequence position of each row.
    pub positions: Vec<u32>,
    pub upstream_chain: UpstreamChain,
    /// The placement epoch the producing stage held when it produced the frame.
    pub placement_epoch: u64,
    /// `tokens x width` elements of `dtype`, row-major, little-endian.
    pub tensor: Vec<u8>,
}

/// The header every chunk of a frame repeats.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ChunkHeader {
    pub frame_id: u64,
    pub parent_operation_id: u64,
    pub chunk_index: u32,
    pub chunk_total: u32,
    /// Absolute time (milliseconds) after which the frame is dead, signed with the chunk.
    pub signed_absolute_deadline_ms: u64,
    /// Digest of the complete frame bytes, the same in every chunk of the frame.
    pub frame_digest: [u8; 32],
    /// Digest binding this chunk's envelope and bytes ([`chunk_digest`]).
    pub chunk_digest: [u8; 32],
    pub upstream_chain: UpstreamChain,
}

/// One chunk of a frame as it travels: header, signature over the envelope, bytes.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ActivationChunk {
    pub header: ChunkHeader,
    /// Ed25519 signature of the upstream runtime over [`envelope_signing_bytes`].
    pub signature: [u8; 64],
    pub bytes: Vec<u8>,
}

/// The policy payload of a split operation: the split it was composed for. It is signed by
/// the upstream runtime like a chunk ([`policy_signing_bytes`]) and is dead after its
/// signed deadline; an unsigned or unverified policy is never delivered.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SplitPolicy {
    /// The parent operation the policy binds (a frame of another operation is not bound).
    pub operation_id: u64,
    pub split_layer: u32,
    pub width: u32,
    pub dtype: BoundaryDtype,
    /// Absolute time (milliseconds) after which the policy is dead, signed with it.
    pub signed_absolute_deadline_ms: u64,
    /// Ed25519 signature of the upstream runtime over [`policy_signing_bytes`].
    pub signature: [u8; 64],
}

/// A chunk whose bytes still borrow the payload: the header and signature are decoded, the
/// chunk bytes are not copied. The receiver verifies the signature and the chunk digest on
/// the view before it takes a copy.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ChunkView<'a> {
    pub header: ChunkHeader,
    pub signature: [u8; 64],
    pub bytes: &'a [u8],
}

impl ChunkView<'_> {
    /// An owned chunk (copies the bytes).
    pub fn to_chunk(&self) -> ActivationChunk {
        ActivationChunk {
            header: self.header.clone(),
            signature: self.signature,
            bytes: self.bytes.to_vec(),
        }
    }
}

/// A payload of the allowlist, decoded without copying chunk bytes.
// A short-lived value matched at once by the one receiver; boxing it would only add an allocation.
#[allow(clippy::large_enum_variant)]
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum AllowlistedView<'a> {
    Chunk(ChunkView<'a>),
    Policy(SplitPolicy),
}

/// A payload of the allowlist, decoded.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum AllowlistedPayload {
    Chunk(Box<ActivationChunk>),
    Policy(SplitPolicy),
}

/// Why bytes are no instance of an allowlisted schema.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SchemaError {
    /// The `type_id` is not on the allowlist, whatever the bytes are.
    NotAllowlisted(String),
    BadMagic,
    Truncated,
    TrailingBytes,
    UnknownDtype(u8),
    /// A bound of the schema is exceeded.
    OverBound(&'static str),
    /// `tokens x width x dtype` is not the tensor's length.
    ShapeMismatch {
        expected: u64,
        found: u64,
    },
    /// The number of positions is not the number of rows.
    PositionsLength {
        tokens: u32,
        positions: u32,
    },
    /// `chunk_index >= chunk_total`, or a zero total.
    ChunkIndex {
        index: u32,
        total: u32,
    },
    /// An identity that is empty or not UTF-8.
    BadIdentity,
    /// A version 2 lineage input of a kind code that is not Semantic, Memory or Derived.
    UnknownLineageKind(u8),
    /// A version 2 frame of a reuse mode code that is not Exact or Approximate.
    UnknownMode(u8),
    /// A version 2 frame of a taint code that is not Exact or Approximate.
    UnknownTaint(u8),
    /// A version 2 frame whose mode and taint disagree (an Approximate mode without an
    /// Approximate taint, or the reverse).
    ModeTaintMismatch,
    /// A version 2 lineage whose typed inputs are not strictly increasing by (kind, identity,
    /// generation): the bytes of a lineage are canonical or they are refused.
    NonCanonicalLineage,
    /// A version 2 lineage that names one (kind, identity, generation) with two different
    /// digests: the input cannot be both, so the lineage is refused rather than resolved in
    /// favour of either.
    ConflictingLineageDigests,
}

impl fmt::Display for SchemaError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for SchemaError {}

/// Admit a `TypedPayload` of a split operation: the `type_id` must be on the allowlist and
/// the bytes must decode as exactly that schema. Nothing else is admitted.
pub fn admit(payload: &TypedPayload) -> Result<AllowlistedPayload, SchemaError> {
    admit_typed(&payload.type_id, &payload.bytes)
}

/// [`admit`] over the type id and the bytes.
pub fn admit_typed(type_id: &TypeId, bytes: &[u8]) -> Result<AllowlistedPayload, SchemaError> {
    Ok(match admit_view(type_id, bytes)? {
        AllowlistedView::Chunk(view) => AllowlistedPayload::Chunk(Box::new(view.to_chunk())),
        AllowlistedView::Policy(policy) => AllowlistedPayload::Policy(policy),
    })
}

/// The allowlist itself: the `type_id` must be on it and the bytes must decode as exactly
/// that schema. A chunk's bytes are borrowed, not copied, so a receiver can authenticate the
/// sender before it allocates for the content.
pub fn admit_view<'a>(
    type_id: &TypeId,
    bytes: &'a [u8],
) -> Result<AllowlistedView<'a>, SchemaError> {
    match type_id.0.as_str() {
        ACTIVATION_CHUNK_TYPE => decode_chunk_view(bytes).map(AllowlistedView::Chunk),
        SPLIT_POLICY_TYPE => decode_policy(bytes).map(AllowlistedView::Policy),
        other => Err(SchemaError::NotAllowlisted(other.to_owned())),
    }
}

/// Digest of the complete canonical frame bytes.
pub fn frame_digest(frame_bytes: &[u8]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(FRAME_DOMAIN);
    hasher.update((frame_bytes.len() as u64).to_le_bytes());
    hasher.update(frame_bytes);
    hasher.finalize().into()
}

/// The digest a chunk carries: it binds the frame id, the chunk's place, the whole-frame
/// digest, the parent operation, the signed deadline, the upstream chain and the chunk
/// bytes, so a chunk cannot be moved to another frame, operation, deadline or chain.
pub fn chunk_digest(header: &ChunkHeader, bytes: &[u8]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(CHUNK_DOMAIN);
    hasher.update(header.frame_id.to_le_bytes());
    hasher.update(header.chunk_index.to_le_bytes());
    hasher.update(header.chunk_total.to_le_bytes());
    hasher.update(header.frame_digest);
    hasher.update(header.parent_operation_id.to_le_bytes());
    hasher.update(header.signed_absolute_deadline_ms.to_le_bytes());
    let mut chain = Vec::new();
    write_chain(&mut chain, &header.upstream_chain);
    hasher.update((chain.len() as u64).to_le_bytes());
    hasher.update(&chain);
    hasher.update((bytes.len() as u64).to_le_bytes());
    hasher.update(bytes);
    hasher.finalize().into()
}

/// The bytes the upstream runtime signs: the complete envelope (frame id, parent
/// operation, chunk index and count, deadline, whole-frame digest, chunk digest and the
/// chain), domain-separated.
pub fn envelope_signing_bytes(header: &ChunkHeader) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(ENVELOPE_DOMAIN);
    out.extend_from_slice(&header.frame_id.to_le_bytes());
    out.extend_from_slice(&header.parent_operation_id.to_le_bytes());
    out.extend_from_slice(&header.chunk_index.to_le_bytes());
    out.extend_from_slice(&header.chunk_total.to_le_bytes());
    out.extend_from_slice(&header.signed_absolute_deadline_ms.to_le_bytes());
    out.extend_from_slice(&header.frame_digest);
    out.extend_from_slice(&header.chunk_digest);
    write_chain(&mut out, &header.upstream_chain);
    out
}

impl ActivationFrame {
    /// The canonical bytes of the frame: what is split into chunks and what the whole-frame
    /// digest is taken over.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.tensor.len() + 256);
        out.extend_from_slice(FRAME_MAGIC);
        out.extend_from_slice(&self.tokens.to_le_bytes());
        out.extend_from_slice(&self.width.to_le_bytes());
        out.push(self.dtype.code());
        out.extend_from_slice(&(self.positions.len() as u32).to_le_bytes());
        for position in &self.positions {
            out.extend_from_slice(&position.to_le_bytes());
        }
        write_chain(&mut out, &self.upstream_chain);
        out.extend_from_slice(&self.placement_epoch.to_le_bytes());
        out.extend_from_slice(&(self.tensor.len() as u64).to_le_bytes());
        out.extend_from_slice(&self.tensor);
        out
    }

    /// Decode canonical frame bytes, validating the tensor shape (`tokens x width x dtype`)
    /// and the positions length for every frame, whatever else about it is valid.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, SchemaError> {
        let mut reader = Reader::new(bytes);
        if reader.take(8)? != FRAME_MAGIC {
            return Err(SchemaError::BadMagic);
        }
        let tokens = reader.u32()?;
        let width = reader.u32()?;
        let dtype = BoundaryDtype::from_code(reader.u8()?)?;
        if tokens > MAX_FRAME_TOKENS {
            return Err(SchemaError::OverBound("tokens"));
        }
        if width > MAX_FRAME_WIDTH {
            return Err(SchemaError::OverBound("width"));
        }
        let position_count = reader.u32()?;
        if position_count > MAX_FRAME_TOKENS {
            return Err(SchemaError::OverBound("positions"));
        }
        let mut positions = Vec::with_capacity(position_count as usize);
        for _ in 0..position_count {
            positions.push(reader.u32()?);
        }
        let upstream_chain = read_chain(&mut reader)?;
        let placement_epoch = reader.u64()?;
        let tensor_len = reader.u64()?;
        if tensor_len > MAX_FRAME_BYTES {
            return Err(SchemaError::OverBound("tensor"));
        }
        let tensor = reader.take(tensor_len as usize)?.to_vec();
        if !reader.is_empty() {
            return Err(SchemaError::TrailingBytes);
        }
        let expected = u64::from(tokens) * u64::from(width) * dtype.size();
        if expected != tensor_len {
            return Err(SchemaError::ShapeMismatch {
                expected,
                found: tensor_len,
            });
        }
        if position_count != tokens {
            return Err(SchemaError::PositionsLength {
                tokens,
                positions: position_count,
            });
        }
        Ok(Self {
            tokens,
            width,
            dtype,
            positions,
            upstream_chain,
            placement_epoch,
            tensor,
        })
    }

    /// The tensor as `f32` values (F32 is the only dtype).
    pub fn tensor_f32(&self) -> Vec<f32> {
        self.tensor
            .as_chunks::<4>()
            .0
            .iter()
            .map(|word| f32::from_le_bytes(*word))
            .collect()
    }
}

/// Encode a chunk as the bytes of a `TypedPayload` of [`ACTIVATION_CHUNK_TYPE`].
pub fn encode_chunk(chunk: &ActivationChunk) -> Vec<u8> {
    let mut out = Vec::with_capacity(chunk.bytes.len() + 512);
    out.extend_from_slice(CHUNK_MAGIC);
    out.extend_from_slice(&chunk.header.frame_id.to_le_bytes());
    out.extend_from_slice(&chunk.header.parent_operation_id.to_le_bytes());
    out.extend_from_slice(&chunk.header.chunk_index.to_le_bytes());
    out.extend_from_slice(&chunk.header.chunk_total.to_le_bytes());
    out.extend_from_slice(&chunk.header.signed_absolute_deadline_ms.to_le_bytes());
    out.extend_from_slice(&chunk.header.frame_digest);
    out.extend_from_slice(&chunk.header.chunk_digest);
    write_chain(&mut out, &chunk.header.upstream_chain);
    out.extend_from_slice(&chunk.signature);
    out.extend_from_slice(&(chunk.bytes.len() as u64).to_le_bytes());
    out.extend_from_slice(&chunk.bytes);
    out
}

/// Wrap a chunk as a `TypedPayload`.
pub fn chunk_payload(chunk: &ActivationChunk) -> TypedPayload {
    TypedPayload {
        type_id: TypeId::from(ACTIVATION_CHUNK_TYPE),
        bytes: encode_chunk(chunk),
    }
}

/// Decode the bytes of an [`ACTIVATION_CHUNK_TYPE`] payload.
pub fn decode_chunk(bytes: &[u8]) -> Result<ActivationChunk, SchemaError> {
    decode_chunk_view(bytes).map(|view| view.to_chunk())
}

/// [`decode_chunk`] without copying the chunk bytes.
pub fn decode_chunk_view(bytes: &[u8]) -> Result<ChunkView<'_>, SchemaError> {
    let mut reader = Reader::new(bytes);
    if reader.take(8)? != CHUNK_MAGIC {
        return Err(SchemaError::BadMagic);
    }
    let frame_id = reader.u64()?;
    let parent_operation_id = reader.u64()?;
    let chunk_index = reader.u32()?;
    let chunk_total = reader.u32()?;
    if chunk_total == 0 || chunk_index >= chunk_total {
        return Err(SchemaError::ChunkIndex {
            index: chunk_index,
            total: chunk_total,
        });
    }
    if chunk_total > MAX_CHUNKS_PER_FRAME {
        return Err(SchemaError::OverBound("chunk_total"));
    }
    let signed_absolute_deadline_ms = reader.u64()?;
    let frame_digest = reader.array32()?;
    let chunk_digest = reader.array32()?;
    let upstream_chain = read_chain(&mut reader)?;
    let mut signature = [0u8; 64];
    signature.copy_from_slice(reader.take(64)?);
    let length = reader.u64()?;
    if length > MAX_FRAME_BYTES {
        return Err(SchemaError::OverBound("chunk bytes"));
    }
    let chunk_bytes = reader.take(length as usize)?;
    if !reader.is_empty() {
        return Err(SchemaError::TrailingBytes);
    }
    Ok(ChunkView {
        header: ChunkHeader {
            frame_id,
            parent_operation_id,
            chunk_index,
            chunk_total,
            signed_absolute_deadline_ms,
            frame_digest,
            chunk_digest,
            upstream_chain,
        },
        signature,
        bytes: chunk_bytes,
    })
}

/// Encode a policy as the bytes of a `TypedPayload` of [`SPLIT_POLICY_TYPE`].
pub fn encode_policy(policy: &SplitPolicy) -> Vec<u8> {
    let mut out = Vec::with_capacity(97);
    out.extend_from_slice(POLICY_MAGIC);
    out.extend_from_slice(&policy.operation_id.to_le_bytes());
    out.extend_from_slice(&policy.split_layer.to_le_bytes());
    out.extend_from_slice(&policy.width.to_le_bytes());
    out.push(policy.dtype.code());
    out.extend_from_slice(&policy.signed_absolute_deadline_ms.to_le_bytes());
    out.extend_from_slice(&policy.signature);
    out
}

/// The bytes the upstream runtime signs for a policy: every field but the signature,
/// domain-separated from the chunk envelope.
pub fn policy_signing_bytes(policy: &SplitPolicy) -> Vec<u8> {
    let mut out = Vec::with_capacity(64);
    out.extend_from_slice(POLICY_DOMAIN);
    out.extend_from_slice(&policy.operation_id.to_le_bytes());
    out.extend_from_slice(&policy.split_layer.to_le_bytes());
    out.extend_from_slice(&policy.width.to_le_bytes());
    out.push(policy.dtype.code());
    out.extend_from_slice(&policy.signed_absolute_deadline_ms.to_le_bytes());
    out
}

/// Decode the bytes of a [`SPLIT_POLICY_TYPE`] payload.
pub fn decode_policy(bytes: &[u8]) -> Result<SplitPolicy, SchemaError> {
    let mut reader = Reader::new(bytes);
    if reader.take(8)? != POLICY_MAGIC {
        return Err(SchemaError::BadMagic);
    }
    let policy = SplitPolicy {
        operation_id: reader.u64()?,
        split_layer: reader.u32()?,
        width: reader.u32()?,
        dtype: BoundaryDtype::from_code(reader.u8()?)?,
        signed_absolute_deadline_ms: reader.u64()?,
        signature: {
            let mut signature = [0u8; 64];
            signature.copy_from_slice(reader.take(64)?);
            signature
        },
    };
    if !reader.is_empty() {
        return Err(SchemaError::TrailingBytes);
    }
    if policy.width > MAX_FRAME_WIDTH {
        return Err(SchemaError::OverBound("width"));
    }
    Ok(policy)
}

fn write_element(out: &mut Vec<u8>, element: &ChainElement) {
    out.extend_from_slice(&(element.identity.len() as u32).to_le_bytes());
    out.extend_from_slice(element.identity.as_bytes());
    out.extend_from_slice(&element.generation.to_le_bytes());
    out.extend_from_slice(&element.digest);
    out.extend_from_slice(&element.fence_epoch.to_le_bytes());
}

/// The canonical bytes of a chain, for digests that must cover it.
pub fn chain_bytes(chain: &UpstreamChain) -> Vec<u8> {
    let mut out = Vec::new();
    write_chain(&mut out, chain);
    out
}

fn write_chain(out: &mut Vec<u8>, chain: &UpstreamChain) {
    write_element(out, &chain.stage);
    write_element(out, &chain.base_model);
    for list in [&chain.adapters, &chain.experts] {
        out.extend_from_slice(&(list.len() as u32).to_le_bytes());
        for element in list {
            write_element(out, element);
        }
    }
}

fn read_element(reader: &mut Reader<'_>) -> Result<ChainElement, SchemaError> {
    let length = reader.u32()? as usize;
    if length == 0 {
        return Err(SchemaError::BadIdentity);
    }
    if length > MAX_IDENTITY_BYTES {
        return Err(SchemaError::OverBound("identity"));
    }
    let identity = std::str::from_utf8(reader.take(length)?)
        .map_err(|_| SchemaError::BadIdentity)?
        .to_owned();
    Ok(ChainElement {
        identity,
        generation: reader.u64()?,
        digest: reader.array32()?,
        fence_epoch: reader.u64()?,
    })
}

fn read_chain(reader: &mut Reader<'_>) -> Result<UpstreamChain, SchemaError> {
    let stage = read_element(reader)?;
    let base_model = read_element(reader)?;
    let mut lists: [Vec<ChainElement>; 2] = [Vec::new(), Vec::new()];
    for list in &mut lists {
        let count = reader.u32()? as usize;
        if count > MAX_CHAIN_ELEMENTS {
            return Err(SchemaError::OverBound("chain elements"));
        }
        for _ in 0..count {
            list.push(read_element(reader)?);
        }
    }
    let [adapters, experts] = lists;
    Ok(UpstreamChain {
        stage,
        base_model,
        adapters,
        experts,
    })
}

struct Reader<'a> {
    rest: &'a [u8],
}

impl<'a> Reader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { rest: bytes }
    }

    fn is_empty(&self) -> bool {
        self.rest.is_empty()
    }

    fn take(&mut self, count: usize) -> Result<&'a [u8], SchemaError> {
        if self.rest.len() < count {
            return Err(SchemaError::Truncated);
        }
        let (head, tail) = self.rest.split_at(count);
        self.rest = tail;
        Ok(head)
    }

    fn u8(&mut self) -> Result<u8, SchemaError> {
        Ok(self.take(1)?[0])
    }

    fn u32(&mut self) -> Result<u32, SchemaError> {
        let bytes = self.take(4)?;
        Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    fn u64(&mut self) -> Result<u64, SchemaError> {
        let bytes = self.take(8)?;
        let mut array = [0u8; 8];
        array.copy_from_slice(bytes);
        Ok(u64::from_le_bytes(array))
    }

    fn array32(&mut self) -> Result<[u8; 32], SchemaError> {
        let mut array = [0u8; 32];
        array.copy_from_slice(self.take(32)?);
        Ok(array)
    }
}

// Version 2 of the frame (input lineage, mode, taint) and the dependency-kind checker live in
// submodules; version 1 above is unchanged.
mod kinds;
mod lineage;

pub use kinds::*;
pub use lineage::*;

#[cfg(test)]
mod tests {
    use super::*;

    fn element(name: &str, generation: u64) -> ChainElement {
        ChainElement {
            identity: name.to_owned(),
            generation,
            digest: [generation as u8; 32],
            fence_epoch: 0,
        }
    }

    fn chain() -> UpstreamChain {
        UpstreamChain {
            stage: element("stage-a", 1),
            base_model: element("model", 2),
            adapters: vec![element("adapter", 3)],
            experts: vec![],
        }
    }

    fn frame(tokens: u32, width: u32) -> ActivationFrame {
        ActivationFrame {
            tokens,
            width,
            dtype: BoundaryDtype::F32,
            positions: (0..tokens).collect(),
            upstream_chain: chain(),
            placement_epoch: 7,
            tensor: (0..tokens * width * 4).map(|byte| byte as u8).collect(),
        }
    }

    fn chunk_of(frame: &ActivationFrame) -> ActivationChunk {
        let bytes = frame.to_bytes();
        let mut header = ChunkHeader {
            frame_id: 1,
            parent_operation_id: 2,
            chunk_index: 0,
            chunk_total: 1,
            signed_absolute_deadline_ms: 5_000,
            frame_digest: frame_digest(&bytes),
            chunk_digest: [0; 32],
            upstream_chain: frame.upstream_chain.clone(),
        };
        header.chunk_digest = chunk_digest(&header, &bytes);
        ActivationChunk {
            header,
            signature: [3; 64],
            bytes,
        }
    }

    #[test]
    fn a_frame_round_trips_and_its_tensor_reads_back() {
        let original = frame(3, 4);
        let decoded = ActivationFrame::from_bytes(&original.to_bytes()).unwrap();
        assert_eq!(decoded, original);
        assert_eq!(decoded.tensor_f32().len(), 12);
    }

    #[test]
    fn a_wrong_shape_a_wrong_positions_length_and_trailing_bytes_are_refused() {
        let mut short = frame(3, 4);
        short.tensor.pop();
        assert_eq!(
            ActivationFrame::from_bytes(&short.to_bytes()),
            Err(SchemaError::ShapeMismatch {
                expected: 48,
                found: 47
            })
        );
        let mut positions = frame(3, 4);
        positions.positions.pop();
        assert_eq!(
            ActivationFrame::from_bytes(&positions.to_bytes()),
            Err(SchemaError::PositionsLength {
                tokens: 3,
                positions: 2
            })
        );
        let mut bytes = frame(3, 4).to_bytes();
        bytes.push(0);
        assert_eq!(
            ActivationFrame::from_bytes(&bytes),
            Err(SchemaError::TrailingBytes)
        );
        assert_eq!(
            ActivationFrame::from_bytes(&bytes[..bytes.len() - 20]),
            Err(SchemaError::Truncated)
        );
        let mut magic = frame(1, 1).to_bytes();
        magic[0] ^= 1;
        assert_eq!(
            ActivationFrame::from_bytes(&magic),
            Err(SchemaError::BadMagic)
        );
    }

    #[test]
    fn a_frame_over_a_bound_is_refused_before_its_tensor_is_read() {
        let mut huge = frame(1, 1);
        huge.tokens = MAX_FRAME_TOKENS + 1;
        assert_eq!(
            ActivationFrame::from_bytes(&huge.to_bytes()),
            Err(SchemaError::OverBound("tokens"))
        );
        let mut wide = frame(1, 1);
        wide.width = MAX_FRAME_WIDTH + 1;
        assert_eq!(
            ActivationFrame::from_bytes(&wide.to_bytes()),
            Err(SchemaError::OverBound("width"))
        );
    }

    #[test]
    fn a_chunk_round_trips_through_the_allowlist() {
        let chunk = chunk_of(&frame(2, 2));
        let admitted = admit(&chunk_payload(&chunk)).unwrap();
        assert_eq!(admitted, AllowlistedPayload::Chunk(Box::new(chunk)));
    }

    #[test]
    fn the_policy_round_trips_through_the_allowlist() {
        let policy = SplitPolicy {
            operation_id: 9,
            split_layer: 2,
            width: 64,
            dtype: BoundaryDtype::F32,
            signed_absolute_deadline_ms: 5_000,
            signature: [3; 64],
        };
        let payload = TypedPayload {
            type_id: TypeId::from(SPLIT_POLICY_TYPE),
            bytes: encode_policy(&policy),
        };
        assert_eq!(admit(&payload), Ok(AllowlistedPayload::Policy(policy)));
    }

    #[test]
    fn a_type_id_off_the_allowlist_is_refused_whatever_the_bytes() {
        let chunk = chunk_of(&frame(1, 1));
        for name in ["ptr.kv.page", "ptr.split.activation-chunk.v2", "", "Input"] {
            let payload = TypedPayload {
                type_id: TypeId::from(name),
                bytes: encode_chunk(&chunk),
            };
            assert_eq!(
                admit(&payload),
                Err(SchemaError::NotAllowlisted(name.to_owned()))
            );
        }
    }

    #[test]
    fn allowlisted_bytes_of_another_schema_are_refused() {
        // KV-page-looking bytes under the chunk type: not a chunk.
        let payload = TypedPayload {
            type_id: TypeId::from(ACTIVATION_CHUNK_TYPE),
            bytes: vec![0x5a; 4096],
        };
        assert_eq!(admit(&payload), Err(SchemaError::BadMagic));
        // A frame under the chunk type is not a chunk either.
        let payload = TypedPayload {
            type_id: TypeId::from(ACTIVATION_CHUNK_TYPE),
            bytes: frame(1, 1).to_bytes(),
        };
        assert_eq!(admit(&payload), Err(SchemaError::BadMagic));
        // Trailing bytes after a valid chunk.
        let mut bytes = encode_chunk(&chunk_of(&frame(1, 1)));
        bytes.push(1);
        assert_eq!(decode_chunk(&bytes), Err(SchemaError::TrailingBytes));
    }

    #[test]
    fn a_chunk_index_out_of_its_total_and_a_zero_total_are_refused() {
        let mut chunk = chunk_of(&frame(1, 1));
        chunk.header.chunk_index = 1;
        assert_eq!(
            decode_chunk(&encode_chunk(&chunk)),
            Err(SchemaError::ChunkIndex { index: 1, total: 1 })
        );
        chunk.header.chunk_index = 0;
        chunk.header.chunk_total = 0;
        assert_eq!(
            decode_chunk(&encode_chunk(&chunk)),
            Err(SchemaError::ChunkIndex { index: 0, total: 0 })
        );
        chunk.header.chunk_total = MAX_CHUNKS_PER_FRAME + 1;
        assert_eq!(
            decode_chunk(&encode_chunk(&chunk)),
            Err(SchemaError::OverBound("chunk_total"))
        );
    }

    #[test]
    fn the_chunk_digest_binds_every_field_it_names() {
        let chunk = chunk_of(&frame(2, 2));
        let base = chunk_digest(&chunk.header, &chunk.bytes);
        assert_eq!(base, chunk.header.chunk_digest);
        let mut changed = chunk.header.clone();
        changed.frame_id += 1;
        assert_ne!(chunk_digest(&changed, &chunk.bytes), base);
        let mut changed = chunk.header.clone();
        changed.parent_operation_id += 1;
        assert_ne!(chunk_digest(&changed, &chunk.bytes), base);
        let mut changed = chunk.header.clone();
        changed.signed_absolute_deadline_ms += 1;
        assert_ne!(chunk_digest(&changed, &chunk.bytes), base);
        let mut changed = chunk.header.clone();
        changed.frame_digest[0] ^= 1;
        assert_ne!(chunk_digest(&changed, &chunk.bytes), base);
        let mut changed = chunk.header.clone();
        changed.chunk_index += 1;
        assert_ne!(chunk_digest(&changed, &chunk.bytes), base);
        let mut changed = chunk.header.clone();
        changed.upstream_chain.stage.generation += 1;
        assert_ne!(chunk_digest(&changed, &chunk.bytes), base);
        let mut bytes = chunk.bytes.clone();
        bytes[0] ^= 1;
        assert_ne!(chunk_digest(&chunk.header, &bytes), base);
    }

    #[test]
    fn the_signed_envelope_covers_the_digests_and_the_deadline() {
        let chunk = chunk_of(&frame(2, 2));
        let base = envelope_signing_bytes(&chunk.header);
        let mut changed = chunk.header.clone();
        changed.signed_absolute_deadline_ms += 1;
        assert_ne!(envelope_signing_bytes(&changed), base);
        let mut changed = chunk.header.clone();
        changed.chunk_digest[1] ^= 1;
        assert_ne!(envelope_signing_bytes(&changed), base);
        let mut changed = chunk.header.clone();
        changed.chunk_total += 1;
        assert_ne!(envelope_signing_bytes(&changed), base);
    }

    #[test]
    fn an_empty_identity_and_an_oversize_chain_are_refused() {
        let mut bad = chunk_of(&frame(1, 1));
        bad.header.upstream_chain.stage.identity.clear();
        assert_eq!(
            decode_chunk(&encode_chunk(&bad)),
            Err(SchemaError::BadIdentity)
        );
        let mut many = chunk_of(&frame(1, 1));
        many.header.upstream_chain.experts = vec![element("e", 1); MAX_CHAIN_ELEMENTS + 1];
        assert_eq!(
            decode_chunk(&encode_chunk(&many)),
            Err(SchemaError::OverBound("chain elements"))
        );
    }
}
