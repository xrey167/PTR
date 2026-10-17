//! Version 2 of the activation frame: the input lineage, the reuse mode and the taint.
//!
//! Version 1 (`PTRACT01`, `ptr.split.frame.v1`) names the upstream model-side chain and nothing
//! about the *inputs* that went into the activations, and it is Exact-only. Version 2 is a
//! separate schema with its own magics, its own digest domains and its own type id
//! ([`ACTIVATION_CHUNK_V2_TYPE`]); version 1 is untouched and stays decodable and admissible
//! exactly as before, and neither version decodes the other's bytes.
//!
//! A v2 frame adds three signed fields (doc 36, section D, "Dependency invalidation"):
//!
//! * `input_lineage`: the identity, generation and digest of every semantic, memory and
//!   derived input that contributed to the activations, and the **ordered** content digests
//!   of the Raw segments that influenced them. A Raw segment is not a dependency (it has no
//!   identity or generation to revalidate); its (kind Raw, content digest) contribution is
//!   what a downstream cache key must include, so that an earlier Raw segment removed or
//!   changed never reuses a cached entry.
//! * `mode`: [`FrameMode`], the reuse mode the upstream stage ran under.
//! * `taint`: [`FrameTaint`], the provenance of the activations with respect to KV reuse. An
//!   Approximate frame must carry an Approximate taint and an Exact one an Exact taint, so a
//!   frame cannot launder its mode.
//!
//! Every chunk repeats the three fields and its digest and signed envelope cover them.

use super::{
    read_chain, write_chain, ActivationChunk, AllowlistedView, BoundaryDtype, ChunkView, Reader,
    SchemaError, SplitPolicy, UpstreamChain, ACTIVATION_CHUNK_TYPE, MAX_CHAIN_ELEMENTS,
    MAX_CHUNKS_PER_FRAME, MAX_FRAME_BYTES, MAX_FRAME_TOKENS, MAX_FRAME_WIDTH, MAX_IDENTITY_BYTES,
    SPLIT_POLICY_TYPE,
};
use crate::TypedPayload;
use ptr_types::TypeId;
use sha2::{Digest as _, Sha256};

/// Type id of one chunk of a version 2 activation frame.
pub const ACTIVATION_CHUNK_V2_TYPE: &str = "ptr.split.activation-chunk.v2";
/// Every type id an endpoint that speaks both versions admits ([`admit_v2`]). The
/// version 1 allowlist ([`super::ALLOWLISTED_TYPES`]) is unchanged.
pub const ALLOWLISTED_TYPES_V2: [&str; 3] = [
    ACTIVATION_CHUNK_TYPE,
    ACTIVATION_CHUNK_V2_TYPE,
    SPLIT_POLICY_TYPE,
];

/// Bound on the typed (semantic, memory, derived) inputs of one frame's lineage.
pub const MAX_LINEAGE_INPUTS: usize = 64;
/// Bound on the Raw segments of one frame's ancestry.
pub const MAX_RAW_ANCESTRY: usize = 256;

const FRAME_MAGIC_V2: &[u8; 8] = b"PTRACT02";
const CHUNK_MAGIC_V2: &[u8; 8] = b"PTRCHK02";
const FRAME_DOMAIN_V2: &[u8] = b"ptr.split.frame.v2\0";
const CHUNK_DOMAIN_V2: &[u8] = b"ptr.split.chunk.v2\0";
const ENVELOPE_DOMAIN_V2: &[u8] = b"ptr.split.envelope.v2\0";

/// The generation-bearing segment kinds an input can have. A Raw segment is deliberately not
/// one: it has no identity or generation and is carried as an ancestry digest.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum LineageKind {
    Semantic,
    Memory,
    Derived,
}

impl LineageKind {
    const fn code(self) -> u8 {
        match self {
            Self::Semantic => 1,
            Self::Memory => 2,
            Self::Derived => 3,
        }
    }

    /// The kind's stable name, for namespacing lifecycle targets by kind.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Semantic => "semantic",
            Self::Memory => "memory",
            Self::Derived => "derived",
        }
    }

    fn from_code(code: u8) -> Result<Self, SchemaError> {
        match code {
            1 => Ok(Self::Semantic),
            2 => Ok(Self::Memory),
            3 => Ok(Self::Derived),
            other => Err(SchemaError::UnknownLineageKind(other)),
        }
    }
}

/// One generation-bearing input that contributed to the activations.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LineageInput {
    pub kind: LineageKind,
    pub identity: String,
    pub generation: u64,
    /// The content digest committed for that generation.
    pub digest: [u8; 32],
}

impl LineageInput {
    fn sort_key(&self) -> (u8, &[u8], u64) {
        (self.kind.code(), self.identity.as_bytes(), self.generation)
    }

    /// The sort key and the digest: the order [`InputLineage::canonical`] sorts by, so that
    /// fully identical inputs are adjacent and two digests of one key stay visible.
    fn full_key(&self) -> (u8, &[u8], u64, &[u8; 32]) {
        (
            self.kind.code(),
            self.identity.as_bytes(),
            self.generation,
            &self.digest,
        )
    }
}

/// What the activations were computed from: the typed inputs and the ordered Raw ancestry.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct InputLineage {
    /// Strictly increasing by (kind, identity, generation): the canonical order. A frame
    /// whose inputs are not in it does not decode ([`SchemaError::NonCanonicalLineage`]).
    pub inputs: Vec<LineageInput>,
    /// The content digest of every Raw segment that influenced the activations, **in prompt
    /// order**. Order is part of the lineage: the same segments in another order are another
    /// ancestry.
    pub raw_ancestry: Vec<[u8; 32]>,
}

impl InputLineage {
    /// A lineage in canonical order: `inputs` sorted, with fully identical inputs (same kind,
    /// identity, generation **and** digest) collapsed, `raw_ancestry` kept in the order given.
    /// Two inputs with one (kind, identity, generation) and different digests are both kept:
    /// the lineage then fails [`ActivationFrameV2::validate`] with
    /// [`SchemaError::ConflictingLineageDigests`] instead of silently choosing one.
    pub fn canonical(mut inputs: Vec<LineageInput>, raw_ancestry: Vec<[u8; 32]>) -> Self {
        inputs.sort_by(|a, b| a.full_key().cmp(&b.full_key()));
        inputs.dedup();
        Self {
            inputs,
            raw_ancestry,
        }
    }
}

/// The reuse mode the upstream stage ran under (doc 36, section C).
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum FrameMode {
    Exact,
    Approximate,
}

impl FrameMode {
    const fn code(self) -> u8 {
        match self {
            Self::Exact => 1,
            Self::Approximate => 2,
        }
    }

    fn from_code(code: u8) -> Result<Self, SchemaError> {
        match code {
            1 => Ok(Self::Exact),
            2 => Ok(Self::Approximate),
            other => Err(SchemaError::UnknownMode(other)),
        }
    }
}

/// The provenance of the activations with respect to KV reuse. Approximate is sticky: it is
/// carried to every output derived from the frame.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum FrameTaint {
    Exact,
    Approximate {
        /// Digest of the reuse report of the run that introduced the taint.
        report_digest: [u8; 32],
    },
}

impl FrameTaint {
    pub const fn is_approximate(self) -> bool {
        matches!(self, Self::Approximate { .. })
    }
}

/// One frame of activations crossing between the stages of a pipeline split, version 2.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ActivationFrameV2 {
    pub tokens: u32,
    pub width: u32,
    pub dtype: BoundaryDtype,
    /// The sequence position of each row.
    pub positions: Vec<u32>,
    pub upstream_chain: UpstreamChain,
    pub input_lineage: InputLineage,
    pub mode: FrameMode,
    pub taint: FrameTaint,
    /// The placement epoch the producing stage held when it produced the frame.
    pub placement_epoch: u64,
    /// `tokens x width` elements of `dtype`, row-major, little-endian.
    pub tensor: Vec<u8>,
}

/// The header every chunk of a version 2 frame repeats.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ChunkHeaderV2 {
    pub frame_id: u64,
    pub parent_operation_id: u64,
    pub chunk_index: u32,
    pub chunk_total: u32,
    pub signed_absolute_deadline_ms: u64,
    /// Digest of the complete frame bytes ([`frame_digest_v2`]), the same in every chunk.
    pub frame_digest: [u8; 32],
    /// Digest binding this chunk's envelope and bytes ([`chunk_digest_v2`]).
    pub chunk_digest: [u8; 32],
    pub upstream_chain: UpstreamChain,
    pub input_lineage: InputLineage,
    pub mode: FrameMode,
    pub taint: FrameTaint,
}

impl ChunkHeaderV2 {
    /// Whether the chain, lineage, mode and taint the header repeats are the frame's. A
    /// receiver that has assembled a frame checks this before it delivers it.
    pub fn matches_frame(&self, frame: &ActivationFrameV2) -> bool {
        self.upstream_chain == frame.upstream_chain
            && self.input_lineage == frame.input_lineage
            && self.mode == frame.mode
            && self.taint == frame.taint
    }
}

/// One chunk of a version 2 frame as it travels.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ActivationChunkV2 {
    pub header: ChunkHeaderV2,
    /// Ed25519 signature of the upstream runtime over [`envelope_signing_bytes_v2`].
    pub signature: [u8; 64],
    pub bytes: Vec<u8>,
}

/// A version 2 chunk whose bytes still borrow the payload.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ChunkViewV2<'a> {
    pub header: ChunkHeaderV2,
    pub signature: [u8; 64],
    pub bytes: &'a [u8],
}

impl ChunkViewV2<'_> {
    /// An owned chunk (copies the bytes).
    pub fn to_chunk(&self) -> ActivationChunkV2 {
        ActivationChunkV2 {
            header: self.header.clone(),
            signature: self.signature,
            bytes: self.bytes.to_vec(),
        }
    }
}

/// A payload of the version 1 + 2 allowlist, decoded without copying chunk bytes.
// A short-lived value matched at once by the one receiver; boxing it would only add an allocation.
#[allow(clippy::large_enum_variant)]
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum AllowlistedViewV2<'a> {
    ChunkV1(ChunkView<'a>),
    ChunkV2(ChunkViewV2<'a>),
    Policy(SplitPolicy),
}

/// A payload of the version 1 + 2 allowlist, decoded.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum AllowlistedPayloadV2 {
    ChunkV1(Box<ActivationChunk>),
    ChunkV2(Box<ActivationChunkV2>),
    Policy(SplitPolicy),
}

/// [`super::admit`] for an endpoint that speaks both versions: the `type_id` must be on
/// [`ALLOWLISTED_TYPES_V2`] and the bytes must decode as exactly that schema.
pub fn admit_v2(payload: &TypedPayload) -> Result<AllowlistedPayloadV2, SchemaError> {
    admit_typed_v2(&payload.type_id, &payload.bytes)
}

/// [`admit_v2`] over the type id and the bytes.
pub fn admit_typed_v2(type_id: &TypeId, bytes: &[u8]) -> Result<AllowlistedPayloadV2, SchemaError> {
    Ok(match admit_view_v2(type_id, bytes)? {
        AllowlistedViewV2::ChunkV1(view) => {
            AllowlistedPayloadV2::ChunkV1(Box::new(view.to_chunk()))
        }
        AllowlistedViewV2::ChunkV2(view) => {
            AllowlistedPayloadV2::ChunkV2(Box::new(view.to_chunk()))
        }
        AllowlistedViewV2::Policy(policy) => AllowlistedPayloadV2::Policy(policy),
    })
}

/// The both-versions allowlist itself; chunk bytes are borrowed, not copied.
pub fn admit_view_v2<'a>(
    type_id: &TypeId,
    bytes: &'a [u8],
) -> Result<AllowlistedViewV2<'a>, SchemaError> {
    if type_id.0.as_str() == ACTIVATION_CHUNK_V2_TYPE {
        return decode_chunk_view_v2(bytes).map(AllowlistedViewV2::ChunkV2);
    }
    match super::admit_view(type_id, bytes)? {
        AllowlistedView::Chunk(view) => Ok(AllowlistedViewV2::ChunkV1(view)),
        AllowlistedView::Policy(policy) => Ok(AllowlistedViewV2::Policy(policy)),
    }
}

/// Digest of the complete canonical version 2 frame bytes.
pub fn frame_digest_v2(frame_bytes: &[u8]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(FRAME_DOMAIN_V2);
    hasher.update((frame_bytes.len() as u64).to_le_bytes());
    hasher.update(frame_bytes);
    hasher.finalize().into()
}

/// The canonical bytes of an input lineage, for digests that must cover it.
pub fn lineage_bytes(lineage: &InputLineage) -> Vec<u8> {
    let mut out = Vec::new();
    write_lineage(&mut out, lineage);
    out
}

/// The canonical bytes of a mode and a taint, for digests that must cover them.
pub fn mode_taint_bytes(mode: FrameMode, taint: FrameTaint) -> Vec<u8> {
    let mut out = Vec::new();
    write_mode_taint(&mut out, mode, taint);
    out
}

/// The digest a version 2 chunk carries: it binds everything [`super::chunk_digest`] binds
/// and the input lineage, the mode and the taint.
pub fn chunk_digest_v2(header: &ChunkHeaderV2, bytes: &[u8]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(CHUNK_DOMAIN_V2);
    hasher.update(header.frame_id.to_le_bytes());
    hasher.update(header.chunk_index.to_le_bytes());
    hasher.update(header.chunk_total.to_le_bytes());
    hasher.update(header.frame_digest);
    hasher.update(header.parent_operation_id.to_le_bytes());
    hasher.update(header.signed_absolute_deadline_ms.to_le_bytes());
    let mut covered = Vec::new();
    write_chain(&mut covered, &header.upstream_chain);
    write_lineage(&mut covered, &header.input_lineage);
    write_mode_taint(&mut covered, header.mode, header.taint);
    hasher.update((covered.len() as u64).to_le_bytes());
    hasher.update(&covered);
    hasher.update((bytes.len() as u64).to_le_bytes());
    hasher.update(bytes);
    hasher.finalize().into()
}

/// The bytes the upstream runtime signs for a version 2 chunk: the complete envelope,
/// domain-separated from version 1.
pub fn envelope_signing_bytes_v2(header: &ChunkHeaderV2) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(ENVELOPE_DOMAIN_V2);
    out.extend_from_slice(&header.frame_id.to_le_bytes());
    out.extend_from_slice(&header.parent_operation_id.to_le_bytes());
    out.extend_from_slice(&header.chunk_index.to_le_bytes());
    out.extend_from_slice(&header.chunk_total.to_le_bytes());
    out.extend_from_slice(&header.signed_absolute_deadline_ms.to_le_bytes());
    out.extend_from_slice(&header.frame_digest);
    out.extend_from_slice(&header.chunk_digest);
    write_chain(&mut out, &header.upstream_chain);
    write_lineage(&mut out, &header.input_lineage);
    write_mode_taint(&mut out, header.mode, header.taint);
    out
}

impl ActivationFrameV2 {
    /// The canonical bytes of the frame: what is split into chunks and what the whole-frame
    /// digest is taken over.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.tensor.len() + 512);
        out.extend_from_slice(FRAME_MAGIC_V2);
        out.extend_from_slice(&self.tokens.to_le_bytes());
        out.extend_from_slice(&self.width.to_le_bytes());
        out.push(self.dtype.code());
        out.extend_from_slice(&(self.positions.len() as u32).to_le_bytes());
        for position in &self.positions {
            out.extend_from_slice(&position.to_le_bytes());
        }
        write_chain(&mut out, &self.upstream_chain);
        write_lineage(&mut out, &self.input_lineage);
        write_mode_taint(&mut out, self.mode, self.taint);
        out.extend_from_slice(&self.placement_epoch.to_le_bytes());
        out.extend_from_slice(&(self.tensor.len() as u64).to_le_bytes());
        out.extend_from_slice(&self.tensor);
        out
    }

    /// Whether the frame, built in memory, would decode: canonical lineage order, every bound
    /// (tokens, width, positions, chain lists, identities, lineage inputs and ancestry), the
    /// tensor length against the shape and the mode against the taint. It is exactly
    /// [`Self::from_bytes`] over [`Self::to_bytes`], so a frame that passes here cannot be
    /// refused by the first receiver that decodes its bytes.
    pub fn validate(&self) -> Result<(), SchemaError> {
        self.check_in_memory_bounds()?;
        Self::from_bytes(&self.to_bytes()).map(|_| ())
    }

    /// The bounds and the shape of the frame, checked on the fields in memory **before** any
    /// byte is produced, so an oversized frame is refused without serialising it (and a count
    /// that would wrap its `u32` length prefix is refused before it is written). The order and
    /// the errors are those of [`Self::from_bytes`]; the canonical lineage order and the mode
    /// against the taint are left to the decode.
    fn check_in_memory_bounds(&self) -> Result<(), SchemaError> {
        if self.tokens > MAX_FRAME_TOKENS {
            return Err(SchemaError::OverBound("tokens"));
        }
        if self.width > MAX_FRAME_WIDTH {
            return Err(SchemaError::OverBound("width"));
        }
        if self.positions.len() > MAX_FRAME_TOKENS as usize {
            return Err(SchemaError::OverBound("positions"));
        }
        check_chain_bounds(&self.upstream_chain)?;
        check_lineage_bounds(&self.input_lineage)?;
        if self.tensor.len() as u64 > MAX_FRAME_BYTES {
            return Err(SchemaError::OverBound("tensor"));
        }
        let expected = u64::from(self.tokens) * u64::from(self.width) * self.dtype.size();
        if expected != self.tensor.len() as u64 {
            return Err(SchemaError::ShapeMismatch {
                expected,
                found: self.tensor.len() as u64,
            });
        }
        if self.positions.len() != self.tokens as usize {
            return Err(SchemaError::PositionsLength {
                tokens: self.tokens,
                positions: self.positions.len() as u32,
            });
        }
        Ok(())
    }

    /// The digest of the canonical bytes ([`frame_digest_v2`]).
    pub fn digest(&self) -> [u8; 32] {
        frame_digest_v2(&self.to_bytes())
    }

    /// Decode canonical version 2 frame bytes, validating the shape, the positions, the
    /// lineage and the mode against the taint for every frame.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, SchemaError> {
        let mut reader = Reader::new(bytes);
        if reader.take(8)? != FRAME_MAGIC_V2 {
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
        let input_lineage = read_lineage(&mut reader)?;
        let (mode, taint) = read_mode_taint(&mut reader)?;
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
            input_lineage,
            mode,
            taint,
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

/// Encode a version 2 chunk as the bytes of a `TypedPayload` of [`ACTIVATION_CHUNK_V2_TYPE`].
pub fn encode_chunk_v2(chunk: &ActivationChunkV2) -> Vec<u8> {
    let mut out = Vec::with_capacity(chunk.bytes.len() + 768);
    out.extend_from_slice(CHUNK_MAGIC_V2);
    out.extend_from_slice(&chunk.header.frame_id.to_le_bytes());
    out.extend_from_slice(&chunk.header.parent_operation_id.to_le_bytes());
    out.extend_from_slice(&chunk.header.chunk_index.to_le_bytes());
    out.extend_from_slice(&chunk.header.chunk_total.to_le_bytes());
    out.extend_from_slice(&chunk.header.signed_absolute_deadline_ms.to_le_bytes());
    out.extend_from_slice(&chunk.header.frame_digest);
    out.extend_from_slice(&chunk.header.chunk_digest);
    write_chain(&mut out, &chunk.header.upstream_chain);
    write_lineage(&mut out, &chunk.header.input_lineage);
    write_mode_taint(&mut out, chunk.header.mode, chunk.header.taint);
    out.extend_from_slice(&chunk.signature);
    out.extend_from_slice(&(chunk.bytes.len() as u64).to_le_bytes());
    out.extend_from_slice(&chunk.bytes);
    out
}

/// Wrap a version 2 chunk as a `TypedPayload`.
pub fn chunk_payload_v2(chunk: &ActivationChunkV2) -> TypedPayload {
    TypedPayload {
        type_id: TypeId::from(ACTIVATION_CHUNK_V2_TYPE),
        bytes: encode_chunk_v2(chunk),
    }
}

/// Decode the bytes of an [`ACTIVATION_CHUNK_V2_TYPE`] payload.
pub fn decode_chunk_v2(bytes: &[u8]) -> Result<ActivationChunkV2, SchemaError> {
    decode_chunk_view_v2(bytes).map(|view| view.to_chunk())
}

/// [`decode_chunk_v2`] without copying the chunk bytes.
pub fn decode_chunk_view_v2(bytes: &[u8]) -> Result<ChunkViewV2<'_>, SchemaError> {
    let mut reader = Reader::new(bytes);
    if reader.take(8)? != CHUNK_MAGIC_V2 {
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
    let input_lineage = read_lineage(&mut reader)?;
    let (mode, taint) = read_mode_taint(&mut reader)?;
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
    Ok(ChunkViewV2 {
        header: ChunkHeaderV2 {
            frame_id,
            parent_operation_id,
            chunk_index,
            chunk_total,
            signed_absolute_deadline_ms,
            frame_digest,
            chunk_digest,
            upstream_chain,
            input_lineage,
            mode,
            taint,
        },
        signature,
        bytes: chunk_bytes,
    })
}

fn check_identity_bound(identity: &str) -> Result<(), SchemaError> {
    if identity.is_empty() {
        return Err(SchemaError::BadIdentity);
    }
    if identity.len() > MAX_IDENTITY_BYTES {
        return Err(SchemaError::OverBound("identity"));
    }
    Ok(())
}

fn check_chain_bounds(chain: &UpstreamChain) -> Result<(), SchemaError> {
    check_identity_bound(&chain.stage.identity)?;
    check_identity_bound(&chain.base_model.identity)?;
    for list in [&chain.adapters, &chain.experts] {
        if list.len() > MAX_CHAIN_ELEMENTS {
            return Err(SchemaError::OverBound("chain elements"));
        }
        for element in list {
            check_identity_bound(&element.identity)?;
        }
    }
    Ok(())
}

fn check_lineage_bounds(lineage: &InputLineage) -> Result<(), SchemaError> {
    if lineage.inputs.len() > MAX_LINEAGE_INPUTS {
        return Err(SchemaError::OverBound("lineage inputs"));
    }
    for input in &lineage.inputs {
        check_identity_bound(&input.identity)?;
    }
    if lineage.raw_ancestry.len() > MAX_RAW_ANCESTRY {
        return Err(SchemaError::OverBound("raw ancestry"));
    }
    Ok(())
}

fn write_lineage(out: &mut Vec<u8>, lineage: &InputLineage) {
    out.extend_from_slice(&(lineage.inputs.len() as u32).to_le_bytes());
    for input in &lineage.inputs {
        out.push(input.kind.code());
        out.extend_from_slice(&(input.identity.len() as u32).to_le_bytes());
        out.extend_from_slice(input.identity.as_bytes());
        out.extend_from_slice(&input.generation.to_le_bytes());
        out.extend_from_slice(&input.digest);
    }
    out.extend_from_slice(&(lineage.raw_ancestry.len() as u32).to_le_bytes());
    for raw in &lineage.raw_ancestry {
        out.extend_from_slice(raw);
    }
}

fn read_lineage(reader: &mut Reader<'_>) -> Result<InputLineage, SchemaError> {
    let count = reader.u32()? as usize;
    if count > MAX_LINEAGE_INPUTS {
        return Err(SchemaError::OverBound("lineage inputs"));
    }
    let mut inputs: Vec<LineageInput> = Vec::with_capacity(count);
    for _ in 0..count {
        let kind = LineageKind::from_code(reader.u8()?)?;
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
        let input = LineageInput {
            kind,
            identity,
            generation: reader.u64()?,
            digest: reader.array32()?,
        };
        if let Some(previous) = inputs.last() {
            if previous.sort_key() == input.sort_key() && previous.digest != input.digest {
                return Err(SchemaError::ConflictingLineageDigests);
            }
            if previous.sort_key() >= input.sort_key() {
                return Err(SchemaError::NonCanonicalLineage);
            }
        }
        inputs.push(input);
    }
    let raw_count = reader.u32()? as usize;
    if raw_count > MAX_RAW_ANCESTRY {
        return Err(SchemaError::OverBound("raw ancestry"));
    }
    let mut raw_ancestry = Vec::with_capacity(raw_count);
    for _ in 0..raw_count {
        raw_ancestry.push(reader.array32()?);
    }
    Ok(InputLineage {
        inputs,
        raw_ancestry,
    })
}

fn write_mode_taint(out: &mut Vec<u8>, mode: FrameMode, taint: FrameTaint) {
    out.push(mode.code());
    match taint {
        FrameTaint::Exact => out.push(1),
        FrameTaint::Approximate { report_digest } => {
            out.push(2);
            out.extend_from_slice(&report_digest);
        }
    }
}

fn read_mode_taint(reader: &mut Reader<'_>) -> Result<(FrameMode, FrameTaint), SchemaError> {
    let mode = FrameMode::from_code(reader.u8()?)?;
    let taint = match reader.u8()? {
        1 => FrameTaint::Exact,
        2 => FrameTaint::Approximate {
            report_digest: reader.array32()?,
        },
        other => return Err(SchemaError::UnknownTaint(other)),
    };
    if (mode == FrameMode::Approximate) != taint.is_approximate() {
        return Err(SchemaError::ModeTaintMismatch);
    }
    Ok((mode, taint))
}

#[cfg(test)]
mod tests {
    use super::super::{
        admit, chunk_digest, chunk_payload, decode_chunk, encode_chunk, envelope_signing_bytes,
        frame_digest, ActivationFrame, ChainElement, ChunkHeader, ALLOWLISTED_TYPES,
    };
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
            experts: vec![ChainElement {
                fence_epoch: 4,
                ..element("expert", 1)
            }],
        }
    }

    fn input(kind: LineageKind, name: &str, generation: u64) -> LineageInput {
        LineageInput {
            kind,
            identity: name.to_owned(),
            generation,
            digest: [generation as u8 ^ 0x55; 32],
        }
    }

    fn lineage() -> InputLineage {
        InputLineage::canonical(
            vec![
                input(LineageKind::Derived, "derived-1", 1),
                input(LineageKind::Semantic, "sem-1", 2),
                input(LineageKind::Memory, "mem-1", 3),
            ],
            vec![[1; 32], [2; 32], [3; 32]],
        )
    }

    fn frame(tokens: u32, width: u32) -> ActivationFrameV2 {
        ActivationFrameV2 {
            tokens,
            width,
            dtype: BoundaryDtype::F32,
            positions: (0..tokens).collect(),
            upstream_chain: chain(),
            input_lineage: lineage(),
            mode: FrameMode::Exact,
            taint: FrameTaint::Exact,
            placement_epoch: 7,
            tensor: (0..tokens * width * 4).map(|byte| byte as u8).collect(),
        }
    }

    fn v1_frame(tokens: u32, width: u32) -> ActivationFrame {
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

    fn chunk_of(frame: &ActivationFrameV2) -> ActivationChunkV2 {
        let bytes = frame.to_bytes();
        let mut header = ChunkHeaderV2 {
            frame_id: 1,
            parent_operation_id: 2,
            chunk_index: 0,
            chunk_total: 1,
            signed_absolute_deadline_ms: 5_000,
            frame_digest: frame_digest_v2(&bytes),
            chunk_digest: [0; 32],
            upstream_chain: frame.upstream_chain.clone(),
            input_lineage: frame.input_lineage.clone(),
            mode: frame.mode,
            taint: frame.taint,
        };
        header.chunk_digest = chunk_digest_v2(&header, &bytes);
        ActivationChunkV2 {
            header,
            signature: [3; 64],
            bytes,
        }
    }

    #[test]
    fn a_v2_frame_round_trips_with_its_lineage_mode_and_taint() {
        let mut original = frame(3, 4);
        assert_eq!(
            ActivationFrameV2::from_bytes(&original.to_bytes()).unwrap(),
            original
        );
        original.mode = FrameMode::Approximate;
        original.taint = FrameTaint::Approximate {
            report_digest: [9; 32],
        };
        let decoded = ActivationFrameV2::from_bytes(&original.to_bytes()).unwrap();
        assert_eq!(decoded, original);
        assert_eq!(decoded.tensor_f32().len(), 12);
        assert_eq!(decoded.digest(), frame_digest_v2(&original.to_bytes()));
    }

    #[test]
    fn a_v2_chunk_round_trips_through_the_both_versions_allowlist() {
        let frame = frame(2, 2);
        let chunk = chunk_of(&frame);
        assert!(chunk.header.matches_frame(&frame));
        let admitted = admit_v2(&chunk_payload_v2(&chunk)).unwrap();
        assert_eq!(admitted, AllowlistedPayloadV2::ChunkV2(Box::new(chunk)));
    }

    #[test]
    fn the_v1_schema_is_unchanged_and_still_admitted_next_to_v2() {
        // The v1 magic, the v1 allowlist and the v1 digest are exactly what they were.
        let v1 = v1_frame(2, 2);
        assert_eq!(&v1.to_bytes()[..8], b"PTRACT01");
        let hex: String = frame_digest(&v1.to_bytes())
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect();
        assert_eq!(
            hex,
            "75ce99bc2e005cdd0d42e3ca4f0f0d80a32dd720a8a012b95da03488dcac3992"
        );
        assert_eq!(ALLOWLISTED_TYPES.len(), 2);
        assert!(!ALLOWLISTED_TYPES.contains(&ACTIVATION_CHUNK_V2_TYPE));
        // A v1 chunk is admitted by the v1 and by the both-versions allowlist alike.
        let bytes = v1.to_bytes();
        let mut header = ChunkHeader {
            frame_id: 1,
            parent_operation_id: 2,
            chunk_index: 0,
            chunk_total: 1,
            signed_absolute_deadline_ms: 5_000,
            frame_digest: frame_digest(&bytes),
            chunk_digest: [0; 32],
            upstream_chain: v1.upstream_chain.clone(),
        };
        header.chunk_digest = chunk_digest(&header, &bytes);
        let chunk = ActivationChunk {
            header,
            signature: [3; 64],
            bytes,
        };
        let payload = chunk_payload(&chunk);
        assert_eq!(
            admit(&payload),
            Ok(super::super::AllowlistedPayload::Chunk(Box::new(
                chunk.clone()
            )))
        );
        assert_eq!(
            admit_v2(&payload),
            Ok(AllowlistedPayloadV2::ChunkV1(Box::new(chunk)))
        );
        // A v1 frame and a v2 frame of the same tensor never share a digest.
        assert_ne!(frame_digest(&v1.to_bytes()), frame(2, 2).digest());
    }

    #[test]
    fn v1_and_v2_bytes_are_not_decodable_as_the_other_version() {
        let v2 = frame(2, 2);
        let v1 = v1_frame(2, 2);
        assert_eq!(
            ActivationFrame::from_bytes(&v2.to_bytes()),
            Err(SchemaError::BadMagic)
        );
        assert_eq!(
            ActivationFrameV2::from_bytes(&v1.to_bytes()),
            Err(SchemaError::BadMagic)
        );
        // A v2 chunk under the v1 chunk type, and a v1 chunk under the v2 type, are refused;
        // the v1 allowlist refuses the v2 type id whatever the bytes are.
        let v2_chunk = chunk_of(&v2);
        assert_eq!(
            decode_chunk(&encode_chunk_v2(&v2_chunk)),
            Err(SchemaError::BadMagic)
        );
        let v2_under_v1 = TypedPayload {
            type_id: TypeId::from(ACTIVATION_CHUNK_TYPE),
            bytes: encode_chunk_v2(&v2_chunk),
        };
        assert_eq!(admit(&v2_under_v1), Err(SchemaError::BadMagic));
        assert_eq!(admit_v2(&v2_under_v1), Err(SchemaError::BadMagic));
        assert_eq!(
            admit(&chunk_payload_v2(&v2_chunk)),
            Err(SchemaError::NotAllowlisted(
                ACTIVATION_CHUNK_V2_TYPE.to_owned()
            ))
        );
        let v1_chunk_bytes = {
            let bytes = v1.to_bytes();
            let mut header = ChunkHeader {
                frame_id: 1,
                parent_operation_id: 2,
                chunk_index: 0,
                chunk_total: 1,
                signed_absolute_deadline_ms: 5_000,
                frame_digest: frame_digest(&bytes),
                chunk_digest: [0; 32],
                upstream_chain: v1.upstream_chain.clone(),
            };
            header.chunk_digest = chunk_digest(&header, &bytes);
            encode_chunk(&ActivationChunk {
                header,
                signature: [3; 64],
                bytes,
            })
        };
        assert_eq!(decode_chunk_v2(&v1_chunk_bytes), Err(SchemaError::BadMagic));
        let v1_under_v2 = TypedPayload {
            type_id: TypeId::from(ACTIVATION_CHUNK_V2_TYPE),
            bytes: v1_chunk_bytes,
        };
        assert_eq!(admit_v2(&v1_under_v2), Err(SchemaError::BadMagic));
        // Off both allowlists.
        for name in ["ptr.kv.page", "ptr.split.activation-chunk.v3", ""] {
            let payload = TypedPayload {
                type_id: TypeId::from(name),
                bytes: encode_chunk_v2(&v2_chunk),
            };
            assert_eq!(
                admit_v2(&payload),
                Err(SchemaError::NotAllowlisted(name.to_owned()))
            );
        }
    }

    #[test]
    fn every_truncation_of_a_frame_and_of_a_chunk_is_refused_without_a_panic() {
        let bytes = frame(2, 2).to_bytes();
        for length in 0..bytes.len() {
            assert!(
                ActivationFrameV2::from_bytes(&bytes[..length]).is_err(),
                "frame prefix {length}"
            );
        }
        let chunk = encode_chunk_v2(&chunk_of(&frame(1, 1)));
        for length in 0..chunk.len() {
            assert!(
                decode_chunk_v2(&chunk[..length]).is_err(),
                "chunk prefix {length}"
            );
        }
        let mut extra = bytes.clone();
        extra.push(0);
        assert_eq!(
            ActivationFrameV2::from_bytes(&extra),
            Err(SchemaError::TrailingBytes)
        );
        let mut extra = chunk;
        extra.push(0);
        assert_eq!(decode_chunk_v2(&extra), Err(SchemaError::TrailingBytes));
    }

    #[test]
    fn every_bound_of_the_v2_frame_and_chunk_is_enforced_with_a_typed_error() {
        let mut huge = frame(1, 1);
        huge.tokens = MAX_FRAME_TOKENS + 1;
        assert_eq!(
            ActivationFrameV2::from_bytes(&huge.to_bytes()),
            Err(SchemaError::OverBound("tokens"))
        );
        let mut wide = frame(1, 1);
        wide.width = MAX_FRAME_WIDTH + 1;
        assert_eq!(
            ActivationFrameV2::from_bytes(&wide.to_bytes()),
            Err(SchemaError::OverBound("width"))
        );
        let mut positions = frame(1, 1);
        positions.positions = vec![0; MAX_FRAME_TOKENS as usize + 1];
        assert_eq!(
            ActivationFrameV2::from_bytes(&positions.to_bytes()),
            Err(SchemaError::OverBound("positions"))
        );
        let mut many = frame(1, 1);
        many.upstream_chain.adapters = vec![element("a", 1); super::super::MAX_CHAIN_ELEMENTS + 1];
        assert_eq!(
            ActivationFrameV2::from_bytes(&many.to_bytes()),
            Err(SchemaError::OverBound("chain elements"))
        );
        let mut inputs = frame(1, 1);
        inputs.input_lineage.inputs = (0..MAX_LINEAGE_INPUTS + 1)
            .map(|index| input(LineageKind::Semantic, &format!("s{index:03}"), 1))
            .collect();
        assert_eq!(
            ActivationFrameV2::from_bytes(&inputs.to_bytes()),
            Err(SchemaError::OverBound("lineage inputs"))
        );
        let mut raw = frame(1, 1);
        raw.input_lineage.raw_ancestry = vec![[1; 32]; MAX_RAW_ANCESTRY + 1];
        assert_eq!(
            ActivationFrameV2::from_bytes(&raw.to_bytes()),
            Err(SchemaError::OverBound("raw ancestry"))
        );
        // The bounds hold at their limits.
        let mut at_limit = frame(1, 1);
        at_limit.input_lineage.inputs = (0..MAX_LINEAGE_INPUTS)
            .map(|index| input(LineageKind::Memory, &format!("m{index:03}"), 1))
            .collect();
        at_limit.input_lineage.raw_ancestry = vec![[1; 32]; MAX_RAW_ANCESTRY];
        assert_eq!(
            ActivationFrameV2::from_bytes(&at_limit.to_bytes()).unwrap(),
            at_limit
        );
        let mut long = frame(1, 1);
        long.input_lineage.inputs[0].identity = "x".repeat(MAX_IDENTITY_BYTES + 1);
        assert_eq!(
            ActivationFrameV2::from_bytes(&long.to_bytes()),
            Err(SchemaError::OverBound("identity"))
        );
        let mut empty = frame(1, 1);
        empty.input_lineage.inputs[0].identity.clear();
        assert_eq!(
            ActivationFrameV2::from_bytes(&empty.to_bytes()),
            Err(SchemaError::BadIdentity)
        );
        let mut chunk = chunk_of(&frame(1, 1));
        chunk.header.chunk_index = 1;
        assert_eq!(
            decode_chunk_v2(&encode_chunk_v2(&chunk)),
            Err(SchemaError::ChunkIndex { index: 1, total: 1 })
        );
        chunk.header.chunk_index = 0;
        chunk.header.chunk_total = MAX_CHUNKS_PER_FRAME + 1;
        assert_eq!(
            decode_chunk_v2(&encode_chunk_v2(&chunk)),
            Err(SchemaError::OverBound("chunk_total"))
        );
        let mut chunk = chunk_of(&frame(1, 1));
        chunk.header.input_lineage.raw_ancestry = vec![[1; 32]; MAX_RAW_ANCESTRY + 1];
        assert_eq!(
            decode_chunk_v2(&encode_chunk_v2(&chunk)),
            Err(SchemaError::OverBound("raw ancestry"))
        );
    }

    #[test]
    fn an_unknown_code_a_mode_taint_disagreement_and_a_non_canonical_lineage_are_refused() {
        // Unknown lineage kind: the first input's kind byte follows the header fields.
        let good = frame(1, 1);
        let mut bytes = good.to_bytes();
        let kind_at = bytes
            .windows(5)
            .position(|window| window == [3, 0, 0, 0, 1])
            .map(|at| at + 4)
            .unwrap();
        bytes[kind_at] = 9;
        assert_eq!(
            ActivationFrameV2::from_bytes(&bytes),
            Err(SchemaError::UnknownLineageKind(9))
        );
        // Mode and taint codes sit just before the placement epoch and tensor length.
        let tail = 8 + 8 + good.tensor.len();
        let taint_at = good.to_bytes().len() - tail - 1;
        let mode_at = taint_at - 1;
        let mut bytes = good.to_bytes();
        bytes[mode_at] = 7;
        assert_eq!(
            ActivationFrameV2::from_bytes(&bytes),
            Err(SchemaError::UnknownMode(7))
        );
        let mut bytes = good.to_bytes();
        bytes[taint_at] = 5;
        assert_eq!(
            ActivationFrameV2::from_bytes(&bytes),
            Err(SchemaError::UnknownTaint(5))
        );
        // Approximate mode with an Exact taint, and the reverse.
        let mut laundered = frame(1, 1);
        laundered.mode = FrameMode::Approximate;
        assert_eq!(
            ActivationFrameV2::from_bytes(&laundered.to_bytes()),
            Err(SchemaError::ModeTaintMismatch)
        );
        let mut inverse = frame(1, 1);
        inverse.taint = FrameTaint::Approximate {
            report_digest: [1; 32],
        };
        assert_eq!(
            ActivationFrameV2::from_bytes(&inverse.to_bytes()),
            Err(SchemaError::ModeTaintMismatch)
        );
        let mut chunk = chunk_of(&frame(1, 1));
        chunk.header.mode = FrameMode::Approximate;
        assert_eq!(
            decode_chunk_v2(&encode_chunk_v2(&chunk)),
            Err(SchemaError::ModeTaintMismatch)
        );
        // Out of order and duplicated inputs are not canonical.
        let mut unsorted = frame(1, 1);
        unsorted.input_lineage.inputs.reverse();
        assert_eq!(
            ActivationFrameV2::from_bytes(&unsorted.to_bytes()),
            Err(SchemaError::NonCanonicalLineage)
        );
        let mut duplicate = frame(1, 1);
        let first = duplicate.input_lineage.inputs[0].clone();
        duplicate.input_lineage.inputs.insert(1, first);
        assert_eq!(
            ActivationFrameV2::from_bytes(&duplicate.to_bytes()),
            Err(SchemaError::NonCanonicalLineage)
        );
        // `canonical` produces the order the decoder wants and drops the duplicate.
        let rebuilt = InputLineage::canonical(
            duplicate.input_lineage.inputs.clone(),
            duplicate.input_lineage.raw_ancestry.clone(),
        );
        assert_eq!(rebuilt, frame(1, 1).input_lineage);
    }

    #[test]
    fn conflicting_digests_for_one_input_are_refused_not_collapsed() {
        let mut conflicting = frame(1, 1);
        let mut twin = conflicting.input_lineage.inputs[0].clone();
        twin.digest = [0xEE; 32];
        let mut inputs = conflicting.input_lineage.inputs.clone();
        inputs.push(twin);
        // Fully identical inputs still collapse; conflicting ones are both kept.
        let identical = conflicting.input_lineage.inputs[0].clone();
        let mut with_identical = conflicting.input_lineage.inputs.clone();
        with_identical.push(identical);
        assert_eq!(
            InputLineage::canonical(with_identical, vec![]).inputs,
            conflicting.input_lineage.inputs
        );
        conflicting.input_lineage = InputLineage::canonical(inputs, vec![]);
        assert_eq!(
            conflicting.input_lineage.inputs.len(),
            frame(1, 1).input_lineage.inputs.len() + 1
        );
        assert_eq!(
            conflicting.validate(),
            Err(SchemaError::ConflictingLineageDigests)
        );
        assert_eq!(
            ActivationFrameV2::from_bytes(&conflicting.to_bytes()),
            Err(SchemaError::ConflictingLineageDigests)
        );
    }

    #[test]
    fn an_oversized_frame_is_refused_before_it_is_serialised() {
        let ok = frame(1, 1);
        assert_eq!(ok.check_in_memory_bounds(), Ok(()));
        let mut many = frame(1, 1);
        many.input_lineage.raw_ancestry = vec![[1; 32]; MAX_RAW_ANCESTRY + 1];
        assert_eq!(
            many.check_in_memory_bounds(),
            Err(SchemaError::OverBound("raw ancestry"))
        );
        let mut inputs = frame(1, 1);
        inputs.input_lineage.inputs = (0..MAX_LINEAGE_INPUTS + 1)
            .map(|index| LineageInput {
                kind: LineageKind::Memory,
                identity: format!("m{index:03}"),
                generation: 1,
                digest: [1; 32],
            })
            .collect();
        assert_eq!(
            inputs.check_in_memory_bounds(),
            Err(SchemaError::OverBound("lineage inputs"))
        );
        let mut adapters = frame(1, 1);
        adapters.upstream_chain.adapters = vec![element("a", 1); MAX_CHAIN_ELEMENTS + 1];
        assert_eq!(
            adapters.check_in_memory_bounds(),
            Err(SchemaError::OverBound("chain elements"))
        );
        let mut long = frame(1, 1);
        long.upstream_chain.stage.identity = "x".repeat(MAX_IDENTITY_BYTES + 1);
        assert_eq!(
            long.check_in_memory_bounds(),
            Err(SchemaError::OverBound("identity"))
        );
        let mut tokens = frame(1, 1);
        tokens.tokens = MAX_FRAME_TOKENS + 1;
        assert_eq!(
            tokens.check_in_memory_bounds(),
            Err(SchemaError::OverBound("tokens"))
        );
        let mut tensor = frame(1, 1);
        tensor.tensor = vec![0; MAX_FRAME_BYTES as usize + 1];
        assert_eq!(tensor.validate(), Err(SchemaError::OverBound("tensor")));
        // validate() answers from the bounds before it decodes any byte: a frame that is both
        // over a bound and non-canonical is refused for the bound, which a decode of its
        // serialised bytes would reach only after the lineage order.
        let mut both = frame(1, 1);
        both.input_lineage.inputs.reverse();
        both.input_lineage.raw_ancestry = vec![[1; 32]; MAX_RAW_ANCESTRY + 1];
        assert_eq!(both.validate(), Err(SchemaError::OverBound("raw ancestry")));
        let mut shape = frame(1, 1);
        shape.tensor.pop();
        assert!(matches!(
            shape.check_in_memory_bounds(),
            Err(SchemaError::ShapeMismatch { .. })
        ));
    }

    #[test]
    fn the_frame_digest_binds_the_lineage_the_raw_order_the_mode_and_the_taint() {
        let base = frame(2, 2).digest();
        let mut changed = frame(2, 2);
        changed.input_lineage.inputs[0].generation += 1;
        assert_ne!(changed.digest(), base);
        let mut changed = frame(2, 2);
        changed.input_lineage.raw_ancestry.swap(0, 1);
        assert_ne!(changed.digest(), base);
        let mut changed = frame(2, 2);
        changed.input_lineage.raw_ancestry.remove(0);
        assert_ne!(changed.digest(), base);
        let mut changed = frame(2, 2);
        changed.mode = FrameMode::Approximate;
        changed.taint = FrameTaint::Approximate {
            report_digest: [1; 32],
        };
        assert_ne!(changed.digest(), base);
        let mut changed = frame(2, 2);
        changed.upstream_chain.experts[0].fence_epoch += 1;
        assert_ne!(changed.digest(), base);
    }

    #[test]
    fn the_chunk_digest_and_the_signed_envelope_bind_the_lineage_mode_and_taint() {
        let chunk = chunk_of(&frame(2, 2));
        let digest = chunk_digest_v2(&chunk.header, &chunk.bytes);
        assert_eq!(digest, chunk.header.chunk_digest);
        let envelope = envelope_signing_bytes_v2(&chunk.header);
        type Edit = fn(&mut ChunkHeaderV2);
        let edits: [Edit; 5] = [
            |h| h.input_lineage.inputs[1].digest[0] ^= 1,
            |h| h.input_lineage.raw_ancestry.reverse(),
            |h| {
                h.input_lineage.raw_ancestry.pop();
            },
            |h| {
                h.mode = FrameMode::Approximate;
                h.taint = FrameTaint::Approximate {
                    report_digest: [4; 32],
                };
            },
            |h| h.upstream_chain.stage.generation += 1,
        ];
        for edit in edits {
            let mut changed = chunk.header.clone();
            edit(&mut changed);
            assert_ne!(chunk_digest_v2(&changed, &chunk.bytes), digest);
            assert_ne!(envelope_signing_bytes_v2(&changed), envelope);
            assert!(!changed.matches_frame(&frame(2, 2)));
        }
        // The v1 envelope of the same frame is another domain.
        assert_ne!(
            envelope_signing_bytes(&ChunkHeader {
                frame_id: 1,
                parent_operation_id: 2,
                chunk_index: 0,
                chunk_total: 1,
                signed_absolute_deadline_ms: 5_000,
                frame_digest: chunk.header.frame_digest,
                chunk_digest: chunk.header.chunk_digest,
                upstream_chain: chain(),
            }),
            envelope
        );
    }

    #[test]
    fn a_frame_built_in_memory_is_validated_like_its_bytes() {
        assert_eq!(frame(2, 2).validate(), Ok(()));
        let mut unsorted = frame(2, 2);
        unsorted.input_lineage.inputs.reverse();
        assert_eq!(unsorted.validate(), Err(SchemaError::NonCanonicalLineage));
        let mut many = frame(1, 1);
        many.input_lineage.raw_ancestry = vec![[1; 32]; MAX_RAW_ANCESTRY + 1];
        assert_eq!(many.validate(), Err(SchemaError::OverBound("raw ancestry")));
        let mut short = frame(2, 2);
        short.tensor.pop();
        assert!(matches!(
            short.validate(),
            Err(SchemaError::ShapeMismatch { .. })
        ));
        let mut laundered = frame(1, 1);
        laundered.mode = FrameMode::Approximate;
        assert_eq!(laundered.validate(), Err(SchemaError::ModeTaintMismatch));
        assert_eq!(LineageKind::Memory.name(), "memory");
    }

    #[test]
    fn lineage_and_mode_bytes_are_canonical_helpers_for_downstream_digests() {
        let one = lineage();
        assert_eq!(lineage_bytes(&one), lineage_bytes(&one.clone()));
        let mut other = lineage();
        other.raw_ancestry.reverse();
        assert_ne!(lineage_bytes(&one), lineage_bytes(&other));
        assert_ne!(
            mode_taint_bytes(FrameMode::Exact, FrameTaint::Exact),
            mode_taint_bytes(
                FrameMode::Approximate,
                FrameTaint::Approximate {
                    report_digest: [0; 32]
                }
            )
        );
    }
}
