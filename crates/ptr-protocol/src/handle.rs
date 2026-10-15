//! The tensor handle of the Pod data plane (doc 42, section 1): a fenced, signed, single-use
//! claim on a tensor that a producer holds, in place of the tensor bytes.
//!
//! A [`TensorHandle`] is not a pointer. It names a [`Carrier`] and a [`CarrierLocator`] for the
//! tensor, and it carries what an [`ActivationFrame`](crate::activation::ActivationFrame)
//! carries for admission: the [`UpstreamChain`] of the producing stage, the placement epoch it
//! held, a signed absolute deadline and a nonce, all under an Ed25519 signature over
//! [`handle_signing_bytes`]. This module only defines the wire type and its canonical codec; it
//! maps no memory, opens no file and holds no key (the signature is a `[u8; 64]` field, as in
//! the activation schemas). Admission lives in `ptr-runtime`.
//!
//! **Content commitment.** The handle carries `content_digest`, the SHA-256 of the tensor bytes,
//! under the signature. Admission never reads memory, so it cannot check it: the consumer must
//! map the (sealed, immutable) region, call [`verify_content`] on the mapped bytes and use the
//! tensor only if that succeeds. The producer must not write the region after it signed the
//! handle: ownership of the contents passes to the consumer until `Consumed`, and the producer
//! reclaims the buffer only after `Consumed` or after the signed deadline plus the clock skew.
//!
//! **Canonical layout.** A handle describes a dense, row-major, contiguous tensor of
//! `product(shape)` elements of `dtype`, with no offset or strides beyond the locator's own
//! offset. A producer must materialize a contiguous copy of a sliced or transposed view before
//! it signs the handle, and a consumer interprets the bytes only as that layout.
//!
//! **Operation and audience binding.** Like a chunk of the split path, a handle is signed for
//! one operation (`parent_operation_id`) and one consumer stage (`consumer`): admission refuses
//! a handle presented under another operation or addressed to another stage.
//!
//! **What the carrier checks.** The codec checks the handle against itself only. For
//! [`Carrier::HostShm`] the carrier checks `fd_slot` and `offset + byte_len` against the region
//! it actually received over the control socket, and for the other carriers their own locator
//! against what they hold. For [`Carrier::Bytes`] the bytes are in hand, so `validate` checks
//! `content_digest` against the inline payload itself.
//!
//! The codec is hand written, length-prefixed, little-endian and bounded, like the activation
//! codec: every enumeration crosses as an explicit code, and bytes that disagree with
//! themselves (a locator of another carrier, a byte length that is not `product(shape) x
//! dtype`, trailing bytes) are refused whole. A handle whose carrier is
//! [`Carrier::InProcess`] has no locator and can never be encoded: the tensor object itself
//! does not leave its process.

use sha2::{Digest as _, Sha256};
use std::fmt;

use ptr_types::TypeId;

use crate::activation::{
    chain_bytes, BoundaryDtype, ChainElement, UpstreamChain, MAX_CHAIN_ELEMENTS,
    MAX_CHUNKS_PER_FRAME, MAX_FRAME_BYTES, MAX_IDENTITY_BYTES,
};
use crate::TypedPayload;

/// Type id of an encoded [`TensorHandle`].
pub const TENSOR_HANDLE_TYPE: &str = "ptr.data.tensor-handle.v1";

/// Bound on the dimensions of a handle's shape.
pub const MAX_HANDLE_DIMS: usize = 8;
/// Per-handle bound on `byte_len` (64 MiB): the bound of one activation frame, so a
/// [`Carrier::Remote`] handle can name any frame the split path can carry. The sum over a
/// producer's open handles is bounded separately by the runtime's in-flight byte budget.
pub const MAX_HANDLE_BYTES: u64 = MAX_FRAME_BYTES;
/// Bound on the inline payload of a [`Carrier::Bytes`] handle (1 MiB). A handle is a control
/// message; a tensor larger than this travels by another carrier or as chunks.
pub const MAX_INLINE_BYTES: u64 = 1024 * 1024;

const HANDLE_MAGIC: &[u8; 8] = b"PTRHDL01";
const HANDLE_DOMAIN: &[u8] = b"ptr.data.handle.v1\0";
const DIGEST_DOMAIN: &[u8] = b"ptr.data.handle-digest.v1\0";

/// How the bytes of a tensor are reached, ordered from nearest to farthest (the derived order
/// is the declaration order, which is the `InterconnectClass` of doc 40 seen from the data
/// side).
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum Carrier {
    /// The tensor object itself, in the same process. Never serialised.
    InProcess,
    /// The bytes inline in the handle (the universal fallback).
    Bytes,
    /// A sealed shared-memory region on the same host.
    HostShm,
    /// A CUDA IPC memory handle on the same host and the same GPU.
    CudaIpc,
    /// A buffer of another GPU on the same host, reached by peer copy.
    DevicePeer,
    /// A pinned host buffer.
    PinnedHost,
    /// A file-backed chunk on NVMe.
    NvmeFile,
    /// A frame held by another host.
    Remote,
}

impl Carrier {
    /// The stable wire code of the carrier. Codes are never reused.
    pub const fn code(self) -> u8 {
        match self {
            Self::InProcess => 1,
            Self::Bytes => 2,
            Self::HostShm => 3,
            Self::CudaIpc => 4,
            Self::DevicePeer => 5,
            Self::PinnedHost => 6,
            Self::NvmeFile => 7,
            Self::Remote => 8,
        }
    }

    /// The carrier with wire code `code`.
    pub fn from_code(code: u8) -> Result<Self, HandleError> {
        match code {
            1 => Ok(Self::InProcess),
            2 => Ok(Self::Bytes),
            3 => Ok(Self::HostShm),
            4 => Ok(Self::CudaIpc),
            5 => Ok(Self::DevicePeer),
            6 => Ok(Self::PinnedHost),
            7 => Ok(Self::NvmeFile),
            8 => Ok(Self::Remote),
            other => Err(HandleError::UnknownCarrier(other)),
        }
    }
}

/// How a CUDA buffer is exported across processes.
///
/// The exported memory is an isolated staging buffer owned by the supervisor (the broker), not
/// by the producer worker, so a producer crash cannot free it under the consumer. The consumer
/// copies it into receiver-owned memory and verifies `content_digest` on that copy. The tensor
/// starts at offset zero of the export.
#[derive(Clone, Eq, PartialEq)]
pub enum CudaExport {
    /// A legacy allocation exported with `cuIpcGetMemHandle` and opened with
    /// `cuIpcOpenMemHandle` (with peer access for [`Carrier::DevicePeer`]).
    Legacy { mem_handle: [u8; 64] },
    /// A VMM allocation (`cuMemCreate`), which `cuIpcGetMemHandle` cannot export. It is exported
    /// with `cuMemExportToShareableHandle` as a POSIX fd that the supervisor passes with
    /// `SCM_RIGHTS` (`fd_slot` names the descriptor slot, bound to the handle nonce); the
    /// consumer uses `cuMemImportFromShareableHandle` and `cuMemMap`. Like the `HostShm`
    /// descriptor slot, `fd_slot` is passed with `SCM_RIGHTS` through the supervisor and bound
    /// to the handle nonce by the carrier and admission layer, not by this codec.
    /// `alloc_size` is the size of the allocation, at least `byte_len` and not necessarily a
    /// multiple of any granularity here (granularity is device-specific).
    Vmm { fd_slot: u32, alloc_size: u64 },
}

impl fmt::Debug for CudaExport {
    /// A legacy `mem_handle` is a capability: only the kind and the non-secret sizes print.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Legacy { .. } => formatter.write_str("Legacy { .. }"),
            Self::Vmm {
                fd_slot,
                alloc_size,
            } => formatter
                .debug_struct("Vmm")
                .field("fd_slot", fd_slot)
                .field("alloc_size", alloc_size)
                .finish(),
        }
    }
}

impl CudaExport {
    /// The stable wire tag: 0 legacy, 1 VMM.
    pub const fn tag(&self) -> u8 {
        match self {
            Self::Legacy { .. } => 0,
            Self::Vmm { .. } => 1,
        }
    }
}

/// Where a tensor is, per carrier. [`CarrierLocator::InProcess`] is a unit variant with nothing
/// to serialise; it exists only for [`InProcessHandle`], and no codec path accepts it.
#[derive(Clone, Eq, PartialEq)]
pub enum CarrierLocator {
    Bytes {
        inline: Vec<u8>,
    },
    /// A sealed memfd region. A memfd cannot be reopened by name: `fd_slot` names the
    /// descriptor slot passed with `SCM_RIGHTS` over the control socket and bound to the handle
    /// nonce, and `offset` the start of the tensor in the mapping.
    HostShm {
        fd_slot: u32,
        offset: u64,
    },
    /// A buffer on the same GPU, exported by the supervisor (see [`CudaExport`]). The consumer
    /// imports the export and must copy it into a receiver-owned allocation and verify
    /// `content_digest` on that copy before use; zero-copy use of the staging buffer is outside
    /// the contract. `event_handle` is the `cuIpcGetEventHandle` bytes for stream ordering.
    CudaIpc {
        device: u32,
        export: CudaExport,
        event_handle: [u8; 64],
    },
    /// A buffer on another GPU of the same host. There is no raw pointer: a pointer is only
    /// meaningful inside the producer's process, so the consumer imports the same kind of export
    /// as for [`Carrier::CudaIpc`] (with peer access for the legacy kind), then copies
    /// peer-to-peer from `source_device` into its own allocation and verifies `content_digest`
    /// on that copy before use.
    DevicePeer {
        source_device: u32,
        export: CudaExport,
        event_handle: [u8; 64],
    },
    PinnedHost {
        region: u64,
        offset: u64,
    },
    NvmeFile {
        chunk_digest: [u8; 32],
    },
    /// A complete frame held by another host: the frame id (under the handle's signed
    /// `parent_operation_id`), the digest of the whole frame and its chunk count
    /// (`1..=MAX_CHUNKS_PER_FRAME`).
    Remote {
        frame_id: u64,
        frame_digest: [u8; 32],
        chunk_total: u32,
    },
    /// The tensor object itself, in the producer's process: there is nothing to locate and
    /// nothing to serialise. Only an [`InProcessHandle`] carries it; [`TensorHandle::validate`],
    /// [`TensorHandle::encode`] and [`TensorHandle::decode`] refuse every handle that names it.
    InProcess,
}

impl fmt::Debug for CarrierLocator {
    /// Redacted: the CUDA export and event handles are capabilities and an inline payload is
    /// tensor data, so none of them print; sizes, slots and devices do.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Bytes { inline } => formatter
                .debug_struct("Bytes")
                .field("inline_len", &inline.len())
                .finish(),
            Self::HostShm { fd_slot, offset } => formatter
                .debug_struct("HostShm")
                .field("fd_slot", fd_slot)
                .field("offset", offset)
                .finish(),
            Self::CudaIpc { device, export, .. } => formatter
                .debug_struct("CudaIpc")
                .field("device", device)
                .field("export", export)
                .field("event_handle", &format_args!(".."))
                .finish(),
            Self::DevicePeer {
                source_device,
                export,
                ..
            } => formatter
                .debug_struct("DevicePeer")
                .field("source_device", source_device)
                .field("export", export)
                .field("event_handle", &format_args!(".."))
                .finish(),
            Self::PinnedHost { region, offset } => formatter
                .debug_struct("PinnedHost")
                .field("region", region)
                .field("offset", offset)
                .finish(),
            Self::NvmeFile { chunk_digest } => formatter
                .debug_struct("NvmeFile")
                .field("chunk_digest", chunk_digest)
                .finish(),
            Self::Remote {
                frame_id,
                frame_digest,
                chunk_total,
            } => formatter
                .debug_struct("Remote")
                .field("frame_id", frame_id)
                .field("frame_digest", frame_digest)
                .field("chunk_total", chunk_total)
                .finish(),
            Self::InProcess => formatter.write_str("InProcess"),
        }
    }
}

impl CarrierLocator {
    /// The carrier this locator belongs to.
    pub fn carrier(&self) -> Carrier {
        match self {
            Self::Bytes { .. } => Carrier::Bytes,
            Self::HostShm { .. } => Carrier::HostShm,
            Self::CudaIpc { .. } => Carrier::CudaIpc,
            Self::DevicePeer { .. } => Carrier::DevicePeer,
            Self::PinnedHost { .. } => Carrier::PinnedHost,
            Self::NvmeFile { .. } => Carrier::NvmeFile,
            Self::Remote { .. } => Carrier::Remote,
            Self::InProcess => Carrier::InProcess,
        }
    }
}

/// A fenced, signed, single-use claim on a tensor held by a producer.
///
/// The tensor is dense, row-major and contiguous: `product(shape)` elements of `dtype`, no
/// strides, and no offset other than the locator's own. A producer materializes a contiguous
/// copy of any sliced or transposed view before signing; a consumer reads the bytes only as
/// that layout.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TensorHandle {
    pub carrier: Carrier,
    pub locator: CarrierLocator,
    pub dtype: BoundaryDtype,
    /// The dimensions, outermost first: one to [`MAX_HANDLE_DIMS`], each non-zero.
    pub shape: Vec<u64>,
    /// `product(shape) x dtype.size()`; for [`Carrier::Bytes`] also the inline length.
    pub byte_len: u64,
    /// The stage, base model, adapters and experts the tensor was computed under.
    pub upstream_chain: UpstreamChain,
    /// The placement epoch the producing stage held when it produced the tensor.
    pub placement_epoch: u64,
    /// Absolute time (milliseconds) after which the receiver must not touch the tensor, signed
    /// with the handle.
    pub signed_absolute_deadline_ms: u64,
    /// The single-use nonce of the handle, unique per producer.
    pub handle_nonce: u64,
    /// The operation the handle was produced for, signed like the chunk path's
    /// `parent_operation_id`. Admission refuses a handle presented under another operation.
    pub parent_operation_id: u64,
    /// The identity of the stage that may consume the handle, signed. Non-empty and bounded
    /// like every identity.
    pub consumer: String,
    /// The schema and argument slot of the tensor (for example `ptr.split.activation.v1` or an
    /// operation-specific slot id), signed: a handle replaces `TypedPayload.bytes`, so it must
    /// carry the type the bytes had. Non-empty and bounded like every identity.
    pub payload_type: TypeId,
    /// SHA-256 of the `byte_len` tensor bytes ([`content_digest`]); the consumer verifies it
    /// with [`verify_content`] after mapping and before use.
    pub content_digest: [u8; 32],
    /// Ed25519 signature of the producing runtime over [`handle_signing_bytes`].
    pub signature: [u8; 64],
}

/// A locator-free, non-serialisable [`TensorHandle`] of [`Carrier::InProcess`]: the claim on a
/// tensor object that stays in the producing process and moves to the consumer separately, with
/// unique ownership (doc 42 section 2, rule 3a).
///
/// It carries everything a wire handle carries for admission (chain, placement epoch fence,
/// signed deadline, nonce, operation, audience, content digest, Ed25519 signature) and nothing
/// to locate bytes with. It has no codec and no `Serialize`: it cannot be written to a socket,
/// and [`TensorHandle::encode`] / [`TensorHandle::decode`] still refuse the carrier. It is not
/// `Clone`, so a copy is not made by accident (single use is the nonce tombstone in admission, not
/// this type). It enters admission only through its own source
/// (`HandleSource::InProcess` in `ptr-runtime`), so a wire handle can never be presented as one.
/// The constructor does not verify the signature; admission does, over [`handle_signing_bytes`].
#[derive(Debug, Eq, PartialEq)]
pub struct InProcessHandle {
    handle: TensorHandle,
}

impl InProcessHandle {
    /// Wrap `handle`, which must name [`Carrier::InProcess`] with [`CarrierLocator::InProcess`]
    /// and be well formed in every other field (shape, byte length, identities, chain).
    pub fn new(handle: TensorHandle) -> Result<Self, HandleError> {
        if handle.carrier != Carrier::InProcess {
            return Err(HandleError::LocatorMismatch {
                carrier: handle.carrier,
                locator: Carrier::InProcess,
            });
        }
        handle.validate_fields()?;
        Ok(Self { handle })
    }

    /// The handle fields, for admission and signing. The value is not a wire handle.
    pub fn as_tensor_handle(&self) -> &TensorHandle {
        &self.handle
    }

    /// The bytes the producing runtime signs ([`handle_signing_bytes`]).
    pub fn signing_bytes(&self) -> Vec<u8> {
        handle_signing_bytes(&self.handle)
    }

    /// The digest of the handle ([`handle_digest`]).
    pub fn digest(&self) -> [u8; 32] {
        handle_digest(&self.handle)
    }
}

/// Why bytes are no instance of the handle schema, or why a handle cannot be encoded.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum HandleError {
    BadMagic,
    Truncated,
    TrailingBytes,
    /// A carrier code that names no carrier.
    UnknownCarrier(u8),
    UnknownDtype(u8),
    /// A handle of [`Carrier::InProcess`] has no locator and is never encoded.
    InProcessNotEncodable,
    /// The locator belongs to another carrier than the handle states.
    LocatorMismatch {
        carrier: Carrier,
        locator: Carrier,
    },
    /// A bound of the schema is exceeded.
    OverBound(&'static str),
    EmptyShape,
    ZeroDimension,
    /// `product(shape)` overflows `u64`.
    ShapeOverflow,
    /// `product(shape) x dtype.size()` is not `byte_len`.
    ShapeMismatch {
        expected: u64,
        found: u64,
    },
    /// A CUDA export tag that names no export kind.
    UnknownExport(u8),
    /// A VMM export with an allocation size of zero.
    ZeroAllocSize,
    /// A VMM export smaller than the tensor.
    AllocTooSmall {
        byte_len: u64,
        alloc_size: u64,
    },
    /// A [`Carrier::Remote`] locator with a chunk count of zero.
    ZeroChunkTotal,
    /// The bytes presented to [`verify_content`] are not `byte_len` long.
    ContentLength {
        expected: u64,
        found: u64,
    },
    /// The bytes presented to [`verify_content`] do not hash to `content_digest`.
    ContentMismatch,
    /// The inline payload of a [`Carrier::Bytes`] handle is not `byte_len` long.
    InlineLength {
        expected: u64,
        found: u64,
    },
    /// An identity that is empty or not UTF-8.
    BadIdentity,
}

impl fmt::Display for HandleError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for HandleError {}

const fn dtype_code(dtype: BoundaryDtype) -> u8 {
    // Exhaustive: a new dtype must be given a code here before this compiles. The codes are
    // those of the activation codec.
    match dtype {
        BoundaryDtype::F32 => 1,
    }
}

fn dtype_from_code(code: u8) -> Result<BoundaryDtype, HandleError> {
    match code {
        1 => Ok(BoundaryDtype::F32),
        other => Err(HandleError::UnknownDtype(other)),
    }
}

impl TensorHandle {
    /// Whether the handle is well formed: a serialisable carrier with its own locator, a
    /// bounded non-empty shape whose product matches `byte_len`, an inline payload of exactly
    /// `byte_len` bytes for [`Carrier::Bytes`], and a chain within its bounds. Encoding and
    /// decoding both run this check.
    pub fn validate(&self) -> Result<(), HandleError> {
        if self.carrier == Carrier::InProcess {
            return Err(HandleError::InProcessNotEncodable);
        }
        self.validate_fields()
    }

    /// The checks of [`Self::validate`] without the refusal of [`Carrier::InProcess`], shared
    /// with [`InProcessHandle::new`].
    fn validate_fields(&self) -> Result<(), HandleError> {
        let locator = self.locator.carrier();
        if locator != self.carrier {
            return Err(HandleError::LocatorMismatch {
                carrier: self.carrier,
                locator,
            });
        }
        if self.shape.is_empty() {
            return Err(HandleError::EmptyShape);
        }
        if self.shape.len() > MAX_HANDLE_DIMS {
            return Err(HandleError::OverBound("dims"));
        }
        if self.byte_len > MAX_HANDLE_BYTES {
            return Err(HandleError::OverBound("byte_len"));
        }
        let mut elements: u64 = 1;
        for dimension in &self.shape {
            if *dimension == 0 {
                return Err(HandleError::ZeroDimension);
            }
            elements = elements
                .checked_mul(*dimension)
                .ok_or(HandleError::ShapeOverflow)?;
        }
        let expected = elements
            .checked_mul(self.dtype.size())
            .ok_or(HandleError::ShapeOverflow)?;
        if expected != self.byte_len {
            return Err(HandleError::ShapeMismatch {
                expected,
                found: self.byte_len,
            });
        }
        match &self.locator {
            CarrierLocator::Bytes { inline } => {
                let found = inline.len() as u64;
                if found > MAX_INLINE_BYTES {
                    return Err(HandleError::OverBound("inline"));
                }
                if found != self.byte_len {
                    return Err(HandleError::InlineLength {
                        expected: self.byte_len,
                        found,
                    });
                }
                if content_digest(inline) != self.content_digest {
                    return Err(HandleError::ContentMismatch);
                }
            }
            CarrierLocator::Remote { chunk_total, .. } => {
                if *chunk_total == 0 {
                    return Err(HandleError::ZeroChunkTotal);
                }
                if *chunk_total > MAX_CHUNKS_PER_FRAME {
                    return Err(HandleError::OverBound("chunk_total"));
                }
            }
            CarrierLocator::CudaIpc { export, .. } | CarrierLocator::DevicePeer { export, .. } => {
                check_export(export, self.byte_len)?;
            }
            CarrierLocator::HostShm { .. }
            | CarrierLocator::PinnedHost { .. }
            | CarrierLocator::NvmeFile { .. }
            | CarrierLocator::InProcess => {}
        }
        check_identity(&self.consumer)?;
        check_identity(&self.payload_type.0)?;
        check_chain(&self.upstream_chain)
    }

    /// The canonical bytes of the handle: magic, the body of [`handle_signing_bytes`] without
    /// its domain, and the signature. A handle that is not [`Self::validate`]-clean, and every
    /// handle of [`Carrier::InProcess`], is refused.
    pub fn encode(&self) -> Result<Vec<u8>, HandleError> {
        self.validate()?;
        let mut out = Vec::with_capacity(self.byte_len_hint() + 512);
        out.extend_from_slice(HANDLE_MAGIC);
        write_body(&mut out, self);
        out.extend_from_slice(&self.signature);
        Ok(out)
    }

    fn byte_len_hint(&self) -> usize {
        match &self.locator {
            CarrierLocator::Bytes { inline } => inline.len(),
            CarrierLocator::HostShm { .. }
            | CarrierLocator::CudaIpc { .. }
            | CarrierLocator::DevicePeer { .. }
            | CarrierLocator::PinnedHost { .. }
            | CarrierLocator::NvmeFile { .. }
            | CarrierLocator::Remote { .. }
            | CarrierLocator::InProcess => 0,
        }
    }

    /// Decode canonical handle bytes. Bytes that are not exactly one well-formed handle are
    /// refused: a wrong magic, an unknown carrier, dtype or locator, a locator of another
    /// carrier, a bound exceeded (checked before the content is read), a shape that does not
    /// match `byte_len` and trailing bytes.
    pub fn decode(bytes: &[u8]) -> Result<Self, HandleError> {
        let mut reader = Reader::new(bytes);
        if reader.take(8)? != HANDLE_MAGIC {
            return Err(HandleError::BadMagic);
        }
        let carrier = Carrier::from_code(reader.u8()?)?;
        if carrier == Carrier::InProcess {
            return Err(HandleError::InProcessNotEncodable);
        }
        let locator_carrier = Carrier::from_code(reader.u8()?)?;
        if locator_carrier != carrier {
            return Err(HandleError::LocatorMismatch {
                carrier,
                locator: locator_carrier,
            });
        }
        let locator = read_locator(&mut reader, carrier)?;
        let dtype = dtype_from_code(reader.u8()?)?;
        let dims = reader.u32()? as usize;
        if dims == 0 {
            return Err(HandleError::EmptyShape);
        }
        if dims > MAX_HANDLE_DIMS {
            return Err(HandleError::OverBound("dims"));
        }
        let mut shape = Vec::with_capacity(dims);
        for _ in 0..dims {
            shape.push(reader.u64()?);
        }
        let byte_len = reader.u64()?;
        if byte_len > MAX_HANDLE_BYTES {
            return Err(HandleError::OverBound("byte_len"));
        }
        let upstream_chain = read_chain(&mut reader)?;
        let placement_epoch = reader.u64()?;
        let signed_absolute_deadline_ms = reader.u64()?;
        let handle_nonce = reader.u64()?;
        let parent_operation_id = reader.u64()?;
        let consumer = read_identity(&mut reader)?;
        let payload_type = TypeId(read_identity(&mut reader)?);
        let content_digest = reader.array32()?;
        let signature = reader.array64()?;
        if !reader.is_empty() {
            return Err(HandleError::TrailingBytes);
        }
        let handle = Self {
            carrier,
            locator,
            dtype,
            shape,
            byte_len,
            upstream_chain,
            placement_epoch,
            signed_absolute_deadline_ms,
            handle_nonce,
            parent_operation_id,
            consumer,
            payload_type,
            content_digest,
            signature,
        };
        handle.validate()?;
        Ok(handle)
    }
}

/// The digest a handle commits to: SHA-256 of the tensor bytes.
pub fn content_digest(bytes: &[u8]) -> [u8; 32] {
    Sha256::digest(bytes).into()
}

/// Verify the mapped tensor bytes against the handle's commitment. The carriers call this after
/// mapping the region and before any use: the length must be `byte_len` and the SHA-256 must be
/// `content_digest`. Admission cannot do it, because admission reads no memory.
pub fn verify_content(handle: &TensorHandle, bytes: &[u8]) -> Result<(), HandleError> {
    let found = bytes.len() as u64;
    if found != handle.byte_len {
        return Err(HandleError::ContentLength {
            expected: handle.byte_len,
            found,
        });
    }
    if content_digest(bytes) != handle.content_digest {
        return Err(HandleError::ContentMismatch);
    }
    Ok(())
}

/// Wrap a handle as a `TypedPayload` of [`TENSOR_HANDLE_TYPE`].
pub fn handle_payload(handle: &TensorHandle) -> Result<TypedPayload, HandleError> {
    Ok(TypedPayload {
        type_id: TypeId::from(TENSOR_HANDLE_TYPE),
        bytes: handle.encode()?,
    })
}

/// The bytes the producing runtime signs: every field of the handle except the signature,
/// domain-separated from the activation envelope, the policy and the digest. Infallible: it
/// covers whatever the handle says, and admission refuses a handle that is not well formed
/// before it looks at the signature.
pub fn handle_signing_bytes(handle: &TensorHandle) -> Vec<u8> {
    let mut out = Vec::with_capacity(HANDLE_DOMAIN.len() + 512);
    out.extend_from_slice(HANDLE_DOMAIN);
    write_body(&mut out, handle);
    out
}

/// The digest of a handle: its signed bytes and its signature, domain-separated. Two handles
/// have the same digest only if they agree in every field, the signature included.
pub fn handle_digest(handle: &TensorHandle) -> [u8; 32] {
    let signed = handle_signing_bytes(handle);
    let mut hasher = Sha256::new();
    hasher.update(DIGEST_DOMAIN);
    hasher.update((signed.len() as u64).to_le_bytes());
    hasher.update(&signed);
    hasher.update(handle.signature);
    hasher.finalize().into()
}

fn write_body(out: &mut Vec<u8>, handle: &TensorHandle) {
    out.push(handle.carrier.code());
    write_locator(out, &handle.locator);
    out.push(dtype_code(handle.dtype));
    out.extend_from_slice(&(handle.shape.len() as u32).to_le_bytes());
    for dimension in &handle.shape {
        out.extend_from_slice(&dimension.to_le_bytes());
    }
    out.extend_from_slice(&handle.byte_len.to_le_bytes());
    out.extend_from_slice(&chain_bytes(&handle.upstream_chain));
    out.extend_from_slice(&handle.placement_epoch.to_le_bytes());
    out.extend_from_slice(&handle.signed_absolute_deadline_ms.to_le_bytes());
    out.extend_from_slice(&handle.handle_nonce.to_le_bytes());
    out.extend_from_slice(&handle.parent_operation_id.to_le_bytes());
    out.extend_from_slice(&(handle.consumer.len() as u32).to_le_bytes());
    out.extend_from_slice(handle.consumer.as_bytes());
    out.extend_from_slice(&(handle.payload_type.0.len() as u32).to_le_bytes());
    out.extend_from_slice(handle.payload_type.0.as_bytes());
    out.extend_from_slice(&handle.content_digest);
}

fn write_locator(out: &mut Vec<u8>, locator: &CarrierLocator) {
    out.push(locator.carrier().code());
    match locator {
        CarrierLocator::Bytes { inline } => {
            out.extend_from_slice(&(inline.len() as u32).to_le_bytes());
            out.extend_from_slice(inline);
        }
        CarrierLocator::HostShm { fd_slot, offset } => {
            out.extend_from_slice(&fd_slot.to_le_bytes());
            out.extend_from_slice(&offset.to_le_bytes());
        }
        CarrierLocator::PinnedHost { region, offset } => {
            out.extend_from_slice(&region.to_le_bytes());
            out.extend_from_slice(&offset.to_le_bytes());
        }
        CarrierLocator::CudaIpc {
            device,
            export,
            event_handle,
        }
        | CarrierLocator::DevicePeer {
            source_device: device,
            export,
            event_handle,
        } => {
            out.extend_from_slice(&device.to_le_bytes());
            write_export(out, export);
            out.extend_from_slice(event_handle);
        }
        CarrierLocator::NvmeFile { chunk_digest } => out.extend_from_slice(chunk_digest),
        CarrierLocator::Remote {
            frame_id,
            frame_digest,
            chunk_total,
        } => {
            out.extend_from_slice(&frame_id.to_le_bytes());
            out.extend_from_slice(frame_digest);
            out.extend_from_slice(&chunk_total.to_le_bytes());
        }
        // Nothing follows the carrier code: there is no locator. Reached only through the
        // signing bytes and digest of an `InProcessHandle`; no codec path writes it.
        CarrierLocator::InProcess => {}
    }
}

fn read_locator(reader: &mut Reader<'_>, carrier: Carrier) -> Result<CarrierLocator, HandleError> {
    match carrier {
        Carrier::InProcess => Err(HandleError::InProcessNotEncodable),
        Carrier::Bytes => {
            let length = reader.u32()? as usize;
            if length as u64 > MAX_INLINE_BYTES {
                return Err(HandleError::OverBound("inline"));
            }
            Ok(CarrierLocator::Bytes {
                inline: reader.take(length)?.to_vec(),
            })
        }
        Carrier::HostShm => Ok(CarrierLocator::HostShm {
            fd_slot: reader.u32()?,
            offset: reader.u64()?,
        }),
        Carrier::CudaIpc => Ok(CarrierLocator::CudaIpc {
            device: reader.u32()?,
            export: read_export(reader)?,
            event_handle: reader.array64()?,
        }),
        Carrier::DevicePeer => Ok(CarrierLocator::DevicePeer {
            source_device: reader.u32()?,
            export: read_export(reader)?,
            event_handle: reader.array64()?,
        }),
        Carrier::PinnedHost => Ok(CarrierLocator::PinnedHost {
            region: reader.u64()?,
            offset: reader.u64()?,
        }),
        Carrier::NvmeFile => Ok(CarrierLocator::NvmeFile {
            chunk_digest: reader.array32()?,
        }),
        Carrier::Remote => Ok(CarrierLocator::Remote {
            frame_id: reader.u64()?,
            frame_digest: reader.array32()?,
            chunk_total: reader.u32()?,
        }),
    }
}

fn check_export(export: &CudaExport, byte_len: u64) -> Result<(), HandleError> {
    match export {
        CudaExport::Legacy { .. } => Ok(()),
        CudaExport::Vmm { alloc_size, .. } => {
            if *alloc_size == 0 {
                return Err(HandleError::ZeroAllocSize);
            }
            if byte_len > *alloc_size {
                return Err(HandleError::AllocTooSmall {
                    byte_len,
                    alloc_size: *alloc_size,
                });
            }
            Ok(())
        }
    }
}

fn write_export(out: &mut Vec<u8>, export: &CudaExport) {
    out.push(export.tag());
    match export {
        CudaExport::Legacy { mem_handle } => out.extend_from_slice(mem_handle),
        CudaExport::Vmm {
            fd_slot,
            alloc_size,
        } => {
            out.extend_from_slice(&fd_slot.to_le_bytes());
            out.extend_from_slice(&alloc_size.to_le_bytes());
        }
    }
}

fn read_export(reader: &mut Reader<'_>) -> Result<CudaExport, HandleError> {
    match reader.u8()? {
        0 => Ok(CudaExport::Legacy {
            mem_handle: reader.array64()?,
        }),
        1 => Ok(CudaExport::Vmm {
            fd_slot: reader.u32()?,
            alloc_size: reader.u64()?,
        }),
        other => Err(HandleError::UnknownExport(other)),
    }
}

fn check_identity(identity: &str) -> Result<(), HandleError> {
    if identity.is_empty() {
        return Err(HandleError::BadIdentity);
    }
    if identity.len() > MAX_IDENTITY_BYTES {
        return Err(HandleError::OverBound("identity"));
    }
    Ok(())
}

fn check_element(element: &ChainElement) -> Result<(), HandleError> {
    check_identity(&element.identity)
}

fn check_chain(chain: &UpstreamChain) -> Result<(), HandleError> {
    check_element(&chain.stage)?;
    check_element(&chain.base_model)?;
    for list in [&chain.adapters, &chain.experts] {
        if list.len() > MAX_CHAIN_ELEMENTS {
            return Err(HandleError::OverBound("chain elements"));
        }
        for element in list {
            check_element(element)?;
        }
    }
    Ok(())
}

fn read_identity(reader: &mut Reader<'_>) -> Result<String, HandleError> {
    let length = reader.u32()? as usize;
    if length == 0 {
        return Err(HandleError::BadIdentity);
    }
    if length > MAX_IDENTITY_BYTES {
        return Err(HandleError::OverBound("identity"));
    }
    Ok(std::str::from_utf8(reader.take(length)?)
        .map_err(|_| HandleError::BadIdentity)?
        .to_owned())
}

fn read_element(reader: &mut Reader<'_>) -> Result<ChainElement, HandleError> {
    Ok(ChainElement {
        identity: read_identity(reader)?,
        generation: reader.u64()?,
        digest: reader.array32()?,
        fence_epoch: reader.u64()?,
    })
}

fn read_chain(reader: &mut Reader<'_>) -> Result<UpstreamChain, HandleError> {
    let stage = read_element(reader)?;
    let base_model = read_element(reader)?;
    let mut lists: [Vec<ChainElement>; 2] = [Vec::new(), Vec::new()];
    for list in &mut lists {
        let count = reader.u32()? as usize;
        if count > MAX_CHAIN_ELEMENTS {
            return Err(HandleError::OverBound("chain elements"));
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

    fn take(&mut self, count: usize) -> Result<&'a [u8], HandleError> {
        if self.rest.len() < count {
            return Err(HandleError::Truncated);
        }
        let (head, tail) = self.rest.split_at(count);
        self.rest = tail;
        Ok(head)
    }

    fn u8(&mut self) -> Result<u8, HandleError> {
        Ok(self.take(1)?[0])
    }

    fn u32(&mut self) -> Result<u32, HandleError> {
        let mut array = [0u8; 4];
        array.copy_from_slice(self.take(4)?);
        Ok(u32::from_le_bytes(array))
    }

    fn u64(&mut self) -> Result<u64, HandleError> {
        let mut array = [0u8; 8];
        array.copy_from_slice(self.take(8)?);
        Ok(u64::from_le_bytes(array))
    }

    fn array32(&mut self) -> Result<[u8; 32], HandleError> {
        let mut array = [0u8; 32];
        array.copy_from_slice(self.take(32)?);
        Ok(array)
    }

    fn array64(&mut self) -> Result<[u8; 64], HandleError> {
        let mut array = [0u8; 64];
        array.copy_from_slice(self.take(64)?);
        Ok(array)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const CARRIERS: [Carrier; 8] = [
        Carrier::InProcess,
        Carrier::Bytes,
        Carrier::HostShm,
        Carrier::CudaIpc,
        Carrier::DevicePeer,
        Carrier::PinnedHost,
        Carrier::NvmeFile,
        Carrier::Remote,
    ];

    /// Offset of the dtype byte of a `HostShm` handle: magic, carrier, locator tag, fd_slot,
    /// offset.
    const SHM_DTYPE_AT: usize = 8 + 1 + 1 + 4 + 8;

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

    fn handle_with(locator: CarrierLocator) -> TensorHandle {
        let inline_len = match &locator {
            CarrierLocator::Bytes { inline } => Some(inline.len() as u64),
            CarrierLocator::HostShm { .. }
            | CarrierLocator::CudaIpc { .. }
            | CarrierLocator::DevicePeer { .. }
            | CarrierLocator::PinnedHost { .. }
            | CarrierLocator::NvmeFile { .. }
            | CarrierLocator::Remote { .. }
            | CarrierLocator::InProcess => None,
        };
        let digest = match &locator {
            CarrierLocator::Bytes { inline } => content_digest(inline),
            CarrierLocator::HostShm { .. }
            | CarrierLocator::CudaIpc { .. }
            | CarrierLocator::DevicePeer { .. }
            | CarrierLocator::PinnedHost { .. }
            | CarrierLocator::NvmeFile { .. }
            | CarrierLocator::Remote { .. }
            | CarrierLocator::InProcess => [4; 32],
        };
        TensorHandle {
            carrier: locator.carrier(),
            locator,
            dtype: BoundaryDtype::F32,
            shape: vec![2, inline_len.map_or(3, |len| len / 8)],
            byte_len: inline_len.unwrap_or(24),
            upstream_chain: chain(),
            placement_epoch: 7,
            signed_absolute_deadline_ms: 5_000,
            handle_nonce: 11,
            parent_operation_id: 21,
            consumer: "stage-b".to_owned(),
            payload_type: TypeId::from("ptr.split.activation.v1"),
            content_digest: digest,
            signature: [3; 64],
        }
    }

    fn shm() -> TensorHandle {
        handle_with(CarrierLocator::HostShm {
            fd_slot: 4,
            offset: 8,
        })
    }

    fn every_locator() -> Vec<CarrierLocator> {
        vec![
            CarrierLocator::Bytes {
                inline: (0..24).collect(),
            },
            CarrierLocator::HostShm {
                fd_slot: 4,
                offset: 8,
            },
            CarrierLocator::CudaIpc {
                device: 1,
                export: CudaExport::Legacy {
                    mem_handle: [5; 64],
                },
                event_handle: [6; 64],
            },
            CarrierLocator::CudaIpc {
                device: 1,
                export: CudaExport::Vmm {
                    fd_slot: 3,
                    alloc_size: 4096,
                },
                event_handle: [6; 64],
            },
            CarrierLocator::DevicePeer {
                source_device: 2,
                export: CudaExport::Legacy {
                    mem_handle: [7; 64],
                },
                event_handle: [6; 64],
            },
            CarrierLocator::DevicePeer {
                source_device: 2,
                export: CudaExport::Vmm {
                    fd_slot: 4,
                    alloc_size: 24,
                },
                event_handle: [6; 64],
            },
            CarrierLocator::PinnedHost {
                region: 5,
                offset: 16,
            },
            CarrierLocator::NvmeFile {
                chunk_digest: [7; 32],
            },
            CarrierLocator::Remote {
                frame_id: 3,
                frame_digest: [8; 32],
                chunk_total: 4,
            },
        ]
    }

    /// The wire bytes of a handle without the validation `encode` does, for corrupt inputs.
    fn raw(handle: &TensorHandle) -> Vec<u8> {
        let mut out = HANDLE_MAGIC.to_vec();
        write_body(&mut out, handle);
        out.extend_from_slice(&handle.signature);
        out
    }

    #[test]
    fn carriers_are_ordered_nearest_to_farthest_with_unique_stable_codes() {
        for pair in CARRIERS.windows(2) {
            assert!(pair[0] < pair[1]);
        }
        let codes: Vec<u8> = CARRIERS.iter().map(|carrier| carrier.code()).collect();
        assert_eq!(codes, vec![1, 2, 3, 4, 5, 6, 7, 8]);
        for carrier in CARRIERS {
            assert_eq!(Carrier::from_code(carrier.code()), Ok(carrier));
        }
        assert_eq!(Carrier::from_code(0), Err(HandleError::UnknownCarrier(0)));
        assert_eq!(Carrier::from_code(9), Err(HandleError::UnknownCarrier(9)));
    }

    #[test]
    fn a_handle_of_every_serialisable_carrier_round_trips() {
        for locator in every_locator() {
            let handle = handle_with(locator);
            let bytes = handle.encode().expect("a well-formed handle encodes");
            assert_eq!(TensorHandle::decode(&bytes), Ok(handle.clone()));
            let payload = handle_payload(&handle).expect("payload");
            assert_eq!(payload.type_id, TypeId::from(TENSOR_HANDLE_TYPE));
            assert_eq!(payload.bytes, bytes);
        }
    }

    #[test]
    fn a_handle_of_the_in_process_carrier_never_encodes_or_decodes() {
        let mut handle = shm();
        handle.carrier = Carrier::InProcess;
        assert_eq!(handle.encode(), Err(HandleError::InProcessNotEncodable));
        assert_eq!(
            handle_payload(&handle),
            Err(HandleError::InProcessNotEncodable)
        );
        // The code of InProcess on the wire is refused too.
        let mut bytes = shm().encode().expect("encodes");
        bytes[8] = Carrier::InProcess.code();
        assert_eq!(
            TensorHandle::decode(&bytes),
            Err(HandleError::InProcessNotEncodable)
        );
    }

    #[test]
    fn a_locator_of_another_carrier_is_refused_on_encode_and_decode() {
        let mut handle = shm();
        handle.carrier = Carrier::Remote;
        assert_eq!(
            handle.encode(),
            Err(HandleError::LocatorMismatch {
                carrier: Carrier::Remote,
                locator: Carrier::HostShm,
            })
        );
        let mut bytes = shm().encode().expect("encodes");
        bytes[9] = Carrier::NvmeFile.code();
        assert_eq!(
            TensorHandle::decode(&bytes),
            Err(HandleError::LocatorMismatch {
                carrier: Carrier::HostShm,
                locator: Carrier::NvmeFile,
            })
        );
        bytes[9] = Carrier::InProcess.code();
        assert_eq!(
            TensorHandle::decode(&bytes),
            Err(HandleError::LocatorMismatch {
                carrier: Carrier::HostShm,
                locator: Carrier::InProcess,
            })
        );
    }

    #[test]
    fn framing_errors_are_refused_whole() {
        let bytes = shm().encode().expect("encodes");
        let mut magic = bytes.clone();
        magic[0] ^= 1;
        assert_eq!(TensorHandle::decode(&magic), Err(HandleError::BadMagic));
        assert_eq!(TensorHandle::decode(&[]), Err(HandleError::Truncated));
        assert_eq!(
            TensorHandle::decode(&bytes[..bytes.len() - 1]),
            Err(HandleError::Truncated)
        );
        let mut trailing = bytes.clone();
        trailing.push(0);
        assert_eq!(
            TensorHandle::decode(&trailing),
            Err(HandleError::TrailingBytes)
        );
        let mut carrier = bytes.clone();
        carrier[8] = 99;
        assert_eq!(
            TensorHandle::decode(&carrier),
            Err(HandleError::UnknownCarrier(99))
        );
        let mut dtype = bytes.clone();
        dtype[SHM_DTYPE_AT] = 9;
        assert_eq!(
            TensorHandle::decode(&dtype),
            Err(HandleError::UnknownDtype(9))
        );
        // An activation chunk or frame is not a handle.
        let mut other = bytes;
        other[..8].copy_from_slice(b"PTRACT01");
        assert_eq!(TensorHandle::decode(&other), Err(HandleError::BadMagic));
    }

    #[test]
    fn the_dimension_count_is_bounded_before_the_dimensions_are_read() {
        let mut bytes = shm().encode().expect("encodes");
        let dims_at = SHM_DTYPE_AT + 1;
        bytes[dims_at..dims_at + 4].copy_from_slice(&(MAX_HANDLE_DIMS as u32 + 1).to_le_bytes());
        assert_eq!(
            TensorHandle::decode(&bytes),
            Err(HandleError::OverBound("dims"))
        );
        bytes[dims_at..dims_at + 4].copy_from_slice(&0u32.to_le_bytes());
        assert_eq!(TensorHandle::decode(&bytes), Err(HandleError::EmptyShape));
    }

    #[test]
    fn shape_and_byte_len_must_agree() {
        let mut handle = shm();
        handle.byte_len = 23;
        let refusal = HandleError::ShapeMismatch {
            expected: 24,
            found: 23,
        };
        assert_eq!(handle.encode(), Err(refusal.clone()));
        assert_eq!(TensorHandle::decode(&raw(&handle)), Err(refusal));

        let mut overflow = shm();
        overflow.shape = vec![u64::MAX, 2];
        assert_eq!(overflow.encode(), Err(HandleError::ShapeOverflow));
        assert_eq!(
            TensorHandle::decode(&raw(&overflow)),
            Err(HandleError::ShapeOverflow)
        );

        let mut zero = shm();
        zero.shape = vec![2, 0];
        zero.byte_len = 0;
        assert_eq!(zero.encode(), Err(HandleError::ZeroDimension));

        let mut empty = shm();
        empty.shape = vec![];
        assert_eq!(empty.encode(), Err(HandleError::EmptyShape));
    }

    #[test]
    fn dims_and_byte_len_over_their_bounds_are_refused() {
        let mut many = shm();
        many.shape = vec![1; MAX_HANDLE_DIMS + 1];
        many.byte_len = 4;
        assert_eq!(many.encode(), Err(HandleError::OverBound("dims")));

        let mut full = shm();
        full.shape = vec![1; MAX_HANDLE_DIMS];
        full.byte_len = 4;
        assert_eq!(
            TensorHandle::decode(&full.encode().expect("at the bound")),
            Ok(full)
        );

        let mut large = shm();
        large.shape = vec![MAX_HANDLE_BYTES / 4 + 1];
        large.byte_len = MAX_HANDLE_BYTES + 4;
        assert_eq!(large.encode(), Err(HandleError::OverBound("byte_len")));
        assert_eq!(
            TensorHandle::decode(&raw(&large)),
            Err(HandleError::OverBound("byte_len"))
        );

        let mut at_bound = shm();
        at_bound.shape = vec![MAX_HANDLE_BYTES / 4];
        at_bound.byte_len = MAX_HANDLE_BYTES;
        assert_eq!(at_bound.validate(), Ok(()));
    }

    #[test]
    fn an_inline_payload_must_match_byte_len_and_stay_within_its_bound() {
        let mut handle = handle_with(CarrierLocator::Bytes {
            inline: vec![1; 24],
        });
        handle.byte_len = 24;
        handle.shape = vec![6];
        assert_eq!(handle.validate(), Ok(()));
        handle.locator = CarrierLocator::Bytes {
            inline: vec![1; 20],
        };
        let refusal = HandleError::InlineLength {
            expected: 24,
            found: 20,
        };
        assert_eq!(handle.encode(), Err(refusal.clone()));
        assert_eq!(TensorHandle::decode(&raw(&handle)), Err(refusal));

        let inline = MAX_INLINE_BYTES as usize + 4;
        let big = TensorHandle {
            shape: vec![inline as u64 / 4],
            byte_len: inline as u64,
            locator: CarrierLocator::Bytes {
                inline: vec![0; inline],
            },
            ..handle
        };
        assert_eq!(big.encode(), Err(HandleError::OverBound("inline")));
        assert_eq!(
            TensorHandle::decode(&raw(&big)),
            Err(HandleError::OverBound("inline"))
        );
    }

    #[test]
    fn an_empty_identity_and_an_oversize_chain_are_refused() {
        let mut bad = shm();
        bad.upstream_chain.stage.identity.clear();
        assert_eq!(bad.encode(), Err(HandleError::BadIdentity));
        assert_eq!(
            TensorHandle::decode(&raw(&bad)),
            Err(HandleError::BadIdentity)
        );
        let mut long = shm();
        long.upstream_chain.base_model.identity = "x".repeat(MAX_IDENTITY_BYTES + 1);
        assert_eq!(long.encode(), Err(HandleError::OverBound("identity")));
        let mut many = shm();
        many.upstream_chain.experts = vec![element("e", 1); MAX_CHAIN_ELEMENTS + 1];
        assert_eq!(many.encode(), Err(HandleError::OverBound("chain elements")));
        assert_eq!(
            TensorHandle::decode(&raw(&many)),
            Err(HandleError::OverBound("chain elements"))
        );
    }

    #[test]
    fn the_signing_bytes_bind_every_field_but_the_signature() {
        let base = shm();
        let signed = handle_signing_bytes(&base);
        let mut resigned = base.clone();
        resigned.signature = [9; 64];
        assert_eq!(handle_signing_bytes(&resigned), signed);

        let changes: Vec<(&str, TensorHandle)> = vec![
            (
                "carrier",
                TensorHandle {
                    carrier: Carrier::Remote,
                    ..base.clone()
                },
            ),
            (
                "locator fd_slot",
                TensorHandle {
                    locator: CarrierLocator::HostShm {
                        fd_slot: 5,
                        offset: 8,
                    },
                    ..base.clone()
                },
            ),
            (
                "locator offset",
                TensorHandle {
                    locator: CarrierLocator::HostShm {
                        fd_slot: 4,
                        offset: 9,
                    },
                    ..base.clone()
                },
            ),
            (
                "locator kind",
                TensorHandle {
                    locator: CarrierLocator::PinnedHost {
                        region: 4,
                        offset: 8,
                    },
                    ..base.clone()
                },
            ),
            (
                "shape",
                TensorHandle {
                    shape: vec![3, 2],
                    ..base.clone()
                },
            ),
            (
                "byte_len",
                TensorHandle {
                    byte_len: 25,
                    ..base.clone()
                },
            ),
            (
                "chain",
                TensorHandle {
                    upstream_chain: UpstreamChain {
                        stage: element("stage-a", 2),
                        ..chain()
                    },
                    ..base.clone()
                },
            ),
            (
                "placement_epoch",
                TensorHandle {
                    placement_epoch: 8,
                    ..base.clone()
                },
            ),
            (
                "deadline",
                TensorHandle {
                    signed_absolute_deadline_ms: 5_001,
                    ..base.clone()
                },
            ),
            (
                "nonce",
                TensorHandle {
                    handle_nonce: 12,
                    ..base.clone()
                },
            ),
        ];
        let mut other_operation = base.clone();
        other_operation.parent_operation_id += 1;
        assert_ne!(handle_signing_bytes(&other_operation), signed, "operation");
        let mut other_type = base.clone();
        other_type.payload_type = TypeId::from("ptr.split.activation.v2");
        assert_ne!(handle_signing_bytes(&other_type), signed, "payload_type");
        let mut other_consumer = base.clone();
        other_consumer.consumer = "stage-c".to_owned();
        assert_ne!(handle_signing_bytes(&other_consumer), signed, "consumer");
        for (field, changed) in changes {
            assert_ne!(handle_signing_bytes(&changed), signed, "{field}");
        }
        // The CUDA IPC handles and the Bytes payload are bound as well.
        let cuda = |mem: u8, event: u8| {
            handle_with(CarrierLocator::CudaIpc {
                device: 1,
                export: CudaExport::Legacy {
                    mem_handle: [mem; 64],
                },
                event_handle: [event; 64],
            })
        };
        assert_ne!(
            handle_signing_bytes(&cuda(1, 2)),
            handle_signing_bytes(&cuda(3, 2))
        );
        assert_ne!(
            handle_signing_bytes(&cuda(1, 2)),
            handle_signing_bytes(&cuda(1, 3))
        );
        let inline = |fill: u8| {
            handle_with(CarrierLocator::Bytes {
                inline: vec![fill; 24],
            })
        };
        assert_ne!(
            handle_signing_bytes(&inline(1)),
            handle_signing_bytes(&inline(2))
        );
        // The dtype has one variant (F32), so its code is bound but cannot be varied here.
    }

    fn cuda(export: CudaExport) -> TensorHandle {
        handle_with(CarrierLocator::CudaIpc {
            device: 1,
            export,
            event_handle: [6; 64],
        })
    }

    fn peer(export: CudaExport) -> TensorHandle {
        handle_with(CarrierLocator::DevicePeer {
            source_device: 1,
            export,
            event_handle: [6; 64],
        })
    }

    #[test]
    fn a_cuda_export_is_a_closed_tagged_enum_with_a_stable_wire_form() {
        let legacy = CudaExport::Legacy {
            mem_handle: [1; 64],
        };
        let vmm = CudaExport::Vmm {
            fd_slot: 2,
            alloc_size: 24,
        };
        assert_eq!((legacy.tag(), vmm.tag()), (0, 1));
        for make in [cuda, peer] {
            for export in [legacy.clone(), vmm.clone()] {
                let handle = make(export);
                let bytes = handle.encode().expect("encodes");
                assert_eq!(TensorHandle::decode(&bytes), Ok(handle));
            }
        }
    }

    #[test]
    fn an_unknown_export_tag_and_trailing_bytes_are_refused() {
        let bytes = cuda(CudaExport::Vmm {
            fd_slot: 2,
            alloc_size: 24,
        })
        .encode()
        .expect("encodes");
        // magic(8), carrier(1), locator tag(1), device(4), export tag.
        let tag_at = 8 + 1 + 1 + 4;
        assert_eq!(bytes[tag_at], 1);
        let mut unknown = bytes.clone();
        unknown[tag_at] = 2;
        assert_eq!(
            TensorHandle::decode(&unknown),
            Err(HandleError::UnknownExport(2))
        );
        let mut trailing = bytes;
        trailing.push(0);
        assert_eq!(
            TensorHandle::decode(&trailing),
            Err(HandleError::TrailingBytes)
        );
    }

    #[test]
    fn a_vmm_export_needs_an_allocation_that_holds_the_tensor() {
        for make in [cuda, peer] {
            let zero = make(CudaExport::Vmm {
                fd_slot: 1,
                alloc_size: 0,
            });
            assert_eq!(zero.encode(), Err(HandleError::ZeroAllocSize));
            assert_eq!(
                TensorHandle::decode(&raw(&zero)),
                Err(HandleError::ZeroAllocSize)
            );
            let small = make(CudaExport::Vmm {
                fd_slot: 1,
                alloc_size: 23,
            });
            let refusal = HandleError::AllocTooSmall {
                byte_len: 24,
                alloc_size: 23,
            };
            assert_eq!(small.encode(), Err(refusal.clone()));
            assert_eq!(TensorHandle::decode(&raw(&small)), Err(refusal));
            let exact = make(CudaExport::Vmm {
                fd_slot: 1,
                alloc_size: 24,
            });
            assert_eq!(exact.validate(), Ok(()));
        }
    }

    #[test]
    fn the_signing_bytes_bind_the_export_kind_and_its_fields() {
        let legacy = cuda(CudaExport::Legacy {
            mem_handle: [1; 64],
        });
        let vmm = |fd_slot: u32, alloc_size: u64| {
            cuda(CudaExport::Vmm {
                fd_slot,
                alloc_size,
            })
        };
        let base = handle_signing_bytes(&vmm(2, 24));
        assert_ne!(handle_signing_bytes(&legacy), base, "export kind");
        assert_ne!(handle_signing_bytes(&vmm(3, 24)), base, "fd_slot");
        assert_ne!(handle_signing_bytes(&vmm(2, 48)), base, "alloc_size");
        // CudaIpc and DevicePeer with the same export differ (the carrier is signed).
        assert_ne!(
            handle_signing_bytes(&cuda(CudaExport::Legacy {
                mem_handle: [1; 64]
            })),
            handle_signing_bytes(&peer(CudaExport::Legacy {
                mem_handle: [1; 64]
            }))
        );
    }

    #[test]
    fn the_debug_output_of_a_cuda_handle_prints_no_capability_bytes() {
        let secret_mem = [0xAB_u8; 64];
        let secret_event = [0xCD_u8; 64];
        let cases = [
            (
                "CudaIpc",
                "Legacy",
                handle_with(CarrierLocator::CudaIpc {
                    device: 1,
                    export: CudaExport::Legacy {
                        mem_handle: secret_mem,
                    },
                    event_handle: secret_event,
                }),
            ),
            (
                "DevicePeer",
                "Legacy",
                handle_with(CarrierLocator::DevicePeer {
                    source_device: 1,
                    export: CudaExport::Legacy {
                        mem_handle: secret_mem,
                    },
                    event_handle: secret_event,
                }),
            ),
            (
                "CudaIpc",
                "Vmm",
                handle_with(CarrierLocator::CudaIpc {
                    device: 1,
                    export: CudaExport::Vmm {
                        fd_slot: 2,
                        alloc_size: 24,
                    },
                    event_handle: secret_event,
                }),
            ),
        ];
        for (carrier, kind, handle) in cases {
            for shown in [format!("{handle:?}"), format!("{:?}", handle.locator)] {
                // 0xAB is 171 and 0xCD is 205 in a derived `[u8; 64]` Debug.
                assert!(!shown.contains("171"), "{shown}");
                assert!(!shown.contains("205"), "{shown}");
                assert!(!shown.contains("mem_handle"), "{shown}");
                assert!(shown.contains("event_handle: .."), "{shown}");
                assert!(shown.contains(carrier), "{shown}");
                assert!(shown.contains(kind), "{shown}");
            }
        }
        let vmm = format!(
            "{:?}",
            CudaExport::Vmm {
                fd_slot: 2,
                alloc_size: 24
            }
        );
        assert_eq!(vmm, "Vmm { fd_slot: 2, alloc_size: 24 }");
        assert_eq!(
            format!(
                "{:?}",
                CudaExport::Legacy {
                    mem_handle: secret_mem
                }
            ),
            "Legacy { .. }"
        );
    }

    #[test]
    fn the_signing_bytes_bind_the_devices_and_the_event_handle_for_both_export_kinds() {
        let exports = [
            CudaExport::Legacy {
                mem_handle: [1; 64],
            },
            CudaExport::Vmm {
                fd_slot: 2,
                alloc_size: 24,
            },
        ];
        for export in exports {
            let ipc = |device: u32, event: u8| {
                handle_signing_bytes(&handle_with(CarrierLocator::CudaIpc {
                    device,
                    export: export.clone(),
                    event_handle: [event; 64],
                }))
            };
            let peer = |source_device: u32, event: u8| {
                handle_signing_bytes(&handle_with(CarrierLocator::DevicePeer {
                    source_device,
                    export: export.clone(),
                    event_handle: [event; 64],
                }))
            };
            assert_ne!(ipc(1, 6), ipc(2, 6), "CudaIpc device");
            assert_ne!(ipc(1, 6), ipc(1, 7), "CudaIpc event_handle");
            assert_ne!(peer(1, 6), peer(2, 6), "DevicePeer source_device");
            assert_ne!(peer(1, 6), peer(1, 7), "DevicePeer event_handle");
        }
    }

    #[test]
    fn a_remote_locator_names_the_whole_frame_and_bounds_its_chunk_count() {
        let remote = |frame_id: u64, digest: u8, chunk_total: u32| {
            handle_with(CarrierLocator::Remote {
                frame_id,
                frame_digest: [digest; 32],
                chunk_total,
            })
        };
        assert_eq!(remote(3, 8, MAX_CHUNKS_PER_FRAME).validate(), Ok(()));
        assert_eq!(remote(3, 8, 0).encode(), Err(HandleError::ZeroChunkTotal));
        assert_eq!(
            remote(3, 8, MAX_CHUNKS_PER_FRAME + 1).encode(),
            Err(HandleError::OverBound("chunk_total"))
        );
        assert_eq!(
            TensorHandle::decode(&raw(&remote(3, 8, 0))),
            Err(HandleError::ZeroChunkTotal)
        );
        let base = handle_signing_bytes(&remote(3, 8, 4));
        assert_ne!(handle_signing_bytes(&remote(4, 8, 4)), base, "frame_id");
        assert_ne!(handle_signing_bytes(&remote(3, 9, 4)), base, "frame_digest");
        assert_ne!(handle_signing_bytes(&remote(3, 8, 5)), base, "chunk_total");
    }

    #[test]
    fn operation_and_consumer_round_trip_and_are_bounded() {
        let handle = shm();
        let decoded = TensorHandle::decode(&handle.encode().expect("encodes")).expect("decodes");
        assert_eq!(decoded.parent_operation_id, 21);
        assert_eq!(decoded.consumer, "stage-b");
        let mut empty = shm();
        empty.consumer.clear();
        assert_eq!(empty.encode(), Err(HandleError::BadIdentity));
        let mut long = shm();
        long.consumer = "x".repeat(MAX_IDENTITY_BYTES + 1);
        assert_eq!(long.encode(), Err(HandleError::OverBound("identity")));
        assert_eq!(
            TensorHandle::decode(&raw(&empty)),
            Err(HandleError::BadIdentity)
        );
    }

    #[test]
    fn the_payload_type_round_trips_and_is_bounded() {
        let decoded = TensorHandle::decode(&shm().encode().expect("encodes")).expect("decodes");
        assert_eq!(
            decoded.payload_type,
            TypeId::from("ptr.split.activation.v1")
        );
        let mut empty = shm();
        empty.payload_type = TypeId::from("");
        assert_eq!(empty.encode(), Err(HandleError::BadIdentity));
        assert_eq!(
            TensorHandle::decode(&raw(&empty)),
            Err(HandleError::BadIdentity)
        );
        let mut long = shm();
        long.payload_type = TypeId("t".repeat(MAX_IDENTITY_BYTES + 1));
        assert_eq!(long.encode(), Err(HandleError::OverBound("identity")));
        assert_eq!(
            TensorHandle::decode(&raw(&long)),
            Err(HandleError::OverBound("identity"))
        );
    }

    #[test]
    fn an_inline_payload_is_checked_against_the_content_digest_at_validate_and_decode() {
        let mut handle = handle_with(CarrierLocator::Bytes {
            inline: vec![7; 24],
        });
        assert_eq!(handle.validate(), Ok(()));
        handle.content_digest[0] ^= 1;
        assert_eq!(handle.validate(), Err(HandleError::ContentMismatch));
        assert_eq!(handle.encode(), Err(HandleError::ContentMismatch));
        assert_eq!(
            TensorHandle::decode(&raw(&handle)),
            Err(HandleError::ContentMismatch)
        );
    }

    #[test]
    fn a_claimed_length_over_a_bound_is_refused_not_reported_as_truncated() {
        // A claimed inline length of u32::MAX with almost no bytes after it.
        let mut bytes = HANDLE_MAGIC.to_vec();
        bytes.push(Carrier::Bytes.code());
        bytes.push(Carrier::Bytes.code());
        bytes.extend_from_slice(&u32::MAX.to_le_bytes());
        assert_eq!(
            TensorHandle::decode(&bytes),
            Err(HandleError::OverBound("inline"))
        );
        // A claimed dimension count of 9 with the buffer cut right after it.
        let full = shm().encode().expect("encodes");
        let dims_at = SHM_DTYPE_AT + 1;
        let mut short = full[..dims_at + 4].to_vec();
        short[dims_at..dims_at + 4].copy_from_slice(&(MAX_HANDLE_DIMS as u32 + 1).to_le_bytes());
        assert_eq!(
            TensorHandle::decode(&short),
            Err(HandleError::OverBound("dims"))
        );
    }

    #[test]
    fn the_duplicated_dtype_code_is_the_activation_codecs_code() {
        use crate::activation::ActivationFrame;
        let frame = ActivationFrame {
            tokens: 1,
            width: 1,
            dtype: BoundaryDtype::F32,
            positions: vec![0],
            upstream_chain: chain(),
            placement_epoch: 0,
            tensor: vec![0; 4],
        };
        // Frame layout: magic(8), tokens(4), width(4), dtype(1).
        assert_eq!(frame.to_bytes()[16], dtype_code(BoundaryDtype::F32));
    }

    #[test]
    fn a_shape_whose_product_overflows_is_refused_without_wrapping() {
        let mut huge = shm();
        huge.shape = vec![u64::MAX / 2, 3];
        huge.byte_len = 12;
        assert_eq!(huge.validate(), Err(HandleError::ShapeOverflow));
        let mut times_dtype = shm();
        times_dtype.shape = vec![u64::MAX / 2];
        times_dtype.byte_len = 8;
        assert_eq!(times_dtype.validate(), Err(HandleError::ShapeOverflow));
    }

    #[test]
    fn the_content_digest_is_bound_by_the_signing_bytes_and_the_codec() {
        let base = shm();
        let mut changed = base.clone();
        changed.content_digest[0] ^= 1;
        assert_ne!(handle_signing_bytes(&changed), handle_signing_bytes(&base));
        assert_ne!(handle_digest(&changed), handle_digest(&base));
        let decoded = TensorHandle::decode(&changed.encode().expect("encodes")).expect("decodes");
        assert_eq!(decoded.content_digest, changed.content_digest);
    }

    #[test]
    fn verify_content_checks_length_and_digest() {
        let bytes: Vec<u8> = (0..24).collect();
        let mut handle = shm();
        handle.content_digest = content_digest(&bytes);
        assert_eq!(verify_content(&handle, &bytes), Ok(()));
        let mut flipped = bytes.clone();
        flipped[3] ^= 1;
        assert_eq!(
            verify_content(&handle, &flipped),
            Err(HandleError::ContentMismatch)
        );
        assert_eq!(
            verify_content(&handle, &bytes[..23]),
            Err(HandleError::ContentLength {
                expected: 24,
                found: 23
            })
        );
    }

    #[test]
    fn the_digest_binds_the_signature_and_the_signed_fields() {
        let base = shm();
        let digest = handle_digest(&base);
        assert_eq!(digest, handle_digest(&base.clone()));
        let mut resigned = base.clone();
        resigned.signature[0] ^= 1;
        assert_ne!(handle_digest(&resigned), digest);
        let mut moved = base;
        moved.handle_nonce += 1;
        assert_ne!(handle_digest(&moved), digest);
    }
}
