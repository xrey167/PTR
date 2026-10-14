//! The tensor object of an `InProcess` handle (doc 42 section 2, rule 3a).
//!
//! An [`InProcessHandle`](ptr_protocol::InProcessHandle) is a claim with no locator; the tensor
//! it claims moves **separately**, as a [`MovedTensor`]: one value, owned by exactly one side at
//! a time. It is the moved-ownership token. It is not `Clone`, not `Copy` and has no
//! serialisation, so the type system, not a convention, says that the producer no longer holds
//! the tensor once it has moved it and that two consumers cannot both receive it. A tensor whose
//! storage is shared (a candle tensor with shared storage, a DLPack capsule across a language
//! boundary) is not a [`MovedTensor`]: the producer copies it into a receiver-owned buffer
//! first, and that copy is what moves.
//!
//! [`MovedTensor::receive`] ties the two halves together after admission and consumption: the
//! consumed handle must be an `InProcess` one, and the tensor must have the signed length and
//! content digest. It decides nothing about the handle (that is `HandleTable`) and maps nothing.

use std::fmt;

use ptr_protocol::{content_digest, Carrier};

use crate::handle_admission::{ConsumedHandle, HandleRejection};

/// Why a moved tensor was not accepted. The tensor is dropped with the refusal: ownership had
/// moved, and nothing may use bytes that do not match the signed commitment.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TensorMoveRefused {
    /// The consumed handle names another carrier; only an `InProcess` handle claims a
    /// [`MovedTensor`].
    NotInProcess { carrier: Carrier },
    /// The tensor does not have the signed length or content digest.
    Content(HandleRejection),
}

/// A tensor with unique ownership, moved from a producer to a consumer inside one process.
///
/// There is no way to duplicate one:
///
/// ```compile_fail
/// let tensor = ptr_runtime::MovedTensor::new(vec![0; 8]);
/// let _copy = tensor.clone();
/// ```
#[must_use = "a moved tensor is the only copy; receive it or drop it on purpose"]
#[derive(Eq, PartialEq)]
pub struct MovedTensor {
    bytes: Vec<u8>,
}

impl fmt::Debug for MovedTensor {
    /// The length only: the bytes are tensor data.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("MovedTensor")
            .field("len", &self.bytes.len())
            .finish_non_exhaustive()
    }
}

impl MovedTensor {
    /// Take ownership of `bytes`, the dense row-major bytes the handle will describe.
    pub fn new(bytes: Vec<u8>) -> Self {
        Self { bytes }
    }

    pub fn len(&self) -> usize {
        self.bytes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }

    /// The SHA-256 the producer signs as `content_digest` of the handle.
    pub fn content_digest(&self) -> [u8; 32] {
        content_digest(&self.bytes)
    }

    /// Hand the tensor to the consumer of `consumed`: the handle must be an `InProcess` one and
    /// the bytes must have its signed length and digest. On success the consumer owns the bytes
    /// and the caller finishes the handle (`HandleTable::finish`), which releases the claim.
    pub fn receive(self, consumed: &ConsumedHandle) -> Result<Vec<u8>, TensorMoveRefused> {
        let carrier = consumed.handle().carrier;
        match carrier {
            Carrier::InProcess => {}
            Carrier::Bytes
            | Carrier::HostShm
            | Carrier::CudaIpc
            | Carrier::DevicePeer
            | Carrier::PinnedHost
            | Carrier::NvmeFile
            | Carrier::Remote => return Err(TensorMoveRefused::NotInProcess { carrier }),
        }
        consumed
            .verify_content(&self.bytes)
            .map_err(TensorMoveRefused::Content)?;
        Ok(self.bytes)
    }
}
