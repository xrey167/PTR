//! The `Bytes` carrier: the tensor travels inline in the handle (the universal fallback).

use ptr_protocol::{content_digest, Carrier, CarrierLocator, TensorHandle, MAX_INLINE_BYTES};

use crate::FabricError;

/// Export a tensor as an inline locator. Refuses an empty tensor and one above
/// `MAX_INLINE_BYTES`: a handle is a control message, and a larger tensor needs another
/// carrier.
pub fn export_bytes(bytes: &[u8]) -> Result<CarrierLocator, FabricError> {
    let len = bytes.len() as u64;
    if len == 0 {
        return Err(FabricError::EmptyTensor);
    }
    if len > MAX_INLINE_BYTES {
        return Err(FabricError::TooLarge {
            len,
            max: MAX_INLINE_BYTES,
        });
    }
    Ok(CarrierLocator::Bytes {
        inline: bytes.to_vec(),
    })
}

/// Import the bytes of a `Bytes` handle: the carrier and locator must be `Bytes`, the length
/// must equal `byte_len` and the SHA-256 must equal `content_digest`. The bytes are returned
/// only when all of that holds.
pub fn import_bytes(handle: &TensorHandle) -> Result<&[u8], FabricError> {
    if handle.carrier != Carrier::Bytes {
        return Err(FabricError::WrongCarrier);
    }
    let CarrierLocator::Bytes { inline } = &handle.locator else {
        return Err(FabricError::WrongLocator);
    };
    let found = inline.len() as u64;
    // Self-contained bound: a hand-built handle is not hashed beyond the inline limit.
    if found > MAX_INLINE_BYTES {
        return Err(FabricError::TooLarge {
            len: found,
            max: MAX_INLINE_BYTES,
        });
    }
    if found != handle.byte_len {
        return Err(FabricError::LengthMismatch {
            expected: handle.byte_len,
            found,
        });
    }
    if content_digest(inline) != handle.content_digest {
        return Err(FabricError::DigestMismatch);
    }
    Ok(inline)
}
