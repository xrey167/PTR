//! The `HostShm` carrier (doc 42 rule 3a): a dedicated `memfd` per export holding exactly the
//! tensor (offset 0, file size equal to `byte_len`, never pooled or sub-allocated), sealed
//! against writes and resizing before the handle is sent. The consumer checks the seals, the
//! offset and the file size before mapping and the digest after mapping and before use.
//!
//! Passing the descriptor to the consumer (`SCM_RIGHTS`) is not here.

use std::ffi::c_void;
use std::fs::File;
use std::io::Write;
use std::os::fd::{AsFd, BorrowedFd, OwnedFd};
use std::ptr::{self, NonNull};

use ptr_protocol::{content_digest, Carrier, CarrierLocator, TensorHandle, MAX_HANDLE_BYTES};
use rustix::fs::{fcntl_add_seals, fcntl_get_seals, fstat, memfd_create, MemfdFlags, SealFlags};
use rustix::mm::{mmap, munmap, MapFlags, ProtFlags};

use crate::{FabricError, Seal};

/// The seals in the order they are applied; `F_SEAL_SEAL` is last.
const SEALS: [(Seal, SealFlags); 4] = [
    (Seal::Write, SealFlags::WRITE),
    (Seal::Shrink, SealFlags::SHRINK),
    (Seal::Grow, SealFlags::GROW),
    (Seal::Seal, SealFlags::SEAL),
];

fn io_error(error: rustix::io::Errno) -> FabricError {
    FabricError::Io(error.to_string())
}

/// The page size of this host.
pub fn host_page_size() -> u64 {
    rustix::param::page_size() as u64
}

fn file_size(fd: &OwnedFd) -> Result<u64, FabricError> {
    let stat = fstat(fd).map_err(io_error)?;
    Ok(u64::try_from(stat.st_size).unwrap_or(0))
}

/// The producer side: one sealed memfd holding exactly one tensor.
#[derive(Debug)]
pub struct ShmExport {
    fd: OwnedFd,
    byte_len: u64,
    content_digest: [u8; 32],
}

impl ShmExport {
    /// Create a memfd (`MFD_CLOEXEC | MFD_ALLOW_SEALING`), write the tensor once, check the
    /// file size is exactly its length, and apply `F_SEAL_WRITE`, `F_SEAL_SHRINK`,
    /// `F_SEAL_GROW` and then `F_SEAL_SEAL`. Refuses an empty tensor and one above
    /// `MAX_HANDLE_BYTES`.
    pub fn create(bytes: &[u8]) -> Result<Self, FabricError> {
        let byte_len = bytes.len() as u64;
        if byte_len == 0 {
            return Err(FabricError::EmptyTensor);
        }
        if byte_len > MAX_HANDLE_BYTES {
            return Err(FabricError::TooLarge {
                len: byte_len,
                max: MAX_HANDLE_BYTES,
            });
        }
        let fd = memfd_create(
            "ptr-fabric-tensor",
            MemfdFlags::CLOEXEC | MemfdFlags::ALLOW_SEALING,
        )
        .map_err(io_error)?;
        let mut file = File::from(fd);
        file.write_all(bytes)
            .map_err(|error| FabricError::Io(error.to_string()))?;
        let fd = OwnedFd::from(file);
        let size = file_size(&fd)?;
        if size != byte_len {
            return Err(FabricError::LengthMismatch {
                expected: byte_len,
                found: size,
            });
        }
        for (_, flag) in SEALS {
            fcntl_add_seals(&fd, flag).map_err(io_error)?;
        }
        Ok(Self {
            fd,
            byte_len,
            content_digest: content_digest(bytes),
        })
    }

    /// The tensor length, which is also the file size.
    pub fn byte_len(&self) -> u64 {
        self.byte_len
    }

    /// SHA-256 of the tensor, for the handle's `content_digest`.
    pub fn content_digest(&self) -> [u8; 32] {
        self.content_digest
    }

    /// The locator of the handle. `fd_slot` is the descriptor slot the caller passes the
    /// descriptor in; the offset is always 0 because the region holds exactly the tensor.
    pub fn locator(&self, fd_slot: u32) -> CarrierLocator {
        CarrierLocator::HostShm { fd_slot, offset: 0 }
    }

    /// The bytes the region costs the host: the memfd rounded up to whole pages (doc 42 rule
    /// 3b), for `HandleAdmissionRequest::footprint_bytes`.
    pub fn footprint_bytes(&self) -> u64 {
        let page = host_page_size();
        self.byte_len.div_ceil(page).saturating_mul(page)
    }

    /// The descriptor, to pass to the consumer.
    pub fn fd(&self) -> BorrowedFd<'_> {
        self.fd.as_fd()
    }

    /// Give up the descriptor, for example to hand it to the passing code.
    pub fn into_fd(self) -> OwnedFd {
        self.fd
    }
}

/// What the supervisor learns about a region a producer sent it, before it admits the handle.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ShmRegion {
    /// The file size: exactly the tensor.
    pub byte_len: u64,
    /// The bytes the region costs the host: the file size rounded up to whole pages, for
    /// `HandleAdmissionRequest::footprint_bytes`. Measured on the descriptor, never read from
    /// the handle.
    pub footprint_bytes: u64,
}

/// Inspect a received region: it must be a `memfd` that carries all four seals, and its size
/// is measured. The supervisor calls this on the descriptor before it admits the handle, so
/// the footprint it charges is the real one and a writable or resizable region never reaches a
/// consumer.
pub fn inspect_region(fd: BorrowedFd<'_>) -> Result<ShmRegion, FabricError> {
    let held = fcntl_get_seals(fd).map_err(|error| match error {
        rustix::io::Errno::INVAL => FabricError::NotSealable,
        other => io_error(other),
    })?;
    for (seal, flag) in SEALS {
        if !held.contains(flag) {
            return Err(FabricError::MissingSeal(seal));
        }
    }
    let stat = fstat(fd).map_err(io_error)?;
    let byte_len = u64::try_from(stat.st_size).unwrap_or(0);
    if byte_len == 0 {
        return Err(FabricError::EmptyTensor);
    }
    let page = host_page_size();
    Ok(ShmRegion {
        byte_len,
        footprint_bytes: byte_len.div_ceil(page).saturating_mul(page),
    })
}

/// What the consumer expects of a region, taken from an admitted handle.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ShmExpectation {
    pub byte_len: u64,
    pub offset: u64,
    pub content_digest: [u8; 32],
}

impl ShmExpectation {
    /// Read the expectation from a `HostShm` handle; refuses another carrier or locator.
    pub fn from_handle(handle: &TensorHandle) -> Result<Self, FabricError> {
        if handle.carrier != Carrier::HostShm {
            return Err(FabricError::WrongCarrier);
        }
        let CarrierLocator::HostShm { offset, .. } = &handle.locator else {
            return Err(FabricError::WrongLocator);
        };
        if handle.byte_len > MAX_HANDLE_BYTES {
            return Err(FabricError::TooLarge {
                len: handle.byte_len,
                max: MAX_HANDLE_BYTES,
            });
        }
        Ok(Self {
            byte_len: handle.byte_len,
            offset: *offset,
            content_digest: handle.content_digest,
        })
    }
}

/// The consumer side: a read-only mapping of exactly `byte_len` bytes whose digest has been
/// verified. Unmapped on drop.
#[derive(Debug)]
pub struct ShmMapping {
    base: NonNull<c_void>,
    len: usize,
}

impl ShmMapping {
    /// Refuse a length above `MAX_HANDLE_BYTES` (before any system call), check the offset,
    /// the four seals and the file size, map `PROT_READ`, `MAP_SHARED`
    /// exactly `byte_len` bytes, and verify the digest of the mapped bytes. On any failure
    /// nothing stays mapped. The descriptor is closed afterwards; the mapping outlives it.
    pub fn map(fd: OwnedFd, expected: &ShmExpectation) -> Result<Self, FabricError> {
        if expected.byte_len == 0 {
            return Err(FabricError::EmptyTensor);
        }
        // Before any system call on the descriptor: an expectation built by hand (its fields
        // are public) cannot make the consumer size a mapping above the handle bound.
        if expected.byte_len > MAX_HANDLE_BYTES {
            return Err(FabricError::TooLarge {
                len: expected.byte_len,
                max: MAX_HANDLE_BYTES,
            });
        }
        if expected.offset != 0 {
            return Err(FabricError::NonZeroOffset {
                offset: expected.offset,
            });
        }
        // EINVAL: the descriptor does not support seals, so it is not a memfd.
        let held = fcntl_get_seals(&fd).map_err(|error| match error {
            rustix::io::Errno::INVAL => FabricError::NotSealable,
            other => io_error(other),
        })?;
        for (seal, flag) in SEALS {
            if !held.contains(flag) {
                return Err(FabricError::MissingSeal(seal));
            }
        }
        let size = file_size(&fd)?;
        if size != expected.byte_len {
            return Err(FabricError::LengthMismatch {
                expected: expected.byte_len,
                found: size,
            });
        }
        let len =
            usize::try_from(expected.byte_len).map_err(|_| FabricError::LengthNotAddressable {
                len: expected.byte_len,
            })?;
        // SAFETY: a null hint asks for a fresh mapping, so it cannot overlap existing memory;
        // the descriptor is open for the call, `len` is non-zero and equals the sealed file
        // size, and the mapping is read-only.
        let base = unsafe {
            mmap(
                ptr::null_mut(),
                len,
                ProtFlags::READ,
                MapFlags::SHARED,
                &fd,
                0,
            )
        }
        .map_err(io_error)?;
        let Some(base) = NonNull::new(base) else {
            return Err(FabricError::Io("mmap returned a null mapping".to_owned()));
        };
        // From here the guard unmaps on every early return.
        let mapping = Self { base, len };
        if content_digest(mapping.as_bytes()) != expected.content_digest {
            return Err(FabricError::DigestMismatch);
        }
        Ok(mapping)
    }

    /// The verified tensor bytes: exactly `byte_len` of them.
    pub fn as_bytes(&self) -> &[u8] {
        // SAFETY: `base` is a live read-only mapping of `len` bytes owned by `self`; the file
        // is sealed against writes and resizing, so the bytes cannot change or vanish while
        // the mapping lives, and the borrow cannot outlive `self`, which unmaps on drop.
        unsafe { std::slice::from_raw_parts(self.base.as_ptr().cast::<u8>(), self.len) }
    }

    /// The mapped length, equal to `byte_len`.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Always false: an empty tensor is refused.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl Drop for ShmMapping {
    fn drop(&mut self) {
        // SAFETY: `base` and `len` are exactly what `mmap` returned and was asked for, the
        // mapping has not been unmapped before, and no borrow of it can exist in `drop`.
        let _ = unsafe { munmap(self.base.as_ptr(), self.len) };
    }
}
