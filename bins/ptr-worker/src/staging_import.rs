//! The worker side of a supervisor-owned staging buffer (doc 42 rules 3a and 3b): what a producer
//! worker does with the descriptor it is given to write the tensor through, and what a consumer
//! worker does with the one it is given to read it. Linux only, feature `data-plane`.
//!
//! Two imports exist, chosen by `ptr-worker serve --staging fake|cuda` (a
//! [`BackendChoice`](crate::guardian::BackendChoice), the guardian's own, so that the worker
//! imports what the guardian exports):
//!
//! * **Fake** (the guardian's `--backend fake`): the staging buffer is a `memfd` of exactly the
//!   recorded footprint. The producer maps it `PROT_READ | PROT_WRITE`, copies the tensor in and
//!   unmaps; the consumer maps it `PROT_READ`, copies `byte_len` bytes into a `Vec` of its own,
//!   unmaps, and only then is the digest computed (on the copy, never on the shared bytes). This
//!   is the CPU image of the contract of rule 3a: no seal exists on a staging buffer, so the
//!   shared bytes are never used directly.
//! * **Cuda** (the guardian's `--backend cuda`): the consumer is `ptr_fabric::import_and_copy`
//!   (feature `cuda-ipc-core`). The producer side, writing through an imported VMM descriptor,
//!   does not exist in `ptr-fabric` yet, so a CUDA producer refuses with a typed code. Nothing
//!   here has run on a GPU.
//!
//! The shared mapping is read and written through raw pointers and `copy_nonoverlapping`, never
//! through a `&[u8]` over it: another process may change the bytes while they are copied, which a
//! shared reference may not observe.

use std::ffi::c_void;
use std::fs::File;
use std::os::fd::{AsRawFd, OwnedFd};
use std::ptr::{self, NonNull};

use ptr_protocol::MAX_HANDLE_BYTES;

/// Slack above `MAX_HANDLE_BYTES` for the rounding of a footprint up to the allocation unit.
/// It assumes a VMM granularity (or legacy staging unit) of at most 4 MiB; 2 MiB is what the
/// drivers report today. A device with a larger granularity needs this raised, or its largest
/// tensors are refused as `staging-bad-alloc-size`.
const MAX_ROUNDING_BYTES: u64 = 4 << 20;

/// Why an import failed. Every variant becomes a short typed code (`code`); nothing panics.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum StagingImportError {
    /// The tensor is empty or longer than a handle may be.
    BadLength { byte_len: u64 },
    /// The recorded allocation size is shorter than the tensor or above the bound.
    BadAllocSize { alloc_size: u64, byte_len: u64 },
    /// The descriptor's size is not the recorded allocation size.
    SizeMismatch { recorded: u64, found: u64 },
    /// The size query on the descriptor failed.
    Stat(String),
    /// `mmap` failed.
    Map(String),
    /// The mapping length does not fit this host.
    NotAddressable { len: u64 },
}

impl StagingImportError {
    /// A short code for a `Refused` reply.
    pub fn code(&self) -> String {
        match self {
            Self::BadLength { .. } => "staging-bad-length".to_owned(),
            Self::BadAllocSize { .. } => "staging-bad-alloc-size".to_owned(),
            Self::SizeMismatch { .. } => "staging-size-mismatch".to_owned(),
            Self::Stat(_) => "staging-stat-failed".to_owned(),
            Self::Map(_) => "staging-map-failed".to_owned(),
            Self::NotAddressable { .. } => "staging-not-addressable".to_owned(),
        }
    }
}

impl std::fmt::Display for StagingImportError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for StagingImportError {}

/// A shared mapping of the first `len` bytes of a staging descriptor, unmapped on drop.
struct Mapped {
    base: NonNull<c_void>,
    len: usize,
}

impl Mapped {
    fn new(fd: &OwnedFd, len: usize, writable: bool) -> Result<Self, StagingImportError> {
        let protection = if writable {
            libc::PROT_READ | libc::PROT_WRITE
        } else {
            libc::PROT_READ
        };
        // SAFETY: a null hint asks for a fresh mapping, so it cannot overlap existing memory;
        // `fd` is open for the call, `len` is non-zero and no longer than the descriptor's
        // size (checked by the caller), and offset 0 is page aligned. A failure is reported as
        // `MAP_FAILED`, which is checked below.
        let base = unsafe {
            libc::mmap(
                ptr::null_mut(),
                len,
                protection,
                libc::MAP_SHARED,
                fd.as_raw_fd(),
                0,
            )
        };
        if base == libc::MAP_FAILED {
            return Err(StagingImportError::Map(
                std::io::Error::last_os_error().to_string(),
            ));
        }
        let Some(base) = NonNull::new(base) else {
            return Err(StagingImportError::Map("a null mapping".to_owned()));
        };
        Ok(Self { base, len })
    }
}

impl Drop for Mapped {
    fn drop(&mut self) {
        // SAFETY: `base` and `len` are exactly what `mmap` returned and was asked for, the
        // mapping has not been unmapped before, and no pointer into it outlives `self`.
        let _ = unsafe { libc::munmap(self.base.as_ptr(), self.len) };
    }
}

/// The size of the descriptor, by `fstat`; the descriptor is handed back unchanged.
fn size_of_descriptor(fd: OwnedFd) -> Result<(OwnedFd, u64), StagingImportError> {
    let file = File::from(fd);
    let length = file
        .metadata()
        .map_err(|error| StagingImportError::Stat(error.to_string()))?
        .len();
    Ok((OwnedFd::from(file), length))
}

/// Check the facts both directions rely on and return the descriptor with the mapping length.
fn checked(
    fd: OwnedFd,
    alloc_size: u64,
    byte_len: u64,
) -> Result<(OwnedFd, usize), StagingImportError> {
    if byte_len == 0 || byte_len > MAX_HANDLE_BYTES {
        return Err(StagingImportError::BadLength { byte_len });
    }
    if alloc_size < byte_len || alloc_size > MAX_HANDLE_BYTES.saturating_add(MAX_ROUNDING_BYTES) {
        return Err(StagingImportError::BadAllocSize {
            alloc_size,
            byte_len,
        });
    }
    let (fd, found) = size_of_descriptor(fd)?;
    if found != alloc_size {
        return Err(StagingImportError::SizeMismatch {
            recorded: alloc_size,
            found,
        });
    }
    let len = usize::try_from(byte_len)
        .map_err(|_| StagingImportError::NotAddressable { len: byte_len })?;
    Ok((fd, len))
}

/// The producer side of the fake import: map the staging descriptor writable, copy the tensor
/// in at offset 0, unmap. The bytes past the tensor are not touched (they stay zero). The
/// descriptor is closed when this returns; the producer keeps no mapping.
pub fn write_through(
    fd: OwnedFd,
    alloc_size: u64,
    tensor: &[u8],
) -> Result<(), StagingImportError> {
    let (fd, len) = checked(fd, alloc_size, tensor.len() as u64)?;
    let mapping = Mapped::new(&fd, len, true)?;
    // SAFETY: the mapping is `len` writable bytes owned by `mapping`, `tensor` is `len` readable
    // bytes, and a mapping of a descriptor cannot overlap an ordinary slice.
    unsafe {
        ptr::copy_nonoverlapping(tensor.as_ptr(), mapping.base.as_ptr().cast::<u8>(), len);
    }
    drop(mapping);
    Ok(())
}

/// The consumer side of the fake import: map the staging descriptor read-only, copy exactly
/// `byte_len` bytes into memory of the worker's own, unmap. The caller verifies the digest on
/// the returned copy.
pub fn read_copy(
    fd: OwnedFd,
    alloc_size: u64,
    byte_len: u64,
) -> Result<Vec<u8>, StagingImportError> {
    let (fd, len) = checked(fd, alloc_size, byte_len)?;
    let mapping = Mapped::new(&fd, len, false)?;
    let mut copy = vec![0u8; len];
    // SAFETY: the mapping is `len` readable bytes owned by `mapping`, `copy` is `len` writable
    // bytes, and they cannot overlap. The source may be changed by another process meanwhile;
    // that is why no reference to it is made, and the copy is what is verified and used.
    unsafe {
        ptr::copy_nonoverlapping(
            mapping.base.as_ptr().cast::<u8>().cast_const(),
            copy.as_mut_ptr(),
            len,
        );
    }
    drop(mapping);
    Ok(copy)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::fs::FileExt;

    fn memfd(len: u64) -> (OwnedFd, File) {
        let fd = new_memfd();
        let file = File::from(fd);
        file.set_len(len).expect("size");
        let shared = file.try_clone().expect("dup");
        (OwnedFd::from(file), shared)
    }

    /// A fresh `memfd` through libc.
    fn new_memfd() -> OwnedFd {
        use std::os::fd::FromRawFd;
        // SAFETY: the name is NUL terminated; a negative result is an error, otherwise the
        // descriptor is new and owned by nothing else.
        let raw = unsafe { libc::memfd_create(c"staging-import-test".as_ptr(), libc::MFD_CLOEXEC) };
        assert!(raw >= 0, "memfd_create");
        // SAFETY: `raw` is a fresh descriptor that nothing else owns.
        unsafe { OwnedFd::from_raw_fd(raw) }
    }

    #[test]
    fn a_tensor_written_through_one_descriptor_is_read_through_another() {
        let (writer, shared) = memfd(8192);
        let tensor: Vec<u8> = (0..5000u32).map(|n| (n % 251) as u8).collect();
        write_through(writer, 8192, &tensor).expect("write");
        let mut direct = vec![0u8; 8192];
        shared.read_exact_at(&mut direct, 0).expect("read");
        assert_eq!(&direct[..5000], &tensor[..]);
        assert!(
            direct[5000..].iter().all(|byte| *byte == 0),
            "padding stays zero"
        );
        let (reader, _keep) = {
            let dup = shared.try_clone().expect("dup");
            (OwnedFd::from(dup), shared)
        };
        assert_eq!(read_copy(reader, 8192, 5000).expect("read"), tensor);
    }

    #[test]
    fn a_wrong_size_or_length_is_refused_with_a_code() {
        let (fd, _shared) = memfd(4096);
        let error = write_through(fd, 8192, &[1, 2, 3]).unwrap_err();
        assert_eq!(
            error,
            StagingImportError::SizeMismatch {
                recorded: 8192,
                found: 4096
            }
        );
        assert_eq!(error.code(), "staging-size-mismatch");
        let (fd, _shared) = memfd(4096);
        assert_eq!(
            write_through(fd, 4096, &[]).unwrap_err().code(),
            "staging-bad-length"
        );
        let (fd, _shared) = memfd(4096);
        assert_eq!(
            read_copy(fd, 100, 200).unwrap_err().code(),
            "staging-bad-alloc-size"
        );
        let (fd, _shared) = memfd(4096);
        assert_eq!(
            read_copy(fd, u64::MAX, 200).unwrap_err().code(),
            "staging-bad-alloc-size"
        );
        let (fd, _shared) = memfd(4096);
        assert_eq!(
            read_copy(fd, 4096, MAX_HANDLE_BYTES + 1)
                .unwrap_err()
                .code(),
            "staging-bad-length"
        );
    }

    #[test]
    fn a_descriptor_that_cannot_be_mapped_is_a_typed_error() {
        // A pipe has no size and cannot be mapped.
        let (read_end, _write_end) = std::io::pipe().expect("pipe");
        let error = read_copy(OwnedFd::from(read_end), 4096, 10).unwrap_err();
        assert!(
            matches!(
                error,
                StagingImportError::SizeMismatch { .. } | StagingImportError::Map(_)
            ),
            "{error:?}"
        );
    }
}
