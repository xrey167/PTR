//! Direct (`O_DIRECT`) read path of `FileTierBackend` (doc 42 stage S4, Linux only).
//!
//! The read buffer is a dedicated anonymous `MAP_SHARED` mapping, not heap memory: `open(2)`
//! warns that `O_DIRECT` into a private mapping (heap included) must not run concurrently with
//! `fork(2)`, and a shared mapping is the documented way out. The mapping is page-aligned, and
//! the alignment the file actually needs is taken from `statx(STATX_DIOALIGN)` where the kernel
//! reports it. Writes are not touched by this module.

use rustix::fs::{statx, AtFlags, OFlags, StatxFlags};
use rustix::io::Errno;
use rustix::mm::{mmap_anonymous, munmap, MapFlags, ProtFlags};
use std::fs::File;
use std::io::{self, Read};
use std::os::unix::fs::OpenOptionsExt;
use std::path::Path;
use std::ptr::NonNull;

/// Alignment assumed for the file offset and the read length when the kernel does not report
/// `STATX_DIOALIGN` (before Linux 6.1, or a filesystem that leaves it unset). 4096 covers the
/// logical block size of common block devices.
pub(crate) const DIRECT_ALIGNMENT: usize = 4096;

pub(crate) enum DirectRead {
    /// The whole file was read through `O_DIRECT`.
    Direct(Vec<u8>),
    /// The filesystem refused `O_DIRECT`, needs an alignment this path does not provide, or the
    /// file changed size during the read; the caller must read buffered.
    Refused,
}

fn is_refusal(error: &io::Error) -> bool {
    error.raw_os_error() == Some(Errno::INVAL.raw_os_error())
}

/// What the file needs for direct I/O, as far as the kernel reports it.
#[derive(Debug, PartialEq, Eq)]
enum DirectAlignment {
    /// Offsets and lengths are multiples of this many bytes; memory is page-aligned.
    Usable(usize),
    /// Direct I/O is unsupported for this file, or needs a memory or offset alignment larger
    /// than a page, which this path does not provide.
    Unusable,
}

/// Turns the `statx` direct-I/O fields into the alignment this path uses. `None` means the
/// kernel did not report them, so the fixed 4096-byte assumption applies.
fn direct_alignment(reported: Option<(u32, u32)>, page: usize) -> DirectAlignment {
    let Some((memory, offset)) = reported else {
        return DirectAlignment::Usable(DIRECT_ALIGNMENT);
    };
    let (memory, offset) = (memory as usize, offset as usize);
    if memory == 0 || offset == 0 || memory > page || offset > page {
        return DirectAlignment::Unusable;
    }
    DirectAlignment::Usable(offset.max(DIRECT_ALIGNMENT).min(page))
}

fn reported_alignment(file: &File) -> Option<(u32, u32)> {
    let status = statx(file, "", AtFlags::EMPTY_PATH, StatxFlags::DIOALIGN).ok()?;
    let mask = StatxFlags::from_bits_retain(status.stx_mask);
    mask.contains(StatxFlags::DIOALIGN)
        .then_some((status.stx_dio_mem_align, status.stx_dio_offset_align))
}

/// A page-aligned anonymous `MAP_SHARED` read buffer, unmapped on drop.
struct SharedBuffer {
    ptr: NonNull<u8>,
    length: usize,
}

impl SharedBuffer {
    fn new(length: usize) -> io::Result<Self> {
        // SAFETY: a null hint with a fresh anonymous mapping aliases nothing; `length` is non-zero
        // (callers round up to at least one alignment unit).
        let raw = unsafe {
            mmap_anonymous(
                std::ptr::null_mut(),
                length,
                ProtFlags::READ | ProtFlags::WRITE,
                MapFlags::SHARED,
            )
        }
        .map_err(io::Error::from)?;
        let ptr = NonNull::new(raw.cast::<u8>())
            .ok_or_else(|| io::Error::other("mmap returned a null mapping"))?;
        Ok(Self { ptr, length })
    }

    fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: the mapping is live for the lifetime of `self`, readable and writable,
        // zero-filled by the kernel and exclusively owned by this value.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.length) }
    }
}

impl Drop for SharedBuffer {
    fn drop(&mut self) {
        // SAFETY: the mapping is owned by `self` and no slice borrowed from it outlives `&mut self`.
        unsafe {
            let _ = munmap(self.ptr.as_ptr().cast(), self.length);
        }
    }
}

/// Reads the whole file with `O_DIRECT`. `NotFound` and other errors are returned as is.
pub(crate) fn read_direct(path: &Path) -> io::Result<DirectRead> {
    let mut file = match File::options()
        .read(true)
        .custom_flags(OFlags::DIRECT.bits() as i32)
        .open(path)
    {
        Ok(file) => file,
        Err(error) if is_refusal(&error) => return Ok(DirectRead::Refused),
        Err(error) => return Err(error),
    };
    let page = rustix::param::page_size();
    let alignment = match direct_alignment(reported_alignment(&file), page) {
        DirectAlignment::Usable(alignment) => alignment,
        DirectAlignment::Unusable => return Ok(DirectRead::Refused),
    };
    let length = usize::try_from(file.metadata()?.len())
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "chunk file too large"))?;
    // Read length rounded up to the alignment; the tail read returns short at EOF. The mapping
    // is rounded up to whole pages and is page-aligned, which satisfies any memory alignment up
    // to a page.
    let too_large = || io::Error::new(io::ErrorKind::InvalidData, "chunk file too large");
    let capacity = length
        .div_ceil(alignment)
        .max(1)
        .checked_mul(alignment)
        .ok_or_else(too_large)?;
    let mapped = capacity
        .div_ceil(page)
        .checked_mul(page)
        .ok_or_else(too_large)?;
    let mut shared = SharedBuffer::new(mapped)?;
    let buffer = &mut shared.as_mut_slice()[..capacity];
    let mut filled = 0;
    while filled < length {
        match file.read(&mut buffer[filled..]) {
            Ok(0) => break,
            Ok(read) => filled += read,
            Err(error) if error.kind() == io::ErrorKind::Interrupted => {}
            // EINVAL mid-file is also treated as a refusal: the caller then re-reads the whole
            // file buffered and validates it, so a real EINVAL costs a second read, never a
            // wrong result.
            Err(error) if is_refusal(&error) => return Ok(DirectRead::Refused),
            Err(error) => return Err(error),
        }
    }
    // A file that changed size while it was read (for example extended in place after the
    // snapshot above) is not served from this read: the caller re-reads it buffered through EOF,
    // and validation then rejects the changed object as the buffered path always does.
    if filled != length || file.metadata()?.len() != length as u64 {
        return Ok(DirectRead::Refused);
    }
    Ok(DirectRead::Direct(buffer[..length].to_vec()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_einval_is_a_refusal() {
        assert!(is_refusal(&io::Error::from_raw_os_error(
            Errno::INVAL.raw_os_error()
        )));
        assert!(!is_refusal(&io::Error::from_raw_os_error(
            Errno::IO.raw_os_error()
        )));
        assert!(!is_refusal(&io::Error::from(io::ErrorKind::NotFound)));
    }

    #[test]
    fn the_reported_direct_io_alignment_decides_the_path() {
        let page = 4096;
        // Not reported: the fixed assumption.
        assert_eq!(direct_alignment(None, page), DirectAlignment::Usable(4096));
        // Reported and within a page: offsets and lengths use at least 4096.
        assert_eq!(
            direct_alignment(Some((512, 512)), page),
            DirectAlignment::Usable(4096)
        );
        assert_eq!(
            direct_alignment(Some((4096, 4096)), page),
            DirectAlignment::Usable(4096)
        );
        assert_eq!(
            direct_alignment(Some((4096, 16384)), 65536),
            DirectAlignment::Usable(16384)
        );
        // Zero means direct I/O is unsupported for this file.
        assert_eq!(
            direct_alignment(Some((0, 0)), page),
            DirectAlignment::Unusable
        );
        // An alignment above a page is not provided: refused, so the read is never a misaligned
        // request the kernel might silently serve buffered.
        assert_eq!(
            direct_alignment(Some((4096, 8192)), page),
            DirectAlignment::Unusable
        );
        assert_eq!(
            direct_alignment(Some((8192, 4096)), page),
            DirectAlignment::Unusable
        );
    }

    #[test]
    fn the_shared_buffer_is_page_aligned_zeroed_and_writable() {
        let page = rustix::param::page_size();
        let mut buffer = SharedBuffer::new(2 * page).expect("mmap");
        let slice = buffer.as_mut_slice();
        assert_eq!(slice.as_ptr() as usize % page, 0);
        assert!(slice.iter().all(|byte| *byte == 0));
        slice[2 * page - 1] = 7;
        assert_eq!(slice[2 * page - 1], 7);
    }
}
