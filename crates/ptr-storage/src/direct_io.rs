//! Direct (`O_DIRECT`) read path of `FileTierBackend` (doc 42 stage S4, Linux only).
//!
//! The buffer is aligned with plain safe code: an over-allocated `Vec` is sliced at the first
//! aligned offset. Writes are not touched by this module.

use rustix::fs::OFlags;
use rustix::io::Errno;
use std::fs::File;
use std::io::{self, Read};
use std::os::unix::fs::OpenOptionsExt;
use std::path::Path;

/// Alignment of the buffer address, the file offset and the read length. 4096 covers the logical
/// block size of common block devices and the page size of x86-64 and most aarch64 kernels.
pub(crate) const DIRECT_ALIGNMENT: usize = 4096;

pub(crate) enum DirectRead {
    /// The whole file was read through `O_DIRECT`.
    Direct(Vec<u8>),
    /// The filesystem refused `O_DIRECT`; the caller must read buffered.
    Refused,
}

fn is_refusal(error: &io::Error) -> bool {
    error.raw_os_error() == Some(Errno::INVAL.raw_os_error())
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
    let length = usize::try_from(file.metadata()?.len())
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "chunk file too large"))?;
    // Read length rounded up to the alignment; the tail read returns short at EOF.
    let capacity = length
        .div_ceil(DIRECT_ALIGNMENT)
        .max(1)
        .checked_mul(DIRECT_ALIGNMENT)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "chunk file too large"))?;
    let mut backing = vec![0u8; capacity + DIRECT_ALIGNMENT];
    let skew = backing.as_ptr().align_offset(DIRECT_ALIGNMENT);
    let buffer = &mut backing[skew..skew + capacity];
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
}
