//! Whether a file still holds bytes the kernel has not written to its disk.
//!
//! The crash-image model needs to know what a file had synced when the process stopped, and the
//! harness must not take the product's word for it. This asks the kernel. `posix_fadvise` with
//! `DONTNEED` drops the clean pages of a file from the page cache and keeps the dirty ones, and
//! `mincore` says which pages of a mapping are still cached: a page that stays is a page that
//! holds bytes no flush has written. Called where the product claims a record durable (after
//! its sync, before it answers), a cached page there means the product answered without its
//! fsync, so that a power loss would have taken an acknowledged record.
//!
//! It sees file data pages only. A directory entry (the rename of an anchor, the directory
//! fsync after it) is not in a file's page cache, so a missing directory fsync is outside what
//! this can see and the preregistration says so. It needs a filesystem that writes back to a
//! disk: [`calibrate`] checks that a written file reads dirty and a synced one reads clean on
//! the directory the run uses, and a run on one where it does not (tmpfs, say) declares the
//! probe unavailable, which leaves a coverage prerequisite unmet. Linux only.

use std::fs::File;
use std::io::Write;
use std::path::Path;

#[cfg(target_os = "linux")]
mod sys {
    use std::os::raw::{c_int, c_void};

    extern "C" {
        pub fn posix_fadvise(fd: c_int, offset: i64, len: i64, advice: c_int) -> c_int;
        pub fn mmap(
            addr: *mut c_void,
            len: usize,
            prot: c_int,
            flags: c_int,
            fd: c_int,
            offset: i64,
        ) -> *mut c_void;
        pub fn munmap(addr: *mut c_void, len: usize) -> c_int;
        pub fn mincore(addr: *mut c_void, len: usize, vec: *mut u8) -> c_int;
    }

    pub const POSIX_FADV_DONTNEED: c_int = 4;
    pub const PROT_READ: c_int = 1;
    pub const MAP_SHARED: c_int = 1;
}

/// Whether any page of the file at `path` is cached after the clean ones were dropped, i.e.
/// holds bytes not yet written to disk. `None` when the question cannot be asked (not Linux, a
/// file that cannot be opened or mapped).
#[cfg(target_os = "linux")]
pub fn has_unsynced_pages(path: &Path) -> Option<bool> {
    use std::os::fd::AsRawFd;
    let file = File::open(path).ok()?;
    let length = usize::try_from(file.metadata().ok()?.len()).ok()?;
    if length == 0 {
        return Some(false);
    }
    let fd = file.as_raw_fd();
    let page = 4096usize;
    let pages = length.div_ceil(page);
    let mut resident = vec![0u8; pages];
    // SAFETY: `fd` is open for the duration of the calls; the mapping is read-only, at most as
    // long as the file, unmapped before returning and never dereferenced; `resident` holds one
    // byte per page of the mapping, which is what `mincore` writes.
    unsafe {
        if sys::posix_fadvise(fd, 0, 0, sys::POSIX_FADV_DONTNEED) != 0 {
            return None;
        }
        let map = sys::mmap(
            std::ptr::null_mut(),
            length,
            sys::PROT_READ,
            sys::MAP_SHARED,
            fd,
            0,
        );
        if map as isize == -1 {
            return None;
        }
        let status = sys::mincore(map, length, resident.as_mut_ptr());
        sys::munmap(map, length);
        if status != 0 {
            return None;
        }
    }
    Some(resident.iter().any(|byte| byte & 1 == 1))
}

#[cfg(not(target_os = "linux"))]
pub fn has_unsynced_pages(_path: &Path) -> Option<bool> {
    None
}

/// What the probe can tell on one directory.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Calibration {
    /// A file written and not synced reads as holding unsynced pages.
    pub sees_dirty: bool,
    /// The same file, synced, reads as holding none.
    pub sees_clean: bool,
}

impl Calibration {
    pub fn available(self) -> bool {
        self.sees_dirty && self.sees_clean
    }
}

/// Whether the probe can tell a written file from a synced one on the directory `dir`. A
/// filesystem that does not write back to a disk (tmpfs) reads dirty before and after.
pub fn calibrate(dir: &Path) -> Calibration {
    let path = dir.join("probe-calibration");
    let outcome = (|| -> std::io::Result<Calibration> {
        let mut file = File::create(&path)?;
        file.write_all(&[0x5a; 6000])?;
        let written = has_unsynced_pages(&path);
        file.sync_all()?;
        let synced = has_unsynced_pages(&path);
        Ok(Calibration {
            sees_dirty: written == Some(true),
            sees_clean: synced == Some(false),
        })
    })();
    let _ = std::fs::remove_file(&path);
    outcome.unwrap_or(Calibration {
        sees_dirty: false,
        sees_clean: false,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(label: &str) -> std::path::PathBuf {
        let dir =
            std::env::temp_dir().join(format!("ptr-l001-probe-{label}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn a_missing_or_empty_file_is_not_dirty_and_a_nonlinux_probe_is_none() {
        let dir = scratch("empty");
        assert_ne!(has_unsynced_pages(&dir.join("absent")), Some(true));
        File::create(dir.join("empty")).unwrap();
        if cfg!(target_os = "linux") {
            assert_eq!(has_unsynced_pages(&dir.join("empty")), Some(false));
        }
        std::fs::remove_dir_all(dir).unwrap();
    }

    /// On a disk-backed directory a written file reads dirty and a synced one clean; elsewhere
    /// calibration says the probe is unavailable instead of guessing.
    #[test]
    fn calibration_agrees_with_what_the_probe_reads() {
        let dir = scratch("calibrate");
        let available = calibrate(&dir).available();
        let path = dir.join("f");
        let mut file = File::create(&path).unwrap();
        file.write_all(&[1; 5000]).unwrap();
        let before = has_unsynced_pages(&path);
        file.sync_all().unwrap();
        let after = has_unsynced_pages(&path);
        if available {
            assert_eq!((before, after), (Some(true), Some(false)));
        } else {
            assert!(!(before == Some(true) && after == Some(false)));
        }
        std::fs::remove_dir_all(dir).unwrap();
    }
}
