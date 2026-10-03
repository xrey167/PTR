//! A private directory for the files of durable round trips.
//!
//! A run creates it exclusively, under a name of its own, and removes it when it
//! is done. A name that is taken is never reused and what is there is never
//! removed, so another process that put something at a name this one would
//! choose can make a run fail but cannot make it delete anything.

use std::io::ErrorKind;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

/// The most names tried before giving up.
const ATTEMPTS: usize = 16;

pub struct Scratch {
    dir: PathBuf,
}

impl Scratch {
    /// A new directory `ptr-s003-<label>-<process>-<n>` under the temporary
    /// directory, created by this call (mode 0700 on Unix).
    /// Removal is attempted on drop; cleanup errors are ignored.
    ///
    /// # Errors
    /// Returns a directory-creation error or fails after 16 occupied names.
    pub fn create(label: &str) -> Result<Self, String> {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        Self::create_in(&std::env::temp_dir(), label, || {
            NEXT.fetch_add(1, Ordering::Relaxed)
        })
    }

    /// The same under `base`, numbering by `next`: a name that exists is
    /// skipped, whatever it is. Returns other creation errors immediately,
    /// or an error after 16 occupied names.
    fn create_in(base: &Path, label: &str, mut next: impl FnMut() -> u64) -> Result<Self, String> {
        for _ in 0..ATTEMPTS {
            let dir = base.join(format!(
                "ptr-s003-{label}-{}-{}",
                std::process::id(),
                next()
            ));
            let mut builder = std::fs::DirBuilder::new();
            #[cfg(unix)]
            {
                use std::os::unix::fs::DirBuilderExt;
                builder.mode(0o700);
            }
            match builder.create(&dir) {
                Ok(()) => return Ok(Self { dir }),
                Err(error) if error.kind() == ErrorKind::AlreadyExists => continue,
                Err(error) => return Err(format!("creating {}: {error}", dir.display())),
            }
        }
        Err(format!("no free scratch directory named for {label}"))
    }

    /// The path of the file `name` in the directory.
    pub fn file(&self, name: &str) -> PathBuf {
        self.dir.join(name)
    }

    #[cfg(test)]
    pub fn dir(&self) -> &Path {
        &self.dir
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_scratch_directory_is_new_private_and_gone_when_dropped() {
        let first = Scratch::create("test").expect("a directory");
        let second = Scratch::create("test").expect("another directory");
        assert_ne!(first.dir(), second.dir());
        std::fs::write(first.file("journal.log"), b"x").expect("a file");
        let dir = first.dir().to_path_buf();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = std::fs::metadata(&dir)
                .expect("metadata")
                .permissions()
                .mode();
            assert_eq!(mode & 0o077, 0, "no access for group or others");
        }
        drop(first);
        assert!(!dir.exists());
        assert!(second.dir().exists());
    }

    #[test]
    fn a_name_that_is_taken_is_skipped_and_what_is_there_stays() {
        let base = Scratch::create("base").expect("a base directory");
        let taken = base.file(&format!("ptr-s003-t-{}-0", std::process::id()));
        std::fs::create_dir(&taken).expect("a squatter");
        std::fs::write(taken.join("mine"), b"not yours").expect("a file in it");
        let mut number = 0;
        let scratch = Scratch::create_in(base.dir(), "t", || {
            number += 1;
            number - 1
        })
        .expect("the next name");
        assert_ne!(scratch.dir(), taken);
        drop(scratch);
        assert!(
            taken.join("mine").exists(),
            "the squatter's file is untouched"
        );
    }

    #[test]
    fn every_name_taken_is_an_error_and_nothing_is_removed() {
        let base = Scratch::create("base").expect("a base directory");
        let taken = base.file(&format!("ptr-s003-t-{}-7", std::process::id()));
        std::fs::create_dir(&taken).expect("a squatter");
        let error = Scratch::create_in(base.dir(), "t", || 7)
            .err()
            .expect("no free name");
        assert!(error.contains("no free scratch directory"));
        assert!(taken.exists());
    }
}
