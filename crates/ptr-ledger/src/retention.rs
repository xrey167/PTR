//! What deleting something actually removes, and what still holds it.
//!
//! PTR has no `erase(key)`. Removing a semantic value commits a *removal record*:
//! the value stops being current, and the record that carried it stays in the log
//! exactly where it was. Logical deletion and erasure are different operations,
//! and conflating them is how a system ends up reporting that data is gone while
//! the bytes sit in a file.
//!
//! So erasure here is achieved by **not retaining**, never by overwriting. A
//! committed byte leaves the retained set when the log's floor rises past its
//! record and every artifact that copied it is reclaimed. Overwriting in place is
//! not offered, because on a copy-on-write filesystem, a wear-levelling SSD or a
//! storage layer taking its own snapshots, writing zeros over a file does not
//! destroy the blocks that held the old contents — and a guarantee that cannot be
//! kept is worse than an absent one.
//!
//! This module supplies the measurement. [`ErasureAudit`] answers one question —
//! does this plaintext still appear in anything we retain, and if so where — so
//! that "erased" is a checked claim rather than an assumption.
use crate::compaction::LogPaths;
use crate::integrity::MAX_LOG_BYTES;
use ptr_types::CommitIndex;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

/// Something erasure cannot reach, named so a caller cannot quietly forget it.
///
/// These are not failures of a particular audit; they are permanent properties of
/// the boundary. An audit reports them every time.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum OutOfReach {
    /// Artifacts the host retains elsewhere: recovery snapshots, compacted
    /// snapshots, backups, replicas. An audit cannot discover these; fold each one
    /// in with [`ErasureAudit::with_retained_elsewhere`].
    HostRetainedArtifacts,
    /// Blocks a filesystem or device still holds after a file is removed or
    /// rewritten. Absence from a file is not absence from the medium.
    StorageResidue,
    /// Model, KV and checkpoint state. Not implemented in PTR, so nothing here can
    /// speak to it either way.
    ModelDerivedState,
}

impl OutOfReach {
    /// Stable diagnostic code for this audit boundary.
    pub fn code(self) -> &'static str {
        match self {
            Self::HostRetainedArtifacts => "PTR_ERASURE_HOST_RETAINED_ARTIFACTS",
            Self::StorageResidue => "PTR_ERASURE_STORAGE_RESIDUE",
            Self::ModelDerivedState => "PTR_ERASURE_MODEL_DERIVED_STATE",
        }
    }

    /// Every boundary, reported by every audit.
    pub const ALL: [Self; 3] = [
        Self::HostRetainedArtifacts,
        Self::StorageResidue,
        Self::ModelDerivedState,
    ];
}

/// An artifact that still contains the audited plaintext.
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum Retainer {
    /// A log file of the audited path set, live or superseded.
    Log(PathBuf),
    /// An artifact the caller folded in, named by the caller.
    Elsewhere(String),
}

/// The result of looking for a plaintext across everything reachable.
///
/// A byte search, not a record scan. Erasure asks whether the bytes are present
/// at all, so framing, encoding and coincidence are irrelevant: a match anywhere
/// means "still retained", which is the safe direction for this question. The
/// converse bias — a record scan that misses a byte sequence it does not
/// recognize — would let an audit report erasure that did not happen.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ErasureAudit {
    retainers: Vec<Retainer>,
    scanned_logs: usize,
    live_base: CommitIndex,
    /// Retained so a caller can fold in further artifacts after the log scan
    /// without restating what is being looked for.
    needle: Vec<u8>,
}

impl ErasureAudit {
    /// True only when nothing reachable contains the plaintext.
    ///
    /// This is never a statement about [`OutOfReach`]; see [`Self::out_of_reach`].
    pub fn erased_where_reachable(&self) -> bool {
        self.retainers.is_empty()
    }

    /// Artifacts that still contain it.
    pub fn retainers(&self) -> &[Retainer] {
        &self.retainers
    }

    /// How many log files were searched.
    pub fn scanned_logs(&self) -> usize {
        self.scanned_logs
    }

    /// The live log's floor at audit time.
    ///
    /// A record at or below this index is no longer in the live log. Raising the
    /// floor further is the only way the live log stops retaining a record, and
    /// [`CompactionBarrier`](crate::CompactionBarrier) bounds how far it may rise.
    pub fn live_base(&self) -> CommitIndex {
        self.live_base
    }

    /// The boundaries no audit can cross.
    pub fn out_of_reach(&self) -> [OutOfReach; 3] {
        OutOfReach::ALL
    }

    /// Fold in an artifact the caller retains outside the audited path set.
    ///
    /// Snapshots are the case that matters: a recovery snapshot embeds the whole
    /// journal and a compacted snapshot embeds committed state, so either can hold
    /// a value the log has already discarded. Nothing discovers them for you,
    /// which is why they are a parameter rather than a search path.
    pub fn with_retained_elsewhere(mut self, name: impl Into<String>, artifact: &[u8]) -> Self {
        let name = name.into();
        if retains(artifact, self.needle()) {
            self.retainers.push(Retainer::Elsewhere(name));
        }
        self
    }

    /// Plaintext retained so callers can fold in external artifacts.
    fn needle(&self) -> &[u8] {
        &self.needle
    }
}

/// Whether `artifact` contains `plaintext` anywhere.
pub fn retains(artifact: &[u8], plaintext: &[u8]) -> bool {
    if plaintext.is_empty() || plaintext.len() > artifact.len() {
        return false;
    }
    artifact
        .windows(plaintext.len())
        .any(|window| window == plaintext)
}

/// Read a log only when it remains within the framing size bound.
fn read_bounded(path: &Path) -> io::Result<Vec<u8>> {
    let bytes = fs::read(path)?;
    if bytes.len() > MAX_LOG_BYTES {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "PTR_LOG_SIZE_LIMIT",
        ));
    }
    Ok(bytes)
}

impl LogPaths {
    /// Search every log file of this set — live and superseded — for `plaintext`,
    /// with **no ledger open**.
    ///
    /// The live log is read through an independent handle, which Windows refuses
    /// while a writer holds it, because its advisory locks are mandatory for I/O.
    /// Auditing a running system therefore goes through
    /// [`AcknowledgedLedger::audit_erasure`](crate::AcknowledgedLedger::audit_erasure),
    /// which reads the live log through the handle that owns it.
    ///
    /// `live_base` is the floor from protected anchor storage, so the audit
    /// describes the same live log a reader would open. Superseded files are
    /// included because retention, not liveness, is what erasure is about: an
    /// orphan left by a cutover holds its records until it is reclaimed.
    pub fn audit_erasure(
        &self,
        plaintext: &[u8],
        live_base: CommitIndex,
    ) -> io::Result<ErasureAudit> {
        self.audit_with_live(plaintext, live_base, None)
    }

    /// Audit a set while optionally supplying bytes from an already-open live log.
    pub(crate) fn audit_with_live(
        &self,
        plaintext: &[u8],
        live_base: CommitIndex,
        live_bytes: Option<&[u8]>,
    ) -> io::Result<ErasureAudit> {
        let mut retainers = Vec::new();
        let mut scanned_logs = 0;
        let live = self.log_path(live_base);
        // Orphans are held by nobody, so an independent handle always reaches them.
        for path in self.orphans(live_base)? {
            scanned_logs += 1;
            if retains(&read_bounded(&path)?, plaintext) {
                retainers.push(Retainer::Log(path));
            }
        }
        match live_bytes {
            Some(bytes) => {
                scanned_logs += 1;
                if retains(bytes, plaintext) {
                    retainers.push(Retainer::Log(live));
                }
            }
            None if live.exists() => {
                scanned_logs += 1;
                if retains(&read_bounded(&live)?, plaintext) {
                    retainers.push(Retainer::Log(live));
                }
            }
            None => {}
        }
        retainers.sort();
        Ok(ErasureAudit {
            retainers,
            scanned_logs,
            live_base,
            needle: plaintext.to_vec(),
        })
    }
}
