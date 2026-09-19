//! Portable recovery snapshots replay the complete checked journal. They are not
//! compacted materialized-state snapshots, neural checkpoints or execution tokens.
use super::{PtrRuntime, RuntimeError, RuntimeLedger};
use ptr_config::PtrConfig;
use ptr_ledger::{
    integrity::{self, LogAnchor, MAX_LOG_BYTES},
    FileLedger,
};
use ptr_types::{CommitIndex, Revision};
use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::path::Path;

const MAGIC: &[u8; 8] = b"PTRSN001";
const HEADER: usize = 64;
const DIGEST_BYTES: usize = 32;
pub const MAX_SNAPSHOT_BYTES: usize = MAX_LOG_BYTES + HEADER + DIGEST_BYTES;

/// Named snapshot failure categories, independent of backend I/O diagnostics.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SnapshotError {
    SizeLimit,
    UnsupportedVersion,
    LengthMismatch,
    AnchorMismatch,
    StateMismatch,
    CommitMismatch,
}
impl SnapshotError {
    pub fn code(&self) -> &'static str {
        match self {
            Self::SizeLimit => "PTR_SNAPSHOT_SIZE_LIMIT",
            Self::UnsupportedVersion => "PTR_SNAPSHOT_VERSION",
            Self::LengthMismatch => "PTR_SNAPSHOT_LENGTH",
            Self::AnchorMismatch => "PTR_SNAPSHOT_ANCHOR_MISMATCH",
            Self::StateMismatch => "PTR_SNAPSHOT_STATE_MISMATCH",
            Self::CommitMismatch => "PTR_SNAPSHOT_COMMIT_MISMATCH",
        }
    }
}
impl std::fmt::Display for SnapshotError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.code())
    }
}
impl std::error::Error for SnapshotError {}

/// Trust input from a separately retained checkpoint catalog, not the untrusted
/// snapshot being restored. This value is not authenticated by its Rust type.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SnapshotAnchor {
    pub revision: Revision,
    pub log: LogAnchor,
    pub digest: [u8; 32],
}

/// An immutable, complete replay-backed snapshot of committed runtime state.
/// It excludes configuration, permissions, sessions, permits, model/KV state and
/// transient telemetry. The caller must retain its anchor outside this artifact.
#[derive(Debug)]
pub struct RecoverySnapshot {
    bytes: Vec<u8>,
    anchor: SnapshotAnchor,
}
impl RecoverySnapshot {
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }
    pub fn anchor(&self) -> SnapshotAnchor {
        self.anchor
    }

    /// Immutable create-new publication: no existing file is replaced. A failed
    /// write may leave an invalid destination; readers reject it. On Unix the
    /// parent directory is synchronized; other platforms have file-sync only.
    pub fn write_new(&self, path: impl AsRef<Path>) -> Result<(), RuntimeError> {
        let path = path.as_ref();
        let mut file = OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(path)
            .map_err(io_error)?;
        file.write_all(&self.bytes).map_err(io_error)?;
        file.sync_all().map_err(io_error)?;
        #[cfg(unix)]
        {
            let parent = path
                .parent()
                .filter(|p| !p.as_os_str().is_empty())
                .unwrap_or(Path::new("."));
            File::open(parent)
                .and_then(|file| file.sync_all())
                .map_err(io_error)?;
        }
        Ok(())
    }
}
fn io_error(error: std::io::Error) -> RuntimeError {
    RuntimeError::Ledger(error.to_string())
}
fn invalid(kind: SnapshotError) -> RuntimeError {
    RuntimeError::Snapshot(kind)
}

impl PtrRuntime {
    /// A checkable complete-history identity, only while the runtime is healthy.
    pub fn journal_anchor(&self) -> Result<LogAnchor, RuntimeError> {
        if self.execution.is_fenced() {
            return Err(RuntimeError::ExecutionFenced);
        }
        match &self.ledger {
            RuntimeLedger::File(log) => log.anchor().map_err(io_error),
            RuntimeLedger::Memory(_) => {
                let bytes = integrity::encode_log(self.committed_events()).map_err(io_error)?;
                Ok(integrity::decode_log(&bytes).map_err(io_error)?.anchor())
            }
        }
    }

    /// Strict reopen with externally supplied exact history identity. A valid
    /// shorter chain, different history or missing file cannot silently pass.
    pub fn open_durable_at(
        config: PtrConfig,
        path: impl AsRef<Path>,
        trusted: LogAnchor,
    ) -> Result<Self, RuntimeError> {
        let log = FileLedger::open_at(path, trusted).map_err(io_error)?;
        Self::from_durable_ledger(config, log)
    }

    pub fn export_recovery_snapshot(&self) -> Result<RecoverySnapshot, RuntimeError> {
        let log_anchor = self.journal_anchor()?;
        let log = integrity::encode_log(self.committed_events()).map_err(io_error)?;
        integrity::decode_log(&log)
            .map_err(io_error)?
            .require_anchor(log_anchor)
            .map_err(io_error)?;
        if self.state.last_applied != log_anchor.index.0 {
            return Err(invalid(SnapshotError::CommitMismatch));
        }
        let mut bytes = Vec::with_capacity(HEADER + log.len() + DIGEST_BYTES);
        bytes.extend_from_slice(MAGIC);
        bytes.extend_from_slice(&self.revision().0.to_le_bytes());
        bytes.extend_from_slice(&log_anchor.index.0.to_le_bytes());
        bytes.extend_from_slice(&(log.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&log_anchor.digest);
        bytes.extend_from_slice(&log);
        let digest = integrity::sha256(&bytes);
        bytes.extend_from_slice(&digest);
        Ok(RecoverySnapshot {
            bytes,
            anchor: SnapshotAnchor {
                revision: self.revision(),
                log: log_anchor,
                digest,
            },
        })
    }

    /// Verify byte identity, version, lengths, complete journal, ordering,
    /// lifecycle and semantic transitions before returning any restored state.
    pub fn restore_recovery_snapshot(
        config: PtrConfig,
        bytes: &[u8],
        trusted: SnapshotAnchor,
    ) -> Result<Self, RuntimeError> {
        let log_bytes = snapshot_log(bytes, trusted)?;
        let log = integrity::decode_log(log_bytes).map_err(io_error)?;
        log.require_anchor(trusted.log).map_err(io_error)?;
        let restored = Self::replay(config, log.events())?;
        if restored.revision() != trusted.revision
            || restored.state.last_applied != trusted.log.index.0
        {
            return Err(invalid(SnapshotError::StateMismatch));
        }
        Ok(restored)
    }

    /// Complete validation precedes creating the destination. Restoration never
    /// replaces an existing log and never replays authority or opaque model state.
    pub fn restore_durable_snapshot(
        config: PtrConfig,
        bytes: &[u8],
        trusted: SnapshotAnchor,
        destination: impl AsRef<Path>,
    ) -> Result<Self, RuntimeError> {
        let mut restored = Self::restore_recovery_snapshot(config, bytes, trusted)?;
        let log_bytes = snapshot_log(bytes, trusted)?;
        let log =
            FileLedger::create_from_log(destination, log_bytes, trusted.log).map_err(io_error)?;
        restored.ledger = RuntimeLedger::File(log);
        Ok(restored)
    }

    pub fn read_recovery_snapshot(
        config: PtrConfig,
        path: impl AsRef<Path>,
        trusted: SnapshotAnchor,
    ) -> Result<Self, RuntimeError> {
        let file = File::open(path).map_err(io_error)?;
        if file.metadata().map_err(io_error)?.len() > MAX_SNAPSHOT_BYTES as u64 {
            return Err(invalid(SnapshotError::SizeLimit));
        }
        let mut bytes = Vec::new();
        file.take(MAX_SNAPSHOT_BYTES as u64 + 1)
            .read_to_end(&mut bytes)
            .map_err(io_error)?;
        Self::restore_recovery_snapshot(config, &bytes, trusted)
    }

    /// Explicit migration of the old unchecked event format. The source is never
    /// rewritten. No checksum can retrospectively prove legacy provenance.
    /// Replay validation completes before a create-new v2 destination is opened.
    pub fn migrate_legacy_log(
        config: PtrConfig,
        source: impl AsRef<Path>,
        destination: impl AsRef<Path>,
    ) -> Result<Self, RuntimeError> {
        let legacy = FileLedger::open_legacy_for_migration(source).map_err(io_error)?;
        let mut restored = Self::replay(config, legacy.events())?;
        let bytes = integrity::encode_log(legacy.events()).map_err(io_error)?;
        let anchor = integrity::decode_log(&bytes).map_err(io_error)?.anchor();
        restored.ledger = RuntimeLedger::File(
            FileLedger::create_from_log(destination, &bytes, anchor).map_err(io_error)?,
        );
        Ok(restored)
    }
}

fn snapshot_log(bytes: &[u8], trusted: SnapshotAnchor) -> Result<&[u8], RuntimeError> {
    if bytes.len() < HEADER + DIGEST_BYTES || bytes.len() > MAX_SNAPSHOT_BYTES {
        return Err(invalid(SnapshotError::SizeLimit));
    }
    if &bytes[..8] != MAGIC {
        return Err(invalid(SnapshotError::UnsupportedVersion));
    }
    let revision = Revision(u64::from_le_bytes(
        bytes[8..16].try_into().expect("snapshot revision"),
    ));
    let index = CommitIndex(u64::from_le_bytes(
        bytes[16..24].try_into().expect("snapshot index"),
    ));
    let length = u64::from_le_bytes(bytes[24..32].try_into().expect("snapshot length"));
    if length > MAX_LOG_BYTES as u64 || length != (bytes.len() - HEADER - DIGEST_BYTES) as u64 {
        return Err(invalid(SnapshotError::LengthMismatch));
    }
    let log = LogAnchor {
        index,
        digest: bytes[32..64].try_into().expect("snapshot log digest"),
    };
    let end = bytes.len() - DIGEST_BYTES;
    let digest = integrity::sha256(&bytes[..end]);
    if bytes[end..] != digest
        || digest != trusted.digest
        || revision != trusted.revision
        || log != trusted.log
    {
        return Err(invalid(SnapshotError::AnchorMismatch));
    }
    Ok(&bytes[HEADER..end])
}
