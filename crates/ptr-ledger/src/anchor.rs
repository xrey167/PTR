//! Authenticated, atomically published anchor storage.
//!
//! A hash chain proves internal consistency. It cannot prove that the chain in
//! front of you is the chain you committed, because an attacker who rewrites the
//! log can rehash it end to end. Detection therefore requires an expected value
//! held somewhere the log cannot influence, and this module is that place: a
//! fixed-size record authenticated with HMAC-SHA256 under a host-held key,
//! published by atomic rename, carrying a monotonic epoch and the compaction
//! floor.
//!
//! Deliberate non-goals. This is authentication, not encryption: the record is
//! readable and only its integrity and origin are protected. It is a local
//! artifact, not a distributed fence — a file rename orders writers on one
//! filesystem and says nothing about another node. And it cannot by itself
//! detect replacement of the anchor file with an *older authentic* record,
//! because that record carries a valid MAC; only the retained epoch witness
//! described on [`AnchorStore::open_expecting`] closes that gap.
use crate::integrity::LogAnchor;
use ptr_types::CommitIndex;
use sha2::{Digest, Sha256};
use std::fmt;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};

const MAGIC: &[u8; 8] = b"PTRANC01";
const MAC_DOMAIN: &[u8] = b"PTRANC01-MAC";
const AUTHENTICATED_BYTES: usize = 136;
/// Exact on-disk size of one anchor record. A different length is rejected
/// rather than parsed, so a partially written file can never authenticate.
pub const ANCHOR_BYTES: usize = AUTHENTICATED_BYTES + 32;
const BLOCK_BYTES: usize = 64;

/// Identity of the log a [`ProtectedAnchor`] belongs to, taken as the record
/// digest at commit index 1.
///
/// The empty-log digest is a constant of the format and so is identical for
/// every log; the first record's digest is not. Compaction may discard index 1,
/// after which the origin is carried forward by the anchor instead of being
/// recomputed — see [`AnchorStore::acknowledge`].
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ChainOrigin([u8; 32]);

impl ChainOrigin {
    /// Origin of a log that has no records yet.
    pub const UNSET: Self = Self([0; 32]);
    /// Constructs a chain origin from the first record digest.
    pub fn from_first_record(digest: [u8; 32]) -> Self {
        Self(digest)
    }
    /// Returns whether this origin identifies a first record.
    pub fn is_set(self) -> bool {
        self != Self::UNSET
    }
    /// Returns the canonical encoded bytes.
    pub fn bytes(self) -> [u8; 32] {
        self.0
    }
}

/// Host-held anchor authentication key.
///
/// The key must come from the embedding host's secret material and must never be
/// stored beside the log or the anchor it protects; an attacker who can rewrite
/// both the log and the key can forge any history. [`fmt::Debug`] is redacted
/// (global invariant 13) and [`Drop`] overwrites the bytes on a best-effort
/// basis — Rust cannot guarantee erasure of copies the optimizer, the allocator
/// or a page swap retained.
pub struct AnchorKey([u8; 32]);

impl AnchorKey {
    /// Constructs a host-held authentication key from raw bytes.
    pub fn from_bytes(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }
}

impl fmt::Debug for AnchorKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("AnchorKey(<redacted>)")
    }
}

impl Drop for AnchorKey {
    fn drop(&mut self) {
        self.0.fill(0);
        let _ = std::hint::black_box(&self.0);
    }
}

/// Named anchor failure categories. Every variant is a refusal: this module
/// never repairs, replaces or invents an anchor.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AnchorError {
    /// No anchor exists. A missing anchor is never silently initialized,
    /// because an empty anchor authenticates an empty history.
    Absent,
    /// Wrong length, wrong magic or reserved bits set.
    Format,
    /// Unknown format version.
    UnsupportedVersion,
    /// The MAC does not verify under the supplied key: forged, corrupted, or
    /// authenticated by a different key.
    Unauthenticated,
    /// The stored epoch is older than the witness the host retained, so the
    /// anchor file itself was rolled back.
    StaleEpoch { stored: u64, expected: u64 },
    /// An advance would move the covered index, the epoch or the compaction
    /// floor backwards, or would restate a covered index with a new digest.
    NonMonotonic,
    /// The anchor authenticates a different log.
    OriginMismatch,
    /// The compaction floor lies above the covered index.
    BaseRange,
    /// Initialization would replace an existing anchor.
    Exists,
    /// The anchor could not be read or installed. The underlying operating-system
    /// diagnostic is deliberately not folded into a validity verdict: a
    /// permission or capacity failure is not evidence about the history.
    Storage,
}

impl AnchorError {
    /// Returns the stable machine-readable diagnostic code.
    pub fn code(self) -> &'static str {
        match self {
            Self::Absent => "PTR_ANCHOR_ABSENT",
            Self::Format => "PTR_ANCHOR_FORMAT",
            Self::UnsupportedVersion => "PTR_ANCHOR_VERSION",
            Self::Unauthenticated => "PTR_ANCHOR_UNAUTHENTICATED",
            Self::StaleEpoch { .. } => "PTR_ANCHOR_STALE_EPOCH",
            Self::NonMonotonic => "PTR_ANCHOR_NONMONOTONIC",
            Self::OriginMismatch => "PTR_ANCHOR_ORIGIN_MISMATCH",
            Self::BaseRange => "PTR_ANCHOR_BASE_RANGE",
            Self::Exists => "PTR_ANCHOR_EXISTS",
            Self::Storage => "PTR_ANCHOR_STORAGE",
        }
    }
}

impl fmt::Display for AnchorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.code())
    }
}

impl std::error::Error for AnchorError {}

impl From<AnchorError> for io::Error {
    fn from(error: AnchorError) -> Self {
        match error {
            AnchorError::Absent => io::Error::new(io::ErrorKind::NotFound, error.code()),
            AnchorError::Storage => io::Error::other(error.code()),
            _ => io::Error::new(io::ErrorKind::InvalidData, error.code()),
        }
    }
}

/// The authenticated commitment: which history, how far it is acknowledged, how
/// far it has been compacted, and how many times it has advanced.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ProtectedAnchor {
    /// Monotonic advancement counter. The host retains the last value it saw so
    /// that a rollback of the anchor file is detectable.
    pub epoch: u64,
    /// Acknowledged complete-history identity.
    pub log: LogAnchor,
    /// Identity of the log this anchor belongs to.
    pub origin: ChainOrigin,
    /// Compaction floor: the log file is expected to begin directly above this
    /// index and to chain from this digest. Equal to [`LogAnchor::empty`] while
    /// no records have been discarded.
    pub base: LogAnchor,
}

impl ProtectedAnchor {
    /// The commitment for a log that exists but holds no records.
    pub fn initial() -> Self {
        Self {
            epoch: 1,
            log: LogAnchor::empty(),
            origin: ChainOrigin::UNSET,
            base: LogAnchor::empty(),
        }
    }

    /// True when compaction has discarded at least one record from the log.
    pub fn is_compacted(self) -> bool {
        self.base != LogAnchor::empty()
    }

    fn check_self(&self) -> Result<(), AnchorError> {
        if self.base.index.0 > self.log.index.0 {
            return Err(AnchorError::BaseRange);
        }
        if self.base != LogAnchor::empty() && self.base.index.0 == 0 {
            return Err(AnchorError::BaseRange);
        }
        Ok(())
    }

    fn encode(&self) -> [u8; AUTHENTICATED_BYTES] {
        let mut bytes = [0u8; AUTHENTICATED_BYTES];
        bytes[..8].copy_from_slice(MAGIC);
        bytes[8..16].copy_from_slice(&self.epoch.to_le_bytes());
        bytes[16..24].copy_from_slice(&self.log.index.0.to_le_bytes());
        bytes[24..56].copy_from_slice(&self.log.digest);
        bytes[56..88].copy_from_slice(&self.origin.0);
        bytes[88..96].copy_from_slice(&self.base.index.0.to_le_bytes());
        bytes[96..128].copy_from_slice(&self.base.digest);
        // bytes[128..136] stay zero: reserved flags.
        bytes
    }

    fn decode(bytes: &[u8]) -> Result<Self, AnchorError> {
        if bytes.len() != ANCHOR_BYTES {
            return Err(AnchorError::Format);
        }
        if &bytes[..8] != MAGIC {
            // A different but well-formed magic is a version problem; anything
            // else is not this format at all.
            return Err(if bytes[..3] == *b"PTR" {
                AnchorError::UnsupportedVersion
            } else {
                AnchorError::Format
            });
        }
        if bytes[128..136] != [0; 8] {
            return Err(AnchorError::Format);
        }
        let anchor = Self {
            epoch: u64::from_le_bytes(bytes[8..16].try_into().expect("fixed epoch")),
            log: LogAnchor {
                index: CommitIndex(u64::from_le_bytes(
                    bytes[16..24].try_into().expect("fixed index"),
                )),
                digest: bytes[24..56].try_into().expect("fixed digest"),
            },
            origin: ChainOrigin(bytes[56..88].try_into().expect("fixed origin")),
            base: LogAnchor {
                index: CommitIndex(u64::from_le_bytes(
                    bytes[88..96].try_into().expect("fixed base index"),
                )),
                digest: bytes[96..128].try_into().expect("fixed base digest"),
            },
        };
        anchor.check_self()?;
        Ok(anchor)
    }
}

/// An anchor file held open for the lifetime of its writer.
///
/// Every mutation validates the requested transition, writes a complete
/// replacement record to a scratch name, synchronizes it, and installs it by
/// rename. A reader therefore observes the previous record or the next one and
/// never a blend of the two.
#[derive(Debug)]
pub struct AnchorStore {
    path: PathBuf,
    key: AnchorKey,
    current: ProtectedAnchor,
}

impl AnchorStore {
    /// Create the first anchor for a log. An existing anchor is never replaced,
    /// so a lost key cannot be papered over by re-initializing.
    ///
    /// The existence check is not a concurrency primitive. Callers serialize
    /// through the log's single-writer lock; see
    /// [`AcknowledgedLedger`](crate::AcknowledgedLedger), which acquires that
    /// lock before touching the anchor.
    pub fn initialize(
        path: impl AsRef<Path>,
        key: AnchorKey,
        anchor: ProtectedAnchor,
    ) -> Result<Self, AnchorError> {
        let path = path.as_ref();
        anchor.check_self()?;
        if path.exists() {
            return Err(AnchorError::Exists);
        }
        let store = Self {
            path: path.to_owned(),
            key,
            current: anchor,
        };
        store.publish(anchor)?;
        Ok(store)
    }

    /// Open an existing anchor and authenticate it under `key`.
    ///
    /// This verifies that the record was produced by a holder of the key and
    /// arrived intact. It does **not** establish freshness: an older record from
    /// the same key is equally authentic. Use [`Self::open_expecting`] wherever
    /// rollback of the anchor file is in scope.
    pub fn open(path: impl AsRef<Path>, key: AnchorKey) -> Result<Self, AnchorError> {
        Self::open_expecting(path, key, 0)
    }

    /// Open an existing anchor, authenticate it, and reject an epoch below
    /// `minimum_epoch`.
    ///
    /// `minimum_epoch` must come from a monotonic counter the host keeps outside
    /// this file — a sealed counter, a secret manager, a replicated register.
    /// Reading it back from the anchor being checked would make the check
    /// vacuous, which is the same mistake as trusting a log's own checksum to
    /// prove the log was not replaced.
    pub fn open_expecting(
        path: impl AsRef<Path>,
        key: AnchorKey,
        minimum_epoch: u64,
    ) -> Result<Self, AnchorError> {
        let path = path.as_ref();
        let bytes = match read_exact_anchor(path) {
            Ok(bytes) => bytes,
            Err(error) => {
                return Err(match error.kind() {
                    io::ErrorKind::NotFound => AnchorError::Absent,
                    io::ErrorKind::InvalidData => AnchorError::Format,
                    _ => AnchorError::Storage,
                })
            }
        };
        let expected = mac(&key, &bytes[..AUTHENTICATED_BYTES]);
        if !constant_time_eq(&bytes[AUTHENTICATED_BYTES..], &expected) {
            return Err(AnchorError::Unauthenticated);
        }
        let current = ProtectedAnchor::decode(&bytes)?;
        if current.epoch < minimum_epoch {
            return Err(AnchorError::StaleEpoch {
                stored: current.epoch,
                expected: minimum_epoch,
            });
        }
        Ok(Self {
            path: path.to_owned(),
            key,
            current,
        })
    }
    /// Returns the backing path.
    pub fn path(&self) -> &Path {
        &self.path
    }
    /// Returns the current protected anchor.
    pub fn current(&self) -> ProtectedAnchor {
        self.current
    }
    /// Returns the current monotonic anchor epoch.
    pub fn epoch(&self) -> u64 {
        self.current.epoch
    }

    /// Acknowledge a longer prefix of the same log, adopting its origin the
    /// first time one exists.
    ///
    /// The covered index may only grow, and restating the covered index is
    /// accepted only when the digest is identical, so an alternative history of
    /// the same length is refused. An already established origin is carried
    /// forward and must match: rebinding it is exactly the substitution the
    /// field exists to detect. Origin adoption shares this transition rather
    /// than following it, so no epoch ever describes a log that holds records
    /// under no origin.
    pub fn acknowledge(
        &mut self,
        log: LogAnchor,
        origin: ChainOrigin,
    ) -> Result<ProtectedAnchor, AnchorError> {
        let next = ProtectedAnchor {
            epoch: self
                .current
                .epoch
                .checked_add(1)
                .ok_or(AnchorError::NonMonotonic)?,
            log,
            origin: if self.current.origin.is_set() {
                self.current.origin
            } else {
                origin
            },
            base: self.current.base,
        };
        self.commit(next)
    }

    /// Raise the compaction floor and the acknowledged prefix together.
    ///
    /// Compaction and acknowledgment are one transition because they must not be
    /// separately observable: an anchor whose floor has moved but whose covered
    /// index has not would describe a log no writer ever produced.
    pub fn advance_compacted(
        &mut self,
        base: LogAnchor,
        log: LogAnchor,
    ) -> Result<ProtectedAnchor, AnchorError> {
        if base.index.0 < self.current.base.index.0 {
            return Err(AnchorError::NonMonotonic);
        }
        let next = ProtectedAnchor {
            epoch: self
                .current
                .epoch
                .checked_add(1)
                .ok_or(AnchorError::NonMonotonic)?,
            log,
            origin: self.current.origin,
            base,
        };
        self.commit(next)
    }

    fn commit(&mut self, next: ProtectedAnchor) -> Result<ProtectedAnchor, AnchorError> {
        next.check_self()?;
        let current = self.current;
        if next.epoch <= current.epoch || next.log.index.0 < current.log.index.0 {
            return Err(AnchorError::NonMonotonic);
        }
        if next.log.index == current.log.index && next.log.digest != current.log.digest {
            return Err(AnchorError::NonMonotonic);
        }
        if next.origin != current.origin && current.origin.is_set() {
            return Err(AnchorError::OriginMismatch);
        }
        self.publish(next)?;
        self.current = next;
        Ok(next)
    }

    /// Install a complete record by atomic rename.
    ///
    /// A crash before the rename leaves the previous record authoritative and a
    /// scratch file that the next publication overwrites; a crash after it
    /// leaves the new record. On Unix the parent directory is synchronized so the
    /// new name survives. Other platforms synchronize the file only, and no
    /// hardware power-loss guarantee is claimed on any platform.
    fn publish(&self, anchor: ProtectedAnchor) -> Result<(), AnchorError> {
        let mut bytes = [0u8; ANCHOR_BYTES];
        let authenticated = anchor.encode();
        bytes[..AUTHENTICATED_BYTES].copy_from_slice(&authenticated);
        bytes[AUTHENTICATED_BYTES..].copy_from_slice(&mac(&self.key, &authenticated));
        write_atomically(&self.path, &bytes).map_err(|_| AnchorError::Storage)
    }
}

fn scratch_path(path: &Path) -> PathBuf {
    let mut name = path.file_name().unwrap_or_default().to_os_string();
    name.push(".publishing");
    path.with_file_name(name)
}

fn write_atomically(path: &Path, bytes: &[u8]) -> io::Result<()> {
    let scratch = scratch_path(path);
    // A scratch file left by an interrupted publication is never authoritative,
    // so replacing it is safe and keeps recovery free of manual cleanup.
    let mut file = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(&scratch)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    drop(file);
    fs::rename(&scratch, path)?;
    crate::file::sync_parent(path)
}

fn read_exact_anchor(path: &Path) -> io::Result<Vec<u8>> {
    let mut file = File::open(path)?;
    if file.metadata()?.len() != ANCHOR_BYTES as u64 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            AnchorError::Format.code(),
        ));
    }
    let mut bytes = Vec::with_capacity(ANCHOR_BYTES);
    file.read_to_end(&mut bytes)?;
    if bytes.len() != ANCHOR_BYTES {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            AnchorError::Format.code(),
        ));
    }
    Ok(bytes)
}

fn mac(key: &AnchorKey, message: &[u8]) -> [u8; 32] {
    let mut domained = Vec::with_capacity(MAC_DOMAIN.len() + message.len());
    domained.extend_from_slice(MAC_DOMAIN);
    domained.extend_from_slice(message);
    hmac_sha256(&key.0, &domained)
}

/// HMAC-SHA256 per RFC 2104, verified against the RFC 4231 vectors in this
/// module's tests. Implemented here so anchor authentication adds no dependency
/// outside the vendor-patch policy.
fn hmac_sha256(key: &[u8], message: &[u8]) -> [u8; 32] {
    let mut block = [0u8; BLOCK_BYTES];
    if key.len() > BLOCK_BYTES {
        block[..32].copy_from_slice(&<[u8; 32]>::from(Sha256::digest(key)));
    } else {
        block[..key.len()].copy_from_slice(key);
    }
    let mut inner_key = [0u8; BLOCK_BYTES];
    let mut outer_key = [0u8; BLOCK_BYTES];
    for (index, byte) in block.iter().enumerate() {
        inner_key[index] = byte ^ 0x36;
        outer_key[index] = byte ^ 0x5c;
    }
    let mut inner = Sha256::new();
    inner.update(inner_key);
    inner.update(message);
    let inner_digest = <[u8; 32]>::from(inner.finalize());
    let mut outer = Sha256::new();
    outer.update(outer_key);
    outer.update(inner_digest);
    <[u8; 32]>::from(outer.finalize())
}

/// Compare without an early exit, so a rejected MAC does not report how much of
/// it matched.
fn constant_time_eq(left: &[u8], right: &[u8]) -> bool {
    if left.len() != right.len() {
        return false;
    }
    let difference = left
        .iter()
        .zip(right)
        .fold(0u8, |accumulated, (a, b)| accumulated | (a ^ b));
    difference == 0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn decode_hex(text: &str) -> Vec<u8> {
        text.as_bytes()
            .chunks(2)
            .map(|pair| {
                let digits = std::str::from_utf8(pair).expect("ascii hex");
                u8::from_str_radix(digits, 16).expect("hex byte")
            })
            .collect()
    }

    /// Verifies that hmac matches rfc 4231 vectors.
    #[test]
    fn hmac_matches_rfc_4231_vectors() {
        // Cases 1, 2, 3 and 6; case 6 exercises the key-longer-than-block path.
        let cases: [(Vec<u8>, Vec<u8>, &str); 4] = [
            (
                vec![0x0b; 20],
                b"Hi There".to_vec(),
                "b0344c61d8db38535ca8afceaf0bf12b881dc200c9833da726e9376c2e32cff7",
            ),
            (
                b"Jefe".to_vec(),
                b"what do ya want for nothing?".to_vec(),
                "5bdcc146bf60754e6a042426089575c75a003f089d2739839dec58b964ec3843",
            ),
            (
                vec![0xaa; 20],
                vec![0xdd; 50],
                "773ea91e36800e46854db8ebd09181a72959098b3ef8c122d9635514ced565fe",
            ),
            (
                vec![0xaa; 131],
                b"Test Using Larger Than Block-Size Key - Hash Key First".to_vec(),
                "60e431591ee0b67f0d8a26aacbf5b77f8e0bc6213728c5140546040f0ee37f54",
            ),
        ];
        for (key, message, expected) in cases {
            assert_eq!(hmac_sha256(&key, &message).as_slice(), decode_hex(expected));
        }
    }

    /// Verifies that constant time eq rejects length and content differences.
    #[test]
    fn constant_time_eq_rejects_length_and_content_differences() {
        assert!(constant_time_eq(&[1, 2, 3], &[1, 2, 3]));
        assert!(!constant_time_eq(&[1, 2, 3], &[1, 2, 4]));
        assert!(!constant_time_eq(&[1, 2, 3], &[1, 2]));
    }

    /// Verifies that redacted debug hides key material.
    #[test]
    fn redacted_debug_hides_key_material() {
        let rendered = format!("{:?}", AnchorKey::from_bytes([7; 32]));
        assert_eq!(rendered, "AnchorKey(<redacted>)");
        assert!(!rendered.contains('7'));
    }

    /// Verifies that reserved flags and base range are rejected.
    #[test]
    fn reserved_flags_and_base_range_are_rejected() {
        let anchor = ProtectedAnchor::initial();
        let mut bytes = [0u8; ANCHOR_BYTES];
        bytes[..AUTHENTICATED_BYTES].copy_from_slice(&anchor.encode());
        assert_eq!(ProtectedAnchor::decode(&bytes), Ok(anchor));

        let mut flagged = bytes;
        flagged[128] = 1;
        assert_eq!(ProtectedAnchor::decode(&flagged), Err(AnchorError::Format));

        let mut short = bytes.to_vec();
        short.pop();
        assert_eq!(ProtectedAnchor::decode(&short), Err(AnchorError::Format));

        let floating_base = ProtectedAnchor {
            base: LogAnchor {
                index: CommitIndex(9),
                digest: [1; 32],
            },
            ..anchor
        };
        assert_eq!(floating_base.check_self(), Err(AnchorError::BaseRange));
    }
}
