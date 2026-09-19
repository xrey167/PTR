//! Versioned, bounded SHA-256 chained records. Hashes detect corruption; only an
//! independently trusted anchor detects a rewritten or rolled-back whole log.
//! This is not a MAC, signature, distributed fence or secure-erasure mechanism.
use crate::{decode_event, encode_event, CommittedEvent, LedgerEvent};
use ptr_types::CommitIndex;
use sha2::{Digest, Sha256};
use std::io;

pub const LOG_MAGIC: &[u8; 8] = b"PTRLOG02";
const FRAME_MAGIC: &[u8; 8] = b"PTRFR002";
const END_MAGIC: &[u8; 8] = b"PTREND02";
pub const FRAME_HEADER_BYTES: usize = 84;
pub const FRAME_TRAILER_BYTES: usize = 40;
pub const MAX_RECORD_BYTES: usize = 8 * 1024 * 1024;
pub const MAX_LOG_BYTES: usize = 128 * 1024 * 1024;
pub const MAX_RECORDS: usize = 100_000;

/// A commitment to a complete ordered prefix. Retain this outside the log being
/// checked. Accepting an anchor supplied by an untrusted file defeats its purpose.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LogAnchor {
    pub index: CommitIndex,
    pub digest: [u8; 32],
}
impl LogAnchor {
    pub fn empty() -> Self {
        Self {
            index: CommitIndex(0),
            digest: sha256(LOG_MAGIC),
        }
    }
}

#[derive(Debug)]
pub struct VerifiedLog {
    pub(crate) events: Vec<CommittedEvent>,
    pub(crate) anchor: LogAnchor,
}
impl VerifiedLog {
    pub fn events(&self) -> &[CommittedEvent] {
        &self.events
    }
    pub fn anchor(&self) -> LogAnchor {
        self.anchor
    }
    pub fn require_anchor(&self, trusted: LogAnchor) -> io::Result<()> {
        if self.anchor != trusted {
            return Err(invalid("PTR_LOG_ANCHOR_MISMATCH"));
        }
        Ok(())
    }
}

pub fn sha256(bytes: &[u8]) -> [u8; 32] {
    Sha256::digest(bytes).into()
}
fn hash_parts(domain: &[u8], a: &[u8], b: &[u8]) -> [u8; 32] {
    let mut hash = Sha256::new();
    hash.update(domain);
    hash.update(a);
    hash.update(b);
    hash.finalize().into()
}
pub(crate) fn invalid(message: &'static str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}

pub(crate) fn encode_record(
    event: &LedgerEvent,
    previous: LogAnchor,
) -> io::Result<(Vec<u8>, LogAnchor)> {
    let index = previous
        .index
        .0
        .checked_add(1)
        .ok_or_else(|| invalid("PTR_LOG_INDEX_EXHAUSTED"))?;
    if index > MAX_RECORDS as u64 {
        return Err(invalid("PTR_LOG_RECORD_LIMIT"));
    }
    let payload = encode_event(event);
    if payload.is_empty() || payload.len() > MAX_RECORD_BYTES {
        return Err(invalid("PTR_LOG_PAYLOAD_LIMIT"));
    }
    let mut frame = Vec::with_capacity(FRAME_HEADER_BYTES + payload.len() + FRAME_TRAILER_BYTES);
    frame.extend_from_slice(FRAME_MAGIC);
    frame.extend_from_slice(&index.to_le_bytes());
    frame.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    frame.extend_from_slice(&previous.digest);
    let header_hash = hash_parts(b"PTRHDR02", &frame, &[]);
    frame.extend_from_slice(&header_hash);
    let record_hash = hash_parts(b"PTRREC02", &frame, &payload);
    frame.extend_from_slice(&payload);
    frame.extend_from_slice(&record_hash);
    frame.extend_from_slice(END_MAGIC);
    Ok((
        frame,
        LogAnchor {
            index: CommitIndex(index),
            digest: record_hash,
        },
    ))
}

/// Serialize a complete ordered log, with identical framing to FileLedger.
pub fn encode_log(events: &[CommittedEvent]) -> io::Result<Vec<u8>> {
    encode_log_from(events, LogAnchor::empty())
}

/// Serialize a log that continues above a compaction floor.
///
/// `base` must come from protected anchor storage. Deriving it from the bytes
/// being encoded or decoded would let a substituted log choose its own starting
/// point, which is the whole failure this parameter exists to prevent.
pub fn encode_log_from(events: &[CommittedEvent], base: LogAnchor) -> io::Result<Vec<u8>> {
    if events.len() > MAX_RECORDS {
        return Err(invalid("PTR_LOG_RECORD_LIMIT"));
    }
    let mut bytes = LOG_MAGIC.to_vec();
    let mut anchor = base;
    for committed in events {
        let (record, next) = encode_record(&committed.event, anchor)?;
        if committed.index != next.index {
            return Err(invalid("PTR_LOG_INDEX_MISMATCH"));
        }
        if bytes.len().saturating_add(record.len()) > MAX_LOG_BYTES {
            return Err(invalid("PTR_LOG_SIZE_LIMIT"));
        }
        bytes.extend_from_slice(&record);
        anchor = next;
    }
    Ok(bytes)
}

/// Anchor produced by each event of an ordered list continuing above `base`.
///
/// Compaction needs the anchor at a chosen floor, and a `FileLedger` keeps only
/// its tail. Recomputing through the canonical encoder means the floor a cutover
/// commits to is derived the same way the records were written, not by a second
/// implementation that could disagree.
pub fn chain_anchors(events: &[CommittedEvent], base: LogAnchor) -> io::Result<Vec<LogAnchor>> {
    if events.len() > MAX_RECORDS {
        return Err(invalid("PTR_LOG_RECORD_LIMIT"));
    }
    let mut anchors = Vec::with_capacity(events.len());
    let mut anchor = base;
    for committed in events {
        let (_, next) = encode_record(&committed.event, anchor)?;
        if committed.index != next.index {
            return Err(invalid("PTR_LOG_INDEX_MISMATCH"));
        }
        anchors.push(next);
        anchor = next;
    }
    Ok(anchors)
}

pub(crate) struct Prefix {
    pub verified: VerifiedLog,
    pub complete_bytes: usize,
    pub incomplete: bool,
    /// One entry per complete-record boundary, in order, starting with the base
    /// itself: the byte offset just past the record and the anchor it produced.
    /// Anchor-directed tail repair needs the byte position of a specific anchor,
    /// which is not recoverable from the events alone.
    pub milestones: Vec<(usize, LogAnchor)>,
}

impl Prefix {
    /// Byte offset just past the record that produced `anchor`, when this log
    /// actually contains that boundary.
    pub fn offset_of(&self, anchor: LogAnchor) -> Option<usize> {
        self.milestones
            .iter()
            .find(|(_, candidate)| *candidate == anchor)
            .map(|(offset, _)| *offset)
    }

    /// Complete records that follow `anchor`, when this log contains it.
    pub fn records_after(&self, anchor: LogAnchor) -> Option<usize> {
        let position = self
            .milestones
            .iter()
            .position(|(_, candidate)| *candidate == anchor)?;
        Some(self.milestones.len() - 1 - position)
    }
}

/// Internal scanner retains an incomplete *next* frame only for explicit
/// anchor-checked recovery. An invalid complete header/payload is always an error.
pub(crate) fn scan(bytes: &[u8]) -> io::Result<Prefix> {
    scan_from(bytes, LogAnchor::empty())
}

/// Scan a log that continues above a compaction floor. `base` is trusted input
/// from protected anchor storage, never read from `bytes`.
pub(crate) fn scan_from(bytes: &[u8], base: LogAnchor) -> io::Result<Prefix> {
    if bytes.len() > MAX_LOG_BYTES {
        return Err(invalid("PTR_LOG_SIZE_LIMIT"));
    }
    if !bytes.starts_with(LOG_MAGIC) {
        return Err(invalid("PTR_LOG_FORMAT_OR_HEADER"));
    }
    let mut offset = LOG_MAGIC.len();
    let mut events = Vec::new();
    let mut anchor = base;
    let mut milestones = vec![(offset, anchor)];
    while offset < bytes.len() {
        if events.len() >= MAX_RECORDS {
            return Err(invalid("PTR_LOG_RECORD_LIMIT"));
        }
        let remaining = &bytes[offset..];
        let prefix_len = remaining.len().min(FRAME_MAGIC.len());
        if remaining[..prefix_len] != FRAME_MAGIC[..prefix_len] {
            return Err(invalid("PTR_LOG_FRAME_MAGIC"));
        }
        // Even a short tail must agree with all available known chain fields.
        let next = anchor
            .index
            .0
            .checked_add(1)
            .ok_or_else(|| invalid("PTR_LOG_INDEX_EXHAUSTED"))?
            .to_le_bytes();
        for (start, expected) in [(8, next.as_slice()), (20, anchor.digest.as_slice())] {
            if remaining.len() > start {
                let available = (remaining.len() - start).min(expected.len());
                if remaining[start..start + available] != expected[..available] {
                    return Err(invalid("PTR_LOG_CHAIN_OR_ORDER"));
                }
            }
        }
        if remaining.len() < FRAME_HEADER_BYTES {
            break;
        }
        let header = &remaining[..FRAME_HEADER_BYTES];
        if header[52..84] != hash_parts(b"PTRHDR02", &header[..52], &[]) {
            return Err(invalid("PTR_LOG_HEADER_DIGEST"));
        }
        let index = u64::from_le_bytes(header[8..16].try_into().expect("fixed index"));
        if Some(index) != anchor.index.0.checked_add(1) || header[20..52] != anchor.digest {
            return Err(invalid("PTR_LOG_CHAIN_OR_ORDER"));
        }
        let length = u32::from_le_bytes(header[16..20].try_into().expect("fixed length")) as usize;
        if length == 0 || length > MAX_RECORD_BYTES {
            return Err(invalid("PTR_LOG_PAYLOAD_LIMIT"));
        }
        let end = FRAME_HEADER_BYTES + length;
        let total = end + FRAME_TRAILER_BYTES;
        if remaining.len() < total {
            break;
        }
        let payload = &remaining[FRAME_HEADER_BYTES..end];
        let record_hash = hash_parts(b"PTRREC02", header, payload);
        if remaining[end..end + 32] != record_hash || &remaining[end + 32..total] != END_MAGIC {
            return Err(invalid("PTR_LOG_RECORD_DIGEST_OR_TRAILER"));
        }
        let event = decode_event(payload)?;
        // Keep the versioned representation canonical, including future decoders.
        if encode_event(&event) != payload {
            return Err(invalid("PTR_LOG_NONCANONICAL_EVENT"));
        }
        events.push(CommittedEvent {
            index: CommitIndex(index),
            event,
        });
        anchor = LogAnchor {
            index: CommitIndex(index),
            digest: record_hash,
        };
        offset += total;
        milestones.push((offset, anchor));
    }
    Ok(Prefix {
        verified: VerifiedLog { events, anchor },
        complete_bytes: offset,
        incomplete: offset != bytes.len(),
        milestones,
    })
}

/// Strict decoding never repairs input. Truncation at a complete frame boundary
/// additionally requires [`VerifiedLog::require_anchor`] to detect loss of a
/// committed suffix.
pub fn decode_log(bytes: &[u8]) -> io::Result<VerifiedLog> {
    decode_log_from(bytes, LogAnchor::empty())
}

/// Strictly decode a log that continues above a compaction floor.
pub fn decode_log_from(bytes: &[u8], base: LogAnchor) -> io::Result<VerifiedLog> {
    let prefix = scan_from(bytes, base)?;
    if prefix.incomplete {
        return Err(invalid("PTR_LOG_INCOMPLETE_FRAME"));
    }
    Ok(prefix.verified)
}

/// Explicit legacy decoding; never used as an automatic fallback from a corrupt
/// v2 header. Legacy records have no checksum or evidence of authenticity.
pub fn decode_legacy_log(bytes: &[u8]) -> io::Result<Vec<CommittedEvent>> {
    if bytes.len() > MAX_LOG_BYTES {
        return Err(invalid("PTR_LOG_SIZE_LIMIT"));
    }
    let mut offset = 0usize;
    let mut events = Vec::new();
    while offset < bytes.len() {
        if events.len() >= MAX_RECORDS {
            return Err(invalid("PTR_LOG_RECORD_LIMIT"));
        }
        let prefix = bytes
            .get(offset..offset + 4)
            .ok_or_else(|| invalid("PTR_LEGACY_INCOMPLETE"))?;
        let length = u32::from_le_bytes(prefix.try_into().expect("legacy length")) as usize;
        if length == 0 || length > MAX_RECORD_BYTES {
            return Err(invalid("PTR_LOG_PAYLOAD_LIMIT"));
        }
        offset += 4;
        let payload = bytes
            .get(offset..offset + length)
            .ok_or_else(|| invalid("PTR_LEGACY_INCOMPLETE"))?;
        let event = decode_event(payload)?;
        if encode_event(&event) != payload {
            return Err(invalid("PTR_LOG_NONCANONICAL_EVENT"));
        }
        events.push(CommittedEvent {
            index: CommitIndex(events.len() as u64 + 1),
            event,
        });
        offset += length;
    }
    Ok(events)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn sha256_known_answer_and_checked_index() {
        assert_eq!(
            sha256(b"abc"),
            [
                0xba, 0x78, 0x16, 0xbf, 0x8f, 0x01, 0xcf, 0xea, 0x41, 0x41, 0x40, 0xde, 0x5d, 0xae,
                0x22, 0x23, 0xb0, 0x03, 0x61, 0xa3, 0x96, 0x17, 0x7a, 0x9c, 0xb4, 0x10, 0xff, 0x61,
                0xf2, 0x00, 0x15, 0xad
            ]
        );
        let event = LedgerEvent::VerifierAttested {
            subject: "x".into(),
            passed: true,
        };
        assert!(encode_record(
            &event,
            LogAnchor {
                index: CommitIndex(u64::MAX),
                digest: [0; 32]
            }
        )
        .is_err());
    }
}
