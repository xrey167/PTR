//! The wire format for a batch of raft messages.
//!
//! One frame carries a batch rather than a single message because a step can
//! produce several, and a transport that takes them one at a time turns one
//! decision into several round trips whose interleaving nobody asked for.
//!
//! Every field is length-prefixed and bounded, and a frame that disagrees with
//! itself is refused rather than partially read. A raft message is a claim about a
//! term and a log position; half of one claims nothing.
use std::fmt;

/// Marks a raft batch. A foreign frame is refused before anything is interpreted.
const MAGIC: &[u8; 8] = b"PTRRAFTW";

/// The only frame layout this build writes and reads.
pub const FORMAT_V1: u16 = 1;

/// Bound on one frame. A peer does not get to decide how much a receiver reads.
pub const MAX_FRAME_BYTES: usize = 32 * 1024 * 1024;

/// Bound on the number of messages in one batch.
pub const MAX_BATCH_MESSAGES: usize = 4096;

/// Why a frame was refused.
///
/// Every variant is a refusal, and none of them yields a partial batch: a caller
/// that stepped half a frame would have applied messages whose companions it
/// rejected.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum FrameError {
    /// The bytes do not begin with the raft-batch magic.
    NotABatch,
    /// A frame layout this build does not know, rather than one parsed anyway.
    UnknownFormat { format: u16 },
    /// The frame ends inside a field.
    Truncated { field: &'static str },
    /// Bytes after the last message: the writer and the reader disagree.
    TrailingBytes { extra: usize },
    /// The frame, or one message in it, is over the bound.
    TooLarge { bytes: usize, limit: usize },
    /// More messages than one batch may carry.
    TooManyMessages { count: usize, limit: usize },
    /// A message that does not decode as a raft message.
    Undecodable { position: usize, reason: String },
}

impl FrameError {
    /// Stable diagnostic code for this refusal.
    pub fn code(&self) -> &'static str {
        match self {
            Self::NotABatch => "PTR_RAFTW_NOT_A_BATCH",
            Self::UnknownFormat { .. } => "PTR_RAFTW_UNKNOWN_FORMAT",
            Self::Truncated { .. } => "PTR_RAFTW_TRUNCATED",
            Self::TrailingBytes { .. } => "PTR_RAFTW_TRAILING_BYTES",
            Self::TooLarge { .. } => "PTR_RAFTW_TOO_LARGE",
            Self::TooManyMessages { .. } => "PTR_RAFTW_TOO_MANY_MESSAGES",
            Self::Undecodable { .. } => "PTR_RAFTW_UNDECODABLE",
        }
    }
}

impl fmt::Display for FrameError {
    /// Render the stable refusal code.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.code())
    }
}

impl std::error::Error for FrameError {}

/// Encode a batch of already-encoded messages.
///
/// The messages arrive encoded because encoding one is `ptr-ledger`'s business:
/// this crate frames what it is handed and does not interpret it.
pub fn encode_batch(messages: &[Vec<u8>]) -> Result<Vec<u8>, FrameError> {
    if messages.len() > MAX_BATCH_MESSAGES {
        return Err(FrameError::TooManyMessages {
            count: messages.len(),
            limit: MAX_BATCH_MESSAGES,
        });
    }
    let mut frame = Vec::with_capacity(16 + messages.iter().map(|m| m.len() + 4).sum::<usize>());
    frame.extend_from_slice(MAGIC);
    frame.extend_from_slice(&FORMAT_V1.to_le_bytes());
    frame.extend_from_slice(&(messages.len() as u32).to_le_bytes());
    for message in messages {
        frame.extend_from_slice(&(message.len() as u32).to_le_bytes());
        frame.extend_from_slice(message);
    }
    if frame.len() > MAX_FRAME_BYTES {
        return Err(FrameError::TooLarge {
            bytes: frame.len(),
            limit: MAX_FRAME_BYTES,
        });
    }
    Ok(frame)
}

/// Decode a batch into the encoded messages it carries.
pub fn decode_batch(bytes: &[u8]) -> Result<Vec<Vec<u8>>, FrameError> {
    if bytes.len() > MAX_FRAME_BYTES {
        return Err(FrameError::TooLarge {
            bytes: bytes.len(),
            limit: MAX_FRAME_BYTES,
        });
    }
    if bytes.len() < MAGIC.len() || &bytes[..MAGIC.len()] != MAGIC {
        return Err(FrameError::NotABatch);
    }
    let mut at = MAGIC.len();
    let format = take_u16(bytes, &mut at, "format")?;
    if format != FORMAT_V1 {
        return Err(FrameError::UnknownFormat { format });
    }
    let count = take_u32(bytes, &mut at, "count")? as usize;
    if count > MAX_BATCH_MESSAGES {
        return Err(FrameError::TooManyMessages {
            count,
            limit: MAX_BATCH_MESSAGES,
        });
    }
    let mut messages = Vec::with_capacity(count.min(64));
    for _ in 0..count {
        let length = take_u32(bytes, &mut at, "message_length")? as usize;
        let end = at
            .checked_add(length)
            .ok_or(FrameError::Truncated { field: "message" })?;
        let message = bytes
            .get(at..end)
            .ok_or(FrameError::Truncated { field: "message" })?;
        messages.push(message.to_vec());
        at = end;
    }
    if at != bytes.len() {
        return Err(FrameError::TrailingBytes {
            extra: bytes.len() - at,
        });
    }
    Ok(messages)
}

/// Read a little-endian `u16`.
fn take_u16(bytes: &[u8], at: &mut usize, field: &'static str) -> Result<u16, FrameError> {
    let slice = bytes
        .get(*at..*at + 2)
        .ok_or(FrameError::Truncated { field })?;
    *at += 2;
    Ok(u16::from_le_bytes([slice[0], slice[1]]))
}

/// Read a little-endian `u32`.
fn take_u32(bytes: &[u8], at: &mut usize, field: &'static str) -> Result<u32, FrameError> {
    let slice = bytes
        .get(*at..*at + 4)
        .ok_or(FrameError::Truncated { field })?;
    *at += 4;
    Ok(u32::from_le_bytes([slice[0], slice[1], slice[2], slice[3]]))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_batch_round_trips_and_an_empty_one_is_a_batch() {
        let messages = vec![vec![1_u8, 2, 3], vec![], vec![9; 128]];
        let frame = encode_batch(&messages).unwrap();
        assert_eq!(decode_batch(&frame).unwrap(), messages);
        // An empty batch is well formed: "I have nothing to send" is an answer.
        let empty = encode_batch(&[]).unwrap();
        assert!(decode_batch(&empty).unwrap().is_empty());
    }

    #[test]
    fn every_prefix_of_a_frame_is_refused() {
        let frame = encode_batch(&[vec![7; 20], vec![8; 20]]).unwrap();
        for length in 0..frame.len() {
            let error = decode_batch(&frame[..length]).expect_err("a prefix is not a frame");
            assert!(
                matches!(error, FrameError::NotABatch | FrameError::Truncated { .. }),
                "prefix of {length}: {error:?}"
            );
        }
        decode_batch(&frame).expect("the whole frame still decodes");
    }

    #[test]
    fn foreign_bytes_trailing_bytes_and_an_unknown_format_are_each_refused() {
        assert_eq!(
            decode_batch(b"PTRCKPT\x00rest").expect_err("another format's magic"),
            FrameError::NotABatch
        );

        let mut trailing = encode_batch(&[vec![1, 2]]).unwrap();
        trailing.extend_from_slice(&[0, 0, 0]);
        assert_eq!(
            decode_batch(&trailing).expect_err("bytes after the batch"),
            FrameError::TrailingBytes { extra: 3 }
        );

        let mut future = encode_batch(&[vec![1, 2]]).unwrap();
        future[8..10].copy_from_slice(&2_u16.to_le_bytes());
        assert_eq!(
            decode_batch(&future).expect_err("a layout this build does not know"),
            FrameError::UnknownFormat { format: 2 }
        );
    }

    #[test]
    fn a_count_a_peer_invented_does_not_allocate_for_it() {
        // The count is a claim, not an instruction. A frame that says it carries a
        // million messages and does not is refused, and the refusal comes from the
        // bound rather than from running out of bytes a million times.
        let mut lying = encode_batch(&[vec![1]]).unwrap();
        lying[10..14].copy_from_slice(&(MAX_BATCH_MESSAGES as u32 + 1).to_le_bytes());
        assert_eq!(
            decode_batch(&lying).expect_err("more than a batch may carry"),
            FrameError::TooManyMessages {
                count: MAX_BATCH_MESSAGES + 1,
                limit: MAX_BATCH_MESSAGES,
            }
        );

        // A plausible count with nothing behind it runs out of bytes instead.
        let mut short = encode_batch(&[vec![1]]).unwrap();
        short[10..14].copy_from_slice(&64_u32.to_le_bytes());
        assert!(matches!(
            decode_batch(&short).expect_err("sixty-four messages are not there"),
            FrameError::Truncated { .. }
        ));
    }

    #[test]
    fn a_frame_over_the_bound_is_refused_before_it_is_parsed() {
        let error = decode_batch(&vec![0; MAX_FRAME_BYTES + 1]).expect_err("over the bound");
        assert_eq!(
            error,
            FrameError::TooLarge {
                bytes: MAX_FRAME_BYTES + 1,
                limit: MAX_FRAME_BYTES,
            }
        );
    }

    #[test]
    fn each_refusal_carries_its_own_code() {
        let errors = [
            FrameError::NotABatch,
            FrameError::UnknownFormat { format: 2 },
            FrameError::Truncated { field: "count" },
            FrameError::TrailingBytes { extra: 1 },
            FrameError::TooLarge { bytes: 1, limit: 0 },
            FrameError::TooManyMessages { count: 1, limit: 0 },
            FrameError::Undecodable {
                position: 0,
                reason: "x".to_owned(),
            },
        ];
        let mut codes: Vec<&str> = errors.iter().map(FrameError::code).collect();
        let total = codes.len();
        codes.sort_unstable();
        codes.dedup();
        assert_eq!(codes.len(), total, "two refusals share a code");
    }
}
