use sha2::{Digest, Sha256};

use crate::config::{check_config, FastMemoryConfig};
use crate::error::FastMemoryError;
use crate::state::FastWeightState;
use crate::write::WriteSeq;

/// Magic and layout version of an encoded state.
pub const STATE_MAGIC: &[u8; 8] = b"PTRFW001";

const HEADER_LEN: usize = 8 + 5 * 4 + 8;
const DIGEST_LEN: usize = 32;

/// Encode a state as `PTRFW001` bytes: header, little-endian `f32` cells and a
/// SHA-256 over everything before it.
///
/// Cells are stored at full `f32` precision. A narrower encoding (bf16, f16)
/// would halve the bytes but a checkpoint restored from it would no longer be
/// the fold of its journal, and revocation promises bit-identical refolds.
pub fn encode_state(state: &FastWeightState) -> Vec<u8> {
    let config = state.config();
    let mut bytes = Vec::with_capacity(HEADER_LEN + state.cells().len() * 4 + DIGEST_LEN);
    bytes.extend_from_slice(STATE_MAGIC);
    for dimension in [config.heads, config.key_dim, config.value_dim] {
        bytes.extend_from_slice(&(dimension as u32).to_le_bytes());
    }
    bytes.extend_from_slice(&config.checkpoint_interval.to_le_bytes());
    bytes.extend_from_slice(&config.max_writes.to_le_bytes());
    bytes.extend_from_slice(&state.applied().0.to_le_bytes());
    for cell in state.cells() {
        bytes.extend_from_slice(&cell.to_le_bytes());
    }
    let digest = Sha256::digest(&bytes);
    bytes.extend_from_slice(&digest);
    bytes
}

/// Decode `PTRFW001` bytes, refusing anything that is not exactly one valid,
/// intact state.
pub fn decode_state(bytes: &[u8]) -> Result<FastWeightState, FastMemoryError> {
    if bytes.len() < HEADER_LEN + DIGEST_LEN {
        return Err(FastMemoryError::CorruptState {
            reason: "shorter than header and digest",
        });
    }
    if &bytes[..8] != STATE_MAGIC {
        return Err(FastMemoryError::CorruptState {
            reason: "unknown magic or layout version",
        });
    }
    let config = FastMemoryConfig {
        heads: read_u32(bytes, 8) as usize,
        key_dim: read_u32(bytes, 12) as usize,
        value_dim: read_u32(bytes, 16) as usize,
        checkpoint_interval: read_u32(bytes, 20),
        max_writes: read_u32(bytes, 24),
    };
    check_config(&config)?;
    let applied = WriteSeq(u64::from_le_bytes(
        bytes[28..36].try_into().expect("eight header bytes"),
    ));
    let expected_len = HEADER_LEN + config.state_cells() * 4 + DIGEST_LEN;
    if bytes.len() != expected_len {
        return Err(FastMemoryError::CorruptState {
            reason: "length does not match the declared shape",
        });
    }
    let (body, digest) = bytes.split_at(expected_len - DIGEST_LEN);
    if Sha256::digest(body).as_slice() != digest {
        return Err(FastMemoryError::DigestMismatch);
    }
    let mut cells = Vec::with_capacity(config.state_cells());
    for (index, chunk) in body[HEADER_LEN..].chunks_exact(4).enumerate() {
        let cell = f32::from_le_bytes(chunk.try_into().expect("four cell bytes"));
        if !cell.is_finite() {
            return Err(FastMemoryError::NonFinite {
                field: "cells",
                index,
            });
        }
        cells.push(cell);
    }
    Ok(FastWeightState::from_parts(config, cells, applied))
}

fn read_u32(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(
        bytes[offset..offset + 4]
            .try_into()
            .expect("four header bytes"),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn state() -> FastWeightState {
        let config = FastMemoryConfig {
            heads: 2,
            key_dim: 2,
            value_dim: 3,
            checkpoint_interval: 4,
            max_writes: 64,
        };
        let cells = (0..config.state_cells())
            .map(|index| index as f32 * 0.25 - 1.0)
            .collect();
        FastWeightState::from_parts(config, cells, WriteSeq(7))
    }

    #[test]
    fn a_state_round_trips_bit_for_bit() {
        let original = state();
        let decoded = decode_state(&encode_state(&original)).unwrap();
        assert_eq!(decoded, original);
    }

    #[test]
    fn a_flipped_cell_bit_is_a_digest_mismatch() {
        let mut bytes = encode_state(&state());
        bytes[HEADER_LEN + 5] ^= 0x01;
        assert_eq!(decode_state(&bytes), Err(FastMemoryError::DigestMismatch));
    }

    #[test]
    fn a_truncated_state_is_refused_before_its_digest_is_read() {
        let bytes = encode_state(&state());
        assert!(matches!(
            decode_state(&bytes[..bytes.len() - 1]),
            Err(FastMemoryError::CorruptState { .. })
        ));
    }

    #[test]
    fn a_foreign_magic_is_refused() {
        let mut bytes = encode_state(&state());
        bytes[7] = b'2';
        assert!(matches!(
            decode_state(&bytes),
            Err(FastMemoryError::CorruptState { .. })
        ));
    }
}
