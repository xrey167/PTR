use crate::error::FastMemoryError;

/// Largest supported number of heads.
pub const MAX_HEADS: usize = 64;
/// Largest supported key or value width per head.
pub const MAX_HEAD_DIM: usize = 1024;
/// Largest supported state, in `f32` cells. 16 Mi cells are 64 MiB, the same
/// bound `ptr-runtime` places on an opaque neural-state payload, so a state this
/// crate accepts is always one the admission layer can seal. A
/// [`crate::SeededProjection`]'s rows are held to the same number of cells.
pub const MAX_STATE_CELLS: usize = 16 * 1024 * 1024;
/// Largest supported distance between two checkpoints, in writes.
pub const MAX_CHECKPOINT_INTERVAL: u32 = 1_000_000;
/// Largest supported journal. A memory's state depends on every write in its
/// journal, and a neural-state binding lists every input it depends on; 65,536
/// is the binding item bound, so a memory can always be declared in full.
pub const MAX_WRITES: u32 = 65_536;
/// Largest magnitude a written value cell may have: `2^24`.
///
/// Admission checks it per write, independently of the state, so that no fold
/// of admitted writes can leave `f32`'s range. A write grows a head's Frobenius
/// norm by at most `beta * ||v||` (the rest of the update is a contraction), and
/// a head's value slice of at most [`MAX_HEAD_DIM`] cells has `||v|| <= 2^5 *
/// 2^24`, so [`MAX_WRITES`] writes keep every head below `2^16 * 2^29 = 2^45`
/// in exact arithmetic; `f32` rounding over the longest journal multiplies that
/// by less than `2^6`. Every cell, readout and squared error a fold computes
/// therefore stays below `2^112`, well inside `f32`'s `2^128`. Because the bound
/// does not depend on the state, every order and every subset of admitted
/// writes folds to finite cells: a write, a restore of its journal and a refold
/// after revocation admit exactly the same writes, and every checkpoint they
/// produce can be decoded again.
pub const MAX_VALUE_MAGNITUDE: f32 = 16_777_216.0;

/// Shape of one fast-weight memory.
///
/// The state is `heads` independent matrices of `key_dim x value_dim`. A write
/// associates a unit key with a value in each head; a read multiplies a query
/// through each matrix.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FastMemoryConfig {
    pub heads: usize,
    pub key_dim: usize,
    pub value_dim: usize,
    /// A full copy of the state is kept after every `checkpoint_interval`
    /// writes. Revocation replays at most this many writes before the first
    /// revoked one, plus every write after it.
    pub checkpoint_interval: u32,
    /// Most writes the journal may hold. A memory that needs more is sharded
    /// into bounded windows, each declared and rebuilt on its own.
    pub max_writes: u32,
}

impl FastMemoryConfig {
    /// Number of `f32` cells in the whole state.
    pub fn state_cells(&self) -> usize {
        self.heads * self.key_dim * self.value_dim
    }

    /// Length of a key or query vector: one unit key per head, concatenated.
    pub fn key_len(&self) -> usize {
        self.heads * self.key_dim
    }

    /// Length of a value or readout vector: one value per head, concatenated.
    pub fn value_len(&self) -> usize {
        self.heads * self.value_dim
    }
}

/// Refuse a configuration outside the supported ranges.
pub fn check_config(config: &FastMemoryConfig) -> Result<(), FastMemoryError> {
    check_range("heads", config.heads, MAX_HEADS)?;
    check_range("key_dim", config.key_dim, MAX_HEAD_DIM)?;
    check_range("value_dim", config.value_dim, MAX_HEAD_DIM)?;
    let cells = config
        .heads
        .checked_mul(config.key_dim)
        .and_then(|cells| cells.checked_mul(config.value_dim));
    match cells {
        Some(cells) if cells <= MAX_STATE_CELLS => {}
        Some(cells) => {
            return Err(FastMemoryError::InvalidConfig {
                field: "state_cells",
                value: cells as u64,
                message: "state exceeds the 64 MiB neural-state payload bound",
            })
        }
        None => {
            return Err(FastMemoryError::InvalidConfig {
                field: "state_cells",
                value: u64::MAX,
                message: "state size overflows",
            })
        }
    }
    if config.max_writes == 0 || config.max_writes > MAX_WRITES {
        return Err(FastMemoryError::InvalidConfig {
            field: "max_writes",
            value: u64::from(config.max_writes),
            message: "journal bound must be in 1..=65_536",
        });
    }
    if config.checkpoint_interval == 0 || config.checkpoint_interval > MAX_CHECKPOINT_INTERVAL {
        return Err(FastMemoryError::InvalidConfig {
            field: "checkpoint_interval",
            value: u64::from(config.checkpoint_interval),
            message: "checkpoint interval must be in 1..=1_000_000",
        });
    }
    Ok(())
}

fn check_range(field: &'static str, value: usize, max: usize) -> Result<(), FastMemoryError> {
    if value == 0 || value > max {
        return Err(FastMemoryError::InvalidConfig {
            field,
            value: value as u64,
            message: "dimension must be at least 1 and within the supported maximum",
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> FastMemoryConfig {
        FastMemoryConfig {
            heads: 4,
            key_dim: 16,
            value_dim: 16,
            checkpoint_interval: 8,
            max_writes: 1024,
        }
    }

    #[test]
    fn a_zero_dimension_is_refused_by_name() {
        let error = check_config(&FastMemoryConfig {
            key_dim: 0,
            ..config()
        })
        .unwrap_err();
        assert!(matches!(
            error,
            FastMemoryError::InvalidConfig {
                field: "key_dim",
                ..
            }
        ));
    }

    #[test]
    fn a_state_beyond_the_payload_bound_is_refused() {
        let error = check_config(&FastMemoryConfig {
            heads: 64,
            key_dim: 1024,
            value_dim: 1024,
            checkpoint_interval: 1,
            max_writes: 1,
        })
        .unwrap_err();
        assert!(matches!(
            error,
            FastMemoryError::InvalidConfig {
                field: "state_cells",
                ..
            }
        ));
    }

    #[test]
    fn a_zero_checkpoint_interval_is_refused() {
        assert!(check_config(&FastMemoryConfig {
            checkpoint_interval: 0,
            ..config()
        })
        .is_err());
        assert!(check_config(&config()).is_ok());
    }
}
