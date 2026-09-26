use crate::config::FastMemoryConfig;
use crate::write::{Decay, MemoryWrite, Query, WriteSeq};

/// The fast-weight matrices of one memory, after a known number of writes.
///
/// Head `h` is a `key_dim x value_dim` matrix `S_h`, stored row-major. A write
/// applies the gated delta rule
///
/// ```text
/// S_h <- (I - beta k_h k_h^T) Diag(alpha_h) S_h + beta k_h v_h^T
/// ```
///
/// with a unit key `k_h`, strength `beta` in `(0, 1]` and decay `alpha_h` in
/// `(0, 1]^key_dim`. Written as an error correction it is
/// `S_h <- A S_h + beta k_h (v_h - (A S_h)^T k_h)^T` with `A = Diag(alpha_h)`:
/// the value already stored under `k_h` is read out and moved towards `v_h`
/// instead of being added to. A read returns `o_h = S_h^T q_h`.
///
/// Every operation is plain IEEE-754 `f32` arithmetic in a fixed order with no
/// fused multiply-add, so the same writes applied to the same starting state
/// produce bit-identical cells on every platform. Exact revocation depends on
/// that.
#[derive(Clone, Debug, PartialEq)]
pub struct FastWeightState {
    config: FastMemoryConfig,
    cells: Vec<f32>,
    applied: WriteSeq,
}

/// One read: `heads * value_dim` values, head by head.
#[derive(Clone, Debug, PartialEq)]
pub struct Readout {
    pub values: Vec<f32>,
    /// The last write this readout reflects.
    pub as_of: WriteSeq,
}

impl FastWeightState {
    /// The empty memory: every matrix is zero and no write has been applied.
    pub(crate) fn empty(config: FastMemoryConfig) -> Self {
        Self {
            config,
            cells: vec![0.0; config.state_cells()],
            applied: WriteSeq(0),
        }
    }

    pub(crate) fn from_parts(config: FastMemoryConfig, cells: Vec<f32>, applied: WriteSeq) -> Self {
        Self {
            config,
            cells,
            applied,
        }
    }

    pub fn config(&self) -> &FastMemoryConfig {
        &self.config
    }

    pub fn cells(&self) -> &[f32] {
        &self.cells
    }

    /// The sequence number of the last write folded into these cells.
    pub fn applied(&self) -> WriteSeq {
        self.applied
    }

    /// Fold one admitted write into the state and return its surprise,
    /// `beta * ||v - (A S)^T k||` summed over heads: how far the memory's
    /// existing answer for this key was from the value written. The caller
    /// guarantees the write was admitted against this state's configuration.
    pub(crate) fn apply(&mut self, write: &MemoryWrite) -> f32 {
        let key_dim = self.config.key_dim;
        let value_dim = self.config.value_dim;
        let block_len = key_dim * value_dim;
        let mut read_back = vec![0.0_f32; value_dim];
        let mut surprise = 0.0_f32;
        for (head, block) in self.cells.chunks_mut(block_len).enumerate() {
            let key = &write.key()[head * key_dim..(head + 1) * key_dim];
            let value = &write.value()[head * value_dim..(head + 1) * value_dim];
            decay_block(block, write.decay(), head, key_dim, value_dim);

            // read_back = (A S)^T k: what the decayed state already returns for k.
            read_back.iter_mut().for_each(|cell| *cell = 0.0);
            for (row, &k) in block.chunks(value_dim).zip(key) {
                for (out, &cell) in read_back.iter_mut().zip(row) {
                    *out += k * cell;
                }
            }

            let error_norm = value
                .iter()
                .zip(&read_back)
                .map(|(target, current)| (target - current) * (target - current))
                .sum::<f32>()
                .sqrt();
            surprise += write.beta() * error_norm;

            // S += beta k (v - read_back)^T
            for (row, &k) in block.chunks_mut(value_dim).zip(key) {
                let scale = write.beta() * k;
                for ((cell, &target), &current) in row.iter_mut().zip(value).zip(&read_back) {
                    *cell += scale * (target - current);
                }
            }
        }
        self.applied = write.seq();
        surprise
    }

    /// `o_h = S_h^T q_h` for every head. Crate-internal: callers read through
    /// [`crate::FastMemory::read_admitted`], which checks the query's head
    /// shape against this configuration and every input first.
    pub(crate) fn read(&self, query: &Query) -> Readout {
        let key_dim = self.config.key_dim;
        let value_dim = self.config.value_dim;
        let mut values = vec![0.0_f32; self.config.value_len()];
        for ((block, out), q) in self
            .cells
            .chunks(key_dim * value_dim)
            .zip(values.chunks_mut(value_dim))
            .zip(query.key().chunks(key_dim))
        {
            for (row, &weight) in block.chunks(value_dim).zip(q) {
                for (cell_out, &cell) in out.iter_mut().zip(row) {
                    *cell_out += weight * cell;
                }
            }
        }
        Readout {
            values,
            as_of: self.applied,
        }
    }
}

fn decay_block(block: &mut [f32], decay: &Decay, head: usize, key_dim: usize, value_dim: usize) {
    match decay {
        Decay::None => {}
        Decay::Scalar(factor) => block.iter_mut().for_each(|cell| *cell *= factor),
        Decay::PerChannel(factors) => {
            let head_factors = &factors[head * key_dim..(head + 1) * key_dim];
            for (row, factor) in block.chunks_mut(value_dim).zip(head_factors) {
                row.iter_mut().for_each(|cell| *cell *= factor);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::write::{admit_write, SourceRef, WriteRequest};
    use ptr_types::Generation;

    fn config() -> FastMemoryConfig {
        FastMemoryConfig {
            heads: 1,
            key_dim: 2,
            value_dim: 2,
            checkpoint_interval: 16,
            max_writes: 1024,
        }
    }

    fn write(seq: u64, key: [f32; 2], value: [f32; 2], beta: f32, decay: Decay) -> MemoryWrite {
        admit_write(
            &config(),
            WriteSeq(seq),
            WriteRequest {
                source: SourceRef {
                    key: format!("s{seq}"),
                    generation: Generation(1),
                    input_digest: [0; 32],
                },
                key: key.to_vec(),
                value: value.to_vec(),
                beta,
                decay,
            },
        )
        .unwrap()
    }

    fn query(key: [f32; 2]) -> Query {
        Query::new(&config(), key.to_vec()).unwrap()
    }

    #[test]
    fn surprise_is_the_scaled_error_the_write_corrected() {
        let mut state = FastWeightState::empty(config());
        let first = state.apply(&write(1, [1.0, 0.0], [3.0, 4.0], 1.0, Decay::None));
        assert_eq!(first, 5.0);
        let repeat = state.apply(&write(2, [1.0, 0.0], [3.0, 4.0], 0.5, Decay::None));
        assert_eq!(repeat, 0.0);
    }

    #[test]
    fn a_full_strength_write_is_recalled_exactly_under_its_key() {
        let mut state = FastWeightState::empty(config());
        state.apply(&write(1, [1.0, 0.0], [3.0, -2.0], 1.0, Decay::None));
        assert_eq!(state.read(&query([1.0, 0.0])).values, vec![3.0, -2.0]);
        assert_eq!(state.read(&query([0.0, 1.0])).values, vec![0.0, 0.0]);
    }

    #[test]
    fn a_second_full_strength_write_under_the_same_key_replaces_rather_than_adds() {
        // This is what separates the delta rule from a Hebbian sum: the second
        // write reads back the first value and corrects it.
        let mut state = FastWeightState::empty(config());
        state.apply(&write(1, [1.0, 0.0], [3.0, -2.0], 1.0, Decay::None));
        state.apply(&write(2, [1.0, 0.0], [5.0, 7.0], 1.0, Decay::None));
        assert_eq!(state.read(&query([1.0, 0.0])).values, vec![5.0, 7.0]);
    }

    #[test]
    fn a_partial_write_moves_the_stored_value_by_beta_of_the_error() {
        let mut state = FastWeightState::empty(config());
        state.apply(&write(1, [1.0, 0.0], [4.0, 0.0], 1.0, Decay::None));
        state.apply(&write(2, [1.0, 0.0], [8.0, 0.0], 0.25, Decay::None));
        assert_eq!(state.read(&query([1.0, 0.0])).values, vec![5.0, 0.0]);
    }

    #[test]
    fn orthogonal_keys_do_not_interfere() {
        let mut state = FastWeightState::empty(config());
        state.apply(&write(1, [1.0, 0.0], [1.0, 2.0], 1.0, Decay::None));
        state.apply(&write(2, [0.0, 1.0], [3.0, 4.0], 1.0, Decay::None));
        assert_eq!(state.read(&query([1.0, 0.0])).values, vec![1.0, 2.0]);
        assert_eq!(state.read(&query([0.0, 1.0])).values, vec![3.0, 4.0]);
    }

    #[test]
    fn per_channel_decay_fades_only_the_decayed_key_direction() {
        let mut state = FastWeightState::empty(config());
        state.apply(&write(1, [1.0, 0.0], [2.0, 2.0], 1.0, Decay::None));
        state.apply(&write(2, [0.0, 1.0], [4.0, 4.0], 1.0, Decay::None));
        // Decay channel 0 by half, keep channel 1; write along channel 1 again
        // with its current value so only the decay is visible.
        state.apply(&write(
            3,
            [0.0, 1.0],
            [4.0, 4.0],
            1.0,
            Decay::PerChannel(vec![0.5, 1.0]),
        ));
        assert_eq!(state.read(&query([1.0, 0.0])).values, vec![1.0, 1.0]);
        assert_eq!(state.read(&query([0.0, 1.0])).values, vec![4.0, 4.0]);
    }

    #[test]
    fn the_update_never_grows_the_state_along_a_unit_key_beyond_the_written_value() {
        let mut state = FastWeightState::empty(config());
        for seq in 1..=200 {
            state.apply(&write(
                seq,
                [0.6, 0.8],
                [1.0, -1.0],
                0.9,
                Decay::Scalar(0.99),
            ));
        }
        let recalled = state.read(&query([0.6, 0.8])).values;
        assert!(recalled.iter().all(|value| value.abs() <= 1.0 + 1e-5));
        assert_eq!(state.applied(), WriteSeq(200));
    }
}
