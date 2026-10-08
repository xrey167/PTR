//! The 52-cell invalidation matrix of R006 (case C9). **Not built yet.**
//!
//! The cells of `config.toml` (`invalidation_coverage_matrix`) are the four paths normal, lag,
//! restore and in-flight of six dependency kinds under revocation and supersession, plus the
//! expert fence. They are to reuse the shared matrix machinery of `layered_kv` (owner decision
//! 2), with its own arms on the one-layer backend and the positional sessions; those do not
//! exist in this tree, so the entry point returns [`HarnessError::NotBuilt`] and produces no
//! cell record.

use super::cases::{HarnessError, Part};
use super::params;

/// The names of the matrix cells, in the order of `config.toml`.
pub fn cell_names() -> Vec<&'static str> {
    params::INVALIDATION_COVERAGE_MATRIX_CELLS.to_vec()
}

/// Run every cell for `seed`. Not built yet: always [`HarnessError::NotBuilt`].
pub fn run_all(_seed: u64) -> Result<(), HarnessError> {
    Err(HarnessError::NotBuilt { part: Part::Matrix })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn running_the_matrix_is_a_typed_not_built_error_and_the_cells_are_the_52_of_the_config() {
        assert_eq!(
            run_all(17),
            Err(HarnessError::NotBuilt { part: Part::Matrix })
        );
        let cells = cell_names();
        assert_eq!(cells.len(), 52);
        let unique: std::collections::BTreeSet<&str> = cells.iter().copied().collect();
        assert_eq!(unique.len(), 52);
    }
}
