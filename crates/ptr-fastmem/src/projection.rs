use sha2::{Digest, Sha256};

use crate::config::{MAX_EMBEDDING_DIM, MAX_HEADS, MAX_HEAD_DIM};
use crate::error::FastMemoryError;

/// A fixed linear map from an embedding space into per-head key or value
/// coordinates, with orthonormal rows inside every head.
///
/// No training is involved: rows are drawn from a seeded Rademacher (`±1`)
/// source and orthonormalised with modified Gram-Schmidt in `f64`. Only
/// addition, multiplication, division and square root are used, all of which
/// IEEE-754 rounds exactly, so a seed yields bit-identical rows on every
/// platform. Orthonormal rows preserve inner products within a head up to the
/// Johnson-Lindenstrauss distortion of dropping the remaining directions, which
/// is what lets a frozen embedding model supply keys and values for the delta
/// rule.
#[derive(Clone, Debug, PartialEq)]
pub struct SeededProjection {
    input_dim: usize,
    heads: usize,
    head_dim: usize,
    rows: Vec<f32>,
}

/// Parameters of a projection.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ProjectionSpec {
    pub input_dim: usize,
    pub heads: usize,
    pub head_dim: usize,
    pub seed: u64,
}

impl SeededProjection {
    /// Draw the projection. `head_dim` may not exceed `input_dim`, because more
    /// orthonormal rows than input dimensions do not exist.
    ///
    /// The rows are `heads * head_dim` rows of `input_dim` `f32` cells: at
    /// most `MAX_HEADS * MAX_HEAD_DIM` rows, the longest key or value vector a
    /// configuration admits, from an embedding at most [`MAX_EMBEDDING_DIM`]
    /// wide. Both are checked, the row count with checked arithmetic, before
    /// anything is allocated, so no spec overflows, wraps or requests more
    /// than 512 Mi cells, and every memory `check_config` accepts has a key
    /// and a value projection from every embedding width up to
    /// [`MAX_EMBEDDING_DIM`] that is at least its head width.
    pub fn new(spec: ProjectionSpec) -> Result<Self, FastMemoryError> {
        let cells = check_spec(&spec)?;
        let mut source = SplitMix64(spec.seed);
        let mut rows = Vec::with_capacity(cells);
        for _ in 0..spec.heads {
            let mut basis: Vec<Vec<f64>> = Vec::with_capacity(spec.head_dim);
            while basis.len() < spec.head_dim {
                let mut row: Vec<f64> = (0..spec.input_dim)
                    .map(|_| if source.next() & 1 == 0 { 1.0 } else { -1.0 })
                    .collect();
                for previous in &basis {
                    let projection = dot64(&row, previous);
                    for (cell, &base) in row.iter_mut().zip(previous) {
                        *cell -= projection * base;
                    }
                }
                let norm = dot64(&row, &row).sqrt();
                // A draw that is (numerically) inside the span already found is
                // discarded and redrawn; the next draw is still seed-determined.
                if norm < 1e-6 {
                    continue;
                }
                row.iter_mut().for_each(|cell| *cell /= norm);
                basis.push(row);
            }
            for row in basis {
                rows.extend(row.into_iter().map(|cell| cell as f32));
            }
        }
        Ok(Self {
            input_dim: spec.input_dim,
            heads: spec.heads,
            head_dim: spec.head_dim,
            rows,
        })
    }

    pub fn input_dim(&self) -> usize {
        self.input_dim
    }

    /// `heads * head_dim`.
    pub fn output_len(&self) -> usize {
        self.heads * self.head_dim
    }

    /// Project one embedding.
    pub fn project(&self, embedding: &[f32]) -> Result<Vec<f32>, FastMemoryError> {
        if embedding.len() != self.input_dim {
            return Err(FastMemoryError::DimensionMismatch {
                field: "embedding",
                expected: self.input_dim,
                actual: embedding.len(),
            });
        }
        if let Some(index) = embedding.iter().position(|value| !value.is_finite()) {
            return Err(FastMemoryError::NonFinite {
                field: "embedding",
                index,
            });
        }
        Ok(self
            .rows
            .chunks(self.input_dim)
            .map(|row| row.iter().zip(embedding).map(|(r, e)| r * e).sum())
            .collect())
    }
}

impl SeededProjection {
    /// Digest of the projection's exact rows (their `f32` bit patterns) and
    /// shape. A memory's keys are only meaningful under the projection that
    /// produced them, so a sealed memory binds this digest the way a neural
    /// state binds its codebook fingerprint.
    pub fn digest(&self) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(b"ptr-fastmem/projection/v1");
        for dimension in [self.input_dim, self.heads, self.head_dim] {
            hasher.update((dimension as u64).to_le_bytes());
        }
        for row in &self.rows {
            hasher.update(row.to_bits().to_le_bytes());
        }
        hasher.finalize().into()
    }
}

/// Seeded quasi-orthogonal value codes, one per fact identity.
///
/// A value stored in the memory is a code for *which* fact was written, not an
/// embedding of *what* it says. Embeddings of related facts ("lives in
/// Hamburg", "lives in Munich") are strongly correlated and a readout could not
/// tell them apart; independent `±1/sqrt(n)` codes derived from the fact id are
/// nearly orthogonal, so `<S^T q, code(id)>` estimates the memory's weight on
/// that fact plus crosstalk of standard deviation about
/// `sqrt(sum_j w_j^2 / n)`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct IdentifierCodebook {
    seed: u64,
    len: usize,
}

impl IdentifierCodebook {
    /// Codes of length `len` (the memory's `heads * value_dim`), at most
    /// `MAX_HEADS * MAX_HEAD_DIM`: the longest value vector a configuration
    /// admits. A longer code could never be compared with a readout, and every
    /// [`Self::code`] would allocate it.
    pub fn new(seed: u64, len: usize) -> Result<Self, FastMemoryError> {
        if len == 0 {
            return Err(FastMemoryError::InvalidConfig {
                field: "code_len",
                value: 0,
                message: "codes need at least one component",
            });
        }
        if len > MAX_HEADS * MAX_HEAD_DIM {
            return Err(FastMemoryError::InvalidConfig {
                field: "code_len",
                value: len as u64,
                message: "codes cannot be longer than the longest value vector",
            });
        }
        Ok(Self { seed, len })
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The code of `id`: signs from SHA-256 blocks of `(seed, id, block)`,
    /// scaled to unit length. Integer arithmetic and one square root, so every
    /// platform derives the same bits.
    pub fn code(&self, id: &str) -> Vec<f32> {
        let scale = 1.0 / (self.len as f32).sqrt();
        let mut code = Vec::with_capacity(self.len);
        let mut block: u64 = 0;
        while code.len() < self.len {
            let mut hasher = Sha256::new();
            hasher.update(b"ptr-fastmem/identifier-code/v1");
            hasher.update(self.seed.to_le_bytes());
            hasher.update((id.len() as u64).to_le_bytes());
            hasher.update(id.as_bytes());
            hasher.update(block.to_le_bytes());
            for byte in hasher.finalize() {
                for bit in 0..8 {
                    if code.len() == self.len {
                        break;
                    }
                    let sign = if (byte >> bit) & 1 == 0 { 1.0 } else { -1.0 };
                    code.push(sign * scale);
                }
            }
            block += 1;
        }
        code
    }
}

/// Refuse a spec [`SeededProjection::new`] cannot draw, before anything is
/// allocated, and return its number of `f32` cells.
fn check_spec(spec: &ProjectionSpec) -> Result<usize, FastMemoryError> {
    if spec.input_dim == 0 || spec.heads == 0 || spec.head_dim == 0 {
        return Err(FastMemoryError::InvalidConfig {
            field: "projection",
            value: 0,
            message: "projection dimensions must be positive",
        });
    }
    if spec.head_dim > spec.input_dim {
        return Err(FastMemoryError::InvalidConfig {
            field: "head_dim",
            value: spec.head_dim as u64,
            message: "a head cannot have more orthonormal rows than input dimensions",
        });
    }
    if spec.input_dim > MAX_EMBEDDING_DIM {
        return Err(FastMemoryError::InvalidConfig {
            field: "input_dim",
            value: spec.input_dim as u64,
            message: "embedding is wider than MAX_EMBEDDING_DIM",
        });
    }
    let rows = match spec.heads.checked_mul(spec.head_dim) {
        Some(rows) if rows <= MAX_HEADS * MAX_HEAD_DIM => rows,
        Some(rows) => {
            return Err(FastMemoryError::InvalidConfig {
                field: "projection_rows",
                value: rows as u64,
                message: "projection has more rows than the longest key or value vector",
            })
        }
        None => {
            return Err(FastMemoryError::InvalidConfig {
                field: "projection_rows",
                value: u64::MAX,
                message: "projection size overflows",
            })
        }
    };
    // At most 2^16 rows of 2^13 cells: 2^29 fits a 32-bit usize, which
    // MAX_STATE_CELLS already needs.
    Ok(rows * spec.input_dim)
}

fn dot64(left: &[f64], right: &[f64]) -> f64 {
    left.iter().zip(right).map(|(l, r)| l * r).sum()
}

/// Steele, Lea and Flood's SplitMix64: a small, fully specified generator whose
/// output is defined by integer arithmetic alone.
struct SplitMix64(u64);

impl SplitMix64 {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(seed: u64) -> ProjectionSpec {
        ProjectionSpec {
            input_dim: 12,
            heads: 2,
            head_dim: 4,
            seed,
        }
    }

    #[test]
    fn rows_within_a_head_are_orthonormal() {
        let projection = SeededProjection::new(spec(7)).unwrap();
        for head in projection.rows.chunks(4 * 12) {
            let rows: Vec<&[f32]> = head.chunks(12).collect();
            for (i, left) in rows.iter().enumerate() {
                for (j, right) in rows.iter().enumerate() {
                    let dot: f32 = left.iter().zip(right.iter()).map(|(l, r)| l * r).sum();
                    let expected = if i == j { 1.0 } else { 0.0 };
                    assert!((dot - expected).abs() < 1e-5, "rows {i},{j}: {dot}");
                }
            }
        }
    }

    #[test]
    fn the_same_seed_draws_the_same_rows_and_another_seed_does_not() {
        assert_eq!(
            SeededProjection::new(spec(7)).unwrap(),
            SeededProjection::new(spec(7)).unwrap()
        );
        assert_ne!(
            SeededProjection::new(spec(7)).unwrap(),
            SeededProjection::new(spec(8)).unwrap()
        );
    }

    #[test]
    fn more_rows_than_input_dimensions_is_refused() {
        let error = SeededProjection::new(ProjectionSpec {
            head_dim: 13,
            ..spec(1)
        })
        .unwrap_err();
        assert!(matches!(
            error,
            FastMemoryError::InvalidConfig {
                field: "head_dim",
                ..
            }
        ));
    }

    #[test]
    fn a_projection_wider_than_any_embedding_or_longer_than_any_key_is_refused_before_allocating() {
        let refused = |input_dim, heads, head_dim| {
            SeededProjection::new(ProjectionSpec {
                input_dim,
                heads,
                head_dim,
                seed: 1,
            })
            .unwrap_err()
        };
        // The row product is checked: this spec used to panic on overflow in
        // debug builds and wrap in release builds.
        assert_eq!(
            refused(2, usize::MAX, 2),
            FastMemoryError::InvalidConfig {
                field: "projection_rows",
                value: u64::MAX,
                message: "projection size overflows",
            }
        );
        // One head more than any memory has, at the widest head.
        assert_eq!(
            refused(MAX_HEAD_DIM, MAX_HEADS + 1, MAX_HEAD_DIM),
            FastMemoryError::InvalidConfig {
                field: "projection_rows",
                value: ((MAX_HEADS + 1) * MAX_HEAD_DIM) as u64,
                message: "projection has more rows than the longest key or value vector",
            }
        );
        // One coordinate wider than the widest supported embedding.
        assert_eq!(
            refused(MAX_EMBEDDING_DIM + 1, 1, 1),
            FastMemoryError::InvalidConfig {
                field: "input_dim",
                value: (MAX_EMBEDDING_DIM + 1) as u64,
                message: "embedding is wider than MAX_EMBEDDING_DIM",
            }
        );
    }

    #[test]
    fn every_admitted_memory_has_key_and_value_projections_from_every_supported_width() {
        // Every head count and head width from one to the largest, including
        // shapes whose state is exactly MAX_STATE_CELLS; the projection of
        // keys 1024 wide into 64 heads alone has four times as many cells.
        let sizes = [1, 2, 64, 128, 256, 1024];
        let widths = [1, 64, 384, 768, 1024, 1536, 3072, 4096, MAX_EMBEDDING_DIM];
        let mut checked = 0;
        for heads in [1, 8, MAX_HEADS] {
            for key_dim in sizes {
                for value_dim in sizes {
                    let config = crate::FastMemoryConfig {
                        heads,
                        key_dim,
                        value_dim,
                        checkpoint_interval: 1,
                        max_writes: 1,
                    };
                    if crate::check_config(&config).is_err() {
                        continue;
                    }
                    for head_dim in [key_dim, value_dim] {
                        for input_dim in widths.into_iter().filter(|&width| width >= head_dim) {
                            let spec = ProjectionSpec {
                                input_dim,
                                heads,
                                head_dim,
                                seed: 1,
                            };
                            assert_eq!(
                                check_spec(&spec),
                                Ok(heads * head_dim * input_dim),
                                "{config:?} {spec:?}"
                            );
                            checked += 1;
                        }
                    }
                }
            }
        }
        assert!(checked > 500, "{checked}");
    }

    #[test]
    fn a_projection_with_more_cells_than_a_state_is_drawn() {
        // 2,049 one-row heads from the widest embedding: 16 Mi + 8 Ki cells,
        // more than a state may hold, and cheap to draw because a head of one
        // row is only normalised.
        let spec = ProjectionSpec {
            input_dim: MAX_EMBEDDING_DIM,
            heads: 2049,
            head_dim: 1,
            seed: 5,
        };
        let projection = SeededProjection::new(spec).unwrap();
        assert!(projection.rows.len() > crate::MAX_STATE_CELLS);
        assert_eq!(projection.output_len(), 2049);
        let embedding = vec![1.0; MAX_EMBEDDING_DIM];
        assert_eq!(projection.project(&embedding).unwrap().len(), 2049);
    }

    #[test]
    fn an_identifier_code_longer_than_any_value_vector_is_refused() {
        let longest = MAX_HEADS * MAX_HEAD_DIM;
        assert_eq!(
            IdentifierCodebook::new(9, longest + 1).unwrap_err(),
            FastMemoryError::InvalidConfig {
                field: "code_len",
                value: (longest + 1) as u64,
                message: "codes cannot be longer than the longest value vector",
            }
        );
        assert_eq!(
            IdentifierCodebook::new(9, longest)
                .unwrap()
                .code("fact:a")
                .len(),
            longest
        );
    }

    #[test]
    fn identifier_codes_are_unit_length_reproducible_and_nearly_orthogonal() {
        let book = IdentifierCodebook::new(9, 256).unwrap();
        let a = book.code("fact:a");
        let b = book.code("fact:b");
        let norm: f32 = a.iter().map(|x| x * x).sum();
        assert!((norm - 1.0).abs() < 1e-5);
        assert_eq!(a, book.code("fact:a"));
        let overlap: f32 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
        // Crosstalk of independent sign codes has standard deviation 1/sqrt(n).
        assert!(overlap.abs() < 4.0 / 16.0, "{overlap}");
        assert_ne!(a, IdentifierCodebook::new(10, 256).unwrap().code("fact:a"));
    }

    #[test]
    fn a_projection_digest_names_its_exact_rows() {
        let a = SeededProjection::new(spec(7)).unwrap();
        assert_eq!(a.digest(), SeededProjection::new(spec(7)).unwrap().digest());
        assert_ne!(a.digest(), SeededProjection::new(spec(8)).unwrap().digest());
    }

    #[test]
    fn an_embedding_of_the_wrong_width_is_refused() {
        let projection = SeededProjection::new(spec(3)).unwrap();
        assert!(matches!(
            projection.project(&[1.0; 11]),
            Err(FastMemoryError::DimensionMismatch { .. })
        ));
        assert_eq!(projection.project(&[1.0; 12]).unwrap().len(), 8);
    }
}
